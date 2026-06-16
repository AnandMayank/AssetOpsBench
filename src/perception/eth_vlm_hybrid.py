"""
AssetOpsBench v2 — ETH-VLM Hybrid Gauge Reader (Experiment 2)

Combines ETH's robust needle-angle geometry with VLM scale reading
to fix the OCR domain mismatch on SyncG synthetic images.

ETH pipeline (steps 1-3) correctly detects the needle angle using
YOLO + DINOv2 keypoints + ellipse fitting — none of these depend on
OCR and they work on synthetic images.

ETH step 4 (OCR → scale mapping) is REPLACED by asking a VLM:
  "What are the min and max numbers on this gauge?"

The hybrid then maps: reading = gauge_min + (needle_frac × gauge_range)
where needle_frac comes from ETH geometry, gauge_range from VLM.

Two modes:
  1. Use existing ETH run results + re-ask VLM for scale only
  2. Full run: re-invoke ETH on new images, then VLM for scale

Usage:
    python src/perception/eth_vlm_hybrid.py \\
        --syncg_dir  src/perception/data/syncg \\
        --eth_results src/perception/eth_results/ \\
        --n_samples  50 \\
        --output     src/perception/hybrid_results/
"""

import argparse
import base64
import json
import random
import subprocess
import time
from pathlib import Path

from vlm_gauge_reader import load_syncg_labels, compute_metrics, print_results, encode_image


SCALE_PROMPT = (
    "Look at the numbers printed around the dial of this analog gauge. "
    "What is the MINIMUM value (start of scale) and MAXIMUM value (end of scale)? "
    "Reply with ONLY two numbers separated by a comma: min,max "
    "Example format: 0,100"
)


# ── VLM scale reader ──────────────────────────────────────────────────────────

def vlm_read_scale(image_path: str) -> dict:
    """Ask Gemini to identify the numeric scale range on the gauge face."""
    import os, re, time
    try:
        from google import genai
        from google.genai import types

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")

        client = genai.Client(api_key=api_key)
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        ext = image_path.lower().rsplit(".", 1)[-1]
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"

        consecutive_429 = 0
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type=mime),
                        SCALE_PROMPT,
                    ],
                )
                raw = response.text.strip()
                # Parse "min,max" response
                nums = re.findall(r"-?\d+\.?\d*", raw)
                if len(nums) < 2:
                    raise ValueError(f"Expected 2 numbers, got: {raw!r}")
                return {"min": float(nums[0]), "max": float(nums[1]), "raw": raw, "success": True}

            except Exception as e:
                err = str(e)
                is_429 = "429" in err or "RESOURCE_EXHAUSTED" in err
                is_503 = "503" in err or "UNAVAILABLE" in err
                if is_429:
                    consecutive_429 += 1
                    if consecutive_429 >= 2:
                        return {"min": None, "max": None, "raw": "QUOTA_EXHAUSTED",
                                "success": False, "error": "quota"}
                    time.sleep(5)
                elif is_503 and attempt < 2:
                    time.sleep(10 * (attempt + 1))
                else:
                    return {"min": None, "max": None, "raw": err, "success": False, "error": err}

        return {"min": None, "max": None, "raw": "max retries", "success": False, "error": "max retries"}

    except Exception as e:
        return {"min": None, "max": None, "raw": str(e), "success": False, "error": str(e)}


# ── ETH result parser ─────────────────────────────────────────────────────────

def parse_eth_needle_angle(eth_run_dir: Path, image_stem: str) -> dict:
    """
    Extract needle angle from an existing ETH run's log.

    ETH logs the rotation angle used before OCR:
      'root - INFO - Rotate image by X degrees'
    This is derived from keypoint geometry, not OCR — it's reliable on SyncG.

    Returns {"angle_deg": float, "success": bool}
    """
    log_path = eth_run_dir / "run.log"
    if not log_path.exists():
        return {"angle_deg": None, "success": False, "error": "run.log not found"}

    target = f"Start processing image at path"
    angle_line_prefix = "Rotate image by"
    in_target = False
    angle = None

    try:
        with open(log_path) as f:
            for line in f:
                if image_stem in line and target in line:
                    in_target = True
                    angle = None
                elif in_target and angle_line_prefix in line:
                    # "root - INFO - Rotate image by -180.27 degrees"
                    parts = line.split("Rotate image by")
                    if len(parts) == 2:
                        angle_str = parts[1].strip().split()[0]
                        angle = float(angle_str)
                elif in_target and "Start processing image" in line and image_stem not in line:
                    break  # moved to next image, stop

        if angle is not None:
            return {"angle_deg": angle, "success": True}
        return {"angle_deg": None, "success": False, "error": "angle not found in log"}

    except Exception as e:
        return {"angle_deg": None, "success": False, "error": str(e)}


def find_latest_eth_run(eth_results_dir: str) -> Path | None:
    """Return the most recent ETH run directory."""
    runs = sorted(Path(eth_results_dir).glob("run_*"))
    return runs[-1] if runs else None


# ── Hybrid reading ────────────────────────────────────────────────────────────

# Standard analog gauge sweep: needle travels 270° clockwise from min to max.
# At min value the needle is at ~225° (bottom-left), at max at ~315° (bottom-right).
# ETH's rotation angle is applied to bring the gauge face upright before OCR.
# The relationship: needle_frac ≈ 1 - (eth_angle + 135) / 270  (approximation)
# This is an approximation — real ETH uses ellipse keypoints for exact mapping.
GAUGE_ARC_SPAN = 270.0


def angle_to_fraction(eth_rotation_deg: float) -> float:
    """
    Convert ETH's image rotation angle to a needle position fraction [0, 1].

    ETH rotates the warped gauge so the zero-point is at the bottom.
    The rotation angle encodes where on the arc the needle sits.
    Larger positive rotation → needle further from start → higher reading.
    """
    # Normalize angle to [-180, 180]
    angle = eth_rotation_deg % 360
    if angle > 180:
        angle -= 360
    frac = (angle + GAUGE_ARC_SPAN / 2) / GAUGE_ARC_SPAN
    return max(0.0, min(1.0, frac))


def hybrid_read(image_path: str, eth_run_dir: Path | None, image_stem: str,
                gt_gauge_min: float, gt_gauge_max: float) -> dict:
    """
    Hybrid: ETH needle angle + VLM scale range.
    Falls back to VLM-only if ETH angle is unavailable.
    """
    # Step 1: get needle angle from ETH
    eth_angle = None
    if eth_run_dir:
        eth_result = parse_eth_needle_angle(eth_run_dir, image_stem)
        if eth_result["success"]:
            eth_angle = eth_result["angle_deg"]

    # Step 2: ask VLM for scale range
    scale = vlm_read_scale(image_path)

    if eth_angle is not None and scale["success"]:
        frac = angle_to_fraction(eth_angle)
        reading = scale["min"] + frac * (scale["max"] - scale["min"])
        return {
            "reading":  round(reading, 3),
            "method":   "eth_angle+vlm_scale",
            "eth_angle": eth_angle,
            "vlm_scale": scale,
            "success":  True,
        }

    elif scale["success"]:
        # ETH angle unavailable but scale was read — mark as scale-only (not counted in hybrid metrics)
        return {"reading": None, "method": "eth_angle_missing", "vlm_scale": scale, "success": False,
                "error": "no ETH angle"}

    else:
        return {"reading": None, "method": "all_failed", "success": False, "error": scale.get("error", "unknown")}


# ── Main ──────────────────────────────────────────────────────────────────────

def _load_eth_angles(eth_run_dir: Path) -> dict[str, float]:
    """Parse all needle rotation angles from ETH run.log."""
    import re as _re
    log = eth_run_dir / "run.log"
    angles: dict[str, float] = {}
    cur = None
    for line in log.read_text().splitlines():
        if "Start processing image at path" in line:
            m = _re.search(r"path (.+)$", line)
            if m:
                cur = Path(m.group(1)).stem
        elif cur and "Rotate image by" in line:
            m = _re.search(r"Rotate image by ([-\d.]+)", line)
            if m:
                angles[cur] = float(m.group(1))
                cur = None
    return angles


def run_hybrid_experiment(syncg_dir: str, eth_results_dir: str,
                          n_samples: int, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    eth_run_dir = find_latest_eth_run(eth_results_dir)
    if not eth_run_dir:
        print("No ETH run found. Run ETH pipeline first.")
        return
    print(f"\nUsing ETH run: {eth_run_dir}")

    # Load angles for all images ETH processed
    eth_angles = _load_eth_angles(eth_run_dir)
    print(f"ETH angles available: {len(eth_angles)} images")

    # Load SyncG GT for those stems (need gauge_min/max for the formula)
    all_labels = load_syncg_labels(syncg_dir)
    label_map = {lab["image_name"]: lab for lab in all_labels}

    # Keep only images that have both ETH angle AND SyncG annotation
    stems = [s for s in eth_angles if s in label_map]
    print(f"Images with both ETH angle + SyncG GT: {len(stems)}")
    stems = stems[:n_samples]  # cap at n_samples if needed

    img_dir = Path(syncg_dir) / "syncG" / "syncG" / "images" / "test"

    results = []
    method_counts: dict[str, int] = {}

    for i, stem in enumerate(stems):
        label    = label_map[stem]
        img_path = str(img_dir / (stem + ".jpg"))
        result   = hybrid_read(img_path, eth_run_dir, stem,
                               label["gauge_min"], label["gauge_max"])

        method_counts[result.get("method", "unknown")] = \
            method_counts.get(result.get("method", "unknown"), 0) + 1

        results.append({
            **label,
            "vlm_reading": result.get("reading"),
            "vlm_success": result.get("success", False),
            "method":      result.get("method"),
            "eth_angle":   result.get("eth_angle"),
            "vlm_scale":   result.get("vlm_scale"),
        })

        if (i + 1) % 10 == 0:
            n_ok = sum(1 for r in results if r["vlm_success"])
            print(f"  {i+1}/{len(stems)} done — {n_ok} successful")

        # Gemini free tier: 15 RPM → 5s between calls
        time.sleep(5)

    print(f"\nMethod breakdown: {method_counts}")

    metrics = compute_metrics(results, "eth_vlm_hybrid")
    print_results(metrics)

    # Compare to VLM-only if summary exists
    vlm_summary = Path("src/perception/vlm_results/summary.json")
    if vlm_summary.exists():
        with open(vlm_summary) as f:
            vlm_data = json.load(f)
        claude_acc = vlm_data.get("results", {}).get("claude", {}).get("accuracy_5pct")
        if claude_acc is not None:
            hybrid_acc = metrics.get("accuracy_5pct", 0)
            delta = hybrid_acc - claude_acc
            print(f"\nVLM-only acc@5%:   {claude_acc:.1%}")
            print(f"Hybrid acc@5%:     {hybrid_acc:.1%}")
            print(f"Delta:             {delta:+.1%}  "
                  f"({'hybrid better' if delta > 0 else 'VLM alone better' if delta < 0 else 'no change'})")

    out = Path(output_dir) / "hybrid_results.json"
    with open(out, "w") as f:
        json.dump({"metrics": metrics, "method_counts": method_counts, "results": results},
                  f, indent=2, default=str)
    print(f"\nSaved: {out}")
    print(f"\nThis = Config C accuracy in the AssetOpsBench v2 paper.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETH-VLM hybrid gauge reader — Experiment 2")
    parser.add_argument("--syncg_dir",    default="src/perception/data/syncg")
    parser.add_argument("--eth_results",  default="src/perception/eth_results/")
    parser.add_argument("--n_samples",    type=int, default=50)
    parser.add_argument("--output",       default="src/perception/hybrid_results/")
    args = parser.parse_args()

    run_hybrid_experiment(args.syncg_dir, args.eth_results, args.n_samples, args.output)
