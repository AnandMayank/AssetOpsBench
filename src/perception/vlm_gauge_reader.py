"""
AssetOpsBench v2 — VLM Gauge Reader (Experiment 1)

Tests VLMs directly on SyncG synthetic gauge images.
Compares against exact Blender-computed ground truth.

Fixes ETH OCR domain mismatch: DB_r18 (ICDAR2015 natural scene text)
cannot detect Blender-rendered dial fonts. VLMs generalize across font
distributions without an explicit OCR step.

SyncG annotation fields used:
  ground_truth        — exact reading from Blender scene geometry
  start_value         — gauge_min
  long_interval_value × long_num → gauge_range; gauge_max = min + range
  gauge_type          — temperature / pressure / bar / sf6gas

Usage:
    python src/perception/vlm_gauge_reader.py \\
        --syncg_dir src/perception/data/syncg \\
        --n_samples 50 \\
        --models claude gpt4v \\
        --output src/perception/vlm_results/
"""

import argparse
import base64
import json
import os
import random
import re
import time
from pathlib import Path


# ── Ground truth loader ───────────────────────────────────────────────────────

def load_syncg_labels(syncg_dir: str) -> list[dict]:
    """
    Load SyncG test annotations.

    Actual structure inside the downloaded archive is doubly nested:
      <syncg_dir>/syncG/syncG/annotations/test/*.json
      <syncg_dir>/syncG/syncG/images/test/*.jpg

    Returns list of dicts with keys:
      image_path, image_name, gt_reading,
      gauge_min, gauge_max, gauge_range, gauge_type
    """
    base = Path(syncg_dir) / "syncG" / "syncG"
    ann_dir = base / "annotations" / "test"
    img_dir = base / "images" / "test"

    if not ann_dir.exists():
        raise FileNotFoundError(
            f"Annotation dir not found: {ann_dir}\n"
            f"Expected structure: {syncg_dir}/syncG/syncG/annotations/test/"
        )

    records = []
    for jf in sorted(ann_dir.glob("*.json")):
        with open(jf) as f:
            rec = json.load(f)

        gt = rec.get("ground_truth")
        if gt is None:
            continue

        g_min = float(rec.get("start_value", 0))
        g_range = float(rec.get("long_interval_value", 1)) * float(rec.get("long_num", 1))
        g_max = g_min + g_range

        img_path = img_dir / (jf.stem + ".jpg")
        if not img_path.exists():
            continue

        records.append({
            "image_path":  str(img_path),
            "image_name":  jf.stem,
            "gt_reading":  float(gt),
            "gauge_min":   g_min,
            "gauge_max":   g_max,
            "gauge_range": g_range,
            "gauge_type":  rec.get("gauge_type", "unknown"),
        })

    return records


def encode_image(image_path: str) -> tuple[str, str]:
    """Base64-encode image; return (b64_data, media_type)."""
    ext = Path(image_path).suffix.lower()
    media = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode(), media


# ── VLM readers ───────────────────────────────────────────────────────────────

GAUGE_PROMPT = (
    "This is an analog gauge image. "
    "The scale runs from {min:.4g} to {max:.4g}. "
    "Look at where the needle is pointing and read the exact numeric value. "
    "Reply with ONLY a single number — no units, no explanation."
)

GAUGE_PROMPT_SIMPLE = (
    "What value is the needle pointing to on this gauge? "
    "The scale goes from {min:.4g} to {max:.4g}. "
    "Reply with one number only."
)


def read_with_claude(image_path: str, gauge_min: float, gauge_max: float) -> dict:
    try:
        import anthropic
        client = anthropic.Anthropic()
        img_data, media = encode_image(image_path)

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=64,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media, "data": img_data},
                    },
                    {
                        "type": "text",
                        "text": GAUGE_PROMPT.format(min=gauge_min, max=gauge_max),
                    },
                ],
            }],
        )

        raw = response.content[0].text.strip()
        reading = float(raw.replace(",", "").split()[0])
        return {"reading": reading, "raw": raw, "model": "claude-sonnet-4-6", "success": True}

    except Exception as e:
        return {"reading": None, "raw": str(e), "model": "claude-sonnet-4-6", "success": False, "error": str(e)}


def read_with_gemini(image_path: str, gauge_min: float, gauge_max: float,
                     model: str = "gemini-2.5-flash") -> dict:
    """
    Read gauge using Google Gemini Vision.
    Requires GOOGLE_API_KEY env var.
    Uses google-genai SDK (not deprecated google-generativeai).
    Retries on 503 (server overload) and 429 (rate limit) with backoff.
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return {"reading": None, "raw": "GOOGLE_API_KEY not set", "model": model,
                "success": False, "error": "GOOGLE_API_KEY not set"}

    client = genai.Client(api_key=api_key)

    with open(image_path, "rb") as f:
        img_bytes = f.read()
    ext = Path(image_path).suffix.lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

    consecutive_429 = 0

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type=mime),
                    GAUGE_PROMPT.format(min=gauge_min, max=gauge_max),
                ],
            )
            raw = response.text.strip()
            nums = re.findall(r"-?\d+\.?\d*", raw)
            if not nums:
                raise ValueError(f"No number in response: {raw!r}")
            reading = float(nums[0])
            return {"reading": reading, "raw": raw, "model": model, "success": True}

        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
            is_overload   = "503" in err_str or "UNAVAILABLE" in err_str
            if is_rate_limit:
                consecutive_429 += 1
                # Daily quota exhausted — stop trying immediately
                if consecutive_429 >= 2:
                    return {"reading": None, "raw": "QUOTA_EXHAUSTED", "model": model,
                            "success": False, "error": "Daily quota exceeded"}
                time.sleep(5)
            elif is_overload and attempt < 2:
                time.sleep(10 * (attempt + 1))
            else:
                return {"reading": None, "raw": err_str, "model": model,
                        "success": False, "error": err_str}

    return {"reading": None, "raw": "max retries exceeded", "model": model,
            "success": False, "error": "max retries exceeded"}


def read_with_ollama(image_path: str, gauge_min: float, gauge_max: float,
                     model: str = "moondream") -> dict:
    """
    Read gauge using a locally running Ollama vision model.
    Requires `ollama serve` to be running with the model pulled.
    Works offline — no API key needed.

    Recommended models (pull with: OLLAMA_MODELS=<path> ollama pull <name>):
      moondream   — 1.7 GB, fast, purpose-built for visual QA
      llava:7b    — 4.7 GB, more capable
    """
    try:
        import requests, base64
        img_data, _ = encode_image(image_path)

        payload = {
            "model": model,
            "prompt": GAUGE_PROMPT_SIMPLE.format(min=gauge_min, max=gauge_max),
            "images": [img_data],
            "stream": False,
        }
        resp = requests.post("http://localhost:11434/api/generate",
                             json=payload, timeout=60)
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        # Extract first number from response
        import re
        nums = re.findall(r"-?\d+\.?\d*", raw)
        if not nums:
            raise ValueError(f"No number in response: {raw!r}")
        reading = float(nums[0])
        return {"reading": reading, "raw": raw, "model": model, "success": True}

    except Exception as e:
        return {"reading": None, "raw": str(e), "model": model, "success": False, "error": str(e)}


def read_with_gpt4v(image_path: str, gauge_min: float, gauge_max: float) -> dict:
    try:
        from openai import OpenAI
        client = OpenAI()
        img_data, media = encode_image(image_path)

        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=64,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{media};base64,{img_data}"}},
                    {"type": "text", "text": GAUGE_PROMPT.format(min=gauge_min, max=gauge_max)},
                ],
            }],
        )

        raw = response.choices[0].message.content.strip()
        reading = float(raw.replace(",", "").split()[0])
        return {"reading": reading, "raw": raw, "model": "gpt-4o", "success": True}

    except Exception as e:
        return {"reading": None, "raw": str(e), "model": "gpt-4o", "success": False, "error": str(e)}


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict], model_name: str) -> dict:
    valid = [r for r in results if r.get("vlm_success") and r.get("vlm_reading") is not None]
    invalid = [r for r in results if not r.get("vlm_success")]

    if not valid:
        return {"model": model_name, "n_evaluated": 0, "n_parse_failed": len(invalid), "error": "no valid results"}

    per_image = []
    fm3 = fm4 = 0

    for r in valid:
        gt, pred, rng = r["gt_reading"], r["vlm_reading"], r["gauge_range"]
        abs_err = abs(pred - gt)
        rel_err = abs_err / rng if rng > 0 else 1.0

        per_image.append({
            "image":        r["image_name"],
            "gauge_type":   r.get("gauge_type", "unknown"),
            "gt":           round(gt, 3),
            "pred":         round(pred, 3),
            "abs_err":      round(abs_err, 3),
            "rel_err":      round(rel_err, 4),
            "within_2pct":  rel_err < 0.02,
            "within_5pct":  rel_err < 0.05,
            "within_10pct": rel_err < 0.10,
        })

        if rel_err > 0.50:
            fm3 += 1
        if rng > 0 and (abs(pred - gt * 10) / rng < 0.05 or abs(pred - gt / 10) / rng < 0.05):
            fm4 += 1

    n = len(per_image)
    return {
        "model":          model_name,
        "n_evaluated":    n,
        "n_parse_failed": len(invalid),
        "accuracy_2pct":  round(sum(e["within_2pct"]  for e in per_image) / n, 4),
        "accuracy_5pct":  round(sum(e["within_5pct"]  for e in per_image) / n, 4),
        "accuracy_10pct": round(sum(e["within_10pct"] for e in per_image) / n, 4),
        "mean_rel_error": round(sum(e["rel_err"]       for e in per_image) / n, 4),
        "fm3_rate":       round(fm3 / n, 4),
        "fm4_rate":       round(fm4 / n, 4),
        "worst_5":        sorted(per_image, key=lambda x: x["rel_err"], reverse=True)[:5],
        "best_5":         sorted(per_image, key=lambda x: x["rel_err"])[:5],
        "per_image":      per_image,
    }


def print_results(metrics: dict):
    print(f"\n{'='*58}")
    print(f"Model: {metrics['model']}")
    print(f"{'='*58}")
    if "error" in metrics:
        print(f"ERROR: {metrics['error']}")
        return

    print(f"Evaluated:         {metrics['n_evaluated']} images")
    print(f"Parse failed:      {metrics['n_parse_failed']} images")
    print(f"\nAccuracy:")
    print(f"  Within  2% range: {metrics['accuracy_2pct']:.1%}  (ETH paper target on real gauges)")
    print(f"  Within  5% range: {metrics['accuracy_5pct']:.1%}  (AssetOpsBench v2 threshold)")
    print(f"  Within 10% range: {metrics['accuracy_10pct']:.1%}")
    print(f"  Mean rel error:   {metrics['mean_rel_error']:.1%}")
    print(f"\nFailure modes:")
    print(f"  FM-3 hallucination (>50% err): {metrics['fm3_rate']:.1%}")
    print(f"  FM-4 scale error  (~10× off):  {metrics['fm4_rate']:.1%}")

    acc = metrics["accuracy_5pct"]
    print(f"\n{'='*58}")
    if acc >= 0.85:
        print(f"VERDICT: SUFFICIENT — acc@5% = {acc:.1%}")
        print(f"  VLM reads SyncG gauges well. Config B baseline = {acc:.1%}")
    elif acc >= 0.50:
        print(f"VERDICT: MARGINAL — acc@5% = {acc:.1%}")
        print(f"  VLM works but below SUFFICIENT. Run hybrid experiment.")
        print(f"  MeasureBench 2026 best VLM on real gauges: ~30%")
    else:
        print(f"VERDICT: POOR — acc@5% = {acc:.1%}")
        print(f"  VLMs also struggle with SyncG synthetic fonts.")
        print(f"  Real facility photos likely required for any approach.")
    print(f"{'='*58}")

    if metrics.get("worst_5"):
        print(f"\nWorst 5:")
        for w in metrics["worst_5"]:
            print(f"  {w['image']} [{w['gauge_type']}]: gt={w['gt']}, pred={w['pred']}, err={w['rel_err']:.1%}")
    if metrics.get("best_5"):
        print(f"\nBest 5:")
        for b in metrics["best_5"]:
            print(f"  {b['image']} [{b['gauge_type']}]: gt={b['gt']}, pred={b['pred']}, err={b['rel_err']:.1%}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_experiment(syncg_dir: str, n_samples: int, models: list[str], output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\nLoading SyncG labels from {syncg_dir} ...")
    all_labels = load_syncg_labels(syncg_dir)
    print(f"Found {len(all_labels)} annotated test images")

    if not all_labels:
        print("ERROR: No labels loaded. Check syncg_dir.")
        return

    random.seed(42)
    labels = random.sample(all_labels, min(n_samples, len(all_labels)))
    print(f"Sampled {len(labels)} images (seed=42 for reproducibility)")
    print(f"Sample: {labels[0]['image_name']}  gt={labels[0]['gt_reading']:.3f}  "
          f"range=[{labels[0]['gauge_min']:.1f}, {labels[0]['gauge_max']:.1f}]")

    all_metrics: dict[str, dict] = {}

    for model_name in models:
        if model_name in ("moondream", "llava", "llava-llama3"):
            reader = lambda p, mn, mx, m=model_name: read_with_ollama(p, mn, mx, model=m)
        elif model_name in ("gemini", "gemini-flash", "gemini-pro",
                            "gemini-2.5-flash", "gemini-2.5-flash-lite",
                            "gemini-3.5-flash"):
            gem_model = {"gemini":                 "gemini-2.5-flash",
                         "gemini-flash":           "gemini-2.5-flash",
                         "gemini-pro":             "gemini-2.5-flash",
                         "gemini-2.5-flash":       "gemini-2.5-flash",
                         "gemini-2.5-flash-lite":  "gemini-2.5-flash-lite",
                         "gemini-3.5-flash":       "gemini-3.5-flash"}[model_name]
            reader = lambda p, mn, mx, m=gem_model: read_with_gemini(p, mn, mx, model=m)
        else:
            reader = {"claude": read_with_claude, "gpt4v": read_with_gpt4v}.get(model_name)
        if reader is None:
            print(f"Unknown model '{model_name}' — skipping")
            continue

        print(f"\n{'─'*58}")
        print(f"Running {model_name} on {len(labels)} images ...")
        print(f"{'─'*58}")

        results = []
        for i, label in enumerate(labels):
            vlm = reader(label["image_path"], label["gauge_min"], label["gauge_max"])
            results.append({
                **label,
                "vlm_reading": vlm.get("reading"),
                "vlm_raw":     vlm.get("raw"),
                "vlm_success": vlm.get("success", False),
                "model":       model_name,
            })

            if (i + 1) % 10 == 0:
                n_ok = sum(1 for r in results if r["vlm_success"])
                print(f"  {i+1}/{len(labels)} done — {n_ok} successful reads so far")

            # Gemini free tier: ~15 RPM → 4s minimum; 5s gives headroom
            if model_name.startswith("gemini"):
                time.sleep(5)
            else:
                time.sleep(0.3)

        metrics = compute_metrics(results, model_name)
        all_metrics[model_name] = metrics
        print_results(metrics)

        out = Path(output_dir) / f"{model_name}_results.json"
        with open(out, "w") as f:
            json.dump({"metrics": metrics, "raw_results": results}, f, indent=2, default=str)
        print(f"\nSaved: {out}")

    # Comparison table
    if len(all_metrics) > 1:
        print(f"\n{'='*58}")
        print(f"COMPARISON")
        print(f"{'='*58}")
        print(f"{'Model':<15} {'Acc@5%':<10} {'MeanErr':<10} {'FM-3':<8} {'FM-4':<8}")
        print(f"{'─'*58}")
        for name, m in all_metrics.items():
            if "error" not in m:
                print(f"{name:<15} {m['accuracy_5pct']:.1%}      "
                      f"{m['mean_rel_error']:.1%}       "
                      f"{m['fm3_rate']:.1%}    {m['fm4_rate']:.1%}")
        print(f"\nMeasureBench 2026 best VLM on real gauges: ~30% acc@5%")

    # Per-gauge-type breakdown
    for name, m in all_metrics.items():
        if "per_image" not in m:
            continue
        types: dict[str, list] = {}
        for e in m["per_image"]:
            types.setdefault(e["gauge_type"], []).append(e["rel_err"])
        if len(types) > 1:
            print(f"\n{name} per gauge type:")
            for t, errs in sorted(types.items()):
                print(f"  {t:<20} n={len(errs):3d}  mean={sum(errs)/len(errs):.1%}")

    # Summary JSON
    summary = {
        "experiment":       "VLM gauge reading on SyncG",
        "n_samples":        len(labels),
        "syncg_dir":        syncg_dir,
        "models_tested":    models,
        "eth_baseline":     {"accuracy_5pct": 0.0513, "pipeline_failure_rate": 0.846, "verdict": "insufficient"},
        "results": {
            name: {k: m[k] for k in ("accuracy_2pct", "accuracy_5pct", "accuracy_10pct",
                                      "mean_rel_error", "fm3_rate", "fm4_rate", "n_evaluated", "n_parse_failed")
                   if k in m}
            for name, m in all_metrics.items()
        },
    }
    summary_path = Path(output_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM gauge reader — Experiment 1")
    parser.add_argument("--syncg_dir",  default="src/perception/data/syncg")
    parser.add_argument("--n_samples",  type=int, default=50)
    parser.add_argument("--models",     nargs="+", default=["gemini"],
                        choices=["claude", "gpt4v", "gemini", "gemini-flash",
                                 "gemini-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
                                 "gemini-3.5-flash", "moondream", "llava", "llava-llama3"])
    parser.add_argument("--output",     default="src/perception/vlm_results/")
    args = parser.parse_args()

    run_experiment(args.syncg_dir, args.n_samples, args.models, args.output)
