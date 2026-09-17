"""
VLM labeling for pmc_windows_clean_full/* windows.

Priority order per window:
  1. If any frame matches a perception_real.csv source_file -> use that
     ground truth directly (label_source="ground_truth"), no API call.
  2. Otherwise, call Gemini ONCE on up to 4 sampled frames spanning the whole
     window (first, last, and evenly-spaced in between) -> label_source="vlm".

Why whole-window, not last-frame-only (decided 2026-07-19, see
compare_labeling_strategies.py): this is a photographer-walkthrough dataset,
not a standardized robot approach. Some bursts zoom IN toward the gauge
(last frame = clearest -- e.g. run138_0: distant unreadable gauges -> clean
close-up), others pan AWAY or transition out (last frame = worst). A fixed
"read the last frame" rule is wrong for whichever pattern it doesn't match.
Asking the model to judge across the whole sampled sequence -- success if
ANY frame gives a confident read, matching how a real multi-read
commit_reading loop works -- fixes this, and costs the SAME one API request
per window (Gemini quotas per request, not per image in the request).

Never overwrites a ground_truth label with a VLM call.
"""
import base64
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokenrouter_backend import call_tokenrouter, load_tokenrouter_credentials  # noqa: E402

DEST = Path("/media/adityapachauri/second_drive/aditya_pmc_work")
CLEAN = DEST / "pmc_windows_clean_full"
PERCEPTION_CSV = Path.home() / "AssetOpsBench/src/orchestrator/data/perception_real.csv"
ENV_FILE = Path.home() / "AssetOpsBench/.env"

VALID_CATEGORIES = {
    "gauge_degradation", "occlusion", "scale_interpretation",
    "glare_lighting", "iot_contradiction", "never_read", "clean",
}

EXTRACT_PROMPT = """You are an expert industrial asset inspector reviewing a short burst of photographs (shown in temporal order, first to last) taken while approaching and examining ONE gauge.

The frames may show the camera zooming IN toward the gauge (getting clearer), panning AWAY from it (getting worse), or staying roughly the same. Judge the INSPECTION OUTCOME across the WHOLE sequence, not just the last frame: if ANY frame gives a clear, confident reading of the gauge, the inspection is a SUCCESS even if other frames are blurry or off-target. If NO frame ever gives a clear reading (always occluded, degraded, too small, glare, or the gauge exits the frame entirely), it is a FAILURE.

Respond ONLY with this exact JSON structure, no other text:
{
  "category": "<one of: gauge_degradation | occlusion | scale_interpretation | glare_lighting | iot_contradiction | never_read | clean>",
  "asset": "<industrial asset type the gauge is monitoring>",
  "gauge_readable": true | false,
  "gauge_value": "<numeric value shown by pointer/needle if readable in the best frame, else 'UNREADABLE'>",
  "best_frame_index": <0-based index, among the frames you were shown, giving the clearest evidence for your verdict>,
  "description": "<one sentence: what visual challenge, if any, a robot inspector would face reading this gauge>"
}

Category definitions:
- gauge_degradation: dial face contaminated (soot, oil film, frost, moisture, dust, corrosion)
- occlusion: gauge physically blocked by pipe, cable, structure, or object
- scale_interpretation: gauge too small/far/dim to read graduation marks reliably
- glare_lighting: strong glare, reflections, or thermal shimmer obscuring the dial
- iot_contradiction: two instruments visible showing contradicting values
- never_read: gauge exists but is inaccessible, behind enclosure, or off the inspection route
- clean: gauge is clear, unobstructed, and fully readable
"""


def load_google_api_key():
    key = os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if line.startswith("GOOGLE_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("GOOGLE_API_KEY not found in environment or .env")


GEMINI_REST_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)


def sample_frame_indices(n_frames: int, max_n: int = 4) -> list:
    """Evenly sample up to max_n frame indices, always including first and
    last (so a whole-window call covers the full span, not just one end)."""
    if n_frames <= max_n:
        return list(range(n_frames))
    idx = sorted(set([0, n_frames - 1] +
                      [round(i * (n_frames - 1) / (max_n - 1)) for i in range(max_n)]))
    return idx[:max_n]


def call_gemini(img_paths, api_key: str, retries: int = 4) -> dict:
    """Direct REST call (no SDK) — the installable google-generativeai SDK
    versions that expose GenerativeModel all require Python>=3.9; this repo's
    GPU env (gauge_reader) is Python 3.8. The REST API is version-agnostic.

    img_paths: a single Path (single-image call) or a list of Paths (one
    request carrying multiple images -- whole-window judgment, same quota
    cost as a single image since Gemini bills per request)."""
    import urllib.request
    import urllib.error

    if isinstance(img_paths, (str, Path)):
        img_paths = [img_paths]

    image_parts = []
    for p in img_paths:
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        image_parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64}})

    body = json.dumps({
        "contents": [{
            "parts": image_parts + [{"text": EXTRACT_PROMPT}]
        }]
    }).encode()

    req = urllib.request.Request(
        f"{GEMINI_REST_URL}?key={api_key}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read())
            text = payload["candidates"][0]["content"]["parts"][0]["text"]
            match = re.search(r"\{[\s\S]+\}", text)
            return json.loads(match.group()) if match else {}
        except urllib.error.HTTPError as e:
            code = e.code
            body_text = e.read().decode(errors="replace")
            if code == 429 and "free_tier_requests" in body_text and "RESOURCE_EXHAUSTED" in body_text:
                # daily free-tier cap, not a transient spike -- retrying won't help
                raise QuotaExhausted(body_text[:400])
            if code == 429 or code >= 500:
                wait = 15 * (attempt + 1)
                print(f"    [HTTP {code}] waiting {wait}s...", file=sys.stderr)
                time.sleep(wait)
            else:
                print(f"    [HTTP {code}] {body_text[:300]}", file=sys.stderr)
                return {}
        except QuotaExhausted:
            raise
        except Exception as e:
            print(f"    [error] {str(e)[:200]}", file=sys.stderr)
            return {}
    return {}


class QuotaExhausted(Exception):
    """Raised when the API reports a daily free-tier quota exhaustion (not a
    transient rate limit) -- the caller should stop the whole run, not retry
    the next window, since it will fail identically until the quota resets."""


def call_vlm(img_paths, backend: str, google_api_key: str = None,
             tokenrouter_creds: tuple = None) -> dict:
    """Dispatches to the selected VLM backend with the same whole-window
    EXTRACT_PROMPT and image list. Gemini bills/quotas per request regardless
    of image count; TokenRouter (OpenAI-compatible) bills per token, so more
    sampled frames costs more there, but 4 frames is still cheap."""
    if backend == "gemini":
        return call_gemini(img_paths, google_api_key)
    if backend == "tokenrouter":
        api_key, base_url = tokenrouter_creds
        return call_tokenrouter(img_paths, EXTRACT_PROMPT, api_key, base_url)
    raise ValueError(f"unknown backend: {backend}")


def main():
    backend = os.environ.get("VLM_BACKEND", "tokenrouter")
    print(f"VLM backend: {backend}")

    google_api_key = None
    tokenrouter_creds = None
    if backend == "gemini":
        google_api_key = load_google_api_key()
    elif backend == "tokenrouter":
        tokenrouter_creds = load_tokenrouter_credentials(ENV_FILE)
    else:
        raise ValueError(f"unknown VLM_BACKEND: {backend}")

    rows = list(csv.DictReader(open(PERCEPTION_CSV)))
    by_basename = {Path(r["source_file"]).name: r for r in rows}

    manifest = json.loads((CLEAN / "manifest.json").read_text())
    out_path = CLEAN / "labels_vlm.json"
    labels = json.loads(out_path.read_text()) if out_path.exists() else {}

    window_names = sorted(manifest.keys())
    print(f"{len(window_names)} windows total, {len(labels)} already labeled from a prior run")

    n_gt, n_vlm, n_skip, n_fail = 0, 0, 0, 0
    for i, name in enumerate(window_names):
        if name in labels:
            n_skip += 1
            continue

        info = manifest[name]
        frames = info["frames"]
        sample_idx = sample_frame_indices(len(frames), max_n=4)
        sampled_frames = [frames[i] for i in sample_idx]

        # 1. ground-truth match on ANY frame in the window
        gt_matches = [(f, by_basename[f]) for f in frames if f in by_basename]
        if gt_matches:
            votes = [m[1]["gauge_readable"] == "true" for m in gt_matches]
            is_readable = sum(votes) > len(votes) / 2
            categories = sorted({m[1]["category"] for m in gt_matches})
            labels[name] = dict(
                label="success" if is_readable else "failure",
                label_source="ground_truth",
                category=categories[0],
                decision_frame=frames[-1],
                matched_frames=[m[0] for m in gt_matches],
            )
            n_gt += 1
            print(f"[{i+1}/{len(window_names)}] {name}: ground_truth -> {labels[name]['label']} ({categories[0]})")
            continue

        # 2. real VLM call, whole-window: up to 4 sampled frames in ONE request
        img_paths = [CLEAN / name / f for f in sampled_frames]
        try:
            vlm = call_vlm(img_paths, backend, google_api_key, tokenrouter_creds)
        except QuotaExhausted as e:
            out_path.write_text(json.dumps(labels, indent=2))
            print(f"\n[QUOTA EXHAUSTED] stopping at window {i+1}/{len(window_names)} "
                  f"({name}). Daily free-tier cap reached: {e}")
            print(f"Progress saved: {len(labels)} windows labeled so far "
                  f"({n_gt} ground_truth + {n_vlm} vlm this run).")
            print("Re-run this script after the quota resets to resume from here "
                  "(already-labeled windows are skipped), or set "
                  "VLM_BACKEND=tokenrouter to switch backends.")
            return
        if not vlm:
            n_fail += 1
            print(f"[{i+1}/{len(window_names)}] {name}: VLM call failed, leaving unlabeled")
            continue

        category = vlm.get("category", "unknown")
        if category not in VALID_CATEGORIES:
            category = "unknown"
        is_readable = bool(vlm.get("gauge_readable", False)) or category == "clean"
        best_i = vlm.get("best_frame_index")
        best_frame = (sampled_frames[best_i] if isinstance(best_i, int)
                      and 0 <= best_i < len(sampled_frames) else None)
        labels[name] = dict(
            label="success" if is_readable else "failure",
            label_source="vlm",
            vlm_backend=backend,
            category=category,
            sampled_frames=sampled_frames,
            best_frame=best_frame,
            gauge_value=str(vlm.get("gauge_value", "UNREADABLE")),
            asset=vlm.get("asset", "unknown"),
            description=vlm.get("description", ""),
        )
        n_vlm += 1
        print(f"[{i+1}/{len(window_names)}] {name}: vlm -> {labels[name]['label']} "
              f"({category}, readable={is_readable})")

        # checkpoint after every call so a rate-limit/crash doesn't lose progress
        out_path.write_text(json.dumps(labels, indent=2))
        time.sleep(2.0)

    out_path.write_text(json.dumps(labels, indent=2))
    print(f"\nDone. ground_truth={n_gt} vlm={n_vlm} skipped_cached={n_skip} failed={n_fail}")
    print(f"Total labeled: {len(labels)}/{len(window_names)}")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
