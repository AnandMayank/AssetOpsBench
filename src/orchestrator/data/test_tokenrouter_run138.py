"""
One-off test: TokenRouter whole-window labeling on run138_0 (the unambiguous
zoom-in case: wide shot of 2 unreadable gauges -> clean close-up read),
already visually confirmed and previously labeled correctly by Gemini's
whole-window call before quota exhaustion cut off further testing.

Goal: verify the TokenRouter backend (OpenAI-compatible chat completions,
no SDK) produces a sane, matching verdict before using it to resume
labeling the ~154 remaining unlabeled windows.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
from tokenrouter_backend import call_tokenrouter, load_tokenrouter_credentials  # noqa: E402
from label_windows_vlm import CLEAN, EXTRACT_PROMPT, ENV_FILE, sample_frame_indices  # noqa: E402

# label_windows_vlm.py reads only os.environ, not python-dotenv; load .env manually
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

api_key, base_url = load_tokenrouter_credentials(ENV_FILE)
print(f"base_url={base_url}")

manifest = json.loads((CLEAN / "manifest.json").read_text())
name = "run138_0"
frames = manifest[name]["frames"]
idx = sample_frame_indices(len(frames), max_n=4)
sampled = [frames[i] for i in idx]
img_paths = [CLEAN / name / f for f in sampled]
print(f"{name}: {len(frames)} frames total, sampling {sampled}")

result = call_tokenrouter(img_paths, EXTRACT_PROMPT, api_key, base_url)
print("\nresult:")
print(json.dumps(result, indent=2))
