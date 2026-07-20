"""
Batch-renders every success-labeled PMC window in pmc_windows_clean_full/ to
an .mp4 file, producing the calibration_dir that GaugeFailClassification's
scripts/inference/calibrate.py expects (a directory of "success" videos used
to fit conformal-prediction thresholds on CosmosWorld's anomaly scores).

Only success-labeled windows go here -- split conformal calibration for
one-class/anomaly-detection setups is fit on known-good examples only, per
calibrate.py's docstring and the paper's success-only CP-calibration split.
Failure-labeled windows are left out for later score_ood.py testing instead.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get(
    "GAUGE_FAIL_CLASSIFICATION_DIR", str(Path.home() / "GaugeFailClassification")) + "/scripts/data_processing")
from frames_to_video import frames_to_video  # noqa: E402

CLEAN = Path(os.environ.get("PMC_CLEAN_DIR", "pmc_windows_clean_full"))
OUT_DIR = Path(os.environ.get("PMC_CALIB_VIDEOS_OUT", "pmc_calibration_videos_success"))
FPS = 2.0  # matches the rough photo-burst cadence observed in windowA/windowB (3-5fps at higher frame counts)

manifest = json.loads((CLEAN / "manifest.json").read_text())
labels = json.loads((CLEAN / "labels_vlm.json").read_text())

success_windows = [name for name, lab in labels.items() if lab["label"] == "success"]
print(f"{len(success_windows)} success-labeled windows out of {len(labels)} total")

OUT_DIR.mkdir(parents=True, exist_ok=True)
rendered, skipped = 0, 0
for i, name in enumerate(sorted(success_windows)):
    n_frames = manifest[name]["n_frames"]
    out_path = OUT_DIR / f"{name}.mp4"
    if out_path.exists():
        skipped += 1
        continue
    if n_frames < 2:
        print(f"[{i+1}/{len(success_windows)}] {name}: only {n_frames} frame(s), skipping (need >=2 for a video)")
        continue
    frames_to_video(
        input_dir=CLEAN / name,
        output_path=out_path,
        fps=FPS,
        pattern="*.jpg",
        codec="mp4v",
    )
    rendered += 1
    if (i + 1) % 20 == 0:
        print(f"...{i+1}/{len(success_windows)} done")

print(f"\nDone. rendered={rendered} skipped_existing={skipped} total_success={len(success_windows)}")
print(f"Calibration video dir: {OUT_DIR}")
