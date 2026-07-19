"""
Batch-renders every failure-labeled PMC window to .mp4, producing the
test_dir that GaugeFailClassification's scripts/inference/classify.py needs
to score against the conformal-prediction bands fit on success-only videos
(see render_success_windows_to_video.py / cp_bands/).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "GaugeFailClassification" / "scripts" / "data_processing"))
from frames_to_video import frames_to_video  # noqa: E402

CLEAN = Path("/media/adityapachauri/second_drive/aditya_pmc_work/pmc_windows_clean_full")
OUT_DIR = Path("/media/adityapachauri/second_drive/aditya_pmc_work/pmc_test_videos_failure")
FPS = 2.0

manifest = json.loads((CLEAN / "manifest.json").read_text())
labels = json.loads((CLEAN / "labels_vlm.json").read_text())

failure_windows = [name for name, lab in labels.items() if lab["label"] == "failure"]
print(f"{len(failure_windows)} failure-labeled windows out of {len(labels)} total")

OUT_DIR.mkdir(parents=True, exist_ok=True)
rendered, skipped = 0, 0
for i, name in enumerate(sorted(failure_windows)):
    n_frames = manifest[name]["n_frames"]
    out_path = OUT_DIR / f"{name}.mp4"
    if out_path.exists():
        skipped += 1
        continue
    if n_frames < 2:
        print(f"[{i+1}/{len(failure_windows)}] {name}: only {n_frames} frame(s), skipping")
        continue
    frames_to_video(
        input_dir=CLEAN / name,
        output_path=out_path,
        fps=FPS,
        pattern="*.jpg",
        codec="mp4v",
    )
    rendered += 1

print(f"\nDone. rendered={rendered} skipped_existing={skipped} total_failure={len(failure_windows)}")
print(f"Test video dir: {OUT_DIR}")
