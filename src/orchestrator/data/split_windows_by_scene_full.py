"""Scale-up version of split_windows_by_scene.py: reads <RAW_DIR>/run*
(timestamp-clustered raw runs, each a directory of .jpg frames extracted from
your own raw PMC photo archive -- see docs/pmc_reproduction/README.md for
where that archive comes from) and any already-trusted <FILTERED_DIR>/window_*
(manually verified ground-truth windows, optional), auto-splits every run on
scene changes, writes clean single-asset windows + manifest.json under
<CLEAN_DIR>/.

Usage:
    python split_windows_by_scene_full.py <RAW_DIR> <CLEAN_DIR> [FILTERED_DIR]
Or via env vars: PMC_RAW_DIR, PMC_CLEAN_DIR, PMC_FILTERED_DIR (optional).
"""
import json
import os
import re
import shutil
import sys
from pathlib import Path

import cv2

RAW_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PMC_RAW_DIR", "pmc_windows_raw_full"))
CLEAN_DIR = Path(sys.argv[2] if len(sys.argv) > 2 else os.environ.get("PMC_CLEAN_DIR", "pmc_windows_clean_full"))
FILTERED_DIR = Path(sys.argv[3] if len(sys.argv) > 3 else os.environ.get("PMC_FILTERED_DIR", "")) or None
CORR_THRESHOLD = 0.55
MIN_SUBWINDOW_LEN = 3


def natural_ts_key(path: Path):
    m = re.search(r"IMG_(\d{8})_(\d{6})(?:_(\d+))?", path.name)
    _, hhmmss, suffix = m.groups()
    return (hhmmss, int(suffix) if suffix else 0)


def hist_corr(img_a, img_b):
    ga = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    ga = cv2.resize(ga, (256, 192))
    gb = cv2.resize(gb, (256, 192))
    ha = cv2.calcHist([ga], [0], None, [64], [0, 256])
    hb = cv2.calcHist([gb], [0], None, [64], [0, 256])
    cv2.normalize(ha, ha)
    cv2.normalize(hb, hb)
    return float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))


def split_one_window(frame_paths, label):
    frame_paths = sorted(frame_paths, key=natural_ts_key)
    if len(frame_paths) < 2:
        return [(f"{label}_0", frame_paths)]
    boundaries = [0]
    prev_img = cv2.imread(str(frame_paths[0]))
    for i in range(1, len(frame_paths)):
        cur_img = cv2.imread(str(frame_paths[i]))
        if cur_img is None or prev_img is None:
            prev_img = cur_img
            continue
        corr = hist_corr(prev_img, cur_img)
        if corr < CORR_THRESHOLD:
            boundaries.append(i)
        prev_img = cur_img
    boundaries.append(len(frame_paths))
    subwindows = []
    for k in range(len(boundaries) - 1):
        seg = frame_paths[boundaries[k]:boundaries[k + 1]]
        if len(seg) >= MIN_SUBWINDOW_LEN:
            subwindows.append((f"{label}_{k}", seg))
    return subwindows


def main():
    CLEAN_DIR.mkdir(exist_ok=True)
    manifest = {}
    total_raw_windows = 0
    dropped_too_short = 0

    run_dirs = sorted(RAW_DIR.glob("run*"))
    for i, win_dir in enumerate(run_dirs):
        label = win_dir.name
        frames = sorted(win_dir.rglob("*.jpg"))
        if not frames:
            continue
        subwindows = split_one_window(frames, label)
        total_raw_windows += 1
        for name, paths in subwindows:
            outdir = CLEAN_DIR / name
            outdir.mkdir(parents=True, exist_ok=True)
            for p in paths:
                shutil.copy(p, outdir / p.name)
            manifest[name] = {
                "source": "raw_zip_extraction",
                "n_frames": len(paths),
                "frames": [p.name for p in sorted(paths, key=natural_ts_key)],
            }
        if (i + 1) % 20 == 0:
            print(f"  ...{i+1}/{len(run_dirs)} raw runs processed, {len(manifest)} clean windows so far")

    if FILTERED_DIR is not None:
        for win_dir in sorted(FILTERED_DIR.glob("window_*")):
            name = win_dir.name
            frames = sorted(win_dir.glob("*.jpg"), key=natural_ts_key)
            outdir = CLEAN_DIR / name
            outdir.mkdir(parents=True, exist_ok=True)
            for p in frames:
                shutil.copy(p, outdir / p.name)
            manifest[name] = {
                "source": "manually_verified",
                "n_frames": len(frames),
                "frames": [p.name for p in frames],
            }

    (CLEAN_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{total_raw_windows} raw runs -> {len(manifest)} clean single-asset windows")
    print(f"Total frames in clean windows: {sum(v['n_frames'] for v in manifest.values())}")
    print(f"Written to {CLEAN_DIR}")


if __name__ == "__main__":
    main()
