"""
Scene-change-aware window splitter for PMC raw timestamp-clustered frame bursts.

Timestamp clustering alone (max_gap<=Ns) is necessary but not sufficient: a
photographer can walk to a *different* asset within the gap threshold (the
angle_sequence case, confirmed by manual inspection: 134139 -> 134149 was a
10s gap that crossed to a different gauge cabinet). This script automates
that check via grayscale-histogram correlation between consecutive frames:
a sharp drop in correlation marks an asset boundary.

Usage:
    python3 split_windows_by_scene.py

Reads:  pmc_windows_raw/<label>/images/{test,train}/*.jpg  (+ the already
        manually-verified pmc_windows_filtered/window_{A,B}_*)
Writes: pmc_windows_clean/<label>_<subwindow>/*.jpg
        pmc_windows_clean/manifest.json  (per-window frame list + boundary scores)
"""
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
RAW_DIR = HERE / "pmc_windows_raw"
FILTERED_DIR = HERE / "pmc_windows_filtered"   # already manually verified (angle_sequence)
CLEAN_DIR = HERE / "pmc_windows_clean"
CORR_THRESHOLD = 0.55   # below this histogram correlation -> scene cut
MIN_SUBWINDOW_LEN = 3


def natural_ts_key(path: Path):
    m = re.search(r"IMG_(\d{8})_(\d{6})(?:_(\d+))?", path.name)
    date, hhmmss, suffix = m.groups()
    return (hhmmss, int(suffix) if suffix else 0)


def hist_corr(img_a, img_b):
    """Grayscale histogram correlation, robust to needle/lighting jitter
    but sensitive to a genuine change of physical asset/background."""
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
    """Returns list of (subwindow_name, [frame_paths]) split at scene cuts."""
    frame_paths = sorted(frame_paths, key=natural_ts_key)
    if len(frame_paths) < 2:
        return [(f"{label}_0", frame_paths)]

    boundaries = [0]
    scores = []
    prev_img = cv2.imread(str(frame_paths[0]))
    for i in range(1, len(frame_paths)):
        cur_img = cv2.imread(str(frame_paths[i]))
        corr = hist_corr(prev_img, cur_img)
        scores.append(round(corr, 4))
        if corr < CORR_THRESHOLD:
            boundaries.append(i)
        prev_img = cur_img
    boundaries.append(len(frame_paths))

    subwindows = []
    for k in range(len(boundaries) - 1):
        seg = frame_paths[boundaries[k]:boundaries[k + 1]]
        if len(seg) >= MIN_SUBWINDOW_LEN:
            subwindows.append((f"{label}_{k}", seg))
        else:
            print(f"  [drop] {label}_{k}: only {len(seg)} frames after scene split (< {MIN_SUBWINDOW_LEN})")
    return subwindows, scores


def main():
    CLEAN_DIR.mkdir(exist_ok=True)
    manifest = {}

    # 1. Raw timestamp-clustered windows straight from the zip (need splitting)
    for win_dir in sorted(RAW_DIR.glob("win*")):
        label = win_dir.name
        frames = sorted(win_dir.rglob("*.jpg"))
        if not frames:
            continue
        subwindows, scores = split_one_window(frames, label)
        print(f"{label}: {len(frames)} frames -> {len(subwindows)} sub-window(s) "
              f"(min corr {min(scores):.3f})" if scores else f"{label}: {len(frames)} frames")
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
            print(f"  -> {name}: {len(paths)} frames")

    # 2. Already manually-verified windows (angle_sequence Group A/B) — trust as-is
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
        print(f"{name}: {len(frames)} frames (manually verified, copied as-is)")

    (CLEAN_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest)} clean single-asset windows written to {CLEAN_DIR}")
    print(f"Manifest: {CLEAN_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
