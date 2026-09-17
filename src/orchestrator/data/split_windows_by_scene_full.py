"""Scale-up version of split_windows_by_scene.py: reads pmc_windows_raw_full/run*
(all 144 timestamp-clustered raw runs) and the already-trusted
pmc_windows_filtered/window_{A,B} (manual ground truth for the splitter's own
validation), auto-splits every run on scene changes, writes clean single-asset
windows + manifest.json under pmc_windows_clean_full/."""
import json
import re
import shutil
from pathlib import Path

import cv2

DEST = Path("/media/adityapachauri/second_drive/aditya_pmc_work")
RAW_DIR = DEST / "pmc_windows_raw_full"
FILTERED_DIR = DEST / "pmc_windows_filtered"
CLEAN_DIR = DEST / "pmc_windows_clean_full"
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
