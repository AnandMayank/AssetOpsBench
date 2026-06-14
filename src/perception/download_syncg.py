"""
Download and extract SyncG dataset (YihengDeng/syncG) from HuggingFace.

SyncG: 20,000 synthetic gauge images (16K train / 4K test), 24.3 GB total.
- Source: https://huggingface.co/datasets/YihengDeng/syncG
- Target: /media/adityapachauri/second_drive/syncg_data/

SyncG ships as split zip volumes (syncG.zip, syncG.z01, ...).
After download this script merges and extracts them to produce:
  syncg_data/syncG/
    annotations/{train,test}/*.json
    images/{train,test}/*.png
    masks/{train,test}/*.png

Usage:
    uv run python src/perception/download_syncg.py
    uv run python src/perception/download_syncg.py --skip-download   # extract only
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

TARGET_DIR = Path("/media/adityapachauri/second_drive/syncg_data")
EXTRACT_DIR = TARGET_DIR / "syncG"
REPO_ID = "YihengDeng/syncG"
MIN_FREE_GB = 26.0


def check_disk_space(path: Path, min_gb: float) -> None:
    root = path if path.exists() else path.parent
    usage = shutil.disk_usage(root)
    free_gb = usage.free / (1024**3)
    print(f"Free space at {root}: {free_gb:.1f} GB")
    if free_gb < min_gb:
        print(f"ERROR: need {min_gb} GB free, only {free_gb:.1f} GB available.")
        sys.exit(1)


def download() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not installed. Run: uv add huggingface-hub")
        sys.exit(1)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    check_disk_space(TARGET_DIR, MIN_FREE_GB)

    print(f"Downloading {REPO_ID} → {TARGET_DIR}")
    print("SyncG is ~24 GB — this will take a while. Resume-safe if interrupted.\n")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(TARGET_DIR),
        # scene_file/ contains HDRI maps needed only for re-generating images (not evaluation)
        ignore_patterns=["scene_file/**", "scene_file/*"],
    )

    n_files = sum(1 for f in TARGET_DIR.rglob("*") if f.is_file())
    print(f"\nDownload complete: {n_files} files in {TARGET_DIR}")


def extract() -> None:
    """Merge split zip volumes and extract SyncG images + annotations."""
    zip_root = TARGET_DIR / "syncG.zip"
    merged = TARGET_DIR / "syncG_full.zip"

    if not zip_root.exists():
        # Find any .zip file that might be the root volume
        candidates = sorted(TARGET_DIR.glob("syncG*.zip"))
        if not candidates:
            print("No syncG.zip found yet — download may still be in progress.")
            return
        zip_root = candidates[0]

    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.rglob("*.png")):
        print(f"Already extracted: {EXTRACT_DIR}")
        return

    print(f"Merging split zip volumes → {merged}")
    subprocess.run(
        ["zip", "-s-", str(zip_root), "-O", str(merged)],
        check=True,
        cwd=TARGET_DIR,
    )

    print(f"Extracting {merged} → {TARGET_DIR}")
    subprocess.run(
        ["unzip", "-q", str(merged), "-d", str(TARGET_DIR)],
        check=True,
    )
    merged.unlink(missing_ok=True)

    n_img = len(list(EXTRACT_DIR.rglob("*.png")))
    print(f"Extraction complete: {n_img} images in {EXTRACT_DIR}")


def verify() -> None:
    """Quick sanity check on extracted data."""
    import json

    ann_dir = EXTRACT_DIR / "annotations" / "test"
    img_dir = EXTRACT_DIR / "images" / "test"

    if not ann_dir.exists():
        print(f"WARNING: {ann_dir} not found — extraction may not be done yet.")
        return

    annotations = sorted(ann_dir.glob("*.json"))
    images = sorted(img_dir.glob("*.png"))

    print(f"\nVerification:")
    print(f"  Test annotations: {len(annotations)}")
    print(f"  Test images:      {len(images)}")

    if annotations:
        with open(annotations[0]) as f:
            sample = json.load(f)
        print(f"  Sample fields:    {list(sample.keys())[:6]}")
        print(f"  ground_truth:     {sample.get('ground_truth')}")
        print(f"  gauge_type:       {sample.get('gauge_type')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, go straight to extraction")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip extraction (download only)")
    args = parser.parse_args()

    if not args.skip_download:
        download()
    if not args.skip_extract:
        extract()
    verify()
