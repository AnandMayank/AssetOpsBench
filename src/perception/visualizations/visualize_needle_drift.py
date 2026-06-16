"""
Visualize the "needle drift" between Blender ground truth and ETH's OCR-based prediction.

Creates a 3-panel figure:
  Panel 1 — Original gauge image with GT needle (green) and ETH phantom arrow (red dashed)
  Panel 2 — ETH's perspective-warped gauge face (from image_warped.jpg in ETH results)
  Panel 3 — Schematic dial showing GT fraction vs ETH fraction on a 270° arc

Usage:
    python src/perception/visualizations/visualize_needle_drift.py \
        --image_stem sync_17664 \
        --syncg_dir  src/perception/data/syncg \
        --eth_results src/perception/eth_results/ \
        --output src/perception/visualizations/needle_drift_viz.png

How the GT position is derived (from Blender scene geometry):
    The SyncG dataset stores 'pointer_rotate_degree' — the exact needle angle
    set in Blender before rendering.  The ground truth reading is:

        gt_fraction = pointer_rotate_degree / (long_interval_degree × long_num)
        gt_reading  = start_value + gt_fraction × (long_interval_value × long_num)

    This is analytical — zero OCR, zero image processing, zero ambiguity.

Why ETH drifts:
    ETH's OCR step (DB_r18) was trained on ICDAR2015 real scene text and fails
    to detect Blender-rendered gauge numerals (clean anti-aliased fonts on a
    synthetic dial).  Without scale detection, ETH outputs nonsensical readings
    that can be hundreds of units off.
"""

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image


def load_annotation(syncg_dir: str, stem: str) -> dict:
    ann_dir = Path(syncg_dir) / "syncG" / "syncG" / "annotations" / "test"
    path = ann_dir / (stem + ".json")
    if not path.exists():
        raise FileNotFoundError(f"Annotation not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_image(syncg_dir: str, stem: str) -> Image.Image:
    img_dir = Path(syncg_dir) / "syncG" / "syncG" / "images" / "test"
    path = img_dir / (stem + ".jpg")
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path)


def load_eth_result(eth_results_dir: str, stem: str):
    """Return (warped_image_path, eth_reading) from latest ETH run."""
    runs = sorted(Path(eth_results_dir).glob("run_*"))
    if not runs:
        return None, None
    run = runs[-1]
    img_dir = run / (stem + ".jpg")
    warped = img_dir / "image_warped.jpg"
    result_file = img_dir / "result.json"
    eth_reading = None
    if result_file.exists():
        with open(result_file) as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            r = data[0].get("reading")
            if r != "Failed":
                try:
                    eth_reading = float(r)
                except (TypeError, ValueError):
                    pass
    warped_img = Image.open(warped) if warped.exists() else None
    return warped_img, eth_reading


def draw_needle(ax, cx, cy, radius, fraction, arc_span_deg=270.0,
                start_angle_deg=225.0, color="green", lw=2.5,
                linestyle="-", label=None, arrow=True):
    """
    Draw a needle arrow on a gauge schematic.

    Standard gauge convention:
      - Needle sweeps clockwise from bottom-left (225°) to bottom-right (315°)
      - fraction=0 → 225° (gauge_min), fraction=1 → 315° = -45° (gauge_max)
      - In matplotlib angles: 0° = right, 90° = up (CCW)
      - Conversion: mpl_angle = 90 - clockwise_angle
    """
    cw_angle = start_angle_deg - fraction * arc_span_deg
    mpl_angle = math.radians(90.0 - cw_angle)
    tip_x = cx + radius * math.cos(mpl_angle)
    tip_y = cy + radius * math.sin(mpl_angle)
    if arrow:
        ax.annotate("", xy=(tip_x, tip_y), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                   linestyle=linestyle))
    else:
        ax.plot([cx, tip_x], [cy, tip_y], color=color, lw=lw,
                linestyle=linestyle)
    if label:
        lx = cx + (radius * 1.15) * math.cos(mpl_angle)
        ly = cy + (radius * 1.15) * math.sin(mpl_angle)
        ax.text(lx, ly, label, color=color, fontsize=8, ha="center", va="center")


def visualize(image_stem: str, syncg_dir: str, eth_results_dir: str, output: str):
    ann  = load_annotation(syncg_dir, image_stem)
    orig = load_image(syncg_dir, image_stem)
    warped_img, eth_reading = load_eth_result(eth_results_dir, image_stem)

    gt_reading   = float(ann["ground_truth"])
    start_value  = float(ann.get("start_value", 0))
    li_deg       = float(ann.get("long_interval_degree", 10))
    li_val       = float(ann.get("long_interval_value", 1))
    ln           = float(ann.get("long_num", 1))
    ptr_deg      = float(ann.get("pointer_rotate_degree", 0))
    arc_span     = li_deg * ln
    gauge_range  = li_val * ln
    gauge_max    = start_value + gauge_range
    gauge_type   = ann.get("gauge_type", "unknown")

    # GT fraction directly from Blender's pointer_rotate_degree
    gt_frac = ptr_deg / arc_span if arc_span > 0 else 0.0

    # ETH fraction (if available)
    eth_frac = None
    if eth_reading is not None:
        eth_frac = (eth_reading - start_value) / gauge_range

    fig, axes = plt.subplots(1, 3 if warped_img else 2,
                             figsize=(15 if warped_img else 10, 5))
    fig.suptitle(
        f"Needle Drift Visualization — {image_stem} [{gauge_type}]\n"
        f"Scale: {start_value:.4g} to {gauge_max:.4g}   "
        f"GT: {gt_reading:.4g}   "
        f"ETH: {eth_reading:.4g if eth_reading is not None else 'Failed'}   "
        f"Rel err: {abs(eth_reading - gt_reading)/gauge_range:.1%}"
        if eth_reading is not None else
        f"Needle Drift Visualization — {image_stem} [{gauge_type}]\n"
        f"Scale: {start_value:.4g} to {gauge_max:.4g}   GT: {gt_reading:.4g}   ETH: Failed",
        fontsize=11, fontweight="bold"
    )

    # ── Panel 1: Original image with overlaid needle arrows ───────────────────
    ax1 = axes[0]
    ax1.imshow(orig)
    ax1.set_title("Original image\n(green = GT, red = ETH prediction)")
    ax1.axis("off")

    # Find dial bounding box from annotation
    bbox = ann.get("dial_bbox_annotations")  # [x1, y1, x2, y2]
    if bbox:
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        radius = (bbox[2] - bbox[0]) / 2 * 0.8
    else:
        h, w = np.array(orig).shape[:2]
        cx, cy, radius = w / 2, h / 2, min(w, h) * 0.35

    draw_needle(ax1, cx, cy, radius, gt_frac, color="lime", lw=3,
                label=f"GT={gt_reading:.2f}")
    if eth_reading is not None:
        eth_frac_clamped = max(0.0, min(1.0, eth_frac))
        draw_needle(ax1, cx, cy, radius * 0.85, eth_frac_clamped,
                    color="red", lw=2, linestyle="--",
                    label=f"ETH={eth_reading:.2f}")
    ax1.text(cx, cy + radius * 1.1,
             "ETH = FAILED (OCR)" if eth_reading is None else "",
             ha="center", va="bottom", color="red", fontsize=9)

    # ── Panel 2: ETH warped gauge face ────────────────────────────────────────
    panel_idx = 1
    if warped_img:
        ax2 = axes[panel_idx]
        ax2.imshow(warped_img)
        ax2.set_title("ETH's warped gauge face\n(OCR fails on Blender fonts)")
        ax2.axis("off")
        panel_idx += 1

    # ── Panel 3: Schematic dial showing drift magnitude ───────────────────────
    ax3 = axes[panel_idx]
    ax3.set_aspect("equal")
    ax3.set_xlim(-1.4, 1.4)
    ax3.set_ylim(-1.4, 1.4)
    ax3.set_title("Schematic dial\n(needle arc positions)")
    ax3.axis("off")

    # Draw arc
    ARC_START = 225.0  # degrees (clockwise from right), bottom-left
    ARC_SPAN  = 270.0
    theta = np.linspace(
        math.radians(90 - ARC_START),
        math.radians(90 - (ARC_START - ARC_SPAN)),
        300
    )
    ax3.plot(np.cos(theta), np.sin(theta), "k-", lw=1.5, alpha=0.4)

    # GT needle
    draw_needle(ax3, 0, 0, 1.0, gt_frac, color="lime", lw=3,
                label=f"GT={gt_reading:.2g}")

    # ETH needle (clamped)
    if eth_reading is not None:
        eth_frac_clamped = max(0.0, min(1.0, eth_frac))
        draw_needle(ax3, 0, 0, 0.85, eth_frac_clamped,
                    color="red", lw=2.5, linestyle="--",
                    label=f"ETH={eth_reading:.2g}")

    # Scale labels at min/max
    for frac, label in [(0.0, f"{start_value:.4g}"), (1.0, f"{gauge_max:.4g}")]:
        cw = ARC_START - frac * ARC_SPAN
        mpl_a = math.radians(90 - cw)
        lx = 1.25 * math.cos(mpl_a)
        ly = 1.25 * math.sin(mpl_a)
        ax3.text(lx, ly, label, ha="center", va="center", fontsize=8, color="gray")

    # Legend
    patches = [mpatches.Patch(color="lime", label=f"GT = {gt_reading:.4g}")]
    if eth_reading is not None:
        rel = abs(eth_reading - gt_reading) / gauge_range
        patches.append(mpatches.Patch(color="red",
                                      label=f"ETH = {eth_reading:.4g} (+{rel:.1%} err)"))
    else:
        patches.append(mpatches.Patch(color="red", label="ETH = FAILED"))
    ax3.legend(handles=patches, loc="lower center", fontsize=9)

    plt.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize ETH needle drift on SyncG gauge")
    parser.add_argument("--image_stem",   default="sync_17664",
                        help="SyncG image stem (without extension)")
    parser.add_argument("--syncg_dir",    default="src/perception/data/syncg")
    parser.add_argument("--eth_results",  default="src/perception/eth_results/")
    parser.add_argument("--output",       default="src/perception/visualizations/needle_drift_viz.png")
    args = parser.parse_args()

    visualize(args.image_stem, args.syncg_dir, args.eth_results, args.output)
