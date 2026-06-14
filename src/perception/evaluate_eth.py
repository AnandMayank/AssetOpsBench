"""
AssetOpsBench v2 — Evaluate ETH analog gauge reader on SyncG test split.

Compares ETH pipeline output against SyncG ground truth annotations.
Reports accuracy at 2% and 5% relative error thresholds and prints a
VERDICT used to decide whether synthetic data is sufficient for Layer 1.

SyncG annotation format (annotations/test/<name>.json):
  { "ground_truth": float, "start_value": int, ... }

ETH output format ({base_path}/{run_ts}/{image_name}/result.json):
  [ { "reading": float, "unit": str } ]

Usage:
    # Run ETH first (from analog_gauge_reader dir, gauge_reader conda env):
    #   python pipeline.py \\
    #     --input /media/.../syncg_data/syncG/images/test/ \\
    #     --base_path /path/to/AssetOpsBench/src/perception/eth_results/ \\
    #     --detection_model models/gauge_detection_model.pt \\
    #     --key_point_model models/key_point_model.pt \\
    #     --segmentation_model models/segmentation_model.pt

    python src/perception/evaluate_eth.py \\
        --annotations /media/adityapachauri/second_drive/syncg_data/syncG/annotations/test/ \\
        --eth_base    src/perception/eth_results/ \\
        --gauge_range 100
"""

import argparse
import json
import glob
import os
from pathlib import Path


def load_syncg_ground_truth(annotations_dir: str) -> dict[str, dict]:
    """
    Load SyncG ground truth.
    Returns {stem: {"gt": float, "gauge_range": float}}.
    gauge_range = long_interval_value * long_num (full scale span).
    """
    records: dict[str, dict] = {}
    for path in Path(annotations_dir).glob("*.json"):
        with open(path) as f:
            ann = json.load(f)
        gt = ann.get("ground_truth")
        if gt is None:
            continue
        start = ann.get("start_value", 0)
        # Derive range from scale geometry; fall back to caller-supplied default
        span = ann.get("long_interval_value", 0) * ann.get("long_num", 0)
        gauge_range = span if span > 0 else None
        records[path.stem] = {
            "gt": float(gt),
            "start": float(start),
            "gauge_range": float(gauge_range) if gauge_range else None,
            "gauge_type": ann.get("gauge_type", "unknown"),
        }
    return records


def load_eth_results(eth_base: str) -> dict[str, float]:
    """
    Walk ETH output directory structure and collect readings.

    ETH writes: {eth_base}/{run_TIMESTAMP}/{image_filename}/result.json
    result.json: [{"reading": float, "unit": str}]

    Returns {image_stem: reading_float}.
    """
    results: dict[str, float] = {}
    for result_path in Path(eth_base).rglob("result.json"):
        image_name = result_path.parent.name          # e.g. "gauge_0000.png"
        stem = Path(image_name).stem                  # e.g. "gauge_0000"
        try:
            with open(result_path) as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                reading = data[0].get("reading")
            elif isinstance(data, dict):
                reading = data.get("reading") or data.get("value")
            else:
                reading = None
            if reading is not None:
                results[stem] = float(reading)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return results


def evaluate(annotations_dir: str, eth_base: str, default_gauge_range: float = 100.0) -> None:
    gt_map = load_syncg_ground_truth(annotations_dir)
    eth_map = load_eth_results(eth_base)

    print(f"\n{'='*60}")
    print("AssetOpsBench v2 — ETH Pipeline Evaluation on SyncG")
    print(f"{'='*60}")
    print(f"Ground truth annotations: {len(gt_map)}")
    print(f"ETH results found:        {len(eth_map)}")

    if not eth_map:
        print("\nNo ETH results found. Run the pipeline first:")
        print("  cd /home/adityapachauri/analog_gauge_reader")
        print("  conda activate gauge_reader")
        print("  python pipeline.py --input <syncg_test_images> \\")
        print("      --base_path <path>/src/perception/eth_results/ \\")
        print("      --detection_model models/gauge_detection_model.pt \\")
        print("      --key_point_model models/key_point_model.pt \\")
        print("      --segmentation_model models/segmentation_model.pt")
        return

    matched: list[dict] = []
    no_eth: list[str] = []

    for stem, gt_info in gt_map.items():
        if stem not in eth_map:
            no_eth.append(stem)
            continue

        eth_val = eth_map[stem]
        gt_val = gt_info["gt"]
        gauge_range = gt_info["gauge_range"] or default_gauge_range

        abs_err = abs(eth_val - gt_val)
        rel_err = abs_err / gauge_range

        matched.append({
            "image": stem,
            "gt": round(gt_val, 3),
            "eth": round(eth_val, 3),
            "abs_error": round(abs_err, 3),
            "rel_error": round(rel_err, 4),
            "gauge_type": gt_info["gauge_type"],
            "within_2pct": rel_err < 0.02,
            "within_5pct": rel_err < 0.05,
        })

    n = len(matched)
    if n == 0:
        print("\nNo overlapping images between ground truth and ETH output.")
        print("Check that image filenames match between annotations and results.")
        return

    mean_rel = sum(m["rel_error"] for m in matched) / n
    acc_2pct = sum(m["within_2pct"] for m in matched) / n
    acc_5pct = sum(m["within_5pct"] for m in matched) / n

    print(f"\nMatched: {n}/{len(gt_map)} images  |  No ETH output: {len(no_eth)}")
    print(f"\nAccuracy:")
    print(f"  Within 2% of range:  {acc_2pct:.1%}  (ETH paper claims this on clean gauges)")
    print(f"  Within 5% of range:  {acc_5pct:.1%}  (AssetOpsBench v2 threshold)")
    print(f"  Mean relative error: {mean_rel:.1%}")

    # Sim-to-real gap vs ETH paper benchmark (<2% rel error on real data)
    gap = mean_rel - 0.02
    print(f"\nETH paper benchmark:   <2% relative error (real in-the-wild gauges)")
    print(f"Our result:            {mean_rel:.1%} mean relative error")
    print(f"Sim-to-real gap:       {gap:+.1%}")

    # Per-gauge-type breakdown
    types = {}
    for m in matched:
        t = m["gauge_type"]
        types.setdefault(t, []).append(m["rel_error"])
    if len(types) > 1:
        print(f"\nPer gauge type:")
        for t, errs in sorted(types.items()):
            print(f"  {t:20s}  n={len(errs):4d}  mean_err={sum(errs)/len(errs):.1%}")

    # Worst predictions
    worst = sorted(matched, key=lambda x: x["rel_error"], reverse=True)[:5]
    print(f"\nWorst 5 predictions:")
    for w in worst:
        print(f"  {w['image']}: gt={w['gt']}, eth={w['eth']}, err={w['rel_error']:.1%}")

    # Verdict
    print(f"\n{'='*60}")
    if acc_5pct >= 0.85:
        verdict = "sufficient"
        print("VERDICT: SUFFICIENT")
        print("  ETH works well on SyncG synthetic data.")
        print("  Proceed to Robot MCP server implementation.")
        print("  No real gauge photos needed for Layer 1 baseline.")
    elif acc_5pct >= 0.60:
        verdict = "marginal"
        print("VERDICT: MARGINAL — consider augmentation")
        print("  ETH works but sim-to-real gap exists.")
        print("  Add 5-10 real gauge photos from facility to bridge gap.")
        print("  Consider NVIDIA DIG fine-tuning on real images.")
    else:
        verdict = "insufficient"
        print("VERDICT: INSUFFICIENT")
        print("  Synthetic data alone is not sufficient.")
        print("  Prioritize facility visit for real gauge photos.")
        print("  Layer 1 needs real-world training data.")
    print(f"{'='*60}\n")

    # Save results
    out_path = Path(__file__).parent / "evaluation_results.json"
    summary = {
        "summary": {
            "n_gt": len(gt_map),
            "n_matched": n,
            "n_no_eth_output": len(no_eth),
            "accuracy_2pct": round(acc_2pct, 4),
            "accuracy_5pct": round(acc_5pct, 4),
            "mean_rel_error": round(mean_rel, 4),
            "sim_to_real_gap": round(gap, 4),
            "verdict": verdict,
            "eth_paper_benchmark_rel_error": 0.02,
        },
        "per_image_sample": matched[:100],
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Full results saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ETH gauge reader on SyncG test set"
    )
    parser.add_argument(
        "--annotations",
        default="/media/adityapachauri/second_drive/syncg_data/syncG/annotations/test/",
        help="Path to SyncG test annotation JSONs",
    )
    parser.add_argument(
        "--eth_base",
        default="src/perception/eth_results/",
        help="Base path where ETH pipeline wrote its run folders",
    )
    parser.add_argument(
        "--gauge_range",
        type=float,
        default=100.0,
        help="Fallback gauge range if not derivable from annotation",
    )
    args = parser.parse_args()
    evaluate(args.annotations, args.eth_base, args.gauge_range)
