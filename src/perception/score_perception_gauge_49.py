"""Scorer for PerceptionGauge-49: compares each scenario's VLM-drafted
annotation (written by annotate_perception_gauge.py into vlm_annotation)
against the catalog-authored gold preserved as gold_gauge_readable /
gold_recommended_action, and reports accuracy overall and per category.

Usage:
    python score_perception_gauge_49.py --assets-dir /path/to/assets
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets-dir",
                    default="/media/adityapachauri/second_drive/perception_gauge_49/assets")
    args = ap.parse_args()

    assets_path = Path(args.assets_dir)
    scenario_dirs = sorted(d for d in assets_path.iterdir() if d.is_dir())

    n_total = n_annotated = n_action_correct = n_readable_correct = 0
    by_cat = defaultdict(lambda: {"n": 0, "action_correct": 0, "readable_correct": 0})
    rows = []

    for d in scenario_dirs:
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        n_total += 1
        cat = meta.get("category", "unknown")
        by_cat[cat]["n"] += 1

        vlm = meta.get("vlm_annotation")
        if not vlm:
            rows.append({"scenario_id": meta.get("scenario_id"), "category": cat,
                        "annotated": False})
            continue
        n_annotated += 1

        gold_action = meta.get("gold_recommended_action")
        pred_action = vlm.get("recommended_action")
        action_correct = (gold_action == pred_action) if gold_action else None

        gold_readable = meta.get("gold_gauge_readable")
        pred_readable_raw = vlm.get("gauge_readable")
        pred_readable = str(pred_readable_raw).lower() if pred_readable_raw is not None else None
        gold_readable_norm = str(gold_readable).lower() if gold_readable is not None else None
        readable_correct = (gold_readable_norm == pred_readable) if gold_readable is not None else None

        if action_correct:
            n_action_correct += 1
            by_cat[cat]["action_correct"] += 1
        if readable_correct:
            n_readable_correct += 1
            by_cat[cat]["readable_correct"] += 1

        rows.append({
            "scenario_id": meta.get("scenario_id"), "category": cat, "annotated": True,
            "gold_recommended_action": gold_action, "vlm_recommended_action": pred_action,
            "action_correct": action_correct,
            "gold_gauge_readable": gold_readable_norm, "vlm_gauge_readable": pred_readable,
            "readable_correct": readable_correct,
        })

    print(f"N scenarios: {n_total}")
    print(f"N annotated (VLM ran): {n_annotated}")
    if n_annotated:
        print(f"recommended_action accuracy: {n_action_correct}/{n_annotated} = "
             f"{n_action_correct/n_annotated:.3f}")
        print(f"gauge_readable accuracy:     {n_readable_correct}/{n_annotated} = "
             f"{n_readable_correct/n_annotated:.3f}")
    print("\nPer-category:")
    for cat, s in sorted(by_cat.items()):
        n = s["n"]
        print(f"  {cat:22s} n={n:2d}  action_acc={s['action_correct']}/{n}  "
             f"readable_acc={s['readable_correct']}/{n}")

    out = {
        "n_total": n_total, "n_annotated": n_annotated,
        "action_accuracy": n_action_correct / n_annotated if n_annotated else None,
        "readable_accuracy": n_readable_correct / n_annotated if n_annotated else None,
        "by_category": dict(by_cat),
        "rows": rows,
    }
    out_path = assets_path.parent / "perception_gauge_49_score.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
