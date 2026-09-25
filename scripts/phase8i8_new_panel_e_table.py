#!/usr/bin/env python3
"""phase8i8_new_panel_e_table.py -- E (temporal grounding) results for
the 4-model swapped panel (GPT-6-Astra, Claude Opus 5.5, Gemini 3.1 Pro
Preview, DeepSeek V4 Pro 0813), matching the exact reobservation-
necessity confusion-matrix methodology already established for the
original panel (phase8i0_e_expanded_confusion_ci.py). Bootstrap 95% CIs
(2000 resamples, seed 42). Read-only over existing scored data.
"""
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
E_EXP = REPO_ROOT / "reports" / "benchmark" / "v3_full_results" / "e_expanded_90"
OUT_DIR = REPO_ROOT / "reports" / "benchmark" / "new_panel"
N_BOOT, SEED = 2000, 42

MODELS = [
    ("GPT-6-Astra", "GPT-6-Astra"),
    ("Claude Opus 5.5", "Claude_Opus_5.5"),
    ("Gemini 3.1 Pro Preview", "Gemini_3.1-Pro-Preview"),
    ("DeepSeek V4 Pro 0813", "DeepSeek_V4_Pro_0813"),
]


def bootstrap_ci(values, n_boot=N_BOOT, seed=SEED):
    if not values:
        return None, None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return round(means[int(0.025 * n_boot)], 4), round(means[min(int(0.975 * n_boot), n_boot - 1)], 4)


def main():
    out = {}
    pooled_tp = pooled_fp = pooled_fn = pooled_tn = 0

    for display, label in MODELS:
        lines = [json.loads(l) for l in (E_EXP / f"raw_{label}.jsonl").read_text().splitlines()]
        all_eps = []
        for seq in lines:
            if seq.get("infra_failure") or not seq.get("episodes"):
                continue
            all_eps.extend(seq["episodes"])
        n_total = len(all_eps)
        valid = [e for e in all_eps if e.get("verdict")]
        n_valid = len(valid)

        tp = fp = fn = tn = 0
        for e in valid:
            necessary, reacquired = e["reobservation_was_necessary"], e["reacquired_this_episode"]
            if necessary and reacquired:
                tp += 1
            elif not necessary and reacquired:
                fp += 1
            elif necessary and not reacquired:
                fn += 1
            else:
                tn += 1
        pooled_tp += tp; pooled_fp += fp; pooled_fn += fn; pooled_tn += tn

        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        acc_vals = [1 if e["reobservation_was_necessary"] == e["reacquired_this_episode"] else 0 for e in valid]
        acc_mean = sum(acc_vals) / len(acc_vals) if acc_vals else None
        acc_ci = bootstrap_ci(acc_vals)

        out[display] = {
            "n_total": n_total, "n_valid": n_valid,
            "output_validity_rate": round(n_valid / n_total, 4) if n_total else None,
            "confusion": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "decision_accuracy": round(acc_mean, 4) if acc_mean is not None else None,
            "decision_accuracy_ci": list(acc_ci),
        }

    n_pooled = pooled_tp + pooled_fp + pooled_fn + pooled_tn
    out["_pooled_across_4_new_panel_models"] = {
        "n": n_pooled,
        "precision": round(pooled_tp / (pooled_tp + pooled_fp), 4) if (pooled_tp + pooled_fp) else None,
        "recall": round(pooled_tp / (pooled_tp + pooled_fn), 4) if (pooled_tp + pooled_fn) else None,
        "miss_rate": round(pooled_fn / (pooled_tp + pooled_fn), 4) if (pooled_tp + pooled_fn) else None,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "new_panel_e_table_v1.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
