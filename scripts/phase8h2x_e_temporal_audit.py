#!/usr/bin/env python3
"""phase8h2x_e_temporal_audit.py -- audits the E (temporal grounding)
dimension: is there a genuinely FactoryBench-inaccessible scientific
result here, and is N=18/model (6 sequences, no expanded pool exists)
enough to defend it?

The reobservation-necessity confusion matrix (necessary-to-reacquire vs
actually-reacquired) is the natural temporal-specific analogue of D's
CS-precision/recall: it requires PERSISTENT HIDDEN STATE across multiple
episodes in one continuous session -- something a single/few-turn
supplied-evidence QA benchmark (FactoryBench) cannot construct, since it
has no notion of a world that keeps evolving between the agent's turns.

Read-only; writes to reports/benchmark/e_temporal_audit/.
"""
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FROZEN93 = REPO_ROOT / "reports" / "benchmark" / "v3_full_results" / "frozen93"
OUT_DIR = REPO_ROOT / "reports" / "benchmark" / "e_temporal_audit"

MODELS = [
    ("Claude Sonnet 4.6", "Claude_Sonnet_4.6"),
    ("GPT-5.2", "GPT-5.2"),
    ("DeepSeek V4 Pro", "DeepSeek_V4_Pro"),
    ("Mistral Medium 3.5", "Mistral_Medium_3.5"),
    ("Qwen3.5-397B-A17B", "Qwen3.5-397B-A17B"),
]
N_BOOT, SEED = 2000, 42


def bootstrap_ci(values, n_boot=N_BOOT, seed=SEED):
    if not values:
        return None, None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return round(means[int(0.025 * n_boot)], 4), round(means[int(0.975 * n_boot) - 1], 4)


out = {}
pooled_tp = pooled_fp = pooled_fn = pooled_tn = 0

for display, suffix in MODELS:
    rows = [json.loads(l) for l in (FROZEN93 / f"raw_{suffix}.jsonl").read_text().splitlines()]
    e_rows = [r for r in rows if r["dim"] == "E"]
    n_total = len(e_rows)
    n_empty = sum(1 for r in e_rows if r["runner_return"].get("verdict", "") == "")

    tp = fp = fn = tn = 0
    stale = unnecessary = 0
    n_valid = 0
    for r in e_rows:
        rr = r["runner_return"]
        if rr.get("verdict", "") == "":
            continue
        n_valid += 1
        necessary = rr["reobservation_was_necessary"]
        reacquired = rr["reacquired_this_episode"]
        if necessary and reacquired:
            tp += 1
        elif not necessary and reacquired:
            fp += 1
        elif necessary and not reacquired:
            fn += 1
        else:
            tn += 1
        if rr.get("stale_state_reuse"):
            stale += 1
        if rr.get("unnecessary_reobservation"):
            unnecessary += 1

    pooled_tp += tp; pooled_fp += fp; pooled_fn += fn; pooled_tn += tn

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall and (precision + recall) else None

    per_episode_correct = [1 if ((rr := r["runner_return"])["reobservation_was_necessary"] == rr["reacquired_this_episode"]) else 0
                           for r in e_rows if r["runner_return"].get("verdict", "") != ""]
    acc_ci = bootstrap_ci(per_episode_correct)

    out[display] = {
        "n_total": n_total, "n_empty_verdict": n_empty, "n_valid": n_valid,
        "confusion": {"TP_necessary_and_reacquired": tp, "FP_unnecessary_but_reacquired": fp,
                     "FN_necessary_but_not_reacquired": fn, "TN_unnecessary_and_not_reacquired": tn},
        "reacquisition_precision": round(precision, 4) if precision is not None else None,
        "reacquisition_recall": round(recall, 4) if recall is not None else None,
        "reacquisition_F1": round(f1, 4) if f1 is not None else None,
        "stale_state_reuse_count": stale, "unnecessary_reobservation_count": unnecessary,
        "decision_accuracy_necessity_match": round(sum(per_episode_correct) / len(per_episode_correct), 4) if per_episode_correct else None,
        "decision_accuracy_CI": list(acc_ci),
    }

pooled_precision = pooled_tp / (pooled_tp + pooled_fp) if (pooled_tp + pooled_fp) else None
pooled_recall = pooled_tp / (pooled_tp + pooled_fn) if (pooled_tp + pooled_fn) else None
out["_pooled_across_5_models"] = {
    "note": "NOT independent samples in the usual sense -- pooling across models to check if ANY signal exists at all before deciding whether per-model N=18 is defensible",
    "TP": pooled_tp, "FP": pooled_fp, "FN": pooled_fn, "TN": pooled_tn,
    "n_total_datapoints": pooled_tp + pooled_fp + pooled_fn + pooled_tn,
    "pooled_precision": round(pooled_precision, 4) if pooled_precision else None,
    "pooled_recall": round(pooled_recall, 4) if pooled_recall else None,
}

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "e_temporal_confusion_v1.json"
out_path.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
print(f"\nWrote {out_path}")
