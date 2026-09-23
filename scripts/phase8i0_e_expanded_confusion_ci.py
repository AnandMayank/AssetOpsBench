#!/usr/bin/env python3
"""phase8i0_e_expanded_confusion_ci.py -- the reobservation-necessity
confusion matrix (Precision/Recall/F1) on the E-expanded-v1 pool (30
sequences x 3 episodes = 90 episodes/model), with bootstrap 95% CIs
(2000 resamples, seed 42), to check whether per-model differentiation
is now defensible (it was not at E-v1's N=18 -- see
e_temporal_audit/e_temporal_confusion_v1.json).

Data shape differs from the frozen-93 flat-row format: each line in
raw_<model>.jsonl is one SEQUENCE with a nested "episodes" list (3
entries). Read-only.
"""
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
E_EXP = REPO_ROOT / "reports" / "benchmark" / "v3_full_results" / "e_expanded_90"
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


def ci_overlap(ci_a, ci_b):
    if ci_a[0] is None or ci_b[0] is None:
        return None
    return not (ci_a[1] < ci_b[0] or ci_b[1] < ci_a[0])


out = {}
for display, suffix in MODELS:
    lines = [json.loads(l) for l in (E_EXP / f"raw_{suffix}.jsonl").read_text().splitlines()]
    all_episodes = []
    for seq_rec in lines:
        if seq_rec.get("infra_failure") or not seq_rec.get("episodes"):
            continue
        all_episodes.extend(seq_rec["episodes"])

    n_total = len(all_episodes)
    n_empty = sum(1 for e in all_episodes if not e.get("verdict"))
    valid = [e for e in all_episodes if e.get("verdict")]
    n_valid = len(valid)

    tp = fp = fn = tn = 0
    stale = unnecessary = 0
    for e in valid:
        necessary = e["reobservation_was_necessary"]
        reacquired = e["reacquired_this_episode"]
        if necessary and reacquired:
            tp += 1
        elif not necessary and reacquired:
            fp += 1
        elif necessary and not reacquired:
            fn += 1
        else:
            tn += 1
        if e.get("stale_state_reuse"):
            stale += 1
        if e.get("unnecessary_reobservation"):
            unnecessary += 1

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall and (precision + recall) else None

    per_episode_correct = [1 if e["reobservation_was_necessary"] == e["reacquired_this_episode"] else 0 for e in valid]
    acc = sum(per_episode_correct) / len(per_episode_correct) if per_episode_correct else None
    acc_ci = bootstrap_ci(per_episode_correct)

    # precision/recall CIs via bootstrap over the per-episode TP/FP/FN/TN indicator
    def resample_metric(metric_fn):
        rng = random.Random(SEED)
        n = len(valid)
        vals = []
        for _ in range(N_BOOT):
            sample = [valid[rng.randrange(n)] for _ in range(n)]
            v = metric_fn(sample)
            if v is not None:
                vals.append(v)
        if not vals:
            return None, None
        vals.sort()
        lo_idx = int(0.025 * len(vals))
        hi_idx = min(int(0.975 * len(vals)), len(vals) - 1)
        return round(vals[lo_idx], 4), round(vals[hi_idx], 4)

    def _precision(sample):
        stp = sum(1 for e in sample if e["reobservation_was_necessary"] and e["reacquired_this_episode"])
        sfp = sum(1 for e in sample if not e["reobservation_was_necessary"] and e["reacquired_this_episode"])
        return stp / (stp + sfp) if (stp + sfp) else None

    def _recall(sample):
        stp = sum(1 for e in sample if e["reobservation_was_necessary"] and e["reacquired_this_episode"])
        sfn = sum(1 for e in sample if e["reobservation_was_necessary"] and not e["reacquired_this_episode"])
        return stp / (stp + sfn) if (stp + sfn) else None

    precision_ci = resample_metric(_precision)
    recall_ci = resample_metric(_recall)

    out[display] = {
        "n_total": n_total, "n_empty_verdict": n_empty, "n_valid": n_valid,
        "output_validity_rate": round(n_valid / n_total, 4) if n_total else None,
        "confusion": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "precision": round(precision, 4) if precision is not None else None,
        "precision_ci": list(precision_ci),
        "recall": round(recall, 4) if recall is not None else None,
        "recall_ci": list(recall_ci),
        "F1": round(f1, 4) if f1 is not None else None,
        "stale_state_reuse_count": stale, "unnecessary_reobservation_count": unnecessary,
        "decision_accuracy_necessity_match": round(acc, 4) if acc is not None else None,
        "decision_accuracy_CI": list(acc_ci),
    }

# pairwise CI-overlap check (accuracy) across all models, for defensibility summary
names = list(out.keys())
overlaps = {}
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        ov = ci_overlap(out[a]["decision_accuracy_CI"], out[b]["decision_accuracy_CI"])
        overlaps[f"{a} vs {b}"] = ov

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "e_expanded_confusion_ci_v1.json"
result = {"per_model": out, "pairwise_accuracy_CI_overlap": overlaps}
out_path.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
print(f"\nWrote {out_path}")

n_nonoverlap = sum(1 for v in overlaps.values() if v is False)
print(f"\n{n_nonoverlap}/{len(overlaps)} model pairs show a defensible (non-overlapping-CI) "
     f"difference in reobservation decision accuracy at N=90 (vs N=18 previously).")
