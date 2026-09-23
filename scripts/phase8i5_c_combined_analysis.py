#!/usr/bin/env python3
"""phase8i5_c_combined_analysis.py -- combines the original 9-episode C
pool (frozen93) with the new 8-episode C-expanded-v1 pool into a single
17-episode-per-model analysis, with the SAME contamination-aware split
established for A: excludes apparatus_failure (empty-verdict) episodes
and reports procedural_coverage plus ordering_satisfied on the evaluable
subset, alongside output-validity rate. Bootstrap 95% CIs (2000
resamples, seed 42). Read-only.
"""
import json
import random
from pathlib import Path

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
from ordering import procedural_coverage  # noqa: E402

FROZEN93 = REPO_ROOT / "reports" / "benchmark" / "v3_full_results" / "frozen93"
C_EXP = REPO_ROOT / "reports" / "benchmark" / "v3_full_results" / "c_expanded_17"
OUT_DIR = REPO_ROOT / "reports" / "benchmark" / "c_expanded"
N_BOOT, SEED = 2000, 42

MODELS = [
    ("Claude Sonnet 4.6", "Claude_Sonnet_4.6"),
    ("GPT-5.2", "GPT-5.2"),
    ("DeepSeek V4 Pro", "DeepSeek_V4_Pro"),
    ("Mistral Medium 3.5", "Mistral_Medium_3.5"),
    ("Qwen3.5-397B-A17B", "Qwen3.5-397B-A17B"),
]


def bootstrap_ci(values, n_boot=N_BOOT, seed=SEED):
    if not values:
        return None, None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return round(means[int(0.025 * n_boot)], 4), round(means[int(0.975 * n_boot) - 1], 4)


out = {}
for display, suffix in MODELS:
    frozen_rows = [json.loads(l) for l in (FROZEN93 / f"raw_{suffix}.jsonl").read_text().splitlines()]
    orig_c = [r for r in frozen_rows if r["dim"] == "C"]
    exp_rows = [json.loads(l) for l in (C_EXP / f"raw_{suffix}.jsonl").read_text().splitlines()]

    combined = []
    for r in orig_c:
        rr = r["runner_return"]
        combined.append({
            "scenario_id": r["episode_id"].split("::")[1], "verdict": rr.get("verdict", ""),
            "apparatus_failure": bool(rr.get("apparatus_failure")),
            "required_order": rr.get("required_order") or [], "executed_order": rr.get("executed_order") or [],
            "ordering_satisfied": rr.get("ordering_satisfied"), "CC": rr.get("CC"),
            "source": "original_9",
        })
    for r in exp_rows:
        combined.append({
            "scenario_id": r["scenario_id"], "verdict": r.get("verdict", ""),
            "apparatus_failure": bool(r.get("apparatus_failure")) or not r.get("verdict"),
            "required_order": r.get("required_order") or [], "executed_order": r.get("executed_order") or [],
            "ordering_satisfied": r.get("ordering_satisfied"), "CC": r.get("CC"),
            "source": "expanded_8",
        })

    n_total = len(combined)
    evaluable = [c for c in combined if not c["apparatus_failure"]]
    n_eval = len(evaluable)

    coverages = [procedural_coverage(c["required_order"], c["executed_order"]) for c in evaluable]
    coverages = [c for c in coverages if c is not None]
    ordering_vals = [1 if c["ordering_satisfied"] else 0 for c in evaluable]
    cc_vals = [c["CC"] for c in evaluable if c["CC"] is not None]

    cov_mean = sum(coverages) / len(coverages) if coverages else None
    cov_ci = bootstrap_ci(coverages)
    ord_mean = sum(ordering_vals) / len(ordering_vals) if ordering_vals else None
    ord_ci = bootstrap_ci(ordering_vals)
    cc_mean = sum(cc_vals) / len(cc_vals) if cc_vals else None
    cc_ci = bootstrap_ci(cc_vals)

    out[display] = {
        "n_total": n_total, "n_evaluable": n_eval,
        "output_validity_rate": round(n_eval / n_total, 4),
        "procedural_coverage_mean": round(cov_mean, 4) if cov_mean is not None else None,
        "procedural_coverage_ci": list(cov_ci),
        "ordering_satisfied_mean": round(ord_mean, 4) if ord_mean is not None else None,
        "ordering_satisfied_ci": list(ord_ci),
        "CC_mean": round(cc_mean, 4) if cc_mean is not None else None,
        "CC_ci": list(cc_ci),
        "apparatus_failure_scenarios": [c["scenario_id"] for c in combined if c["apparatus_failure"]],
    }

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "c_combined_17_analysis_v1.json"
out_path.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
print(f"\nWrote {out_path}")
