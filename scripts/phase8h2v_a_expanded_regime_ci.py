#!/usr/bin/env python3
"""phase8h2v_a_expanded_regime_ci.py -- scales the matched-regime TDA/GSR
comparison from the original 16-world A pool to the 80-world A-expanded-v1
pool (built and run by a separate concurrent session; verified here before
use: 240/240 rows for GPT-5.2/DeepSeek/Mistral/Claude, 20/240 for Qwen
-- still running, excluded). Adds bootstrap 95% CIs (2000 resamples, seed
42) to check whether the panel-b TDA claims are defensible at this N.
Read-only.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
A_EXP = REPO_ROOT / "reports" / "benchmark" / "v3_full_results" / "a_expanded_240"
OUT_DIR = REPO_ROOT / "reports" / "benchmark" / "matched_regime_delta"

# Qwen excluded: run incomplete (20/240) at time of this audit.
MODELS = [
    ("GPT-5.2", "GPT-5.2"),
    ("DeepSeek V4 Pro", "DeepSeek_V4_Pro"),
    ("Mistral Medium 3.5", "Mistral_Medium_3.5"),
    ("Claude Sonnet 4.6", "Claude_Sonnet_4.6"),
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
    if None in ci_a or None in ci_b:
        return None
    return not (ci_a[1] < ci_b[0] or ci_b[1] < ci_a[0])


out = {}
for display, suffix in MODELS:
    rows = [json.loads(l) for l in (A_EXP / f"raw_{suffix}.jsonl").read_text().splitlines()]
    n_total = len(rows)
    n_missing = sum(1 for r in rows if not r["verdict_present"])

    by_regime_tda = defaultdict(list)
    by_regime_gsr = defaultdict(list)
    for r in rows:
        if not r["verdict_present"]:
            continue
        by_regime_tda[r["regime"]].append(r["metric"]["TDA"])
        by_regime_gsr[r["regime"]].append(r["metric"]["GSR"])

    entry = {"n_total": n_total, "n_missing_verdict": n_missing,
            "output_validity_rate": round((n_total - n_missing) / n_total, 4)}
    cis_tda, cis_gsr = {}, {}
    for regime in ("FULL", "PHYSICAL_ONLY", "DIGITAL_ONLY"):
        tda_vals = by_regime_tda[regime]
        gsr_vals = by_regime_gsr[regime]
        n_eval = len(tda_vals)
        tda_mean = round(sum(tda_vals) / n_eval, 4) if n_eval else None
        gsr_mean = round(sum(gsr_vals) / n_eval, 4) if n_eval else None
        tda_ci = bootstrap_ci(tda_vals)
        gsr_ci = bootstrap_ci(gsr_vals)
        cis_tda[regime] = tda_ci
        cis_gsr[regime] = gsr_ci
        entry[regime] = {"n_evaluable": n_eval, "TDA": tda_mean, "TDA_ci": list(tda_ci),
                         "GSR": gsr_mean, "GSR_ci": list(gsr_ci)}

    entry["TDA_FULL_vs_DIGITAL_ONLY_CIs_overlap"] = ci_overlap(cis_tda["FULL"], cis_tda["DIGITAL_ONLY"])
    entry["TDA_FULL_vs_PHYSICAL_ONLY_CIs_overlap"] = ci_overlap(cis_tda["FULL"], cis_tda["PHYSICAL_ONLY"])
    entry["GSR_FULL_vs_PHYSICAL_ONLY_CIs_overlap"] = ci_overlap(cis_gsr["FULL"], cis_gsr["PHYSICAL_ONLY"])
    out[display] = entry

out_path = OUT_DIR / "a_expanded_regime_ci_v1.json"
out_path.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
print(f"\nWrote {out_path}")

print("\n=== Defensibility summary (n=80 worlds, A-expanded-v1) ===")
for m, e in out.items():
    print(f"{m}: TDA FULL-vs-DIGITAL_ONLY CIs overlap = {e['TDA_FULL_vs_DIGITAL_ONLY_CIs_overlap']}  "
         f"(output validity {e['output_validity_rate']:.1%})")
