#!/usr/bin/env python3
"""phase8h2u_iot_agreement_ci.py -- adds bootstrap 95% CIs (2000
resamples, seed 42, matching the precedent already used elsewhere in
this audit) to the iot_agree vs iot_disagree GSR/TDA split, so the
claim's defensibility is explicit rather than implied by point estimates
alone. At n=7-8 per cell, this is expected to show wide, overlapping
CIs -- which is itself the finding: there is currently NOT enough data
to claim a directional effect either way. Read-only.
"""
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FROZEN93 = REPO_ROOT / "reports" / "benchmark" / "v3_full_results" / "frozen93"
SPLIT = json.loads((REPO_ROOT / "reports" / "benchmark" / "matched_regime_delta" /
                    "iot_agreement_split_v1.json").read_text())

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


def get_values(rows, tag, metric):
    full = [r for r in rows if r["dim"] == "A" and r["arm"] == "FULL"]
    subset = [r for r in full if tag in r["world_id"]]
    evaluable = [r for r in subset if r["runner_return"].get("verdict", "") != ""]
    return [r["runner_return"]["metric"][metric] for r in evaluable]


out = {}
for display, suffix in MODELS:
    rows = [json.loads(l) for l in (FROZEN93 / f"raw_{suffix}.jsonl").read_text().splitlines()]
    entry = {}
    overlap = None
    for tag in ("iot_agree", "iot_disagree"):
        gsr_vals = get_values(rows, tag, "GSR")
        tda_vals = get_values(rows, tag, "TDA")
        lo_g, hi_g = bootstrap_ci(gsr_vals)
        lo_t, hi_t = bootstrap_ci(tda_vals)
        entry[tag] = {
            "n_evaluable": len(gsr_vals),
            "GSR": SPLIT[display][tag]["GSR"], "GSR_ci": [lo_g, hi_g],
            "TDA": SPLIT[display][tag]["TDA"], "TDA_ci": [lo_t, hi_t],
        }
    a, d = entry["iot_agree"], entry["iot_disagree"]
    if a["GSR_ci"][1] is not None and d["GSR_ci"][1] is not None:
        overlap = not (a["GSR_ci"][1] < d["GSR_ci"][0] or d["GSR_ci"][1] < a["GSR_ci"][0])
    entry["GSR_CIs_overlap"] = overlap
    entry["defensible_directional_claim"] = (overlap is False)
    out[display] = entry

out_path = REPO_ROOT / "reports" / "benchmark" / "matched_regime_delta" / "iot_agreement_split_with_ci_v1.json"
out_path.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
print(f"\nWrote {out_path}")

n_defensible = sum(1 for v in out.values() if v["defensible_directional_claim"])
print(f"\nModels with a statistically defensible (non-overlapping-CI) GSR agree/disagree "
     f"difference: {n_defensible}/{len(MODELS)}")
