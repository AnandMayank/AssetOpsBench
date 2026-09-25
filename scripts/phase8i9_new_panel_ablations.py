#!/usr/bin/env python3
"""phase8i9_new_panel_ablations.py -- the three deeper ablations already
done for the original panel, now computed for the 4-model swapped panel
(GPT-6-Astra, Claude Opus 5.5, Gemini 3.1 Pro Preview, DeepSeek V4 Pro
0813). Zero additional API cost -- all data already collected
(a_expanded_240, dphys_pilot). Bootstrap 95% CIs (2000 resamples, seed
42) throughout, matching every existing convention in this project.

1. Matched within-world regime delta (FULL vs PHYSICAL_ONLY vs
   DIGITAL_ONLY), on A-expanded-v1's 80-world triplets -- same
   methodology as phase8h2w_a_expanded_figure.py.
2. D conditional grounding P(CSA|CC) -- same methodology as
   phase8i2_d_conditional_grounding.py.
3. IoT-agreement split (FULL regime only, iot_agree vs iot_disagree) --
   same methodology as phase8h2u_iot_agreement_ci.py, using A-expanded's
   FULL-regime episodes (frozen-93's FULL-only pool is too small at
   N=16/model to bother re-deriving here).

Read-only over existing scored data.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
A_EXP = REPO_ROOT / "reports" / "benchmark" / "v3_full_results" / "a_expanded_240"
DPHYS = REPO_ROOT / "reports" / "benchmark" / "dphys_pilot"
OUT_DIR = REPO_ROOT / "reports" / "benchmark" / "new_panel"
N_BOOT, SEED = 2000, 42

MODELS = [
    ("GPT-6-Astra", "GPT-6-Astra", "tokenrouter_openai_gpt-6-astra"),
    ("Claude Opus 5.5", "Claude_Opus_5.5", "tokenrouter_anthropic_claude-opus-5.5"),
    ("Gemini 3.1 Pro Preview", "Gemini_3.1-Pro-Preview", "tokenrouter_google_gemini-3.1-pro-preview"),
    ("DeepSeek V4 Pro 0813", "DeepSeek_V4_Pro_0813", "tokenrouter_deepseek_deepseek-v4-pro-0813"),
]


def bootstrap_ci(values, n_boot=N_BOOT, seed=SEED):
    if not values:
        return None, None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return round(means[int(0.025 * n_boot)], 4), round(means[min(int(0.975 * n_boot), n_boot - 1)], 4)


def ci_overlap(a, b):
    if a[0] is None or b[0] is None:
        return None
    return not (a[1] < b[0] or b[1] < a[0])


# --- 1. matched regime delta ------------------------------------------
def regime_delta(label):
    rows = [json.loads(l) for l in (A_EXP / f"raw_{label}.jsonl").read_text().splitlines()]
    by_world = defaultdict(dict)
    for r in rows:
        by_world[r["world_id"]][r["regime"]] = r
    triplets = [w for w, d in by_world.items() if all(a in d for a in ("FULL", "PHYSICAL_ONLY", "DIGITAL_ONLY"))
               and all(d[a].get("verdict_present") for a in ("FULL", "PHYSICAL_ONLY", "DIGITAL_ONLY"))]
    out = {"n_full_triplets": len(triplets)}
    for regime in ("FULL", "PHYSICAL_ONLY", "DIGITAL_ONLY"):
        gsr = [by_world[w][regime]["metric"]["GSR"] for w in triplets]
        tda = [by_world[w][regime]["metric"]["TDA"] for w in triplets]
        n = len(gsr)
        out[f"{regime}_GSR"] = round(sum(gsr) / n, 4) if n else None
        out[f"{regime}_GSR_ci"] = list(bootstrap_ci(gsr))
        out[f"{regime}_TDA"] = round(sum(tda) / n, 4) if n else None
        out[f"{regime}_TDA_ci"] = list(bootstrap_ci(tda))
    out["TDA_FULL_vs_DIGITAL_ONLY_CIs_overlap"] = ci_overlap(out["FULL_TDA_ci"], out["DIGITAL_ONLY_TDA_ci"])
    out["GSR_FULL_vs_PHYSICAL_ONLY_CIs_overlap"] = ci_overlap(out["FULL_GSR_ci"], out["PHYSICAL_ONLY_GSR_ci"])
    return out


# --- 2. D conditional grounding -----------------------------------------
def d_conditional(suffix):
    rows = [json.loads(l) for l in (DPHYS / f"dphys_pilot_raw_{suffix}_v3_full.jsonl").read_text().splitlines()]
    n = len(rows)
    cc_vals = [r["score"]["CC"] for r in rows]
    csa_vals = [r["score"]["CSA"] for r in rows]
    joint_vals = [1 if (r["score"]["CC"] and r["score"]["CSA"]) else 0 for r in rows]
    n_cc1 = sum(cc_vals); n_cc0 = n - n_cc1
    p_csa_given_cc1 = (sum(1 for r in rows if r["score"]["CC"] == 1 and r["score"]["CSA"] == 1) / n_cc1) if n_cc1 else None
    p_csa_given_cc0 = (sum(1 for r in rows if r["score"]["CC"] == 0 and r["score"]["CSA"] == 1) / n_cc0) if n_cc0 else None
    return {
        "n": n, "P_CC": round(sum(cc_vals) / n, 4), "P_CSA": round(sum(csa_vals) / n, 4),
        "P_CC_and_CSA": round(sum(joint_vals) / n, 4),
        "n_CC1": n_cc1, "n_CC0": n_cc0,
        "P_CSA_given_CC1": round(p_csa_given_cc1, 4) if p_csa_given_cc1 is not None else None,
        "P_CSA_given_CC0": round(p_csa_given_cc0, 4) if p_csa_given_cc0 is not None else None,
    }


# --- 3. IoT-agreement split (A-expanded FULL regime) ---------------------
def iot_agreement(label):
    rows = [json.loads(l) for l in (A_EXP / f"raw_{label}.jsonl").read_text().splitlines()]
    full = [r for r in rows if r["regime"] == "FULL"]
    out = {}
    for tag in ("iot_agree", "iot_disagree"):
        subset = [r for r in full if tag in r["world_id"]]
        valid = [r for r in subset if r.get("verdict_present")]
        gsr = [r["metric"]["GSR"] for r in valid]
        n = len(gsr)
        out[tag] = {"n": len(subset), "n_valid": n,
                   "GSR": round(sum(gsr) / n, 4) if n else None, "GSR_ci": list(bootstrap_ci(gsr))}
    out["CIs_overlap"] = ci_overlap(out["iot_agree"]["GSR_ci"], out["iot_disagree"]["GSR_ci"])
    return out


def main():
    results = {"regime_delta": {}, "d_conditional_grounding": {}, "iot_agreement": {}}
    for display, label, suffix in MODELS:
        results["regime_delta"][display] = regime_delta(label)
        results["d_conditional_grounding"][display] = d_conditional(suffix)
        results["iot_agreement"][display] = iot_agreement(label)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "new_panel_ablations_v1.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
