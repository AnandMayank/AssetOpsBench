#!/usr/bin/env python3
"""phase8i2_d_conditional_grounding.py -- audits whether D's paper text
("testing whether correct operational decisions are accompanied by
correct identification of their physical constraints") is actually
answered by the currently-reported numbers. CC and CSA are reported as
separate marginals (P(CC), P(CSA)); the paper's own question requires
the JOINT/conditional statistic P(CSA | CC) -- is a correct terminal
decision actually accompanied by correct constraint identification, or
just as often a "right answer for the wrong / absent reason"?

Mirrors the A-family's TDA/GSR relationship (GSR is conditioned on a
correct, evidence-grounded decision) -- D currently lacks this
decomposition. Computes, per model, with bootstrap 95% CIs (2000
resamples, seed 42):
  P(CC=1)                          -- terminal decision correctness (reported)
  P(CSA=1)                          -- constraint-set exact accuracy (reported)
  P(CC=1 AND CSA=1)                 -- "correctly grounded" terminal decision (NEW)
  P(CSA=1 | CC=1)                   -- conditional: given right answer, right reasons? (NEW)
  P(CSA=1 | CC=0)                   -- conditional: given wrong answer, right reasons? (NEW)
  Delta = P(CC=1) - P(CC=1 AND CSA=1)  -- the D-family grounding gap (NEW)

Read-only over existing scored D-physical raw data.
"""
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DPHYS = REPO_ROOT / "reports" / "benchmark" / "dphys_pilot"
OUT_DIR = REPO_ROOT / "reports" / "benchmark" / "d_conditional_grounding"
N_BOOT, SEED = 2000, 42

MODELS = [
    ("Claude Sonnet 4.6", "tokenrouter_anthropic_claude-sonnet-4.6"),
    ("GPT-5.2", "tokenrouter_openai_gpt-5.2"),
    ("DeepSeek V4 Pro", "tokenrouter_deepseek_deepseek-v4-pro"),
    ("Mistral Medium 3.5", "tokenrouter_mistralai_mistral-medium-3-5"),
    ("Qwen3.5-397B-A17B", "tokenrouter_qwen_qwen3.5-397b-a17b"),
]


def bootstrap_ci(values, n_boot=N_BOOT, seed=SEED):
    if not values:
        return None, None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return round(means[int(0.025 * n_boot)], 4), round(means[int(0.975 * n_boot) - 1], 4)


def bootstrap_ci_paired_conditional(rows, cond_key, cond_val, out_key, n_boot=N_BOOT, seed=SEED):
    """Bootstrap CI for P(out_key=1 | cond_key==cond_val), resampling
    episodes (not the conditional subset) so the denominator's own
    sampling variability is included."""
    rng = random.Random(seed)
    n = len(rows)
    means = []
    for _ in range(n_boot):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        subset = [r for r in sample if r["score"][cond_key] == cond_val]
        if subset:
            means.append(sum(r["score"][out_key] for r in subset) / len(subset))
    if not means:
        return None, None
    means.sort()
    return round(means[int(0.025 * len(means))], 4), round(means[min(int(0.975 * len(means)), len(means) - 1)], 4)


def find_file(suffix):
    candidates = list(DPHYS.glob(f"dphys_pilot_raw_{suffix}_v3_full_merged.jsonl"))
    if not candidates:
        candidates = list(DPHYS.glob(f"dphys_pilot_raw_{suffix}_v3_full.jsonl"))
    return candidates[0]


out = {}
for display, suffix in MODELS:
    path = find_file(suffix)
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    n = len(rows)

    cc_vals = [r["score"]["CC"] for r in rows]
    csa_vals = [r["score"]["CSA"] for r in rows]
    joint_vals = [1 if (r["score"]["CC"] and r["score"]["CSA"]) else 0 for r in rows]

    cc_mean = sum(cc_vals) / n
    csa_mean = sum(csa_vals) / n
    joint_mean = sum(joint_vals) / n
    delta = cc_mean - joint_mean

    cc_ci = bootstrap_ci(cc_vals)
    csa_ci = bootstrap_ci(csa_vals)
    joint_ci = bootstrap_ci(joint_vals)

    n_cc1 = sum(cc_vals)
    n_cc0 = n - n_cc1
    p_csa_given_cc1 = (sum(1 for r in rows if r["score"]["CC"] == 1 and r["score"]["CSA"] == 1) / n_cc1) if n_cc1 else None
    p_csa_given_cc0 = (sum(1 for r in rows if r["score"]["CC"] == 0 and r["score"]["CSA"] == 1) / n_cc0) if n_cc0 else None
    ci_given_cc1 = bootstrap_ci_paired_conditional(rows, "CC", 1, "CSA")
    ci_given_cc0 = bootstrap_ci_paired_conditional(rows, "CC", 0, "CSA")

    out[display] = {
        "n": n,
        "P_CC": round(cc_mean, 4), "P_CC_ci": list(cc_ci),
        "P_CSA": round(csa_mean, 4), "P_CSA_ci": list(csa_ci),
        "P_CC_and_CSA": round(joint_mean, 4), "P_CC_and_CSA_ci": list(joint_ci),
        "Delta_gap": round(delta, 4),
        "n_CC1": n_cc1, "n_CC0": n_cc0,
        "P_CSA_given_CC1": round(p_csa_given_cc1, 4) if p_csa_given_cc1 is not None else None,
        "P_CSA_given_CC1_ci": list(ci_given_cc1),
        "P_CSA_given_CC0": round(p_csa_given_cc0, 4) if p_csa_given_cc0 is not None else None,
        "P_CSA_given_CC0_ci": list(ci_given_cc0),
    }

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "d_conditional_grounding_v1.json"
out_path.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
print(f"\nWrote {out_path}")
