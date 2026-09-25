#!/usr/bin/env python3
"""phase8i7_new_panel_scaled_table.py -- scaled, defensible main-results
table for the 4 CHANGED primary-panel models (GPT-6-Astra, Claude Opus
5.5, Gemini 3.1 Pro Preview, DeepSeek V4 Pro 0813), matching the exact
methodology already established for the original 5-model panel:
  A: GSR on the evaluable set, POOLED across frozen-93 (48) and
     A-expanded-v1 (240) -- same fix already applied to Claude Sonnet
     4.6's blank A cell earlier this session, applied here to every
     model in this panel for consistency and defensibility.
  B: B_AGS (branched, corrected), B-Acquisition pool, N=66 (unchanged --
     already well-powered, no expansion needed).
  C: procedural_coverage, POOLED across the original 9-scenario frozen
     set and C-expanded-v1's 8 additional scenarios, N=17.
  D: CSA, D-physical pool, N=70 (unchanged -- already well-powered).
Bootstrap 95% CIs (2000 resamples, seed 42) throughout, matching the
existing main_table convention. Read-only over existing scored data.
"""
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
from ordering import procedural_coverage  # noqa: E402

FROZEN93 = REPO_ROOT / "reports" / "benchmark" / "v3_full_results" / "frozen93"
A_EXP = REPO_ROOT / "reports" / "benchmark" / "v3_full_results" / "a_expanded_240"
C_EXP = REPO_ROOT / "reports" / "benchmark" / "v3_full_results" / "c_expanded_17"
BACQ = REPO_ROOT / "reports" / "benchmark" / "b_acquisition_pilot"
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


def compute_a_pooled(label):
    frozen = [json.loads(l) for l in (FROZEN93 / f"raw_{label}.jsonl").read_text().splitlines()]
    a_frozen = [r for r in frozen if r["dim"] == "A"]
    eval_frozen = [r for r in a_frozen if r["runner_return"].get("verdict", "") != ""]
    gsr_frozen = [r["runner_return"]["metric"]["GSR"] for r in eval_frozen]
    tda_frozen = [r["runner_return"]["metric"]["TDA"] for r in eval_frozen]

    expanded = [json.loads(l) for l in (A_EXP / f"raw_{label}.jsonl").read_text().splitlines()]
    eval_expanded = [r for r in expanded if r.get("verdict_present")]
    gsr_expanded = [r["metric"]["GSR"] for r in eval_expanded]
    tda_expanded = [r["metric"]["TDA"] for r in eval_expanded]

    gsr_all = gsr_frozen + gsr_expanded
    tda_all = tda_frozen + tda_expanded
    n_total_attempted = 48 + 240
    n_valid = len(gsr_all)
    gsr_mean = round(sum(gsr_all) / n_valid, 4)
    tda_mean = round(sum(tda_all) / n_valid, 4)
    gsr_ci = bootstrap_ci(gsr_all)
    tda_ci = bootstrap_ci(tda_all)
    return {"n_valid": n_valid, "n_total": n_total_attempted,
            "output_validity_rate": round(n_valid / n_total_attempted, 4),
            "GSR": gsr_mean, "GSR_ci": list(gsr_ci),
            "TDA": tda_mean, "TDA_ci": list(tda_ci)}


def compute_b(suffix):
    rows = [json.loads(l) for l in (BACQ / f"b_acq_pilot_raw_{suffix}_v3_full.jsonl").read_text().splitlines()]
    values = []
    for r in rows:
        s, g = r["score"], r["gold"]
        if r.get("error") is not None or r.get("response") is None:
            values.append(0.0)
            continue
        ada, tda_post = bool(s["ADA"]), bool(s["TDA_post"])
        asa, uha = (s.get("ASA") == 1.0), (s.get("UHA") == 1.0)
        if not g["acquisition_required"]:
            ok = ada and tda_post
        elif not g["acquisition_genuinely_unavailable"]:
            ok = ada and asa and tda_post
        else:
            ok = ada and asa and uha
        values.append(1.0 if ok else 0.0)
    ci = bootstrap_ci(values)
    return {"n_valid": len(values), "n_total": 66, "B_AGS": round(sum(values) / len(values), 4),
            "B_AGS_ci": list(ci)}


def compute_c_pooled(label):
    frozen = [json.loads(l) for l in (FROZEN93 / f"raw_{label}.jsonl").read_text().splitlines()]
    c_frozen = [r for r in frozen if r["dim"] == "C"]
    eval_frozen = [r for r in c_frozen if r["runner_return"].get("verdict", "") != ""]
    covs_frozen = [procedural_coverage(r["runner_return"].get("required_order") or [],
                                       r["runner_return"].get("executed_order") or [])
                  for r in eval_frozen]
    covs_frozen = [c for c in covs_frozen if c is not None]

    expanded = [json.loads(l) for l in (C_EXP / f"raw_{label}.jsonl").read_text().splitlines()]
    eval_expanded = [r for r in expanded if not r.get("infra_failure") and r.get("verdict")]
    covs_expanded = [procedural_coverage(r.get("required_order") or [], r.get("executed_order") or [])
                     for r in eval_expanded]
    covs_expanded = [c for c in covs_expanded if c is not None]

    covs_all = covs_frozen + covs_expanded
    n_total_attempted = 9 + 8
    n_valid = len(covs_all)
    cov_mean = round(sum(covs_all) / n_valid, 4) if n_valid else None
    ci = bootstrap_ci(covs_all)
    return {"n_valid": n_valid, "n_total": n_total_attempted,
            "output_validity_rate": round(n_valid / n_total_attempted, 4) if n_total_attempted else None,
            "coverage": cov_mean, "coverage_ci": list(ci)}


def compute_d(suffix):
    rows = [json.loads(l) for l in (DPHYS / f"dphys_pilot_raw_{suffix}_v3_full.jsonl").read_text().splitlines()]
    values = [r["score"]["CSA"] for r in rows]
    ci = bootstrap_ci(values)
    return {"n_valid": len(values), "n_total": 70, "CSA": round(sum(values) / len(values), 4),
            "CSA_ci": list(ci)}


def main():
    out = {}
    for display, label, suffix in MODELS:
        out[display] = {
            "A": compute_a_pooled(label),
            "B": compute_b(suffix),
            "C": compute_c_pooled(label),
            "D": compute_d(suffix),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "new_panel_scaled_table_v1.json"
    out_path.write_text(json.dumps(out, indent=2))

    lines = ["| Model | A: GSR (n valid/attempted) | A: TDA | B: B_AGS (N=66) | C: coverage (n valid/attempted) | D: CSA (N=70) |",
             "|---|---|---|---|---|---|"]
    for display, _, _ in MODELS:
        e = out[display]
        a, b, c, d = e["A"], e["B"], e["C"], e["D"]
        lines.append(
            f"| {display} | {a['GSR']:.3f} [{a['GSR_ci'][0]:.3f},{a['GSR_ci'][1]:.3f}] (n={a['n_valid']}/{a['n_total']}) "
            f"| {a['TDA']:.3f} [{a['TDA_ci'][0]:.3f},{a['TDA_ci'][1]:.3f}] "
            f"| {b['B_AGS']:.3f} [{b['B_AGS_ci'][0]:.3f},{b['B_AGS_ci'][1]:.3f}] "
            f"| {c['coverage']:.3f} [{c['coverage_ci'][0]:.3f},{c['coverage_ci'][1]:.3f}] (n={c['n_valid']}/{c['n_total']}) "
            f"| {d['CSA']:.3f} [{d['CSA_ci'][0]:.3f},{d['CSA_ci'][1]:.3f}] |")
    md = "\n".join(lines)
    (OUT_DIR / "new_panel_scaled_table_v1.md").write_text(md + "\n")
    print(md)
    print(f"\nWrote {out_path}")
    print(f"Wrote {OUT_DIR}/new_panel_scaled_table_v1.md")


if __name__ == "__main__":
    main()
