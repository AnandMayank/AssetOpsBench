#!/usr/bin/env python3
"""build_new_panel_table.py -- compiles new_panel_main_table_v1.json for
the swapped panel (GPT-6-Astra, Claude Opus 5.5, Gemini 3.1 Pro Preview,
DeepSeek V4 Pro 0813, Qwen3.5-397B-A17B unchanged), at the SAME N as the
paper's existing Figure 5 (frozen-93 only for A/B_legacy/C/D_enterprise/E,
not the scaled-up pools -- those feed separate prose findings, matching
the paper's own convention for E's 5x scale-up). Qwen's numbers are
reused unchanged from the original panel's main_table_v1.json.
"""
import json, sys
from pathlib import Path
REPO = Path("/home/adityapachauri/AssetOpsBench")
sys.path.insert(0, str(REPO / "src" / "orchestrator"))
from ordering import procedural_coverage

FROZEN93 = REPO / "reports" / "benchmark" / "v3_full_results" / "frozen93"
BACQ = REPO / "reports" / "benchmark" / "b_acquisition_pilot"
DPHYS = REPO / "reports" / "benchmark" / "dphys_pilot"
HERE = REPO / "reports" / "benchmark" / "figures"

NEW_MODELS = [
    ("GPT-6-Astra", "GPT-6-Astra", "tokenrouter_openai_gpt-6-astra"),
    ("Claude Opus 5.5", "Claude_Opus_5.5", "tokenrouter_anthropic_claude-opus-5.5"),
    ("Gemini 3.1 Pro Preview", "Gemini_3.1-Pro-Preview", "tokenrouter_google_gemini-3.1-pro-preview"),
    ("DeepSeek V4 Pro 0813", "DeepSeek_V4_Pro_0813", "tokenrouter_deepseek_deepseek-v4-pro-0813"),
]


def compute(label, suffix):
    rows = [json.loads(l) for l in (FROZEN93 / f"raw_{label}.jsonl").read_text().splitlines()]
    a = [r for r in rows if r["dim"] == "A"]
    eval_a = [r for r in a if r["runner_return"].get("verdict", "") != ""]
    n_a = len(eval_a)
    gsr = sum(r["runner_return"]["metric"]["GSR"] for r in eval_a) / n_a if n_a else None
    tda = sum(r["runner_return"]["metric"]["TDA"] for r in eval_a) / n_a if n_a else None

    brows = [json.loads(l) for l in (BACQ / f"b_acq_pilot_raw_{suffix}_v3_full.jsonl").read_text().splitlines()]
    n_ags = 0
    for r in brows:
        s, g = r["score"], r["gold"]
        if r.get("error") is not None or r.get("response") is None:
            continue
        ada, tdap = bool(s["ADA"]), bool(s["TDA_post"])
        asa, uha = (s.get("ASA") == 1.0), (s.get("UHA") == 1.0)
        if not g["acquisition_required"]:
            ok = ada and tdap
        elif not g["acquisition_genuinely_unavailable"]:
            ok = ada and asa and tdap
        else:
            ok = ada and asa and uha
        n_ags += int(ok)
    bags = n_ags / len(brows)

    c = [r for r in rows if r["dim"] == "C"]
    ceval = [r for r in c if r["runner_return"].get("verdict", "") != ""]
    covs = [procedural_coverage(r["runner_return"].get("required_order") or [],
                                r["runner_return"].get("executed_order") or []) for r in ceval]
    covs = [x for x in covs if x is not None]
    cov = sum(covs) / len(covs) if covs else None

    drows = [json.loads(l) for l in (DPHYS / f"dphys_pilot_raw_{suffix}_v3_full.jsonl").read_text().splitlines()]
    csa = sum(r["score"]["CSA"] for r in drows) / len(drows)
    cc_d = sum(r["score"]["CC"] for r in drows) / len(drows)
    lca_vals = [r["score"]["LCA"] for r in drows if r["score"].get("LCA") is not None]
    lca = sum(lca_vals) / len(lca_vals) if lca_vals else None

    e = [r for r in rows if r["dim"] == "E"]
    egsr = [r["runner_return"].get("CC_grounded") for r in e]
    en = sum(1 for x in egsr if x is not None)
    ecc = sum(x for x in egsr if x is not None) / en if en else None

    return {
        "A": {"value": round(gsr, 4) if gsr is not None else None, "n": n_a, "caveat": n_a < 48},
        "A_TDA": round(tda, 4) if tda is not None else None,
        "B": {"value": round(bags, 4), "n": len(brows), "caveat": False},
        "C": {"value": round(cov, 4) if cov is not None else None, "n": len(ceval), "caveat": len(ceval) < 9},
        "D": {"value": round(csa, 4), "n": len(drows), "caveat": False},
        "D_CC": round(cc_d, 4), "D_LCA": round(lca, 4) if lca is not None else None,
        "E": {"value": round(ecc, 4) if ecc is not None else None, "n": en, "caveat": en < 18},
    }


out = {"_N": {"A": 48, "B": 66, "C": 9, "D": 70, "E": 18}}
for display, label, suffix in NEW_MODELS:
    out[display] = compute(label, suffix)

# Qwen unchanged -- reuse the original panel's already-verified numbers
orig = json.loads((REPO / "reports" / "benchmark" / "main_table" / "main_table_v1.json").read_text())
qrows = [json.loads(l) for l in (FROZEN93 / "raw_Qwen3.5-397B-A17B.jsonl").read_text().splitlines()]
qa = [r for r in qrows if r["dim"] == "A"]
qeval = [r for r in qa if r["runner_return"].get("verdict", "") != ""]
q_tda = sum(r["runner_return"]["metric"]["TDA"] for r in qeval) / len(qeval)
qd = [json.loads(l) for l in (DPHYS / "dphys_pilot_raw_tokenrouter_qwen_qwen3.5-397b-a17b_v3_full_merged.jsonl").read_text().splitlines()]
q_cc = sum(r["score"]["CC"] for r in qd) / len(qd)
q_lca_vals = [r["score"]["LCA"] for r in qd if r["score"].get("LCA") is not None]
q_lca = sum(q_lca_vals) / len(q_lca_vals) if q_lca_vals else None

qm = orig["Qwen3.5-397B-A17B"]
out["Qwen3.5-397B-A17B"] = {
    "A": {"value": qm["A_GSR"]["value"], "n": qm["A_GSR"]["n_valid"], "caveat": qm["A_GSR"]["n_valid"] < 48},
    "A_TDA": round(q_tda, 4),
    "B": {"value": qm["B_B_AGS"]["value"], "n": qm["B_B_AGS"]["n_valid"], "caveat": False},
    "C": {"value": qm["C_coverage"]["value"], "n": qm["C_coverage"]["n_valid"], "caveat": qm["C_coverage"]["n_valid"] < 9},
    "D": {"value": qm["D_CSA"]["value"], "n": qm["D_CSA"]["n_valid"], "caveat": False},
    "D_CC": round(q_cc, 4), "D_LCA": round(q_lca, 4) if q_lca is not None else None,
    "E": {"value": qm["E_CC_grounded"]["value"], "n": qm["E_CC_grounded"]["n_valid"], "caveat": qm["E_CC_grounded"]["n_valid"] < 18},
}

# Scaled C: template-generated 112-episode pool (phase8j3), mean required-action recall.
C_LABELS = {"Claude Opus 5.5": "Claude_Opus_5.5", "GPT-6-Astra": "GPT-6-Astra",
            "DeepSeek V4 Pro 0813": "DeepSeek_V4_Pro_0813", "Gemini 3.1 Pro Preview": "Gemini_3.1_Pro_Preview",
            "Qwen3.5-397B-A17B": "Qwen3.5-397B-A17B"}
for mname, lab in C_LABELS.items():
    rows = [json.loads(l) for l in (REPO / "reports/benchmark/v3_full_results/c_generated_112" / f"raw_{lab}.jsonl").read_text().splitlines()]
    valid = [r for r in rows if not r.get("infra_failure") and not r.get("apparatus_failure")]
    rec = [r["score"]["required_action_prf1"]["recall"] for r in valid if r["score"]["required_action_prf1"]["recall"] is not None]
    out[mname]["C_scaled"] = {"value": round(sum(rec) / len(rec), 4), "n": len(valid), "pool": len(rows)}
out["_N"]["C_scaled"] = 112

(HERE / "new_panel_main_table_v1.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
print(f"\nWrote {HERE}/new_panel_main_table_v1.json")
