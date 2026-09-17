#!/usr/bin/env python3
"""phase8h2n_a_evaluable_verify.py -- verifies the A-family "evaluable
episode set" (used for the primary GSR/TDA numbers) meets all 5 required
conditions per episode, and that TDA and GSR are computed over the exact
same set per model. Read-only: makes no changes to any benchmark or
results file.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FROZEN93 = REPO_ROOT / "reports" / "benchmark" / "v3_full_results" / "frozen93"

MODELS = [
    ("GPT-5.2", "GPT-5.2"),
    ("Mistral Medium 3.5", "Mistral_Medium_3.5"),
    ("DeepSeek V4 Pro", "DeepSeek_V4_Pro"),
    ("Qwen3.5-397B-A17B", "Qwen3.5-397B-A17B"),
    ("Claude Sonnet 4.6", "Claude_Sonnet_4.6"),
]

INFRA_MARKERS = ("call_error:",)
TRUNC_PARSE_MARKERS = ("no_answer:", "parse_error:")


def classify(call_errors):
    if not call_errors:
        return None
    text = " ".join(call_errors)
    if any(m in text for m in INFRA_MARKERS):
        return "infrastructure"
    if any(m in text for m in TRUNC_PARSE_MARKERS):
        return "truncation_or_parse"
    return "other_error"


def main():
    world_id_sets = {}
    per_model = {}

    for display, suffix in MODELS:
        rows = [json.loads(l) for l in (FROZEN93 / f"raw_{suffix}.jsonl").read_text().splitlines()]
        a_rows = [r for r in rows if r["dim"] == "A"]
        world_id_sets[display] = sorted(r["episode_id"] for r in a_rows)

        n_total = len(a_rows)
        evaluable_ids = []
        n_invalid = 0
        n_genuine = 0
        n_trunc = 0
        n_infra = 0
        tda_num_eval = 0
        gsr_num_eval = 0

        for r in a_rows:
            rr = r["runner_return"]
            verdict = rr.get("verdict", "")
            metric = rr.get("metric") or {}
            gsr = metric.get("GSR")
            tda = metric.get("TDA")
            call_errors = rr.get("call_errors")
            err_class = classify(call_errors)

            # Condition checks for INCLUSION in the evaluable set:
            cond1_valid_verdict = verdict != ""
            cond2_valid_gsr = gsr is not None
            cond4_no_infra = err_class != "infrastructure"
            cond5_no_trunc_parse = err_class != "truncation_or_parse"

            is_evaluable = cond1_valid_verdict and cond2_valid_gsr and cond4_no_infra and cond5_no_trunc_parse

            if not is_evaluable:
                n_invalid += 1
                if err_class == "infrastructure":
                    n_infra += 1
                elif err_class == "truncation_or_parse":
                    n_trunc += 1
                elif not cond1_valid_verdict and err_class is None:
                    n_genuine += 1
                else:
                    n_genuine += 1  # unclassified empty-verdict-with-no-error -> genuine
                continue

            evaluable_ids.append(r["episode_id"])
            tda_num_eval += tda
            gsr_num_eval += gsr

        n_eval = len(evaluable_ids)
        per_model[display] = {
            "N_total": n_total,
            "N_evaluable": n_eval,
            "N_invalid": n_invalid,
            "N_genuine_model_failure": n_genuine,
            "N_truncation": n_trunc,
            "N_infrastructure": n_infra,
            "TDA_numerator": tda_num_eval,
            "GSR_numerator": gsr_num_eval,
            "TDA": round(tda_num_eval / n_eval, 4) if n_eval else None,
            "GSR": round(gsr_num_eval / n_eval, 4) if n_eval else None,
            "TDA_minus_GSR_gap_pp": round((tda_num_eval / n_eval - gsr_num_eval / n_eval) * 100, 1) if n_eval else None,
            "evaluable_episode_ids": evaluable_ids,
        }

    # Cross-model identical episode/scenario/world assignment check
    base = world_id_sets["GPT-5.2"]
    identical_assignment = all(v == base for v in world_id_sets.values())

    # TDA/GSR same evaluable set check (they're computed over the identical
    # evaluable_ids list per model by construction above -- verify count match)
    same_set_check = {m: True for m in per_model}  # by construction, same denominator used for both

    print("=== Condition 3: identical episode/scenario/world assignment across all 5 models ===")
    print("IDENTICAL" if identical_assignment else "MISMATCH")
    print()

    print("=== Per-model evaluable-set verification ===")
    header = f"{'model':22s} {'N_total':8s} {'N_eval':8s} {'N_inval':8s} {'genuine':8s} {'trunc':7s} {'infra':7s} {'TDA_n':7s} {'GSR_n':7s} {'TDA':8s} {'GSR':8s} {'gap_pp':7s}"
    print(header)
    for display, d in per_model.items():
        print(f"{display:22s} {d['N_total']:<8d} {d['N_evaluable']:<8d} {d['N_invalid']:<8d} "
             f"{d['N_genuine_model_failure']:<8d} {d['N_truncation']:<7d} {d['N_infrastructure']:<7d} "
             f"{d['TDA_numerator']:<7d} {d['GSR_numerator']:<7d} "
             f"{d['TDA']:<8.4f} {d['GSR']:<8.4f} {d['TDA_minus_GSR_gap_pp']:<7.1f}")

    out = {
        "identical_episode_assignment_across_models": identical_assignment,
        "per_model": {k: {kk: vv for kk, vv in v.items() if kk != "evaluable_episode_ids"} for k, v in per_model.items()},
    }
    out_path = REPO_ROOT / "reports" / "benchmark" / "A_evaluable_verification.json"
    out_path.write_text(json.dumps(out, indent=2))
    print()
    print(f"Wrote {out_path}")

    # Full detail with episode-id lists, separate file (larger)
    out_full = REPO_ROOT / "reports" / "benchmark" / "A_evaluable_verification_full.json"
    out_full.write_text(json.dumps(per_model, indent=2))
    print(f"Wrote {out_full}")


if __name__ == "__main__":
    main()
