#!/usr/bin/env python3
"""phase8h2o_a_appendix_build.py -- builds the A-family output-validity
appendix (A1 TDA, A3 validity rate, hard-zero sensitivity), kept
separate from the main table's A column per the reviewed decision:
GSR-on-evaluable(lenient N) is primary, everything else here.
Read-only over existing raw files; writes only to
reports/benchmark/main_table/A_appendix_v1.md.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FROZEN93 = REPO_ROOT / "reports" / "benchmark" / "v3_full_results" / "frozen93"
OUT = REPO_ROOT / "reports" / "benchmark" / "main_table" / "A_appendix_v1.md"

MODELS = [
    ("GPT-5.2", "GPT-5.2"),
    ("Mistral Medium 3.5", "Mistral_Medium_3.5"),
    ("DeepSeek V4 Pro", "DeepSeek_V4_Pro"),
    ("Qwen3.5-397B-A17B", "Qwen3.5-397B-A17B"),
    ("Claude Sonnet 4.6", "Claude_Sonnet_4.6"),
]

ROOT_CAUSE = {
    "Claude Sonnet 4.6": "46/48 step2_protocol_noncompliance (genuine model behavior, confirmed live via full 46-episode re-run)",
    "DeepSeek V4 Pro": "5/48 empty verdict: truncation/output-budget (finish_reason=length confirmed live on 1/5); 24/48 additional episodes had a logged call_error at an earlier turn but still produced a scored verdict (TDA/GSR both computed; kept in the lenient evaluable set per reviewed decision)",
    "Qwen3.5-397B-A17B": "5/48 empty verdict: 2 truncation/output-budget, 3 infrastructure (network timeout/504)",
    "GPT-5.2": "none",
    "Mistral Medium 3.5": "none",
}


def main():
    lines = ["# A-family output-validity appendix",
             "",
             "Kept separate from the main table's A = GSR(evaluable) column per the reviewed",
             "decision: evidence-grounding capability != response/parser/output validity.",
             "",
             "| Model | N_total | N_evaluable | A1 TDA (evaluable) | A2 GSR (evaluable, = main table A) | A3 output-validity rate | Root cause of exclusions |",
             "|---|---|---|---|---|---|---|"]

    for display, suffix in MODELS:
        rows = [json.loads(l) for l in (FROZEN93 / f"raw_{suffix}.jsonl").read_text().splitlines()]
        a_rows = [r for r in rows if r["dim"] == "A"]
        n_total = len(a_rows)
        evaluable = [r for r in a_rows if r["runner_return"].get("verdict", "") != ""]
        n_eval = len(evaluable)
        tda = sum(r["runner_return"]["metric"]["TDA"] for r in evaluable)
        gsr = sum(r["runner_return"]["metric"]["GSR"] for r in evaluable)
        validity_rate = n_eval / n_total
        if n_eval == 0:
            tda_s, gsr_s = "n/a", "n/a"
        else:
            tda_s = f"{tda}/{n_eval} = {tda/n_eval:.1%}"
            gsr_s = f"{gsr}/{n_eval} = {gsr/n_eval:.1%}"
        lines.append(f"| {display} | {n_total} | {n_eval} | {tda_s} | {gsr_s} | "
                     f"{n_eval}/{n_total} = {validity_rate:.1%} | {ROOT_CAUSE[display]} |")

    lines += ["", "## Sensitivity: hard-zero on missing outputs (NOT primary, appendix only)",
             "", "| Model | TDA (hard-zero, N=48) | GSR (hard-zero, N=48) |", "|---|---|---|"]
    for display, suffix in MODELS:
        rows = [json.loads(l) for l in (FROZEN93 / f"raw_{suffix}.jsonl").read_text().splitlines()]
        a_rows = [r for r in rows if r["dim"] == "A"]
        n_total = len(a_rows)
        tda_hz = sum(r["runner_return"]["metric"]["TDA"] for r in a_rows)
        gsr_hz = sum(r["runner_return"]["metric"]["GSR"] for r in a_rows)
        lines.append(f"| {display} | {tda_hz}/{n_total} = {tda_hz/n_total:.1%} | {gsr_hz}/{n_total} = {gsr_hz/n_total:.1%} |")

    lines += ["", "Full per-episode diagnostic: reports/benchmark/A_evaluable_verification.json, "
             "reports/benchmark/A_evaluable_verification_full.json, "
             "reports/benchmark/claude_A_audit.csv, reports/benchmark/DeepSeek_V4_Pro_A_audit.csv, "
             "reports/benchmark/Qwen3.5-397B-A17B_A_audit.csv."]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
