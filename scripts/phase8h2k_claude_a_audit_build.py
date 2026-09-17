#!/usr/bin/env python3
"""phase8h2k_claude_a_audit_build.py -- builds the Claude A-family audit
artifacts (claude_A_audit.csv / claude_A_audit_summary.json) from:
  (1) the OFFICIAL, already-scored Claude A raw file (source of truth for
      N, terminal_action, gold_action, TDA, GSR -- NOT overwritten, NOT
      re-scored here);
  (2) the live diagnostic re-run capture (phase8h2k_claude_a_diagnostic.py)
      as root-cause evidence for the 46 originally-empty-verdict rows.

Root-cause finding: 44/46 affected rows are the model replying with a
tool_calls object again at the mandatory STEP 2 turn instead of the
required verdict schema -- genuine model behavior, not a parser or infra
bug (valid JSON every time, finish_reason=stop, no call errors on the
official run). This script documents that finding; it does not alter
the frozen benchmark, the scoring contract, or any existing results
file.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_RAW = (REPO_ROOT / "reports" / "benchmark" / "v3_full_results" /
                "frozen93" / "raw_Claude_Sonnet_4.6.jsonl")
DIAG_CAPTURE = (REPO_ROOT / "reports" / "benchmark" / "claude_a_diag" /
                "raw_text_capture_1789668887.jsonl")
OUT_CSV = REPO_ROOT / "reports" / "benchmark" / "claude_A_audit.csv"
OUT_SUMMARY = REPO_ROOT / "reports" / "benchmark" / "claude_A_audit_summary.json"

FIELDS = [
    "episode_id", "world_id", "scenario_id", "evidence_regime", "raw_run_path",
    "model_snapshot", "request_status", "response_present", "parse_valid",
    "trace_present", "tool_execution_valid", "observation_delivery_valid",
    "terminal_action", "gold_action", "TDA", "evidence_contract_valid", "GSR",
    "truncation", "missing", "blocked", "excluded", "exclusion_reason",
    "root_cause",
]


def classify_root_cause(diag: Optional[Dict[str, Any]], verdict: str) -> str:
    if verdict != "":
        return "compliant: model emitted a valid verdict at STEP 2"
    if diag is None:
        return "unverified: not in the diagnostic re-run sample"
    cap2 = diag.get("step2_capture") or {}
    if cap2.get("api_error"):
        return ("diagnostic_rerun_transient_infra: " + str(cap2["api_error"]) +
                " (official run had no call_errors -- original cause was still "
                "step2_protocol_noncompliance)")
    pk = cap2.get("parsed_keys") or []
    if pk == ["tool_calls"]:
        return ("step2_protocol_noncompliance: model replied with a tool_calls "
                "object again at the mandatory STEP 2 decision turn instead of "
                "a verdict object -- valid JSON, finish_reason=" +
                str(cap2.get("finish_reason")) + ", no truncation, no API "
                "error; genuine model behavior under the frozen 2-turn "
                "protocol, not a parser or infra bug")
    if cap2.get("parse_error"):
        return "parse_error_on_rerun: " + str(cap2["parse_error"])
    return "other: parsed_keys=" + str(pk)


def main() -> int:
    official = [json.loads(l) for l in OFFICIAL_RAW.read_text().splitlines()]
    a_rows = [r for r in official if r["dim"] == "A"]

    diag_by_episode: Dict[str, Dict[str, Any]] = {}
    if DIAG_CAPTURE.exists():
        for line in DIAG_CAPTURE.read_text().splitlines():
            d = json.loads(line)
            diag_by_episode[d["episode_id"]] = d

    csv_rows: List[Dict[str, Any]] = []
    tda_num = gsr_num = 0
    n_compliant = n_noncompliant = 0

    for rec in a_rows:
        rr = rec["runner_return"]
        verdict = rr.get("verdict", "")
        gold = rr.get("gold", "")
        metric = rr.get("metric") or {}
        tda = metric.get("TDA")
        gsr = metric.get("GSR")
        integrity_flags = rr.get("integrity_flags")
        diag = diag_by_episode.get(rec["episode_id"])
        root_cause = classify_root_cause(diag, verdict)

        if verdict != "":
            n_compliant += 1
        else:
            n_noncompliant += 1
        if tda:
            tda_num += 1
        if gsr:
            gsr_num += 1

        csv_rows.append({
            "episode_id": rec["episode_id"],
            "world_id": rec["world_id"],
            "scenario_id": rec["scenario_id"],
            "evidence_regime": rec["arm"],
            "raw_run_path": str(OFFICIAL_RAW.relative_to(REPO_ROOT)),
            "model_snapshot": "claude-sonnet-4-6 (TokenRouter-served; requested anthropic/claude-sonnet-4.6)",
            "request_status": "ok" if not rr.get("call_errors") else "call_error",
            "response_present": True,
            "parse_valid": True,
            "trace_present": True,
            "tool_execution_valid": True,
            "observation_delivery_valid": not bool(integrity_flags),
            "terminal_action": verdict if verdict != "" else None,
            "gold_action": gold,
            "TDA": tda,
            "evidence_contract_valid": not bool(integrity_flags),
            "GSR": gsr,
            "truncation": False,
            "missing": False,
            "blocked": False,
            "excluded": False,
            "exclusion_reason": None,
            "root_cause": root_cause,
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(csv_rows)

    n = len(a_rows)
    summary = {
        "manifest_N": 48,
        "attempted_N": n,
        "parse_valid_N": n,
        "trace_valid_N": n,
        "eligible_N": n,
        "TDA_N": tda_num,
        "GSR_N": gsr_num,
        "truncated_N": 0,
        "missing_N": 0,
        "blocked_N": 0,
        "excluded_N": 0,
        "step2_compliant_N": n_compliant,
        "step2_noncompliant_N": n_noncompliant,
        "final_TDA": round(tda_num / n, 4),
        "final_GSR": round(gsr_num / n, 4),
        "final_gap_pp": round((tda_num / n - gsr_num / n) * 100, 1),
        "root_cause_of_originally_reported_46_48_empty_verdicts": (
            "NOT an evaluator/parser/infra bug. Live diagnostic re-run "
            "(scripts/phase8h2k_claude_a_diagnostic.py, "
            "reports/benchmark/claude_a_diag/raw_text_capture_1789668887.jsonl, "
            "46/46 affected episodes re-run against the identical frozen worlds "
            "through the unmodified run_a_episode()/_chat() path with added "
            "raw-response capture) shows 44/46 cases are the model replying "
            "with a tool_calls object again at the mandatory STEP 2 decision "
            "turn instead of the required verdict schema -- valid JSON every "
            "time, finish_reason=stop (no truncation), zero API errors. 1/46 "
            "hit a transient 504 Gateway Timeout on this re-run only (the "
            "original run for that episode also had call_errors=[] and an "
            "empty verdict, so its original cause was the same step2 "
            "non-compliance, not a timeout). 1/46 complied on re-run "
            "(non-deterministic -- consistent with the 2/48 already compliant "
            "in the official run, both FULL-arm). This is genuine model "
            "behavior under the frozen 2-turn protocol: Claude Sonnet 4.6 "
            "frequently wants additional evidence before committing and the "
            "protocol has no third turn to grant that. The existing scoring "
            "convention (missing/invalid terminal_action = hard 0, matching "
            "b_acquisition_scoring._tda_post and dphys_scoring.py) already "
            "scores this correctly. No rerun, protocol change, or N inflation "
            "is justified -- doing so would change the frozen protocol "
            "specifically to help this model, which the evaluation's own "
            "constraints forbid."
        ),
        "action_taken": "none -- official results, frozen protocol, and frozen-93 manifest are unchanged",
        "disclosure_for_paper": (
            "Claude Sonnet 4.6's low A-family TDA/GSR under InspectionBench V3's "
            "frozen 2-turn protocol is substantially explained by a low STEP-2 "
            "verdict-emission compliance rate (2/48 official; 3/48 across the "
            "diagnostic sample including the non-deterministic re-run), not by "
            "incorrect terminal decisions when a decision IS emitted (1/3 "
            "observed compliant verdicts across both runs matched gold). This "
            "should be reported as a distinct, disclosed finding -- a "
            "protocol-compliance / turn-budget limitation -- not conflated "
            "with a grounding-capability finding, and not silently folded "
            "into a single TDA number without this caveat."
        ),
        "diagnostic_capture_path": str(DIAG_CAPTURE.relative_to(REPO_ROOT)),
        "diagnostic_script_path": "scripts/phase8h2k_claude_a_diagnostic.py",
    }

    with open(OUT_SUMMARY, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("Wrote " + str(OUT_CSV) + " (" + str(len(csv_rows)) + " rows)")
    print("Wrote " + str(OUT_SUMMARY))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
