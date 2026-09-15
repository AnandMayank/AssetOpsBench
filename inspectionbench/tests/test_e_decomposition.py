"""Regression tests for e_decomposition.py against the REAL recorded 6-sequence
artifact reports/ec/phase8h1_e_rerun_raw.json (no API, no model calls).

Pins the mechanical-vs-behavioral rule, the pre-episode battery reconstruction
(used_before = battery_reads_used_after - reads_this_episode), and both
denominator conventions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

from e_decomposition import (  # noqa: E402
    decompose_e, classify_e_episode, validate_battery_accounting,
    BEHAVIORAL_GROUNDING_FAILURE, MECHANICAL_FORCED_BUDGET,
    TASK_DECISION_FAILURE, GROUNDED_SUCCESS)

ARTIFACT = REPO_ROOT / "reports" / "ec" / "phase8h1_e_rerun_raw.json"
ROWS = json.loads(ARTIFACT.read_text())["results"]

#: Expected classification for every one of the 18 recorded episodes.
#: The 7 TDA=1/GSR=0 rows match reports/ec/phase8h1_E_decomposition.md's table
#: verbatim; the GROUNDED_SUCCESS / TASK_DECISION_FAILURE rows are the rest.
EXPECTED = {
    ("SEQ-5000", 0): BEHAVIORAL_GROUNDING_FAILURE,
    ("SEQ-5000", 1): GROUNDED_SUCCESS,
    ("SEQ-5000", 2): GROUNDED_SUCCESS,
    ("SEQ-5001", 0): GROUNDED_SUCCESS,
    ("SEQ-5001", 1): GROUNDED_SUCCESS,
    ("SEQ-5001", 2): MECHANICAL_FORCED_BUDGET,
    ("SEQ-5002", 0): BEHAVIORAL_GROUNDING_FAILURE,
    ("SEQ-5002", 1): GROUNDED_SUCCESS,
    ("SEQ-5002", 2): GROUNDED_SUCCESS,
    ("SEQ-5003", 0): GROUNDED_SUCCESS,
    ("SEQ-5003", 1): TASK_DECISION_FAILURE,   # CC flipped to 0 in this rerun
    ("SEQ-5003", 2): MECHANICAL_FORCED_BUDGET,
    ("SEQ-5004", 0): BEHAVIORAL_GROUNDING_FAILURE,
    ("SEQ-5004", 1): BEHAVIORAL_GROUNDING_FAILURE,
    ("SEQ-5004", 2): BEHAVIORAL_GROUNDING_FAILURE,
    ("SEQ-5005", 0): GROUNDED_SUCCESS,
    ("SEQ-5005", 1): GROUNDED_SUCCESS,
    ("SEQ-5005", 2): TASK_DECISION_FAILURE,
}


def _idx(row):
    return int(row["arm"][2:])


def test_every_recorded_episode_classifies_as_expected():
    for row in ROWS:
        k = _idx(row)
        got = classify_e_episode(row, episode_index=k).classification
        assert got == EXPECTED[(row["world_id"], k)], (
            f"{row['world_id']}#{k}: got {got}, expected {EXPECTED[(row['world_id'], k)]}")


def test_pre_episode_battery_reconstruction_is_boundary_consistent():
    per_seq = {}
    for row in ROWS:
        per_seq.setdefault(row["world_id"], []).append(row)
    anomalies = validate_battery_accounting(per_seq)
    assert anomalies == [], f"battery ledger anomalies: {anomalies}"


def test_used_before_matches_the_decomposition_doc_table():
    # doc column "reads used before this ep" for the 7 TDA=1/GSR=0 rows
    doc_used_before = {
        ("SEQ-5000", 0): 0, ("SEQ-5001", 2): 2, ("SEQ-5002", 0): 0,
        ("SEQ-5003", 2): 2, ("SEQ-5004", 0): 0, ("SEQ-5004", 1): 0, ("SEQ-5004", 2): 0,
    }
    for row in ROWS:
        key = (row["world_id"], _idx(row))
        if key not in doc_used_before:
            continue
        c = classify_e_episode(row, episode_index=key[1])
        assert c.used_before == doc_used_before[key], (
            f"{key}: used_before {c.used_before} != doc {doc_used_before[key]}")


def test_forced_budget_and_behavioral_fractions_match_the_doc():
    res = decompose_e(ROWS)
    # these two are the numbers the grounding-gap claim rests on
    assert res["Delta_E_forced_budget_over_18"] == 0.1111      # 2/18, doc-exact
    assert res["Delta_E_residual_behavioral_over_18"] == 0.2778  # 5/18, doc-exact
    assert res["counts"][BEHAVIORAL_GROUNDING_FAILURE] == 5
    assert res["counts"][MECHANICAL_FORCED_BUDGET] == 2
    assert res["n_sequences"] == 6
    assert res["battery_anomalies"] == []


def test_both_denominators_emitted_and_labeled():
    res = decompose_e(ROWS)
    for stem in ("Delta_E_total", "Delta_E_forced_budget",
                 "Delta_E_residual_behavioral", "Delta_E_task_decision_failure"):
        assert f"{stem}_over_18" in res
        assert f"{stem}_over_Npaired" in res


def test_seq5004_is_the_all_behavioral_sequence():
    res = decompose_e(ROWS)
    assert res["per_sequence"]["SEQ-5004"] == [BEHAVIORAL_GROUNDING_FAILURE] * 3


def test_reads_this_episode_from_trace_overrides_the_int_reacquired_fallback():
    # a synthetic episode whose trace shows TWO physical reads in one episode
    from e_decomposition import reads_this_episode_from_trace
    tr = {"events": [
        {"stage": "EXECUTED", "tool": "read_gauge"},
        {"stage": "EXECUTED", "tool": "read_gauge"},
        {"stage": "EXECUTED", "tool": "get_work_order"},
    ]}
    assert reads_this_episode_from_trace(tr) == 2
    row = {"world_id": "SEQ-9", "raw_CC": 1, "raw_CCg": 0,
           "reacquired_this_episode": True, "reobservation_was_necessary": True,
           "battery_reads_used": 2, "battery_budget": 2}
    c = classify_e_episode(row, episode_index=1, reads_this_episode=2)
    assert c.used_before == 0
