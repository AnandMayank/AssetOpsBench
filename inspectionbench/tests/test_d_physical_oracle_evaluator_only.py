"""Phase 3 structural regression: the D-physical design requires the
SpotAdmissibilityVerifier oracle to remain evaluator-only. If
check_admissibility (or any oracle-verdict field) ever becomes agent-visible,
D-physical degenerates into "call the oracle, relay its answer" -- tool
obedience, not constraint integration (see reports/paper/d_candidate_taxonomy.md
Sec. 5). This test fails loudly if that boundary is ever crossed.

Zero model/API calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

from couchdb_executor import TOOLSET  # noqa: E402


def test_check_admissibility_not_in_agent_toolset():
    assert "check_admissibility" not in TOOLSET, (
        "check_admissibility must never be exposed to the agent -- it collapses "
        "D-physical's constraint-integration construct into tool-call obedience"
    )


def test_check_cdc_not_in_agent_toolset():
    """check_cdc (world-model calibration domain) is the other pre-dispatch
    oracle-shaped tool on the robot MCP server; same rule applies."""
    assert "check_cdc" not in TOOLSET


ORACLE_VERDICT_FIELDS = {"admissible", "violated_constraints", "limiting_constraint",
                         "admissibility_verdict", "checks"}


def test_agent_visible_toolset_is_exactly_the_frozen_set():
    """Pin TOOLSET's membership so a future addition of an oracle-shaped tool
    (or accidental removal of an existing one) is caught explicitly rather
    than silently changing the agent's capability surface."""
    expected = {
        "navigate_to", "get_pose", "get_battery", "list_waypoints",
        "safety_gate_check", "open_panel", "capture_image", "read_gauge",
        "read_iot", "get_work_order", "get_asset_state",
        "sit", "stand", "dock", "power_on", "commit_reading",
        "read_thermal_image", "commit_thermal_decision",
        "select_capability", "get_sensor_history", "read_vibration", "escalate",
        "read_acoustic", "commit_acoustic_decision",
        "request_observation",  # Family B (Evidence Acquisition), Phase 8H.2G --
        # a legitimate agent-visible acquisition tool, distinct from the
        # oracle-verdict tools (check_admissibility/check_cdc) this test set
        # exists to keep out; see test_b_acquisition_oracle_evaluator_only.py
        # for the analogous B-specific evaluator-only boundary tests.
    }
    assert set(TOOLSET) == expected
    assert set(TOOLSET) & ORACLE_VERDICT_FIELDS == set(), (
        "no tool name may collide with an oracle-verdict field name"
    )
