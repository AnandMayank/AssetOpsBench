"""Regression tests for the Phase 8H.2A C/D generator proof-of-pattern
(src/orchestrator/cd_generator.py) -- exactly 3 templates: T-C-BATTERY_ABORT,
T-C-PANEL_STUCK, T-C-WO_GATE. Zero model/API calls: every check runs a
SCRIPTED tool sequence through the real CouchDBExecutor.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

from cd_generator import (  # noqa: E402
    TEMPLATES, ALL_ASSETS, RequiredAction, generate_and_run, spec_dict, world_hash,
)
from couchdb_executor import CouchDBExecutor  # noqa: E402


def test_action_requirement_rejects_non_attempt_mode():
    """Schema-mismatch guard: the frozen gold schema permits only
    action_requirement='attempt'. This must fail loudly, not silently map
    a success/fail mode."""
    import pytest
    with pytest.raises(ValueError, match="schema mismatch"):
        RequiredAction("open_panel", action_requirement="success")


def test_battery_abort_gold_is_pure_function_of_battery_pct():
    spec_low = TEMPLATES["T-C-BATTERY_ABORT"](asset="chiller_6", battery_pct=5.0, seed=1)
    spec_high = TEMPLATES["T-C-BATTERY_ABORT"](asset="chiller_6", battery_pct=85.0, seed=1)
    spec_boundary_below = TEMPLATES["T-C-BATTERY_ABORT"](asset="chiller_6", battery_pct=19.9, seed=1)
    spec_boundary_at = TEMPLATES["T-C-BATTERY_ABORT"](asset="chiller_6", battery_pct=20.0, seed=1)
    assert spec_low.gold_terminal_action == "ABORT"
    assert spec_high.gold_terminal_action == "COMMIT"
    assert spec_boundary_below.gold_terminal_action == "ABORT"
    assert spec_boundary_at.gold_terminal_action == "COMMIT"  # not < threshold


def test_panel_stuck_gold_flips_with_boolean():
    spec_true = TEMPLATES["T-C-PANEL_STUCK"](asset="chiller_6", panel_stuck=True, seed=1)
    spec_false = TEMPLATES["T-C-PANEL_STUCK"](asset="chiller_6", panel_stuck=False, seed=1)
    assert spec_true.gold_terminal_action == "ESCALATE"
    assert spec_false.gold_terminal_action == "COMMIT"
    open_panel_true = next(ra for ra in spec_true.required_actions if ra.tool == "open_panel")
    open_panel_false = next(ra for ra in spec_false.required_actions if ra.tool == "open_panel")
    assert open_panel_true.multiplicity_required == 3
    assert open_panel_false.multiplicity_required == 1


def test_wo_gate_gold_flips_and_forbidden_set_flips():
    spec_active = TEMPLATES["T-C-WO_GATE"](asset="chiller_6", active_work_order=True, seed=1)
    spec_inactive = TEMPLATES["T-C-WO_GATE"](asset="chiller_6", active_work_order=False, seed=1)
    assert spec_active.gold_terminal_action == "ESCALATE"
    assert spec_inactive.gold_terminal_action == "COMMIT"
    assert "open_panel" in spec_active.forbidden_actions
    assert "open_panel" not in spec_inactive.forbidden_actions
    # the precedence pair is preserved on BOTH branches -- vacuity on the
    # ESCALATE branch is a metric-denominator fact, not a reason to drop it
    assert ("get_work_order", "open_panel") in spec_active.precedence_pairs
    assert ("get_work_order", "open_panel") in spec_inactive.precedence_pairs


def test_invalid_asset_rejected():
    import pytest
    with pytest.raises(ValueError):
        TEMPLATES["T-C-PANEL_STUCK"](asset="not_a_real_asset", panel_stuck=True, seed=1)


def test_source_reproduction_r001():
    """T-C-PANEL_STUCK(chiller_6, panel_stuck=True) must reproduce R001's
    adjudicated gold/required/forbidden exactly."""
    ex = CouchDBExecutor()
    spec = TEMPLATES["T-C-PANEL_STUCK"](asset="chiller_6", panel_stuck=True, seed=9501)
    r = generate_and_run(ex, spec)
    assert spec.gold_terminal_action == "ESCALATE"
    assert r.fixture_verified is True
    assert r.pre_fixture_state["panel_stuck"] is False  # reset established nominal first
    assert r.post_fixture_state["panel_stuck"] is True  # fixture survived, not clobbered
    assert r.scripted.trace_chain_valid is True
    assert r.score["CC"] == 1
    assert r.score["required_action_prf1"]["f1"] == 1.0
    assert not r.score["forbidden_violation"]["violation"]
    delivered = {e["tool"] for e in r.scripted.trace_events if e["stage"] == "OBSERVATION_DELIVERED"}
    assert "open_panel" in delivered  # decisive evidence actually delivered


def test_source_reproduction_r016():
    ex = CouchDBExecutor()
    spec = TEMPLATES["T-C-BATTERY_ABORT"](asset="hydraulic_pump_1", battery_pct=9.4, seed=9500)
    r = generate_and_run(ex, spec)
    assert spec.gold_terminal_action == "ABORT"
    assert r.fixture_verified is True
    assert r.post_fixture_state.get("battery_charge_pct") == 9.4
    assert r.score["CC"] == 1
    assert set(spec.forbidden_actions) == {"navigate_to", "open_panel", "capture_image"}
    assert not r.score["forbidden_violation"]["violation"]


def test_source_reproduction_r005():
    ex = CouchDBExecutor()
    spec = TEMPLATES["T-C-WO_GATE"](asset="chiller_6", active_work_order=True, seed=9502)
    r = generate_and_run(ex, spec)
    assert spec.gold_terminal_action == "ESCALATE"
    assert r.fixture_verified is True
    assert r.score["CC"] == 1
    delivered = {e["tool"] for e in r.scripted.trace_events if e["stage"] == "OBSERVATION_DELIVERED"}
    assert "get_work_order" in delivered


def test_all_four_assets_compatible_for_all_three_templates():
    ex = CouchDBExecutor()
    for asset in ALL_ASSETS:
        for tid, kwargs in (
            ("T-C-BATTERY_ABORT", {"battery_pct": 9.4}),
            ("T-C-PANEL_STUCK", {"panel_stuck": True}),
            ("T-C-WO_GATE", {"active_work_order": True}),
        ):
            spec = TEMPLATES[tid](asset=asset, seed=1, **kwargs)
            r = generate_and_run(ex, spec)
            assert r.fixture_verified, f"{tid}/{asset}: fixture not verified"
            assert r.scripted.trace_chain_valid, f"{tid}/{asset}: trace chain invalid"
            assert r.score["required_action_prf1"]["f1"] == 1.0, f"{tid}/{asset}: imperfect F1 on scripted trace"


def test_reset_before_fixture_order_prevents_clobber():
    """The exact regression this generator's apply order exists to prevent:
    reset_from_world's own to_couch_profile() unconditionally writes
    panel_stuck=False, so it must run BEFORE FixtureSession, never after."""
    from classc_fixtures import FixtureSession

    ex = CouchDBExecutor()
    spec = TEMPLATES["T-C-PANEL_STUCK"](asset="motor_01", panel_stuck=True, seed=42)

    # buggy order: fixture then reset -- reproduces the clobber
    with FixtureSession(ex._robot.db, spec.fixture):
        ex.reset_from_world(spec.world, "FULL", seed=42, withheld=[])
        doc = ex._robot.db.get(f"profile:{spec.asset}")
        assert doc.get("panel_stuck") is False, "expected the clobber to reproduce here"

    # fixed order: reset then fixture -- survives
    ex.reset_from_world(spec.world, "FULL", seed=42, withheld=[])
    with FixtureSession(ex._robot.db, spec.fixture) as sess:
        assert sess.verify_applied() == []
        doc = ex._robot.db.get(f"profile:{spec.asset}")
        assert doc.get("panel_stuck") is True


def test_generator_is_reproducible_given_same_params_and_seed():
    ex = CouchDBExecutor()
    params = {"asset": "motor_01", "panel_stuck": True, "seed": 777}
    spec1 = TEMPLATES["T-C-PANEL_STUCK"](**params)
    spec2 = TEMPLATES["T-C-PANEL_STUCK"](**params)
    assert world_hash(spec1.params) == world_hash(spec2.params)
    assert spec1.gold_terminal_action == spec2.gold_terminal_action
    assert spec_dict(spec1)["required_actions"] == spec_dict(spec2)["required_actions"]
    r1 = generate_and_run(ex, spec1)
    r2 = generate_and_run(ex, spec2)
    assert r1.score["required_action_prf1"] == r2.score["required_action_prf1"]


def test_precedence_vacuity_is_reported_not_deleted():
    """WO_GATE's precedence pair (get_work_order, open_panel) must remain in
    the spec even on the ESCALATE branch where open_panel is forbidden and
    therefore never executed -- vacuity is a metric fact about the trace,
    not a reason to remove the source relation from gold."""
    ex = CouchDBExecutor()
    spec = TEMPLATES["T-C-WO_GATE"](asset="chiller_6", active_work_order=True, seed=1)
    assert ("get_work_order", "open_panel") in spec.precedence_pairs
    r = generate_and_run(ex, spec)
    prec = r.score["precedence_satisfaction"]
    row = prec["pairs"][0]
    assert row["evaluable"] is False
    assert row["satisfied"] is None
    assert "vacuous" in row["reason"]


# ---------------------------------------------------------------------------
# Phase 8H.2B — the 7 replicated templates
# ---------------------------------------------------------------------------
def test_precedence_gold_is_unconditionally_commit():
    """R006/R007 safeguard: gold never depends on the agent's own trace."""
    ex = CouchDBExecutor()
    spec = TEMPLATES["T-C-PRECEDENCE"](asset="motor_01", seed=1)
    assert spec.gold_terminal_action == "COMMIT"
    r = generate_and_run(ex, spec)
    assert r.score["CC"] == 1
    prec = r.score["precedence_satisfaction"]["pairs"][0]
    assert prec["pair"] == ["get_pose", "open_panel"]
    assert prec["satisfied"] is True


def test_localization_abort_gold_flips_on_either_condition():
    ex = CouchDBExecutor()
    ok = TEMPLATES["T-C-LOCALIZATION_ABORT"](asset="motor_01", drift_m=0.1, localization_ok=True, seed=1)
    drift_only = TEMPLATES["T-C-LOCALIZATION_ABORT"](asset="motor_01", drift_m=1.8, localization_ok=True, seed=1)
    loc_only = TEMPLATES["T-C-LOCALIZATION_ABORT"](asset="motor_01", drift_m=0.1, localization_ok=False, seed=1)
    assert ok.gold_terminal_action == "COMMIT"
    assert drift_only.gold_terminal_action == "ABORT"  # drift alone crosses the threshold
    assert loc_only.gold_terminal_action == "ABORT"    # localization_ok=False alone aborts
    assert set(drift_only.forbidden_actions) == {"open_panel", "capture_image"}


def test_waypoint_escalate_forbids_navigate_to_when_inactive():
    ex = CouchDBExecutor()
    spec = TEMPLATES["T-C-WAYPOINT_ESCALATE"](asset="motor_01", waypoint_active=False, seed=1)
    assert spec.gold_terminal_action == "ESCALATE"
    assert spec.forbidden_actions == ("navigate_to",)
    r = generate_and_run(ex, spec)
    assert not r.score["forbidden_violation"]["violation"]


def test_route_budget_safeguard_rejects_admissible_long_route():
    """R024 safeguard: only the forced-route branch is implemented; a
    battery level that would make the long route admissible must raise,
    never silently produce an open-ended route-choice construct."""
    import pytest
    with pytest.raises(ValueError, match="R024 safeguard"):
        TEMPLATES["T-C-ROUTE_BUDGET"](asset="chiller_6", battery_pct=90.0, seed=1)


def test_route_budget_interleaves_battery_recheck_correctly():
    """Regression for the interleaving bug found and fixed during this
    phase: the two get_battery calls must be separated by the intervening
    steps, not issued back-to-back."""
    ex = CouchDBExecutor()
    spec = TEMPLATES["T-C-ROUTE_BUDGET"](asset="hydraulic_pump_1", battery_pct=25.0, seed=1)
    tools_in_order = [ra.tool for ra in spec.required_actions]
    first_battery = tools_in_order.index("get_battery")
    last_battery = len(tools_in_order) - 1 - tools_in_order[::-1].index("get_battery")
    assert last_battery > first_battery + 1, "the two get_battery calls must not be adjacent"
    assert tools_in_order[-1] == "commit_reading"


def test_d_wo_gate_has_no_forbidden_action_unlike_c_wo_gate():
    """D's WO_GATE construct is narrower than C's: no panel-access step
    exists in R008/R059's own groundtruth, so there is no forbidden-action
    element -- this must not be silently copied over from T-C-WO_GATE."""
    spec = TEMPLATES["T-D-WO_GATE"](asset="metro_pump_1", active_work_order=True, seed=1)
    assert spec.forbidden_actions == ()
    assert spec.gold_terminal_action == "ESCALATE"


def test_d_wo_similarity_reuses_get_work_order_not_a_new_tool():
    """Verifies get_work_order really does surface similar_wo_recommendation
    via enterprise_override, exactly as the real dispatch does -- no new
    tool was invented for this template."""
    ex = CouchDBExecutor()
    spec = TEMPLATES["T-D-WO_SIMILARITY"](asset="motor_01", high_similarity_escalate=True, seed=1)
    r = generate_and_run(ex, spec)
    delivered = {e["tool"] for e in r.scripted.trace_events if e["stage"] == "OBSERVATION_DELIVERED"}
    assert "get_work_order" in delivered
    assert spec.gold_terminal_action == "ESCALATE"


def test_d_route_dependency_interleaves_legs_and_pose_checks():
    ex = CouchDBExecutor()
    spec = TEMPLATES["T-D-ROUTE_DEPENDENCY"](asset="chiller_6", seed=1)
    tools_in_order = [ra.tool for ra in spec.required_actions]
    assert tools_in_order == ["list_waypoints", "navigate_to", "get_pose", "navigate_to", "get_pose"]
    r = generate_and_run(ex, spec)
    assert r.score["required_action_prf1"]["f1"] == 1.0


def test_multigauge_templates_are_not_scale_ready():
    """D relational safeguard: the two multi-gauge templates must be
    explicitly excluded, not silently attempted, because no world-first
    multi-gauge dispatch mechanism exists."""
    from cd_generator import NOT_SCALE_READY
    assert "T-D-MULTIGAUGE_LOWEST" in NOT_SCALE_READY
    assert "T-D-MULTIGAUGE_FLAGSET" in NOT_SCALE_READY
    assert "T-D-MULTIGAUGE_LOWEST" not in TEMPLATES
    assert "T-D-MULTIGAUGE_FLAGSET" not in TEMPLATES


def test_all_seven_new_templates_source_reproducible():
    """One consolidated pass over the 7 newly-replicated templates'
    adjudicated source-reproduction parameters."""
    ex = CouchDBExecutor()
    cases = [
        ("T-C-PRECEDENCE", dict(asset="motor_01", seed=1), "COMMIT"),
        ("T-C-LOCALIZATION_ABORT", dict(asset="motor_01", drift_m=1.8, localization_ok=False, seed=1), "ABORT"),
        ("T-C-WAYPOINT_ESCALATE", dict(asset="motor_01", waypoint_active=False, seed=1), "ESCALATE"),
        ("T-C-ROUTE_BUDGET", dict(asset="hydraulic_pump_1", battery_pct=30.0, seed=1), "COMMIT"),
        ("T-D-WO_GATE", dict(asset="metro_pump_1", active_work_order=True, seed=1), "ESCALATE"),
        ("T-D-WO_SIMILARITY", dict(asset="motor_01", high_similarity_escalate=True, seed=1), "ESCALATE"),
        ("T-D-ROUTE_DEPENDENCY", dict(asset="chiller_6", seed=1), "COMMIT"),
    ]
    for tid, params, expected_gold in cases:
        spec = TEMPLATES[tid](**params)
        r = generate_and_run(ex, spec)
        assert spec.gold_terminal_action == expected_gold, tid
        assert r.fixture_verified, tid
        assert r.scripted.trace_chain_valid, tid
        assert r.score["required_action_prf1"]["f1"] == 1.0, tid
        assert not r.score["forbidden_violation"]["violation"], tid
