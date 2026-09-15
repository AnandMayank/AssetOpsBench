"""test_world_gold.py — Pass R4: WORLD, GOLD=f(W), matched counterfactuals.

Requires the AssetOpsBenchScenarioGeneration checkout (same convention as
test_scenario_contract.py / test_rc004_gold_trees.py). No network calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ORCH_SRC = Path(__file__).resolve().parents[1]  # src/orchestrator
sys.path.insert(0, str(_ORCH_SRC))

_SCENARIOS_DIR = Path.home() / "AssetOpsBenchScenarioGeneration" / "RobotInspection" / "Scenarios"
requires_scenario_repo = pytest.mark.skipif(
    not _SCENARIOS_DIR.exists(), reason="AssetOpsBenchScenarioGeneration checkout not found")

if _SCENARIOS_DIR.exists():
    import scenario_constructor as SCon
    import scenario_contract as SC
    import world as W


# --------------------------------------------------------------------- #1
@requires_scenario_repo
def test_world_round_trips_to_dict():
    w = W.build_world_for_scenario("R049")
    d = w.to_dict()
    assert d["asset_id"] == "motor_01"
    assert d["evidence_provenance"] == "L2_ASSET_CLASS_EVIDENCE_REPLAY"
    # unknown fields are explicit, not omitted
    assert d["site_id"] is None
    assert d["inspection_point_id"] is None


@requires_scenario_repo
def test_typed_gold_and_world_agree_on_asset():
    reg = SC.build_registry()
    for sid in ("R049", "R090", "R009"):
        w = W.build_world_for_scenario(sid)
        assert w.asset_id == reg[sid].asset_id


# --------------------------------------------------------------------- #2
@requires_scenario_repo
def test_deterministic_gold_from_world_matches_authored_gold():
    results = W.verify_gauge_band_derivation()
    assert results, "no gauge-band scenarios verified -- family is empty"
    assert all(results.values()), f"derivation drifted from authored gold: {results}"


@requires_scenario_repo
def test_deterministic_gold_source_is_declared():
    reg = SC.build_registry()
    derived = [sid for sid, g in reg.items() if g.gold_source == "derived_from_world"]
    assert set(derived) == set(W.GAUGE_BAND_DERIVED_SCENARIOS)
    assert len(derived) >= 1  # the stop condition's literal minimum


@requires_scenario_repo
def test_derived_from_world_list_matches_world_module():
    """scenario_contract._GOLD_SOURCE_OVERRIDE duplicates
    world.GAUGE_BAND_DERIVED_SCENARIOS (to avoid a circular import) -- this
    is the guard that catches the two ever diverging."""
    derived_in_contract = {sid for sid, v in SC._GOLD_SOURCE_OVERRIDE.items()
                           if v == "derived_from_world"}
    assert derived_in_contract == set(W.GAUGE_BAND_DERIVED_SCENARIOS)


@requires_scenario_repo
def test_derive_gold_gauge_band_actually_recomputes_not_lookup():
    """Not a lookup table in disguise -- perturbing the hidden value across
    the band boundary must flip the derived gold."""
    w = W.build_world_for_scenario("R009")
    band = w.operational_state["band"]
    w.operational_state["value"] = band[0] - 1  # just outside, below
    assert W.derive_gold_gauge_band(w) == "ESCALATE"
    w.operational_state["value"] = (band[0] + band[1]) / 2  # midpoint, inside
    assert W.derive_gold_gauge_band(w) == "COMMIT"


@requires_scenario_repo
def test_derive_gold_gauge_band_returns_none_when_world_lacks_the_fields():
    """Must not silently default to a verdict for a WORLD this rule cannot
    evaluate (e.g. a thermal scenario, which has no scalar value/band)."""
    w = W.build_world_for_scenario("R049")
    assert W.derive_gold_gauge_band(w) is None


# --------------------------------------------------------------------- #3
@requires_scenario_repo
def test_r004_conditional_gold_preserved_not_collapsed():
    reg = SC.build_registry()
    r004 = reg["R004"]
    assert r004.conditional_gold is not None
    assert r004.conditional_gold.branches == {"true": "COMMIT", "false": "ESCALATE"}
    # gold_action still exists as R1's single chosen value (unresolved
    # conflict, per the binding "preserve conflicts" rule) -- conditional_gold
    # is additive, not a replacement.
    assert r004.gold_action in ("COMMIT", "ESCALATE")


@requires_scenario_repo
def test_conditional_gold_round_trips_through_json():
    reg = SC.build_registry()
    r004 = reg["R004"]
    d = r004.to_dict()
    assert d["conditional_gold"]["branches"] == {"true": "COMMIT", "false": "ESCALATE"}
    reloaded = SC.ScenarioGold.from_dict(d)
    assert reloaded.conditional_gold == r004.conditional_gold


# --------------------------------------------------------------------- #4
@requires_scenario_repo
def test_matched_scenarios_differ_only_in_intended_condition():
    w = W.build_world_for_scenario("R009")
    gold = W.derive_gold_gauge_band(w)
    base = SCon.baseline_spec(
        w, "R009_S0", evidence_visibility=("physical", "digital"),
        modality_access=("capture_image", "read_gauge"),
        tool_access=("navigate_to", "get_pose", "capture_image", "read_gauge"),
        gold_action=gold,
    )
    variants = [
        SCon.vary(base, "R009_S1", evidence_visibility=("physical",)),
        SCon.vary(base, "R009_S2", modality_access=()),
        SCon.vary(base, "R009_S3", tool_access=("navigate_to",)),
        SCon.vary(base, "R009_S4", hint_condition="H2"),
        SCon.vary(base, "R009_S5", temporal_condition="asynchronous"),
    ]
    report = SCon.validate_matched_set(base, variants)
    assert all(v["all_pass"] for v in report["variants"].values()), report


@requires_scenario_repo
def test_vary_rejects_more_than_one_condition_at_once():
    w = W.build_world_for_scenario("R009")
    base = SCon.baseline_spec(w, "S0", evidence_visibility=("physical",),
                              modality_access=(), tool_access=(), gold_action="COMMIT")
    with pytest.raises(ValueError):
        SCon.vary(base, "S_bad", evidence_visibility=("digital",), hint_condition="H1")


@requires_scenario_repo
def test_cross_asset_matched_scenario():
    w9 = W.build_world_for_scenario("R009")
    w58 = W.build_world_for_scenario("R058")
    gold9 = W.derive_gold_gauge_band(w9)
    gold58 = W.derive_gold_gauge_band(w58)
    assert gold9 != gold58, "test needs a genuinely different gold to be meaningful"
    base = SCon.baseline_spec(w9, "R009_S0", evidence_visibility=("physical",),
                              modality_access=("capture_image",), tool_access=("navigate_to",),
                              gold_action=gold9)
    cross = SCon.cross_asset_variant(base, "R058_cross", w58, gold58)
    report = SCon.validate_cross_asset_set(base, [cross])
    assert report["variants"]["R058_cross"]["all_pass"]
    assert cross.asset_id == "hydraulic_pump_1"
    assert cross.gold_action == "ESCALATE"


# --------------------------------------------------------------------- #5
@requires_scenario_repo
def test_no_gold_or_source_label_reachable_through_spec_agent_visible_fields():
    """A ScenarioSpec's agent-visible surface (evidence_visibility,
    modality_access, tool_access, hint fields) must never itself carry the
    gold action as a string inside those tuples/fields -- gold_action is a
    separate, clearly-labelled field a renderer must deliberately choose to
    expose, never accidentally embedded in the tool/evidence surface."""
    w = W.build_world_for_scenario("R009")
    gold = W.derive_gold_gauge_band(w)
    spec = SCon.baseline_spec(w, "S0", evidence_visibility=("physical", "digital"),
                              modality_access=("capture_image",), tool_access=("navigate_to",),
                              gold_action=gold)
    agent_visible = (spec.evidence_visibility + spec.modality_access + spec.tool_access
                     + (spec.hint_condition,))
    assert gold not in agent_visible
    assert "source_label" not in str(agent_visible)


# --------------------------------------------------------------------- #6 (carried from R2, re-verified here)
@requires_scenario_repo
def test_wrong_asset_evidence_still_cannot_satisfy_cc_grounded():
    from l3_grounded_scoring import score_l3_grounded
    from execution_trace import ExecutionTrace, Stage

    trace = ExecutionTrace("R049", "IMAGE_GROUNDED")
    for stage, kw in [
        (Stage.REQUESTED, {"tool": "read_thermal_image"}),
        (Stage.EXECUTED, {"tool": "read_thermal_image", "status": "success"}),
        (Stage.SUCCEEDED, {"tool": "read_thermal_image"}),
        (Stage.OBSERVATION_DELIVERED, {"tool": "read_thermal_image", "observation_id": "obs_x",
                                       "observation_hash": "h", "modality": "thermal",
                                       "asset_id": "reva_induction_motor"}),  # WRONG asset
    ]:
        trace.append(stage, **kw)
    out = score_l3_grounded({"action": "SHUTDOWN"}, {"fm": "FM-26", "asset_id": "motor_01"},
                            {"action": ("SHUTDOWN", "ESCALATE")}, trace)
    assert out["CC"] == 1
    assert out["CC_grounded"] == 0
    assert out["asset_match"] is False


# --------------------------------------------------------------------- #7
@requires_scenario_repo
def test_provenance_classes_preserved_and_never_summed():
    reg = SC.build_registry()
    report = SC.coverage_report(reg)
    dist = report["evidence_provenance_distribution"]
    assert sum(dist.values()) == report["total_scenarios"]
    assert "L1_REAL_ASSET_EVIDENCE" not in dist  # still zero, measured not assumed


# --------------------------------------------------------------------- #8
@requires_scenario_repo
def test_two_level_temporal_semantics_are_distinct():
    """capture_type (record level) and temporal_relation (scenario level)
    must be independently settable, never collapsed into one enum."""
    w_thermal = W.build_world_for_scenario("R049")
    assert w_thermal.temporal_information["capture_type"] == "single_capture"
    assert w_thermal.temporal_information["temporal_relation"] == "not_applicable"
    assert set(W.CAPTURE_TYPE_VALUES).isdisjoint(W.TEMPORAL_RELATION_VALUES)


@requires_scenario_repo
def test_reva_timezone_remains_unverified():
    """R2/R3 review binding: REVA's real EXIF timestamps are real capture
    times but their timezone is not independently established -- must never
    be silently asserted as UTC anywhere in WORLD."""
    w = W.build_world_for_scenario("R049")
    assert w.temporal_information.get("timezone_verified") is not True


# --------------------------------------------------------------------- #9
@requires_scenario_repo
def test_observation_delivery_firewall_blocks_undelivered_id():
    """Executor-level proof (see also test_thermal_image_grounding.py's
    dedicated suite) -- included here since it's one of R4's ten named test
    targets."""
    from couchdb_executor import CouchDBExecutor
    from tool_executor import STATUS_SUCCESS, ToolCall

    ex = CouchDBExecutor()
    ex.reset("R049", "IMAGE_GROUNDED", seed=1, withheld=())
    result = ex.execute(ToolCall("commit_thermal_decision", {
        "observation_id": "obs_thermal_R049", "verdict": "fault", "action": "SHUTDOWN",
    }))
    assert result.status != STATUS_SUCCESS
    assert result.payload["grounded"] is False


# --------------------------------------------------------------------- #10
@requires_scenario_repo
def test_existing_thermal_and_acoustic_scoring_unchanged():
    """CC/PROC/ORDERING/CC_grounded semantics must be byte-identical to
    R2/R3 for the correctly-grounded case -- R4 adds fields, it does not
    change how they're scored."""
    from l3_grounded_scoring import score_l3_grounded
    from execution_trace import ExecutionTrace, Stage

    trace = ExecutionTrace("R049", "IMAGE_GROUNDED")
    for stage, kw in [
        (Stage.REQUESTED, {"tool": "read_thermal_image"}),
        (Stage.EXECUTED, {"tool": "read_thermal_image", "status": "success"}),
        (Stage.SUCCEEDED, {"tool": "read_thermal_image"}),
        (Stage.OBSERVATION_DELIVERED, {"tool": "read_thermal_image", "observation_id": "obs_x",
                                       "observation_hash": "h", "modality": "thermal",
                                       "asset_id": "motor_01"}),
        (Stage.REQUESTED, {"tool": "commit_thermal_decision"}),
        (Stage.EXECUTED, {"tool": "commit_thermal_decision", "status": "success"}),
        (Stage.SUCCEEDED, {"tool": "commit_thermal_decision"}),
    ]:
        trace.append(stage, **kw)
    out = score_l3_grounded({"action": "SHUTDOWN"}, {"fm": "FM-26", "asset_id": "motor_01"},
                            {"action": ("SHUTDOWN", "ESCALATE")}, trace)
    assert out["CC"] == 1 and out["PROC"] == 1 and out["CC_grounded"] == 1
