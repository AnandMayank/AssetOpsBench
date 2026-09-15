"""Phase 4/6 regression tests for dphys_generator.py -- the D-physical
template builders and typed relational gold contract. Zero model/API calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

import pytest  # noqa: E402

ASSET_PROFILES_PATH = (
    Path.home() / "AssetOpsBenchScenarioGeneration" / "RobotInspection" / "Scenarios"
    / "asset_profiles.json"
)
pytestmark = pytest.mark.skipif(
    not ASSET_PROFILES_PATH.exists(),
    reason="AssetOpsBenchScenarioGeneration sibling repo not present in this environment",
)

import dphys_generator as D  # noqa: E402
from couchdb_executor import CouchDBExecutor  # noqa: E402


# --- gold contract -----------------------------------------------------------

def test_relational_gold_has_all_six_constraint_keys():
    spec = D.build_reach_or_energy("chiller_6", "reach", "inadmissible", seed=1)
    assert set(spec.gold.constraints) == set(D.CONSTRAINT_NAMES)


def test_single_binding_inadmissible_violates_exactly_one_constraint():
    """Each single-binding override is verified to isolate exactly its named
    constraint -- including `reach`, which required an explicit
    commanded_arm_qpos override to decouple it from joint_and_collision (see
    dphys_generator._reach_only_override's docstring for the real physical
    coupling this works around)."""
    for constraint in D.CONSTRAINT_NAMES:
        asset = "hydraulic_pump_1" if constraint in (
            "clearance", "grasp_payload", "stability", "energy") else "chiller_6"
        spec = D.build_reach_or_energy(asset, constraint, "inadmissible", seed=1)
        violated = [n for n, v in spec.gold.constraints.items() if v == "VIOLATED"]
        assert violated == [constraint], f"{constraint}: expected exactly one violation, got {violated}"
        assert spec.gold.limiting_constraint == constraint
        assert spec.gold.admissible is False
        assert spec.gold_terminal_action == "ESCALATE"


def test_single_binding_comfortable_is_admissible():
    for constraint in D.CONSTRAINT_NAMES:
        asset = "hydraulic_pump_1" if constraint in (
            "clearance", "grasp_payload", "stability", "energy") else "chiller_6"
        spec = D.build_reach_or_energy(asset, constraint, "comfortable", seed=1)
        assert spec.gold.admissible is True
        assert spec.gold.limiting_constraint is None
        assert spec.gold_terminal_action == "DISPATCH"


@pytest.mark.parametrize("coupling", D.REALIZABLE_PAIRS + (D.REALIZABLE_TRIPLE,))
def test_coupled_violates_at_least_the_named_constraints(coupling):
    spec = D.build_coupled("hydraulic_pump_1", coupling, seed=1)
    violated = {n for n, v in spec.gold.constraints.items() if v == "VIOLATED"}
    assert set(coupling) <= violated
    assert spec.gold.admissible is False
    assert spec.gold.limiting_constraint == D.MULTIPLE  # >=2 violated (len(coupling) >= 2)
    assert spec.gold_terminal_action == "ESCALATE"


@pytest.mark.parametrize("asset", D.ALL_ASSETS)
@pytest.mark.parametrize("config", D.CANDIDATE_CONFIGS)
def test_capx_gold_matches_config_intent(asset, config):
    spec = D.build_capx(asset, config, seed=1)
    if config == "comfortable_baseline":
        # >=1 candidate admissible at unmodified geometry -- literal
        # all-3-admissible is not physically realizable for these assets
        # (see build_capx's docstring), so this checks >=1, not all 3.
        assert spec.gold.admissible is True
        assert spec.gold.selected_standoff is not None
    elif config == "all_inadmissible":
        assert spec.gold.admissible is False
        assert spec.gold.selected_standoff is None
    else:  # mixed_one_admissible: exactly one candidate admissible
        assert spec.gold.admissible is True
        assert spec.gold.selected_standoff is not None
        from spot_admissibility_verifier import SpotAdmissibilityVerifier
        verifier = SpotAdmissibilityVerifier(D.load_registry())
        profiles = D.load_asset_profiles()
        access = D._base_access(profiles, asset)
        if asset in ("hydraulic_pump_1", "metro_pump_1"):
            access["obstacle_clearance_m"] = 0.65
        admissible_count = sum(
            1 for s in spec.params["standoffs"]
            if verifier.verify_candidate(access, s).admissible
        )
        assert admissible_count == 1


# --- scripted execution: proves the scorer/trace mechanics, not a model -----

@pytest.mark.parametrize("constraint", D.CONSTRAINT_NAMES)
def test_scripted_episode_runs_and_produces_valid_trace(constraint):
    asset = "hydraulic_pump_1" if constraint in (
        "clearance", "grasp_payload", "stability", "energy") else "chiller_6"
    spec = D.build_reach_or_energy(asset, constraint, "inadmissible", seed=1)
    ex = CouchDBExecutor()
    result = D.run_scripted_episode(ex, spec)
    assert result["trace_chain_valid"] is True
    assert result["verdict"] == spec.gold_terminal_action


def test_agent_visible_script_never_calls_check_admissibility():
    assert "check_admissibility" not in D.AGENT_VISIBLE_SCRIPT
    assert "check_cdc" not in D.AGENT_VISIBLE_SCRIPT


def test_scripted_episode_payload_has_no_oracle_verdict_leaked_into_trace_events():
    """The trace's REQUESTED/EXECUTED/OBSERVATION_DELIVERED events (what an
    agent would actually see reconstructed from a transcript) must never
    contain the oracle's admissible/violated_constraints/checks fields."""
    spec = D.build_reach_or_energy("chiller_6", "reach", "inadmissible", seed=1)
    ex = CouchDBExecutor()
    result = D.run_scripted_episode(ex, spec)
    forbidden = {"admissible", "violated_constraints", "checks", "admissibility_verdict"}
    for event in result["trace_events"]:
        assert forbidden.isdisjoint(event.keys()), f"oracle field leaked into trace event: {event}"


# --- construct-preserving parameter space (Phase 7 spot-check) --------------

def test_realizable_pairs_are_exactly_six_and_documented_exclusions_are_not_generated():
    assert len(D.REALIZABLE_PAIRS) == 6
    excluded = {("grasp_payload", "stability"), ("joint_and_collision", "clearance"),
               ("joint_and_collision", "energy"), ("grasp_payload", "reach")}
    assert excluded.isdisjoint(set(D.REALIZABLE_PAIRS))


def test_different_seeds_same_params_produce_identical_gold():
    """Gold is a discrete function of the world's physical_access, not of the
    seed -- proves the 'seed jitter alone does not add distinct D-physical
    episodes' claim in d_scaling_analysis.csv rather than just asserting it."""
    spec1 = D.build_reach_or_energy("chiller_6", "reach", "inadmissible", seed=1)
    spec2 = D.build_reach_or_energy("chiller_6", "reach", "inadmissible", seed=999)
    assert spec1.gold.to_dict() == spec2.gold.to_dict()
    assert spec1.params["constraint"] == spec2.params["constraint"]
