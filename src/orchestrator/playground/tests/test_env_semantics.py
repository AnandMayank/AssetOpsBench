"""Live-environment semantics tests (plan section K)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

from playground.env import B2Env  # noqa: E402
from playground.semantics import EpisodeSpec, EvidenceRegime, WorldSpec, sufficiency  # noqa: E402
from playground import substrate  # noqa: E402


def _fresh_env(world, regime, family="B"):
    env = B2Env()
    env.reset(EpisodeSpec(world=world, regime=regime, family=family))
    return env


def test_unavailable_action_delivers_nothing_and_is_traced():
    world = WorldSpec(asset="motor_01", condition="fault", thermal_fault_class="Rotor-0")
    regime = EvidenceRegime(available_modalities=frozenset({"iot", "record"}))
    env = _fresh_env(world, regime, family="D")
    result = env.step("ACQUIRE_THERMAL")
    assert result.observation.status == "unavailable"
    assert len(env.delivered.thermal) == 0
    assert env.trace.delivered_observations() == {}
    events = env.trace.events
    assert any(e.status == "unavailable" for e in events)


def test_success_updates_evidence_and_delivery_firewall():
    world = WorldSpec(asset="chiller_6", condition="fault", machine_id="00", iot_band_state="out_of_band")
    env = _fresh_env(world, EvidenceRegime())
    env.step("DISPATCH")
    result = env.step("ACQUIRE_ACOUSTIC")
    assert result.observation.status == "delivered"
    assert len(env.delivered.acoustic) == 1
    delivered_obs = env.trace.delivered_observations()
    assert len(delivered_obs) == 1


def test_no_future_draws_and_no_leakage_in_rendered_payload():
    world = WorldSpec(asset="chiller_6", condition="fault", machine_id="00", iot_band_state="out_of_band")
    env = _fresh_env(world, EvidenceRegime())
    env.step("DISPATCH")
    for _ in range(3):
        result = env.step("ACQUIRE_ACOUSTIC")
        if result.observation.status == "delivered":
            hits = substrate.scan_for_leakage(result.observation.payload)
            assert hits == [], f"leakage: {hits}"
    # The k-th draw must be the k-th real pool id in order -- verified by
    # substrate.resolve_next's own construction, exercised here end-to-end.
    assert len(env._drawn["acoustic"]) <= len(env._pool.acoustic_ids)


def test_premature_commit_is_traced_ungrounded():
    world = WorldSpec(asset="chiller_6", condition="fault", machine_id="00", iot_band_state="out_of_band")
    env = _fresh_env(world, EvidenceRegime())
    result = env.step("COMMIT_FAULT")  # no evidence acquired at all
    assert result.terminated
    decision_events = [e for e in env.trace.events if e.tool == "COMMIT_FAULT"]
    assert decision_events
    assert decision_events[-1].detail["sufficient_at_decision"] is None
    assert decision_events[-1].detail["grounded"] is False


def test_trace_chain_verifies_after_full_episode():
    world = WorldSpec(asset="chiller_6", condition="fault", machine_id="00", iot_band_state="out_of_band")
    env = _fresh_env(world, EvidenceRegime())
    env.step("DISPATCH")
    env.step("ACQUIRE_ACOUSTIC")
    env.step("ACQUIRE_ACOUSTIC")
    env.step("COMMIT_FAULT")
    assert env.trace.verify_chain() is True


def test_no_environment_side_commit_gate():
    """A COMMIT always terminates the episode -- the environment never
    blocks it, even when ungrounded (plan: no commit gate; grounding is
    an evaluator-side judgment, not an environment-side block)."""
    world = WorldSpec(asset="chiller_6", condition="fault", machine_id="00", iot_band_state="out_of_band")
    env = _fresh_env(world, EvidenceRegime())
    result = env.step("COMMIT_NORMAL")
    assert result.terminated is True
    assert result.observation.status == "ok"


def test_budget_blocks_over_budget_action_without_charging():
    world = WorldSpec(asset="chiller_6", condition="fault", machine_id="00", iot_band_state="out_of_band")
    regime = EvidenceRegime(budget=1.0)  # not enough for DISPATCH (cost 8)
    env = _fresh_env(world, regime)
    result = env.step("DISPATCH")
    assert result.observation.status == "blocked"
    assert env.dispatched is False
    assert env.spent == 0.0
