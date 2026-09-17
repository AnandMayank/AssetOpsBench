"""Verification for ledger B2 (Class-E system prompt pre-empted the failure
mode it existed to measure; re-observation had no cost).

Repair: the persistence-denying prompt sentences are removed
(``run_class_e_pilot.py``); ``SequenceWorld.battery_budget`` caps physical
reads across a whole sequence at a world-fixed value, giving re-observation a
real cost without making gold depend on any model output.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

from sequence_executor import (  # noqa: E402
    SequenceExecutor, derive_sequence_gold, sample_sequence,
)
from tool_executor import STATUS_FAILED, STATUS_SUCCESS, ToolCall  # noqa: E402


def _executor():
    try:
        from couchdb_executor import CouchDBExecutor
        ex = CouchDBExecutor()
        if getattr(ex._robot, "db", None) is None:
            pytest.skip("CouchDB unavailable")
        return ex
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"executor unavailable: {exc}")


# --- battery is a world property, fixed before any model acts --------------

@pytest.mark.parametrize("n_episodes", [1, 2, 3, 5])
def test_battery_budget_is_a_deterministic_world_property(n_episodes):
    w = sample_sequence(seed=42, n_episodes=n_episodes)
    assert w.battery_budget == max(1, n_episodes - 1)
    # Same seed, same budget -- not drawn from any per-call randomness.
    w2 = sample_sequence(seed=42, n_episodes=n_episodes)
    assert w2.battery_budget == w.battery_budget


def test_battery_budget_makes_blind_reobservation_infeasible():
    """The whole point: re-observing every episode (n_episodes reads) must
    exceed the budget, while reading once and reusing must always fit."""
    for n in (2, 3, 4):
        w = sample_sequence(seed=7, n_episodes=n)
        assert w.battery_budget < n, "blind re-observation every episode must not fit"
        assert w.battery_budget >= 1, "reading once must always fit"


# --- gold never depends on battery or any model output ---------------------

def test_gold_signature_has_no_battery_or_model_parameter():
    params = set(inspect.signature(derive_sequence_gold).parameters)
    assert params == {"world", "episode"}, (
        "derive_sequence_gold must be a pure function of world and episode only")


def test_gold_is_identical_regardless_of_simulated_battery_state():
    """Sanity: calling derive_sequence_gold repeatedly (as if under different
    battery histories) never changes its answer -- there is no battery
    argument to vary, so this also documents the invariant directly."""
    w = sample_sequence(seed=99, n_episodes=3)
    for k in range(3):
        results = [derive_sequence_gold(w, k) for _ in range(5)]
        assert all(r == results[0] for r in results)


# --- the executor actually enforces the budget ------------------------------

def test_third_physical_read_fails_once_budget_is_spent():
    ex = _executor()
    se = SequenceExecutor(ex)
    w = sample_sequence(seed=13, n_episodes=3, asset_id="chiller_6")
    assert w.battery_budget == 2
    se.begin_sequence(w)
    se.advance_episode("FULL")

    r1 = se.execute(ToolCall("read_gauge", {"attempt_n": 1}))
    r2 = se.execute(ToolCall("read_gauge", {"attempt_n": 2}))
    r3 = se.execute(ToolCall("read_gauge", {"attempt_n": 3}))
    se.end_sequence()

    assert r1.status == STATUS_SUCCESS
    assert r2.status == STATUS_SUCCESS
    assert r3.executed and r3.status == STATUS_FAILED, (
        "the read past the budget must fail loudly, not silently succeed")
    assert "battery" in (r3.error or "").lower()


def test_budget_is_shared_across_episodes_not_reset_per_episode():
    """The defect this fixes: a per-episode reset would make re-observation
    free again (fresh budget every visit). The count must persist for the
    whole sequence."""
    ex = _executor()
    se = SequenceExecutor(ex)
    w = sample_sequence(seed=21, n_episodes=3, asset_id="motor_01")
    se.begin_sequence(w)

    se.advance_episode("FULL")
    se.execute(ToolCall("read_gauge", {"attempt_n": 1}))
    se.execute(ToolCall("read_gauge", {"attempt_n": 2}))  # spends the whole budget

    se.advance_episode("FULL")   # next episode -- budget must NOT have reset
    r = se.execute(ToolCall("read_gauge", {"attempt_n": 1}))
    se.end_sequence()

    assert r.status == STATUS_FAILED, "budget reset across an episode boundary"


def test_non_physical_tools_are_never_battery_gated():
    """get_pose, get_work_order etc. must keep working after the physical
    budget is spent -- only the sensor read itself is rationed."""
    ex = _executor()
    se = SequenceExecutor(ex)
    w = sample_sequence(seed=33, n_episodes=2, asset_id="hydraulic_pump_1")
    se.begin_sequence(w)
    se.advance_episode("FULL")
    se.execute(ToolCall("read_gauge", {"attempt_n": 1}))
    se.execute(ToolCall("read_gauge", {"attempt_n": 2}))  # budget now spent (n_episodes-1=1... )
    r = se.execute(ToolCall("get_pose", {}))
    se.end_sequence()
    assert r.executed and r.status == STATUS_SUCCESS
