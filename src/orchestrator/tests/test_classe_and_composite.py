"""P1 class-E acceptance tests and P2 composite-verdict tests.

The ten class-E acceptance properties from the design, plus proof that
flattening a composite verdict loses information.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

from composite_verdict import (  # noqa: E402
    R026_GOLD, flatten_loses_information, is_composite, score_verdict,
)
from execution_trace import Stage  # noqa: E402
from metrics_spec import DIAGNOSTIC, PRIMARY, assert_no_blending  # noqa: E402
from sequence_executor import (  # noqa: E402
    LINEAR_DRIFT, PROCESSES, STATIONARY, SequenceTrace, derive_sequence_gold,
    grounded_in_current_episode, sample_sequence, stale_state_reuse,
)
from tool_executor import ToolCall  # noqa: E402


def _executor():
    try:
        from couchdb_executor import CouchDBExecutor
        ex = CouchDBExecutor()
        if getattr(ex._robot, "db", None) is None:
            pytest.skip("CouchDB unavailable")
        return ex
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"executor unavailable: {exc}")


# ==========================================================================
# P0 — frozen metric semantics
# ==========================================================================

def test_primary_and_diagnostic_sets_are_disjoint():
    assert not (PRIMARY & DIAGNOSTIC)
    assert "CC_grounded" in DIAGNOSTIC and "CC" in PRIMARY


def test_a_primary_metric_may_not_incorporate_a_diagnostic():
    assert_no_blending({"CC": 1})
    with pytest.raises(AssertionError):
        assert_no_blending({"CC": 1, "CC_includes_CC_grounded": True})


# ==========================================================================
# P2 — composite verdict
# ==========================================================================

def test_flattening_r026_loses_information():
    rep = flatten_loses_information(R026_GOLD)
    assert rep["information_lost"] is True
    assert not rep["lossless_flattening_exists"]
    # Every candidate misrepresents at least one asset.
    for cand, v in rep["flattenings"].items():
        assert v["n_wrong"] >= 1, f"{cand} claimed lossless"


def test_composite_cc_requires_every_asset_to_match():
    assert score_verdict({"chiller_6": "ABORT", "motor_01": "COMMIT"}, R026_GOLD).CC == 1
    partial = score_verdict({"chiller_6": "ABORT", "motor_01": "ESCALATE"}, R026_GOLD)
    assert partial.CC == 0 and partial.CC_partial == 0.5


def test_flat_verdict_against_composite_gold_is_scored_not_rejected():
    """Scored so the loss is visible in the numbers rather than hidden."""
    s = score_verdict("ESCALATE", R026_GOLD)
    assert s.CC == 0 and s.is_composite


def test_missing_asset_is_a_miss_not_an_inference():
    s = score_verdict({"chiller_6": "ABORT"}, R026_GOLD)
    assert s.CC == 0 and s.missing_assets == ["motor_01"]


def test_single_asset_scenarios_are_unchanged():
    assert not is_composite("COMMIT")
    assert score_verdict("COMMIT", "COMMIT").CC == 1
    assert score_verdict("DISPATCH", "COMMIT").CC == 1     # existing normalisation
    assert score_verdict("ESCALATE", "COMMIT").CC == 0


def test_cc_partial_is_never_blended_into_cc():
    s = score_verdict({"chiller_6": "ABORT", "motor_01": "ESCALATE"}, R026_GOLD)
    assert s.CC == 0 and s.CC_partial > 0     # partial credit visible, CC strict


# ==========================================================================
# P1 — class E acceptance properties (1-10)
# ==========================================================================

# 2. drift sampled before gold, never in a prompt
def test_drift_process_is_sampled_before_gold_and_never_labelled():
    import inspect
    from sequence_executor import sample_sequence as fn
    params = set(inspect.signature(fn).parameters)
    assert not (params & {"gold", "verdict", "label", "expected", "target"})
    src = inspect.getsource(fn)
    for token in ("COMMIT", "ESCALATE", "ABORT"):
        assert token not in src


# 3. gold over a prefix is a pure function of that prefix
def test_sequence_gold_is_a_pure_function_of_the_world_and_prefix():
    w = sample_sequence(seed=11, n_episodes=3)
    for k in range(3):
        assert derive_sequence_gold(w, k) == derive_sequence_gold(w, k)


# 10. replay reproduces every world state
def test_sequence_replay_is_deterministic():
    a = sample_sequence(seed=77, n_episodes=4)
    b = sample_sequence(seed=77, n_episodes=4)
    assert a.to_dict() == b.to_dict()
    assert [a.value_at(k) for k in range(4)] == [b.value_at(k) for k in range(4)]


# 8. a stationary process yields no drift expectation at any k
def test_stationary_process_never_crosses_the_band():
    for seed in range(200):
        w = sample_sequence(seed=seed, n_episodes=4)
        if w.process.kind != STATIONARY:
            continue
        assert all(derive_sequence_gold(w, k)["verdict"] == "COMMIT" for k in range(4))
        return
    pytest.skip("no stationary sequence sampled in range")


def test_drift_process_can_cross_the_band():
    for seed in range(200):
        w = sample_sequence(seed=seed, n_episodes=4)
        if w.process.kind == LINEAR_DRIFT:
            verdicts = [derive_sequence_gold(w, k)["verdict"] for k in range(4)]
            assert "ESCALATE" in verdicts, "drift never becomes detectable"
            return
    pytest.skip("no drift sequence sampled in range")


def test_all_processes_are_reachable():
    kinds = {sample_sequence(seed=s, n_episodes=3).process.kind for s in range(120)}
    assert kinds == set(PROCESSES)


# 4/5. an observation from episode k does not ground episode k+1
def test_prior_episode_observation_does_not_ground_the_current_one():
    st = SequenceTrace("SEQ-1", "FULL")
    t0 = st.new_episode()
    t0.append(Stage.OBSERVATION_DELIVERED, tool="capture_image",
              observation_id="o0", observation_hash="h0", modality="physical")
    st.new_episode()          # t1: nothing observed
    assert grounded_in_current_episode(st, 0) is True
    assert grounded_in_current_episode(st, 1) is False
    assert stale_state_reuse(st, 1) is True


def test_fresh_observation_clears_stale_state():
    st = SequenceTrace("SEQ-2", "FULL")
    t0 = st.new_episode()
    t0.append(Stage.OBSERVATION_DELIVERED, tool="capture_image",
              observation_id="o0", observation_hash="h0", modality="physical")
    t1 = st.new_episode()
    t1.append(Stage.OBSERVATION_DELIVERED, tool="capture_image",
              observation_id="o1", observation_hash="h1", modality="physical")
    assert stale_state_reuse(st, 1) is False
    assert grounded_in_current_episode(st, 1) is True


def test_first_episode_cannot_be_stale():
    st = SequenceTrace("SEQ-3", "FULL")
    st.new_episode()
    assert stale_state_reuse(st, 0) is False


# 6. the hash chain verifies across episode boundaries
def test_sequence_hash_chain_verifies_and_detects_tampering():
    st = SequenceTrace("SEQ-4", "FULL")
    for _ in range(3):
        t = st.new_episode()
        t.append(Stage.REQUESTED, tool="capture_image")
        t.append(Stage.EXECUTED, tool="capture_image", status="success")
    assert st.verify_chain()
    object.__setattr__(st.episodes[1]._events[0], "tool", "read_iot")
    assert not st.verify_chain()


# 1/7/9. sequence-scoped state, isolation, revert
def test_state_persists_within_a_sequence_and_reverts_on_exit():
    from sequence_executor import SequenceExecutor
    ex = _executor()
    se = SequenceExecutor(ex)
    w = sample_sequence(seed=5, n_episodes=3, asset_id="chiller_6")
    before = dict(ex._robot.db.get(f"profile:{w.asset}"))
    se.begin_sequence(w)
    seen = []
    for _ in range(3):
        se.advance_episode("FULL")
        seen.append(ex.execute(ToolCall("read_gauge", {"attempt_n": 1})).payload["reading"])
    se.end_sequence()
    after = ex._robot.db.get(f"profile:{w.asset}")
    assert after["gauge_value"] == before["gauge_value"], "sequence state not reverted"
    if w.process.kind != STATIONARY:
        assert len(set(seen)) > 1, "world did not evolve across episodes"


def test_two_sequences_do_not_contaminate_each_other():
    from sequence_executor import SequenceExecutor
    ex = _executor()
    a = sample_sequence(seed=21, n_episodes=2, asset_id="motor_01")
    b = sample_sequence(seed=22, n_episodes=2, asset_id="motor_01")
    se = SequenceExecutor(ex)
    se.begin_sequence(a); se.advance_episode("FULL")
    va = ex.execute(ToolCall("read_gauge", {"attempt_n": 1})).payload["reading"]
    se.end_sequence()
    se2 = SequenceExecutor(ex)
    se2.begin_sequence(b); se2.advance_episode("FULL")
    vb = ex.execute(ToolCall("read_gauge", {"attempt_n": 1})).payload["reading"]
    se2.end_sequence()
    assert abs(va - a.value_at(0)) < 0.06 * (a.gauge_range[1] - a.gauge_range[0])
    assert abs(vb - b.value_at(0)) < 0.06 * (b.gauge_range[1] - b.gauge_range[0])
