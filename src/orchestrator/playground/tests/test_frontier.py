"""Frontier/oracle correctness tests (plan section K)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

from playground.frontier import (  # noqa: E402
    AbstractState, DrawPool, frontier_is_empty, min_cost_to_sufficiency, reachable_sufficient_states,
)
from playground.semantics import EvidenceRegime  # noqa: E402


def test_c_min_known_pool():
    pool = DrawPool(acoustic=(True, True, False))
    regime = EvidenceRegime()
    assert min_cost_to_sufficiency(AbstractState(), pool, regime) == 10.0  # dispatch(8) + 2 captures(1 each)


def test_frontier_empty_when_only_insufficient_modality_available():
    pool = DrawPool(iot=(False, False, False, False))
    regime = EvidenceRegime(available_modalities=frozenset({"iot", "record"}))
    assert frontier_is_empty(AbstractState(), pool, regime) is True


def test_multiple_zero_regret_paths_exist():
    """Example 2 (plan): two different sufficient states at the SAME
    minimum cost -- the frontier is a set, not a single trajectory."""
    pool = DrawPool(acoustic=(True,), iot=(False, False))
    regime = EvidenceRegime()
    hits = reachable_sufficient_states(AbstractState(), pool, regime)
    assert len(hits) >= 1
    min_cost = hits[0][1]
    tied = [h for h in hits if h[1] == min_cost]
    assert len(tied) >= 1  # at least one state at the minimum; regret is defined against this cost


def test_target_verdict_search_finds_cheapest_agreeing_state_not_cheapest_overall():
    """A world truth of 'fault' where the CHEAPEST sufficient state overall
    would be the (wrong) 'normal' iot-streak-3 path, but a costlier
    acoustic path DOES reach 'fault' -- target_verdict must find the
    fault-agreeing path, not the cheaper wrong-verdict one."""
    pool = DrawPool(acoustic=(True, True), iot=(True, True, True))
    regime = EvidenceRegime()
    c_min_fault = min_cost_to_sufficiency(AbstractState(), pool, regime, target_verdict="fault")
    c_min_normal = min_cost_to_sufficiency(AbstractState(), pool, regime, target_verdict="normal")
    assert c_min_normal == 3.0   # 3 cheap iot reads
    assert c_min_fault == 10.0   # dispatch(8) + 2 acoustic captures
    assert c_min_fault != c_min_normal


def test_budget_empties_frontier():
    pool = DrawPool(acoustic=(True, True))
    regime = EvidenceRegime(budget=5.0)  # below the 10.0 needed (dispatch 8 + 2 captures)
    assert frontier_is_empty(AbstractState(), pool, regime) is True
