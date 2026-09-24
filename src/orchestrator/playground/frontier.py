"""frontier.py — the sufficient-state frontier and two oracle references
(plan section E). Neither reference is "the" correct trajectory:

* `min_cost_to_sufficiency` (C_min) is a CLAIRVOYANT lower bound: exact
  Dijkstra over the reachable-state graph, given full knowledge of what
  each future draw will show. It is a floor, never a policy to imitate.
* `expectimax_reference` (pi*) is NON-CLAIRVOYANT: it only knows the
  world's pool label frequencies (not individual draw outcomes), and
  chooses the cheapest action in expectation at every state. Its realised
  cost on one draw sequence is a fair reference for regret.

Both operate over a small abstract state space (bounded action set, bounded
horizon, deduplicated states) so exact search is tractable -- no ML, no
sampling-based approximation.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, replace
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

from .semantics import (
    ActionCost, AcousticReading, DeliveredEvidence, EvidenceRegime, IoTReading,
    ThermalReading, action_cost, clause_relevant_modalities, sufficiency,
)

__all__ = [
    "DrawPool", "AbstractState", "reachable_sufficient_states",
    "min_cost_to_sufficiency", "frontier_is_empty", "expectimax_reference",
    "wasted_acquisition",
]

#: Actions available at every step. REINSPECT_IOT and ACQUIRE_IOT are the
#: same cost/effect (both draw the next IoT item) -- REINSPECT_IOT exists
#: as a distinct label only for trace/prompt readability (plan section 4's
#: named action list); the frontier treats them identically.
_ACQUIRE_ACTIONS = ("ACQUIRE_IOT", "ACQUIRE_ACOUSTIC", "ACQUIRE_THERMAL")
_MODALITY_OF = {"ACQUIRE_IOT": "iot", "ACQUIRE_ACOUSTIC": "acoustic", "ACQUIRE_THERMAL": "thermal"}


@dataclass(frozen=True)
class DrawPool:
    """The world's ordered per-modality outcome sequences (evaluator-only;
    plan section C, object omega). `acoustic`/`iot`/`thermal` are tuples of
    booleans (already resolved to the R-relevant reading, e.g. "positive"
    for acoustic, "in_band" for iot, "hotspot_present" for thermal) --
    `env.py` and `b2_families.py` are responsible for turning real resolved
    ObservationRecords into these booleans via `semantics.acoustic_indicator`
    etc.; this module never touches raw records."""
    acoustic: Tuple[bool, ...] = ()
    iot: Tuple[bool, ...] = ()
    thermal: Tuple[bool, ...] = ()

    def available(self, modality: str) -> bool:
        return len(getattr(self, modality)) > 0


@dataclass(frozen=True)
class AbstractState:
    """A deduplicable point in the search graph: (# drawn per modality,
    dispatched, unavailable-attempted set, record-queried, budget-spent).
    Draw ORDER within a modality is fixed by DrawPool, so "k drawn" fully
    determines the delivered readings for that modality -- this is what
    keeps the state space small enough for exact search."""
    n_acoustic: int = 0
    n_iot: int = 0
    n_thermal: int = 0
    dispatched: bool = False
    unavailable: FrozenSet[str] = frozenset()
    record_queried: bool = False
    spent: float = 0.0


def _delivered_at(state: AbstractState, pool: DrawPool) -> DeliveredEvidence:
    return DeliveredEvidence(
        acoustic=[AcousticReading(positive=p, indicator_value=1.0 if p else -1.0)
                  for p in pool.acoustic[:state.n_acoustic]],
        iot=[IoTReading(in_band=b, value=1.0 if b else -1.0) for b in pool.iot[:state.n_iot]],
        thermal=[ThermalReading(hotspot_present=h) for h in pool.thermal[:state.n_thermal]],
        record_queried=state.record_queried,
    )


def _available_now(action: str, pool: DrawPool, available_modalities: FrozenSet[str],
                    state: AbstractState) -> bool:
    if action == "DISPATCH":
        return "acoustic" in available_modalities or "thermal" in available_modalities
    if action == "QUERY_RECORD":
        return "record" in available_modalities
    modality = _MODALITY_OF[action]
    if modality not in available_modalities:
        return False
    if action in ("ACQUIRE_ACOUSTIC", "ACQUIRE_THERMAL") and not state.dispatched:
        return False
    if not pool.available(modality):
        return False
    drawn = {"ACQUIRE_IOT": state.n_iot, "ACQUIRE_ACOUSTIC": state.n_acoustic,
             "ACQUIRE_THERMAL": state.n_thermal}[action]
    return drawn < len(getattr(pool, modality))


def _successor(action: str, state: AbstractState, costs: ActionCost) -> AbstractState:
    cost = action_cost(action, costs)
    if action == "DISPATCH":
        return replace(state, dispatched=True, spent=state.spent + cost)
    if action == "QUERY_RECORD":
        return replace(state, record_queried=True, spent=state.spent + cost)
    if action == "ACQUIRE_IOT":
        return replace(state, n_iot=state.n_iot + 1, spent=state.spent + cost)
    if action == "ACQUIRE_ACOUSTIC":
        return replace(state, n_acoustic=state.n_acoustic + 1, spent=state.spent + cost)
    if action == "ACQUIRE_THERMAL":
        return replace(state, n_thermal=state.n_thermal + 1, spent=state.spent + cost)
    raise ValueError(f"unknown action {action!r}")


def _all_actions() -> Tuple[str, ...]:
    return ("DISPATCH", "QUERY_RECORD") + _ACQUIRE_ACTIONS


def reachable_sufficient_states(
    initial: AbstractState, pool: DrawPool, regime: EvidenceRegime, *, max_horizon: int = 10,
    accept: Optional["Callable[[DeliveredEvidence], bool]"] = None,
) -> List[Tuple[AbstractState, float]]:
    """Exact BFS/Dijkstra-style exploration of every state reachable from
    `initial` within `max_horizon` steps and the regime's budget, returning
    every state satisfying `accept` (default: `sufficiency(e) is not None`
    -- ANY sufficient state), paired with the cheapest cost to reach it.
    This IS the frontier (plan section E): a SET of states, not a single
    path -- multiple states can appear here at equal minimum cost.

    Passing `accept=lambda e: sufficiency(e) == target_verdict` is what
    lets `b2_families.compute_gold` search specifically for the CHEAPEST
    state that agrees with world truth, rather than the cheapest
    sufficient state of EITHER verdict -- essential for families (like the
    contradiction family E) where the cheapest available evidence can be
    momentarily misleading and only a costlier path reaches the true
    verdict (plan section C.F/C.G: correctness is judged against R(E_T),
    not against "whatever state is cheapest to reach")."""
    if accept is None:
        accept = lambda e: sufficiency(e) is not None  # noqa: E731
    start_delivered = _delivered_at(initial, pool)
    if accept(start_delivered):
        return [(initial, 0.0)]

    best_cost: Dict[AbstractState, float] = {initial: 0.0}
    frontier_hits: Dict[AbstractState, float] = {}
    heap: List[Tuple[float, int, AbstractState]] = [(0.0, 0, initial)]
    counter = itertools.count(1)
    depth: Dict[AbstractState, int] = {initial: 0}

    while heap:
        cost, _, state = heapq.heappop(heap)
        if cost > best_cost.get(state, float("inf")):
            continue
        if depth[state] >= max_horizon:
            continue
        for action in _all_actions():
            if not _available_now(action, pool, regime.available_modalities, state):
                # Inapplicable this step: either the modality/tool is not in
                # `available_modalities` (permanently unavailable for this
                # regime) or its pool is exhausted at this draw depth. Both
                # are dead branches for reachability -- record the attempt
                # as "unavailable" only for the former, since only that one
                # is a genuine repeat-unavailable outcome an agent could hit.
                if action in ("QUERY_RECORD", "DISPATCH") or _MODALITY_OF.get(action) not in                         regime.available_modalities:
                    nxt = replace(state, unavailable=state.unavailable | {action})
                    if nxt.spent < best_cost.get(nxt, float("inf")):
                        best_cost[nxt] = nxt.spent
                        depth[nxt] = depth[state] + 1
                        heapq.heappush(heap, (nxt.spent, next(counter), nxt))
                continue
            nxt = _successor(action, state, regime.costs)
            if regime.budget is not None and nxt.spent > regime.budget:
                continue
            if nxt.spent < best_cost.get(nxt, float("inf")):
                best_cost[nxt] = nxt.spent
                depth[nxt] = depth[state] + 1
                heapq.heappush(heap, (nxt.spent, next(counter), nxt))
                delivered = _delivered_at(nxt, pool)
                if accept(delivered) and nxt not in frontier_hits:
                    frontier_hits[nxt] = nxt.spent

    return sorted(frontier_hits.items(), key=lambda kv: kv[1])


def min_cost_to_sufficiency(initial: AbstractState, pool: DrawPool,
                             regime: EvidenceRegime, *, max_horizon: int = 10,
                             target_verdict: Optional[str] = None) -> Optional[float]:
    """C_min: the clairvoyant lower-bound cost to reach a sufficient state.
    With `target_verdict` given ('normal' | 'fault'), searches for the
    cheapest state whose OWN R() equals `target_verdict` specifically --
    the correct oracle for a WorldSpec's gold (plan C.F/C.G): the cheapest
    reachable state of EITHER verdict can be the wrong one under real
    sensor noise, and gold must track world truth, not "whatever's
    cheapest". Without `target_verdict`, returns the cheapest state of
    EITHER verdict (used only for cost-referencing, e.g. `expectimax_reference`
    style comparisons, never for gold). Returns None iff no accepting
    state is reachable (the ESCALATE-correct condition, plan C.G)."""
    accept = (lambda e: sufficiency(e) == target_verdict) if target_verdict is not None else None
    hits = reachable_sufficient_states(initial, pool, regime, max_horizon=max_horizon, accept=accept)
    return hits[0][1] if hits else None


def frontier_is_empty(initial: AbstractState, pool: DrawPool, regime: EvidenceRegime,
                       *, max_horizon: int = 10, target_verdict: Optional[str] = None) -> bool:
    return min_cost_to_sufficiency(initial, pool, regime, max_horizon=max_horizon,
                                    target_verdict=target_verdict) is None


# ---------------------------------------------------------------------------
# Non-clairvoyant expectimax reference pi* (plan section E)
# ---------------------------------------------------------------------------

def expectimax_reference(
    initial: AbstractState, regime: EvidenceRegime, *, p_acoustic_positive: float,
    p_iot_in_band: float, p_thermal_hotspot: float, max_horizon: int = 10,
    available_modalities: Optional[FrozenSet[str]] = None,
) -> float:
    """Expected cost of the cost-minimizing policy that knows only the
    world's POOL FREQUENCIES (p_*), not individual future draws. Computed
    by backward induction over depth (small horizon, small branching --
    exact, not sampled). Returns expected cost to reach sufficiency.

    Unlike `min_cost_to_sufficiency` (clairvoyant), this policy cannot see
    the draw outcome before acting, so its expected cost is >= C_min.
    ESCALATE is NOT modeled as a free zero-cost alternative here -- a
    reference that could always "give up" for 0 cost would trivially
    return 0 and defeat the purpose of a cost reference. ESCALATE only
    enters implicitly: if no acquisition action is available at all (every
    remaining modality unavailable/exhausted, or max_horizon reached) the
    recursion bottoms out at 0 additional cost, matching C_min's own
    frontier-empty convention (plan section E) -- it is not a competing
    'cheap' choice among otherwise-available actions.

    This is necessarily an approximation of the true state (it tracks only
    per-modality counts, not which specific booleans were drawn), which is
    the right approximation for a NON-clairvoyant policy: it knows the
    world's aggregate priors, not individual future outcomes."""
    available_modalities = available_modalities or regime.available_modalities
    from functools import lru_cache

    def _delivered_from_counts(a_pos: int, a_neg: int, iot_pattern: Tuple[bool, ...],
                                thermal_hit: bool, thermal_drawn: bool) -> DeliveredEvidence:
        acoustic = [AcousticReading(True, 1.0)] * a_pos + [AcousticReading(False, -1.0)] * a_neg
        iot = [IoTReading(b, 1.0 if b else -1.0) for b in iot_pattern]
        thermal = [ThermalReading(thermal_hit)] if thermal_drawn else []
        return DeliveredEvidence(acoustic=acoustic, iot=iot, thermal=thermal)

    @lru_cache(maxsize=None)
    def value(a_pos: int, a_neg: int, iot_pattern: Tuple[bool, ...],
              thermal_hit: bool, thermal_drawn: bool, dispatched: bool, depth: int) -> float:
        delivered = _delivered_from_counts(a_pos, a_neg, iot_pattern, thermal_hit, thermal_drawn)
        if sufficiency(delivered) is not None:
            return 0.0
        if depth >= max_horizon:
            return 0.0  # unreachable within horizon -> ESCALATE, no further cost (matches C.G)

        candidates: List[float] = []
        if "iot" in available_modalities:
            c = action_cost("ACQUIRE_IOT", regime.costs)
            v_in = value(a_pos, a_neg, iot_pattern + (True,), thermal_hit, thermal_drawn, dispatched, depth + 1)
            v_out = value(a_pos, a_neg, iot_pattern + (False,), thermal_hit, thermal_drawn, dispatched, depth + 1)
            candidates.append(c + p_iot_in_band * v_in + (1 - p_iot_in_band) * v_out)

        needs_dispatch = not dispatched and ("acoustic" in available_modalities or
                                              "thermal" in available_modalities)
        if needs_dispatch:
            c = action_cost("DISPATCH", regime.costs)
            v = value(a_pos, a_neg, iot_pattern, thermal_hit, thermal_drawn, True, depth + 1)
            candidates.append(c + v)

        if dispatched and "acoustic" in available_modalities:
            c = action_cost("ACQUIRE_ACOUSTIC", regime.costs)
            v_pos = value(a_pos + 1, a_neg, iot_pattern, thermal_hit, thermal_drawn, True, depth + 1)
            v_neg = value(a_pos, a_neg + 1, iot_pattern, thermal_hit, thermal_drawn, True, depth + 1)
            candidates.append(c + p_acoustic_positive * v_pos + (1 - p_acoustic_positive) * v_neg)

        if dispatched and "thermal" in available_modalities and not thermal_drawn:
            c = action_cost("ACQUIRE_THERMAL", regime.costs)
            v_hit = value(a_pos, a_neg, iot_pattern, True, True, True, depth + 1)
            v_miss = value(a_pos, a_neg, iot_pattern, False, True, True, depth + 1)
            candidates.append(c + p_thermal_hotspot * v_hit + (1 - p_thermal_hotspot) * v_miss)

        if not candidates:
            return 0.0  # nothing acquirable at all -> ESCALATE, matches frontier_is_empty
        return min(candidates)

    return value(0, 0, (), False, False, initial.dispatched, 0)



# ---------------------------------------------------------------------------
# Wasted acquisition (plan section E) -- non-clairvoyant, needs no cost
# ---------------------------------------------------------------------------

def wasted_acquisition(action: str, state_before: AbstractState, pool: DrawPool) -> bool:
    """True if `action` (an ACQUIRE_* action about to be taken from
    `state_before`) is wasted per plan section E:
      1. sufficiency already holds at state_before (post-sufficiency), or
      2. the action's modality is in state_before.unavailable (repeat of
         an unavailable action), or
      3. the modality feeds no clause of R that is still satisfiable
         (decision-irrelevant) -- approximated here as: the modality
         cannot, by itself or in combination with what's already
         delivered, ever flip an unresolved clause (conservative: only
         flags a modality no live clause mentions at all)."""
    if action not in _MODALITY_OF:
        return False
    delivered = _delivered_at(state_before, pool)
    if sufficiency(delivered) is not None:
        return True
    modality = _MODALITY_OF[action]
    if action in state_before.unavailable:
        return True
    relevant = clause_relevant_modalities()
    fault_live = {k for k, v in {**{c: False for c in relevant}}.items()}
    # A clause is "live" if it has not already been definitively ruled out
    # -- for this conservative check we treat every clause as live unless
    # its required modality has zero remaining draws available in `pool`.
    any_clause_uses_modality = any(modality in mods for mods in relevant.values())
    return not any_clause_uses_modality
