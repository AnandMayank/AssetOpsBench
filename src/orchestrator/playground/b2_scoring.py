"""b2_scoring.py — the B2 metric family (plan section G): primary metrics
EGR/TGS/SD/efficiency/regret, plus diagnostics. Scores a `Trajectory`
(baselines.py or a future model runner) against its `B2Gold`
(b2_families.py). Never recomputes/replaces the frozen B_AGS metric --
that stays defined only in `b_acquisition_scoring.py`, untouched.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

_ORCH_SRC = Path(__file__).resolve().parents[1]
if str(_ORCH_SRC) not in sys.path:
    sys.path.insert(0, str(_ORCH_SRC))

from .b2_families import B2Gold, PRIMARY_HORIZON, world_pool_for  # noqa: E402
from .baselines import Trajectory  # noqa: E402
from .frontier import AbstractState, min_cost_to_sufficiency  # noqa: E402
from .semantics import EpisodeSpec, TERMINAL_ACTIONS, commit_is_grounded, sufficiency  # noqa: E402

__all__ = ["B2Score", "score_trajectory", "aggregate"]

_ACQUIRE_ACTIONS = frozenset({"ACQUIRE_IOT", "REINSPECT_IOT", "ACQUIRE_ACOUSTIC", "ACQUIRE_THERMAL"})


@dataclass
class B2Score:
    episode_id: str
    # --- primary ---
    tgs: bool                      # terminal grounded success
    sd: bool                       # sufficiency detection (no premature commit, no post-sufficiency drift)
    waste_count: int
    egr: bool                      # tgs and sd and waste_count == 0
    total_cost: float
    reg_min: Optional[float]       # (cost - c_min) / c_min, only when TGS and c_min known
    # --- diagnostics ---
    tda: bool                      # terminal action matches g_term, regardless of grounding
    premature_commit: bool
    premature_escalate: bool
    repeat_unavailable: int
    n_acquisitions: int
    escalated: bool


def _delivered_prefix_states(spec: EpisodeSpec, traj: Trajectory):
    """Replays the trajectory's acquire actions against the OFFLINE
    DrawPool (not the live env, which has already been consumed) to
    determine, after each acquire step, whether sufficiency held BEFORE
    that step was taken -- used for premature/redundant/waste checks
    without depending on env internals still being valid."""
    pool = world_pool_for(spec.world)
    counts = {"acoustic": 0, "iot": 0, "thermal": 0}
    states: List[bool] = []  # sufficiency(delivered)-before-this-acquire, per acquire step
    from .semantics import AcousticReading, DeliveredEvidence, IoTReading, ThermalReading
    modality_of = {"ACQUIRE_ACOUSTIC": "acoustic", "ACQUIRE_IOT": "iot", "REINSPECT_IOT": "iot",
                   "ACQUIRE_THERMAL": "thermal"}
    for step in traj.steps:
        if step.action not in _ACQUIRE_ACTIONS or step.observation.status != "delivered":
            continue
        modality = modality_of[step.action]
        delivered = DeliveredEvidence(
            acoustic=[AcousticReading(p, 0.0) for p in pool.acoustic[:counts["acoustic"]]],
            iot=[IoTReading(b, 0.0) for b in pool.iot[:counts["iot"]]],
            thermal=[ThermalReading(h) for h in pool.thermal[:counts["thermal"]]],
        )
        states.append(sufficiency(delivered) is not None)
        counts[modality] += 1
    return states


def score_trajectory(spec: EpisodeSpec, gold: B2Gold, traj: Trajectory) -> B2Score:
    env = traj.env
    delivered = env.delivered
    final_action = traj.final_action

    r_final = sufficiency(delivered)
    if final_action in ("COMMIT_FAULT", "COMMIT_NORMAL"):
        grounded = commit_is_grounded(final_action, delivered)
        matches_world = (final_action == "COMMIT_FAULT" and gold.world_truth == "fault") or \
                         (final_action == "COMMIT_NORMAL" and gold.world_truth == "normal")
        tgs = grounded and matches_world
        premature_commit = r_final is None
    else:  # ESCALATE
        tgs = gold.frontier_empty  # ESCALATE is correct iff no world-agreeing state was reachable
        premature_commit = False

    premature_escalate = (final_action == "ESCALATE") and (not gold.frontier_empty)

    pre_sufficiency_flags = _delivered_prefix_states(spec, traj)
    post_sufficiency_acquisitions = sum(1 for was_sufficient in pre_sufficiency_flags if was_sufficient)
    sd = (not premature_commit) and (post_sufficiency_acquisitions == 0)

    repeat_unavailable = 0
    seen_unavailable_actions = set()
    for step in traj.steps:
        if step.observation.status == "unavailable":
            if step.action in seen_unavailable_actions:
                repeat_unavailable += 1
            seen_unavailable_actions.add(step.action)

    n_acquisitions = sum(1 for s in traj.steps if s.action in _ACQUIRE_ACTIONS and s.observation.status == "delivered")
    waste_count = post_sufficiency_acquisitions + repeat_unavailable
    egr = bool(tgs and sd and waste_count == 0)

    total_cost = env.spent
    reg_min = None
    if tgs and gold.c_min is not None and gold.c_min > 0:
        reg_min = (total_cost - gold.c_min) / gold.c_min

    tda = final_action == gold.g_term

    return B2Score(
        episode_id=spec.episode_id, tgs=tgs, sd=sd, waste_count=waste_count, egr=egr,
        total_cost=total_cost, reg_min=reg_min, tda=tda, premature_commit=premature_commit,
        premature_escalate=premature_escalate, repeat_unavailable=repeat_unavailable,
        n_acquisitions=n_acquisitions, escalated=final_action == "ESCALATE",
    )


def aggregate(scores: List[B2Score]) -> Dict[str, float]:
    n = len(scores)
    if n == 0:
        return {}
    return {
        "n": n,
        "EGR": sum(s.egr for s in scores) / n,
        "TGS": sum(s.tgs for s in scores) / n,
        "SD": sum(s.sd for s in scores) / n,
        "TDA": sum(s.tda for s in scores) / n,
        "mean_waste_count": sum(s.waste_count for s in scores) / n,
        "premature_commit_rate": sum(s.premature_commit for s in scores) / n,
        "premature_escalate_rate": sum(s.premature_escalate for s in scores) / n,
        "escalation_rate": sum(s.escalated for s in scores) / n,
        "mean_total_cost": sum(s.total_cost for s in scores) / n,
        "mean_n_acquisitions": sum(s.n_acquisitions for s in scores) / n,
        "mean_reg_min": (sum(s.reg_min for s in scores if s.reg_min is not None) /
                         max(sum(1 for s in scores if s.reg_min is not None), 1)),
    }
