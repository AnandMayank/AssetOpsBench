"""baselines.py — the five trivial policies plus the scripted, content-aware
reference (plan section K/M). All are deterministic, zero-model-call
policies executed live against `env.B2Env`, so they exercise the real
resolver/trace path exactly like a future model-facing run would.

Trivial policies (content-BLIND unless noted):
  * NeverAcquire   -- commits/escalates from initial evidence only, never
                       acts on any evidence beyond E_0.
  * AlwaysEscalate -- acquires nothing, always ESCALATE.
  * AlwaysCommit   -- acquires nothing, commits to a FIXED verdict
                       (COMMIT_NORMAL) regardless of any evidence.
  * AcquireEverything -- dispatches and exhausts every available modality
                       up to the horizon, THEN commits/escalates from
                       whatever it ends up with (content-AWARE at the very
                       end only -- this is deliberate: it must not be
                       handicapped on grounding, only on cost/redundancy).
  * ContractRule   -- follows a FIXED action sequence and a FIXED terminal
                       rule that never reads any delivered CONTENT (only
                       whether an action was available/unavailable) --
                       the direct analogue of the shortcut that solves the
                       frozen B pool (plan section A).

Scripted reference (content-AWARE, not a trivial baseline): greedily
checks sufficiency after each step and stops as soon as it is reached,
falling back to ESCALATE after `max_horizon` steps.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_ORCH_SRC = Path(__file__).resolve().parents[1]
if str(_ORCH_SRC) not in sys.path:
    sys.path.insert(0, str(_ORCH_SRC))

from .env import B2Env, Observation  # noqa: E402
from .frontier import AbstractState, reachable_sufficient_states  # noqa: E402
from .semantics import EpisodeSpec, sufficiency  # noqa: E402

__all__ = [
    "TrajectoryStep", "Trajectory", "run_policy",
    "policy_never_acquire", "policy_always_escalate", "policy_always_commit",
    "policy_acquire_everything", "policy_contract_rule", "policy_scripted_reference",
    "policy_clairvoyant_reference", "frontier_replay_policy",
    "make_scripted_reference", "make_acquire_everything", "TRIVIAL_POLICIES",
]


@dataclass
class TrajectoryStep:
    action: str
    observation: Observation


@dataclass
class Trajectory:
    episode_id: str
    steps: List[TrajectoryStep] = field(default_factory=list)
    final_action: Optional[str] = None
    env: Optional[B2Env] = None


PolicyFn = Callable[[B2Env, int], str]  # (env, step_index) -> next action


def run_policy(spec: EpisodeSpec, policy: PolicyFn, *, max_steps: int = 12) -> Trajectory:
    env = B2Env()
    env.reset(spec)
    traj = Trajectory(episode_id=spec.episode_id, env=env)
    for i in range(max_steps):
        action = policy(env, i)
        result = env.step(action)
        traj.steps.append(TrajectoryStep(action=action, observation=result.observation))
        if result.terminated:
            traj.final_action = action
            return traj
    # Ran out of steps without a terminal action -- force ESCALATE so every
    # trajectory has a well-defined final action (never silently unscored).
    result = env.step("ESCALATE")
    traj.steps.append(TrajectoryStep(action="ESCALATE", observation=result.observation))
    traj.final_action = "ESCALATE"
    return traj


# ---------------------------------------------------------------------------
# Trivial policies
# ---------------------------------------------------------------------------

def policy_never_acquire(env: B2Env, step: int) -> str:
    if step == 0:
        r = sufficiency(env.delivered)
        if r == "fault":
            return "COMMIT_FAULT"
        if r == "normal":
            return "COMMIT_NORMAL"
        return "ESCALATE"
    raise AssertionError("policy_never_acquire should terminate at step 0")


def policy_always_escalate(env: B2Env, step: int) -> str:
    return "ESCALATE"


def policy_always_commit(env: B2Env, step: int) -> str:
    return "COMMIT_NORMAL"  # fixed, content-blind default -- see module docstring


def make_acquire_everything(rule: str = "primary") -> PolicyFn:
    """Content-blind about WHEN to stop (always exhausts everything it
    can), content-AWARE only for the final verdict (reads sufficiency,
    under `rule`, once it is done acquiring) -- deliberate: AcquireEverything
    should fail on efficiency/redundancy, not be handicapped on grounding
    too, or it would not isolate the construct it is meant to test."""
    def _policy(env: B2Env, step: int) -> str:
        modalities = ("acoustic", "thermal")
        if not env.dispatched and any(m in env.spec.regime.available_modalities for m in modalities):
            return "DISPATCH"
        for action, modality in (("ACQUIRE_IOT", "iot"), ("ACQUIRE_ACOUSTIC", "acoustic"),
                                  ("ACQUIRE_THERMAL", "thermal")):
            if modality not in env.spec.regime.available_modalities:
                continue
            pool_len = {"iot": len(env._pool.iot_ids), "acoustic": len(env._pool.acoustic_ids),
                        "thermal": len(env._pool.thermal_ids)}[modality]
            drawn = len(env._drawn[modality])
            if drawn < pool_len:
                return action
        if "record" in env.spec.regime.available_modalities and not env.delivered.record_queried:
            return "QUERY_RECORD"
        r = sufficiency(env.delivered, rule=rule)
        if r == "fault":
            return "COMMIT_FAULT"
        if r == "normal":
            return "COMMIT_NORMAL"
        return "ESCALATE"
    return _policy


policy_acquire_everything: PolicyFn = make_acquire_everything("primary")


def policy_contract_rule(env: B2Env, step: int) -> str:
    """Content-BLIND: a fixed probe sequence, and a terminal rule that
    reads only ACQUISITION COUNTS (how many readings it has, never their
    VALUES) -- the direct B2 analogue of "read the contract, never the
    evidence" that solves the frozen B pool. This is exactly what plan
    section K's acceptance gate must bound."""
    n_acoustic = len(env.delivered.acoustic)
    n_iot = len(env.delivered.iot)
    n_thermal = len(env.delivered.thermal)
    if not env.dispatched and ("acoustic" in env.spec.regime.available_modalities or
                                "thermal" in env.spec.regime.available_modalities):
        return "DISPATCH"
    if "acoustic" in env.spec.regime.available_modalities and n_acoustic < 2:
        return "ACQUIRE_ACOUSTIC"
    if "thermal" in env.spec.regime.available_modalities and n_thermal < 1:
        return "ACQUIRE_THERMAL"
    if "iot" in env.spec.regime.available_modalities and n_iot < 3:
        return "ACQUIRE_IOT"
    total = len(env.delivered.acoustic) + len(env.delivered.iot) + len(env.delivered.thermal)
    return "COMMIT_FAULT" if total > 0 else "ESCALATE"


TRIVIAL_POLICIES: Dict[str, PolicyFn] = {
    "NeverAcquire": policy_never_acquire,
    "AlwaysEscalate": policy_always_escalate,
    "AlwaysCommit": policy_always_commit,
    "AcquireEverything": policy_acquire_everything,
    "ContractRule": policy_contract_rule,
}


# ---------------------------------------------------------------------------
# Scripted, content-aware reference (plan section K: must reach EGR >= 0.95)
# ---------------------------------------------------------------------------

def make_scripted_reference(rule: str = "primary") -> PolicyFn:
    """FAIR, non-clairvoyant: stops the first time `sufficiency(..., rule)`
    fires on delivered evidence, exactly as any agent (real or trivial)
    must. Under real sensor noise this can occasionally commit the wrong
    verdict (it has no way to know that -- neither does a model) -- that
    is expected, not a defect, and is why this policy is NOT the
    acceptance-gate reference (see `policy_clairvoyant_reference` for that)."""
    def _policy(env: B2Env, step: int) -> str:
        r = sufficiency(env.delivered, rule=rule)
        if r == "fault":
            return "COMMIT_FAULT"
        if r == "normal":
            return "COMMIT_NORMAL"
        if not env.dispatched and ("acoustic" in env.spec.regime.available_modalities or
                                    "thermal" in env.spec.regime.available_modalities):
            return "DISPATCH"
        for action, modality in (("ACQUIRE_ACOUSTIC", "acoustic"), ("ACQUIRE_THERMAL", "thermal"),
                                  ("ACQUIRE_IOT", "iot")):
            if modality not in env.spec.regime.available_modalities:
                continue
            pool_len = {"iot": len(env._pool.iot_ids), "acoustic": len(env._pool.acoustic_ids),
                        "thermal": len(env._pool.thermal_ids)}[modality]
            drawn = len(env._drawn[modality])
            if drawn < pool_len:
                return action
        return "ESCALATE"
    return _policy


policy_scripted_reference: PolicyFn = make_scripted_reference("primary")


def policy_clairvoyant_reference(env: B2Env, step: int) -> str:
    """CLAIRVOYANT: reads `env.spec.world.condition` directly (legitimate
    ONLY for pool-validation/acceptance-gate use, mirroring exactly what
    `frontier.min_cost_to_sufficiency(..., target_verdict=g_world)`
    already encodes as gold -- plan section K's solvability check, never a
    fair agent baseline). Keeps acquiring until either (a) delivered
    evidence supports the TRUE verdict, committing it, or (b) every
    available modality's pool is exhausted, escalating."""
    truth = env.spec.world.condition
    r = sufficiency(env.delivered)
    if r == truth:
        return "COMMIT_FAULT" if truth == "fault" else "COMMIT_NORMAL"
    # Cheapest-first ordering (plan F: record < iot ~ capture << dispatch):
    # try IoT before ever dispatching for a physical capture, and never
    # draw MORE acoustic/thermal than the minimum the fault clause needs
    # (2 positives, or 1 thermal hotspot) -- drawing indefinitely risks
    # crossing BOTH the fault and normal thresholds at once, which makes
    # `sufficiency()` return the PERMANENT contradiction None (both
    # r_fault_clauses and r_normal_clauses satisfied simultaneously) with
    # no way back, for either verdict, for the rest of the episode. A
    # clairvoyant reference must stop drawing a modality once continuing
    # risks that trap, exactly as a rational cost-aware agent should.
    if "iot" in env.spec.regime.available_modalities:
        drawn_iot = len(env._drawn["iot"])
        pool_len_iot = len(env._pool.iot_ids)
        if drawn_iot < min(3, pool_len_iot):
            return "ACQUIRE_IOT"
    if not env.dispatched and ("acoustic" in env.spec.regime.available_modalities or
                                "thermal" in env.spec.regime.available_modalities):
        return "DISPATCH"
    if "thermal" in env.spec.regime.available_modalities and len(env._drawn["thermal"]) < len(env._pool.thermal_ids):
        return "ACQUIRE_THERMAL"
    if "acoustic" in env.spec.regime.available_modalities:
        n_pos = sum(1 for r_ in env.delivered.acoustic if r_.positive)
        n_neg = sum(1 for r_ in env.delivered.acoustic if not r_.positive)
        would_contradict = (n_pos >= 2 and n_neg >= 2) or (n_pos >= 1 and n_neg >= 3)
        if not would_contradict and len(env._drawn["acoustic"]) < len(env._pool.acoustic_ids):
            return "ACQUIRE_ACOUSTIC"
    if "iot" in env.spec.regime.available_modalities and len(env._drawn["iot"]) < len(env._pool.iot_ids):
        return "ACQUIRE_IOT"
    return "ESCALATE"


def frontier_replay_policy(target_state: AbstractState, *, rule: str = "primary") -> "Callable[[B2Env, int], str]":
    """The ACCEPTANCE-GATE reference (plan K): replays exactly the action
    MULTISET frontier.py's exhaustive search found for one specific
    target `AbstractState` (dispatch iff target.dispatched, then
    target.n_acoustic acoustic draws, target.n_iot iot draws,
    target.n_thermal thermal draws, in that fixed order), then commits
    the verdict that state's own delivered evidence supports (or
    escalates if it somehow doesn't -- should not happen for a state the
    search itself certified sufficient). Unlike `policy_clairvoyant_reference`
    (a general heuristic that can, on some pools, walk past the optimal
    state into contradiction -- see module history/tests), this is
    GUARANTEED to reproduce gold exactly, because it takes exactly the
    path gold was computed from. Used ONLY to validate that the pool is
    solvable and its costs/labels self-consistent -- never presented as a
    fair agent baseline (it needs the frontier's own answer to run)."""
    from typing import Callable  # local import: avoids a module-level cycle with frontier.py

    def _policy(env: "B2Env", step: int) -> str:
        if target_state.dispatched and not env.dispatched:
            return "DISPATCH"
        if len(env._drawn["acoustic"]) < target_state.n_acoustic:
            return "ACQUIRE_ACOUSTIC"
        if len(env._drawn["iot"]) < target_state.n_iot:
            return "ACQUIRE_IOT"
        if len(env._drawn["thermal"]) < target_state.n_thermal:
            return "ACQUIRE_THERMAL"
        r = sufficiency(env.delivered, rule=rule)
        if r == "fault":
            return "COMMIT_FAULT"
        if r == "normal":
            return "COMMIT_NORMAL"
        return "ESCALATE"

    return _policy
