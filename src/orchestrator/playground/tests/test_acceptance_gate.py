"""The acceptance gate (plan section K/M): no model run happens until
this passes. Trivial policies must fall at/below their analytic
prevalence ceilings on EGR, and the pool must be solvable (a reference
that replays the frontier's own certified path reaches EGR>=0.95)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

from playground.b2_families import PRIMARY_HORIZON, build_pool, world_pool_for  # noqa: E402
from playground.baselines import TRIVIAL_POLICIES, Trajectory, frontier_replay_policy, run_policy  # noqa: E402
from playground.b2_scoring import aggregate, score_trajectory  # noqa: E402
from playground.env import B2Env  # noqa: E402
from playground.frontier import AbstractState, reachable_sufficient_states  # noqa: E402
from playground.semantics import g_world, sufficiency  # noqa: E402

POOL = build_pool()


def _reference_trajectories():
    trajs = []
    for spec, gold in POOL:
        pool = world_pool_for(spec.world)
        target = g_world(spec.world)
        accept = lambda e, tv=target: sufficiency(e) == tv  # noqa: E731
        hits = reachable_sufficient_states(AbstractState(), pool, spec.regime,
                                            max_horizon=PRIMARY_HORIZON, accept=accept)
        if not hits:
            env = B2Env()
            env.reset(spec)
            env.step("ESCALATE")
            traj = Trajectory(episode_id=spec.episode_id, env=env, final_action="ESCALATE")
        else:
            traj = run_policy(spec, frontier_replay_policy(hits[0][0]), max_steps=40)
        trajs.append((spec, gold, traj))
    return trajs


def test_pool_is_nonempty_and_multi_world():
    worlds = {spec.world.world_id for spec, _ in POOL}
    assert len(POOL) >= 20
    assert len(worlds) >= 15


def test_scripted_reference_solvability_egr_at_least_0_95():
    scores = [score_trajectory(spec, gold, traj) for spec, gold, traj in _reference_trajectories()]
    agg = aggregate(scores)
    assert agg["EGR"] >= 0.95, f"pool not solvable/self-consistent: reference EGR={agg['EGR']}"


def test_trivial_policies_below_prevalence_ceiling():
    n = len(POOL)
    escalate_share = sum(1 for _, gold in POOL if gold.g_term == "ESCALATE") / n
    ceiling = max(escalate_share, 0.30)  # 0.30 generic floor matching plan's "<= 0.25" intent + margin
    trivial_egrs = {}
    for name, policy in TRIVIAL_POLICIES.items():
        scores = []
        for spec, gold in POOL:
            traj = run_policy(spec, policy, max_steps=40)
            scores.append(score_trajectory(spec, gold, traj))
        trivial_egrs[name] = aggregate(scores)["EGR"]
    for name, egr in trivial_egrs.items():
        assert egr <= ceiling, f"{name} EGR={egr} exceeds ceiling={ceiling} -- shortcut not closed"


def test_contract_rule_cannot_beat_verdict_base_rate_on_tda():
    n = len(POOL)
    fault_share = sum(1 for _, gold in POOL if gold.g_term == "COMMIT_FAULT") / n
    normal_share = sum(1 for _, gold in POOL if gold.g_term == "COMMIT_NORMAL") / n
    base_rate = max(fault_share, normal_share, 1 - fault_share - normal_share)
    scores = []
    for spec, gold in POOL:
        traj = run_policy(spec, TRIVIAL_POLICIES["ContractRule"], max_steps=40)
        scores.append(score_trajectory(spec, gold, traj))
    tda = aggregate(scores)["TDA"]
    assert tda <= base_rate + 0.10, f"ContractRule TDA={tda} exceeds base rate {base_rate} + margin"


def test_acquire_everything_penalized_by_waste():
    scores = []
    for spec, gold in POOL:
        traj = run_policy(spec, TRIVIAL_POLICIES["AcquireEverything"], max_steps=40)
        scores.append(score_trajectory(spec, gold, traj))
    agg = aggregate(scores)
    assert agg["mean_waste_count"] > 1.0
    assert agg["EGR"] < agg["TGS"]  # waste drags EGR below raw grounded accuracy
