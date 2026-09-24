"""pg_b2_build.py — Phase 1a: builds the B2 pool + matched groups, scores
the five trivial policies and the frontier-replay solvability reference,
and writes reports/playground/b2/{manifest,acceptance_gate_report}.json.

Offline only. Does not call a model. Does not modify any frozen artifact
(verified by the same checks in playground/tests/test_frozen_unchanged.py
and test_no_id_collision.py, which this script's caller should also run).

This is the go/no-go artifact for `pg_b2_run.py` (NOT YET IMPLEMENTED --
Phase 1a is offline-only per the approved plan): the acceptance gate below
must show every trivial policy at/below its prevalence ceiling and the
solvability reference at EGR >= 0.95 before any model-facing runner exists.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

from playground.b2_families import (  # noqa: E402
    PRIMARY_HORIZON, build_matched_groups, build_pool, world_pool_for,
)
from playground.baselines import (  # noqa: E402
    TRIVIAL_POLICIES, Trajectory, frontier_replay_policy, run_policy,
)
from playground.b2_scoring import aggregate, score_trajectory  # noqa: E402
from playground.env import B2Env  # noqa: E402
from playground.frontier import AbstractState, reachable_sufficient_states  # noqa: E402
from playground.semantics import g_world, sufficiency  # noqa: E402

OUT_DIR = REPO_ROOT / "reports" / "playground" / "b2"


def _reference_trajectory(spec, gold):
    pool = world_pool_for(spec.world)
    target = g_world(spec.world)
    accept = lambda e, tv=target: sufficiency(e) == tv  # noqa: E731
    hits = reachable_sufficient_states(AbstractState(), pool, spec.regime,
                                        max_horizon=PRIMARY_HORIZON, accept=accept)
    if not hits:
        env = B2Env()
        env.reset(spec)
        env.step("ESCALATE")
        return Trajectory(episode_id=spec.episode_id, env=env, final_action="ESCALATE")
    return run_policy(spec, frontier_replay_policy(hits[0][0]), max_steps=40)


def main() -> int:
    pool = build_pool()
    matched = build_matched_groups()

    manifest = {
        "schema": "pg_b2.manifest/1", "namespace": "PG-B2", "phase": "1a_offline",
        "n_episodes": len(pool),
        "n_worlds": len(sorted({spec.world.world_id for spec, _ in pool})),
        "family_counts": {},
        "matched_group_counts": {name: len(specs) for name, specs in matched.items()},
        "episodes": [],
    }
    for spec, gold in pool:
        manifest["family_counts"][spec.family] = manifest["family_counts"].get(spec.family, 0) + 1
        manifest["episodes"].append({
            "episode_id": spec.episode_id, "family": spec.family, "world_id": spec.world.world_id,
            "asset": spec.world.asset, "condition": spec.world.condition,
            "gold_g_term": gold.g_term, "gold_c_min": gold.c_min, "gold_frontier_empty": gold.frontier_empty,
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # --- acceptance gate: trivial policies ---
    n = len(pool)
    escalate_share = sum(1 for _, g in pool if g.g_term == "ESCALATE") / n
    fault_share = sum(1 for _, g in pool if g.g_term == "COMMIT_FAULT") / n
    normal_share = sum(1 for _, g in pool if g.g_term == "COMMIT_NORMAL") / n
    verdict_base_rate = max(fault_share, normal_share)

    trivial_results = {}
    for name, policy in TRIVIAL_POLICIES.items():
        scores = [score_trajectory(spec, gold, run_policy(spec, policy, max_steps=40)) for spec, gold in pool]
        trivial_results[name] = aggregate(scores)

    reference_scores = [score_trajectory(spec, gold, _reference_trajectory(spec, gold)) for spec, gold in pool]
    reference_agg = aggregate(reference_scores)

    ceilings = {
        "NeverAcquire": escalate_share, "AlwaysEscalate": escalate_share,
        "AlwaysCommit": 0.30, "AcquireEverything": 0.30, "ContractRule": verdict_base_rate,
    }
    gate_checks = {
        name: {"egr": trivial_results[name]["EGR"], "ceiling": ceilings[name],
               "pass": trivial_results[name]["EGR"] <= ceilings[name] + 1e-9}
        for name in TRIVIAL_POLICIES
    }
    solvability_pass = reference_agg["EGR"] >= 0.95
    overall_go = solvability_pass and all(c["pass"] for c in gate_checks.values())

    report = {
        "schema": "pg_b2.acceptance_gate_report/1", "n_episodes": n,
        "prevalence": {"escalate_share": escalate_share, "fault_share": fault_share,
                       "normal_share": normal_share, "verdict_base_rate": verdict_base_rate},
        "trivial_policy_results": trivial_results,
        "gate_checks": gate_checks,
        "frontier_replay_reference": reference_agg,
        "solvability_pass": solvability_pass,
        "GO_FOR_PHASE_1A_CONSTRUCTION": overall_go,
        "NOTE": ("This gate covers offline construction only. pg_b2_run.py "
                 "(model-facing) does not exist yet -- per the approved plan, "
                 "building it is a SEPARATE, later decision, gated on this "
                 "report plus G1-G4 (reports/playground/b2/substrate_gates_report.json)."),
    }
    (OUT_DIR / "acceptance_gate_report.json").write_text(json.dumps(report, indent=2, default=str))

    print(f"Pool: {n} episodes, {manifest['n_worlds']} worlds")
    print(f"Family counts: {manifest['family_counts']}")
    print(f"Verdict distribution: escalate={escalate_share:.3f} fault={fault_share:.3f} normal={normal_share:.3f}")
    print()
    for name, c in gate_checks.items():
        status = "PASS" if c["pass"] else "FAIL"
        print(f"  [{status}] {name:<20} EGR={c['egr']:.3f}  ceiling={c['ceiling']:.3f}")
    print(f"  [{'PASS' if solvability_pass else 'FAIL'}] FrontierReplay(ref)   EGR={reference_agg['EGR']:.3f}  target>=0.95")
    print()
    print(f"GO_FOR_PHASE_1A_CONSTRUCTION: {overall_go}")
    print(f"\nWrote {OUT_DIR / 'manifest.json'}")
    print(f"Wrote {OUT_DIR / 'acceptance_gate_report.json'}")
    return 0 if overall_go else 1


if __name__ == "__main__":
    raise SystemExit(main())
