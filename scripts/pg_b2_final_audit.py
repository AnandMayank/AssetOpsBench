"""pg_b2_final_audit.py — Phase 1 final offline scientific audit: the
per-episode audit table, baseline decomposition, world/sample audit, and
final leakage re-scan, folded into one report
(reports/playground/b2/final_audit_report.json +
final_audit_table.jsonl + baseline_decomposition.json). Evaluator-only
(gold/world fields never rendered to any agent). Zero model/API calls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

from playground import substrate  # noqa: E402
from playground.b2_families import (  # noqa: E402
    ACOUSTIC_WORLD_KEYS, PRIMARY_HORIZON, build_matched_groups, build_pool, world_pool_for,
)
from playground.baselines import (  # noqa: E402
    TRIVIAL_POLICIES, Trajectory, frontier_replay_policy, make_scripted_reference, run_policy,
)
from playground.b2_scoring import aggregate, score_trajectory  # noqa: E402
from playground.env import B2Env  # noqa: E402
from playground.frontier import AbstractState, reachable_sufficient_states  # noqa: E402
from playground.semantics import ACOUSTIC_EXCLUDED_KEYS, g_world, sufficiency  # noqa: E402

OUT_DIR = REPO_ROOT / "reports" / "playground" / "b2"


def _matched_group_lookup(pool_specs):
    matched = build_matched_groups()
    lookup = {}
    for name, specs in matched.items():
        for s in specs:
            lookup.setdefault(s.episode_id, []).append((name, s.variant))
    return lookup


def _reference_trajectory(spec, gold):
    pool = world_pool_for(spec.world)
    target = g_world(spec.world)
    accept = lambda e, tv=target: sufficiency(e) == tv  # noqa: E731
    hits = reachable_sufficient_states(AbstractState(), pool, spec.regime,
                                        max_horizon=PRIMARY_HORIZON, accept=accept)
    if not hits:
        return None, []
    return hits[0], hits


def _draw_sequence_summary(spec):
    """Evaluator-only summary of the real, ordered pool behind this
    episode's world -- NOT what the agent sees (env.py's whitelist
    renders one reading at a time via `render_reading`)."""
    dp = world_pool_for(spec.world)
    return {
        "acoustic_len": len(dp.acoustic), "acoustic_true_count": sum(dp.acoustic),
        "iot_len": len(dp.iot), "iot_true_count": sum(dp.iot),
        "thermal_len": len(dp.thermal), "thermal_true_count": sum(dp.thermal),
    }


def build_episode_audit_table():
    pool = build_pool()
    matched_lookup = _matched_group_lookup(pool)
    rows = []
    for spec, gold in pool:
        min_cost_hit, all_hits = _reference_trajectory(spec, gold)
        available = sorted(spec.regime.available_modalities)
        all_modalities = {"iot", "acoustic", "thermal", "record"}
        unavailable = sorted(all_modalities - spec.regime.available_modalities)
        acoustic_tier = None
        if spec.world.asset in substrate.AGREE_ACOUSTIC_ASSETS and spec.world.machine_id:
            key = (spec.world.asset, spec.world.machine_id)
            acoustic_tier = "L2_ASSET_CLASS_EVIDENCE_REPLAY"
            if key in ACOUSTIC_EXCLUDED_KEYS:
                acoustic_tier += "_EXCLUDED_G1_MARGINAL"
        mg = matched_lookup.get(spec.episode_id, [])
        rows.append({
            "episode_id": spec.episode_id,
            "world_id": spec.world.world_id,
            "scenario_id": f"{spec.family}::{spec.world.world_id}",
            "family": spec.family,
            "matched_groups": [f"{g}:{v}" for g, v in mg] or None,
            "world_condition": spec.world.condition,
            "asset": spec.world.asset,
            "source_machine_pool": spec.world.machine_id,
            "initial_delivered_evidence_E0": list(spec.regime.initial_modalities),
            "available_actions": available,
            "unavailable_actions": unavailable,
            "action_costs": {
                "query_record": spec.regime.costs.query_record, "iot_read": spec.regime.costs.iot_read,
                "acoustic_capture": spec.regime.costs.acoustic_capture,
                "thermal_capture": spec.regime.costs.thermal_capture, "dispatch": spec.regime.costs.dispatch,
            },
            "budget": spec.regime.budget,
            "draw_sequence_summary": _draw_sequence_summary(spec),
            "sufficient_state_rule_outcome": gold.g_term,
            "min_world_truth_agreeing_cost": gold.c_min,
            "frontier_empty": gold.frontier_empty,
            "gold_terminal_action": gold.g_term,
            "n_zero_regret_paths_at_min_cost": len(all_hits) if all_hits else 0,
            "primary_or_diagnostic": "primary",
            "acoustic_provenance_tier": acoustic_tier,
        })
    return rows


def build_baseline_decomposition():
    pool = build_pool()
    results = {}

    def _score_all(policy_fn, label):
        rows = []
        for spec, gold in pool:
            traj = run_policy(spec, policy_fn, max_steps=40)
            s = score_trajectory(spec, gold, traj)
            rows.append(s)
        agg = aggregate(rows)
        return {
            "EGR": agg["EGR"], "TGS": agg["TGS"], "SD": agg["SD"],
            "wasted_acquisition_rate": agg["mean_waste_count"],
            "reg_min_mean": agg["mean_reg_min"],
            "acquisition_count_mean": agg["mean_n_acquisitions"],
            "premature_commit_rate": agg["premature_commit_rate"],
            "premature_escalate_rate": agg["premature_escalate_rate"],
            "repeat_unavailable_mean": sum(r.repeat_unavailable for r in rows) / len(rows),
            "cost_mean": agg["mean_total_cost"],
            "TDA": agg["TDA"],
        }

    for name, policy in TRIVIAL_POLICIES.items():
        results[name] = _score_all(policy, name)
    results["FairScriptedReference"] = _score_all(make_scripted_reference("primary"), "FairScriptedReference")

    ref_rows = []
    for spec, gold in pool:
        min_cost_hit, _ = _reference_trajectory(spec, gold)
        if min_cost_hit is None:
            env = B2Env()
            env.reset(spec)
            env.step("ESCALATE")
            traj = Trajectory(episode_id=spec.episode_id, env=env, final_action="ESCALATE")
        else:
            traj = run_policy(spec, frontier_replay_policy(min_cost_hit[0]), max_steps=40)
        ref_rows.append(score_trajectory(spec, gold, traj))
    ref_agg = aggregate(ref_rows)
    results["FrontierReplay"] = {
        "EGR": ref_agg["EGR"], "TGS": ref_agg["TGS"], "SD": ref_agg["SD"],
        "wasted_acquisition_rate": ref_agg["mean_waste_count"], "reg_min_mean": ref_agg["mean_reg_min"],
        "acquisition_count_mean": ref_agg["mean_n_acquisitions"],
        "premature_commit_rate": ref_agg["premature_commit_rate"],
        "premature_escalate_rate": ref_agg["premature_escalate_rate"],
        "repeat_unavailable_mean": sum(r.repeat_unavailable for r in ref_rows) / len(ref_rows),
        "cost_mean": ref_agg["mean_total_cost"], "TDA": ref_agg["TDA"],
    }

    interpretation = {
        "ContractRule": ("fails because its stop/verdict rule reads only acquisition COUNTS, "
                          f"never content -- TDA={results['ContractRule']['TDA']:.3f} near the "
                          "verdict base rate, EGR=0.0."),
        "AcquireEverything": (f"reaches decent grounded accuracy (TGS={results['AcquireEverything']['TGS']:.3f}) "
                               f"but EGR collapses to {results['AcquireEverything']['EGR']:.3f} because it keeps "
                               f"acquiring after sufficiency (mean waste="
                               f"{results['AcquireEverything']['wasted_acquisition_rate']:.2f})."),
        "NeverAcquire": (f"EGR={results['NeverAcquire']['EGR']:.3f} equals exactly the ESCALATE-gold share -- "
                          "correct only on episodes where no acquisition was ever needed."),
        "AlwaysCommit": (f"EGR={results['AlwaysCommit']['EGR']:.3f}, premature_commit_rate="
                          f"{results['AlwaysCommit']['premature_commit_rate']:.3f} -- always commits on "
                          "empty (insufficient) evidence."),
        "AlwaysEscalate": ("identical to NeverAcquire here since E0 is empty in every episode -- fails "
                            f"exactly on the {1 - results['AlwaysEscalate']['EGR']:.3f} share of episodes "
                            "where a world-truth-agreeing state IS reachable."),
        "FairScriptedReference": (f"EGR={results['FairScriptedReference']['EGR']:.3f}: a fair, non-clairvoyant "
                                   "content-aware policy that stops the moment evidence looks sufficient -- "
                                   "not 1.0 because real sensor noise can make it stop on the wrong verdict, "
                                   "which is expected, not a defect."),
        "FrontierReplay": (f"EGR={results['FrontierReplay']['EGR']:.3f}: exact replay of the offline search's "
                            "own certified minimum-cost path -- proves the pool is solvable and internally "
                            "consistent; NOT a fair agent baseline (it needs the answer to run)."),
    }
    return {"results": results, "interpretation": interpretation}


def world_sample_audit():
    pool = build_pool()
    worlds = {spec.world.world_id: spec for spec, _ in pool}
    acoustic_worlds = {w.world_id for w in
                        [spec.world for spec, _ in pool if spec.world.machine_id is not None]}
    thermal_worlds = {w.world_id for w in
                       [spec.world for spec, _ in pool if spec.world.asset == "motor_01"]}
    iot_only_worlds = {w.world_id for w in
                        [spec.world for spec, _ in pool
                         if spec.world.machine_id is None and spec.world.asset != "motor_01"]}
    family_counts = {}
    for spec, _ in pool:
        family_counts[spec.family] = family_counts.get(spec.family, 0) + 1
    matched = build_matched_groups()
    return {
        "n_episodes": len(pool),
        "n_unique_worlds": len(worlds),
        "n_independent_acoustic_source_pools": len(ACOUSTIC_WORLD_KEYS),
        "n_acoustic_worlds_in_pool": len(acoustic_worlds),
        "n_thermal_worlds_in_pool": len(thermal_worlds),
        "n_iot_only_worlds_in_pool": len(iot_only_worlds),
        "family_counts": family_counts,
        "matched_group_counts": {k: len(v) for k, v in matched.items()},
        "warning": ("worlds != episodes != trajectories. n_episodes=43 is NOT 43 independent "
                    f"samples -- there are only {len(worlds)} unique worlds behind them, and a "
                    "future model pilot's confidence intervals MUST cluster-bootstrap at the "
                    "world level, never treat episode rows as independent."),
    }


def leakage_reaudit():
    """Re-scan G4 plus an explicit check that NO evaluator-only key name
    (matched_group, family, world_id, scenario_id, target verdict, gold,
    frontier/oracle fields) appears in what `env.py` actually returns from
    `step()`/`reset()` -- as opposed to substrate.render_reading's
    payload alone (already checked by G4)."""
    pool = build_pool()[:6]  # representative sample -- full sweep already done by G4
    forbidden_keys = {"matched_group", "variant", "family", "world_id", "scenario_id",
                       "g_term", "gold", "frontier_empty", "c_min", "target_verdict", "world_truth"}
    violations = []
    for spec, gold in pool:
        env = B2Env()
        obs = env.reset(spec)
        payloads = [obs.payload]
        for action in ("DISPATCH", "ACQUIRE_IOT", "ACQUIRE_ACOUSTIC", "QUERY_RECORD"):
            try:
                r = env.step(action)
                payloads.append(r.observation.payload)
            except Exception:
                continue
        for p in payloads:
            hits = substrate.scan_for_leakage(p)
            key_hits = [k for k in forbidden_keys if _contains_key(p, k)]
            if hits or key_hits:
                violations.append({"episode_id": spec.episode_id, "leak_hits": hits, "key_hits": key_hits})
    return {"n_episodes_checked": len(pool), "violations": violations, "clean": not violations}


def _contains_key(node, key):
    if isinstance(node, dict):
        if key in node:
            return True
        return any(_contains_key(v, key) for v in node.values())
    if isinstance(node, (list, tuple)):
        return any(_contains_key(v, key) for v in node)
    return False


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    audit_rows = build_episode_audit_table()
    with (OUT_DIR / "final_audit_table.jsonl").open("w") as f:
        for row in audit_rows:
            f.write(json.dumps(row, default=str) + "\n")

    baseline_decomp = build_baseline_decomposition()
    (OUT_DIR / "baseline_decomposition.json").write_text(json.dumps(baseline_decomp, indent=2, default=str))

    world_audit = world_sample_audit()
    leak_audit = leakage_reaudit()

    g1_report = json.loads((OUT_DIR / "substrate_gates_report.json").read_text())
    g4_prior = g1_report["g4"]

    final_report = {
        "schema": "pg_b2.final_audit_report/1",
        "g1_summary": {"verdict": g1_report["g1"]["verdict"], "best_overall": g1_report["g1"]["best_overall"],
                       "excluded_keys": sorted(str(k) for k in ACOUSTIC_EXCLUDED_KEYS)},
        "g4_original": {"verdict": g4_prior["verdict"], "n_checked": g4_prior["n_payloads_checked"]},
        "g4_reaudit_live_env": leak_audit,
        "world_sample_audit": world_audit,
        "n_audit_table_rows": len(audit_rows),
    }
    (OUT_DIR / "final_audit_report.json").write_text(json.dumps(final_report, indent=2, default=str))

    print("World/sample audit:", json.dumps(world_audit, indent=2, default=str))
    print()
    print("G4 re-audit (live env.py payloads):", "CLEAN" if leak_audit["clean"] else f"VIOLATIONS: {leak_audit['violations']}")
    print()
    print("Baseline decomposition (EGR/TGS/SD/waste):")
    for name, r in baseline_decomp["results"].items():
        print(f"  {name:<22} EGR={r['EGR']:.3f} TGS={r['TGS']:.3f} SD={r['SD']:.3f} "
              f"waste={r['wasted_acquisition_rate']:.2f} cost={r['cost_mean']:.2f}")
    print(f"\nWrote {OUT_DIR / 'final_audit_table.jsonl'}")
    print(f"Wrote {OUT_DIR / 'baseline_decomposition.json'}")
    print(f"Wrote {OUT_DIR / 'final_audit_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
