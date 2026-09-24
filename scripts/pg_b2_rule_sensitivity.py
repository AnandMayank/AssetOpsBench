"""pg_b2_rule_sensitivity.py — G3 sensitivity analysis (Phase 1.3): rebuilds
gold and rescoring under R_primary and R_strict, and checks whether the
central benchmark properties (A-F in the request) survive the choice.

R_primary: the originally proposed rule (includes the IoT-only NORMAL
           closure: 3 consecutive in-band reads, no physical evidence).
R_strict:  removes that closure -- NORMAL always requires the
           3-negative-acoustic clause (some physical evidence), matching
           the asymmetry FAULT already has.

This script does NOT choose between them -- it reports consequences.
Zero model/API calls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

from playground.b2_families import PRIMARY_HORIZON, build_pool, world_pool_for  # noqa: E402
from playground.baselines import (  # noqa: E402
    TRIVIAL_POLICIES, Trajectory, frontier_replay_policy, make_acquire_everything,
    make_scripted_reference, run_policy,
)
from playground.b2_scoring import aggregate, score_trajectory  # noqa: E402
from playground.env import B2Env  # noqa: E402
from playground.frontier import AbstractState, reachable_sufficient_states  # noqa: E402
from playground.semantics import g_world, sufficiency  # noqa: E402

OUT_PATH = REPO_ROOT / "reports" / "playground" / "b2" / "rule_sensitivity_report.json"


def _reference_trajectory(spec, gold, rule):
    pool = world_pool_for(spec.world)
    target = g_world(spec.world)
    accept = lambda e, tv=target, r=rule: sufficiency(e, rule=r) == tv  # noqa: E731
    hits = reachable_sufficient_states(AbstractState(), pool, spec.regime,
                                        max_horizon=PRIMARY_HORIZON, accept=accept)
    if not hits:
        env = B2Env()
        env.reset(spec)
        env.step("ESCALATE")
        return Trajectory(episode_id=spec.episode_id, env=env, final_action="ESCALATE")
    return run_policy(spec, frontier_replay_policy(hits[0][0], rule=rule), max_steps=40)


def _run_variant(rule: str) -> dict:
    pool = build_pool(rule=rule)
    n = len(pool)

    family_counts = {}
    for spec, _ in pool:
        family_counts[spec.family] = family_counts.get(spec.family, 0) + 1
    verdicts = {"fault": 0, "normal": 0, "escalate": 0}
    for _, gold in pool:
        key = {"COMMIT_FAULT": "fault", "COMMIT_NORMAL": "normal", "ESCALATE": "escalate"}[gold.g_term]
        verdicts[key] += 1

    trivial_egrs = {}
    trivial_full = {}
    policies = dict(TRIVIAL_POLICIES)
    policies["AcquireEverything"] = make_acquire_everything(rule)  # rule-aware final verdict
    for name, policy in policies.items():
        scores = [score_trajectory(spec, gold, run_policy(spec, policy, max_steps=40)) for spec, gold in pool]
        agg = aggregate(scores)
        trivial_egrs[name] = agg["EGR"]
        trivial_full[name] = agg

    reference_scores = [score_trajectory(spec, gold, _reference_trajectory(spec, gold, rule)) for spec, gold in pool]
    reference_agg = aggregate(reference_scores)

    fault_share = verdicts["fault"] / n
    normal_share = verdicts["normal"] / n
    escalate_share = verdicts["escalate"] / n
    verdict_base_rate = max(fault_share, normal_share)

    properties = {
        "A_contract_rule_below_frontier_replay": trivial_egrs["ContractRule"] < reference_agg["EGR"] - 0.3,
        "B_acquire_everything_penalized": trivial_full["AcquireEverything"]["EGR"] <
                                           trivial_full["AcquireEverything"]["TGS"],
        "C_never_acquire_always_escalate_not_strong": max(trivial_egrs["NeverAcquire"],
                                                            trivial_egrs["AlwaysEscalate"]) <= escalate_share + 0.05,
        "D_pool_solvable": reference_agg["EGR"] >= 0.95,
        "E_headroom_between_trivial_and_reference": (reference_agg["EGR"] -
                                                       max(trivial_egrs.values())) >= 0.3,
        "F_not_reducible_to_bare_classification": trivial_egrs["ContractRule"] <= verdict_base_rate + 0.10,
    }

    return {
        "rule": rule, "n_episodes": n, "family_counts": family_counts,
        "verdict_distribution": {"fault_share": fault_share, "normal_share": normal_share,
                                  "escalate_share": escalate_share},
        "trivial_policy_egr": trivial_egrs,
        "trivial_policy_full": trivial_full,
        "frontier_replay_reference": reference_agg,
        "central_properties": properties,
        "all_central_properties_hold": all(properties.values()),
    }


def main() -> int:
    primary = _run_variant("primary")
    strict = _run_variant("strict")

    delta = {
        "n_episodes_delta": strict["n_episodes"] - primary["n_episodes"],
        "verdict_distribution_delta": {
            k: strict["verdict_distribution"][k] - primary["verdict_distribution"][k]
            for k in primary["verdict_distribution"]
        },
        "contract_rule_egr_delta": strict["trivial_policy_egr"]["ContractRule"] - primary["trivial_policy_egr"]["ContractRule"],
        "frontier_replay_egr_delta": strict["frontier_replay_reference"]["EGR"] - primary["frontier_replay_reference"]["EGR"],
        "acquire_everything_egr_delta": strict["trivial_policy_egr"]["AcquireEverything"] - primary["trivial_policy_egr"]["AcquireEverything"],
    }

    conclusions_highly_sensitive = (
        primary["all_central_properties_hold"] != strict["all_central_properties_hold"]
        or abs(delta["frontier_replay_egr_delta"]) > 0.15
        or abs(delta["contract_rule_egr_delta"]) > 0.15
    )

    report = {
        "schema": "pg_b2.rule_sensitivity_report/1",
        "R_primary": primary, "R_strict": strict, "delta": delta,
        "conclusions_highly_sensitive_to_rule_choice": conclusions_highly_sensitive,
        "note": ("Gold and PRIMARY-pool episode composition are recomputed per variant using the "
                 "SAME 43-episode family/world set (build_pool(rule=...) only changes what counts "
                 "as sufficient, not which worlds/regimes are sampled) -- n_episodes can still "
                 "differ if a world's frontier becomes empty/non-empty under the stricter rule."),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, default=str))

    print("R_primary:", {k: round(v, 3) for k, v in primary["trivial_policy_egr"].items()},
          "| FrontierReplay:", round(primary["frontier_replay_reference"]["EGR"], 3),
          "| properties hold:", primary["all_central_properties_hold"])
    print("R_strict: ", {k: round(v, 3) for k, v in strict["trivial_policy_egr"].items()},
          "| FrontierReplay:", round(strict["frontier_replay_reference"]["EGR"], 3),
          "| properties hold:", strict["all_central_properties_hold"])
    print("verdict distribution delta (strict - primary):", delta["verdict_distribution_delta"])
    print("HIGHLY SENSITIVE TO RULE CHOICE:", conclusions_highly_sensitive)
    print(f"\nWrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
