#!/usr/bin/env python3
"""pg_b2_score_pilot.py — Phase 5/6: scores a completed pg_b2_run.py output
(reports/playground/b2/model_pilot_trajectories.jsonl) against the
evaluator-only gold (model_pilot_gold_evaluator_only.json, never shown to
the model), computes primary + diagnostic metrics per model, and writes
world-clustered bootstrap CIs. Writes model_pilot_results.json and
model_pilot_summary.md. Zero model/API calls -- pure post-hoc scoring.

Replays each recorded action sequence against a FRESH B2Env (deterministic:
same spec -> same real resolver draws) rather than trying to reconstruct
live env state from the saved JSONL directly -- this reuses the
already-tested `b2_scoring.score_trajectory` path exactly as baselines.py
does, instead of a second scoring implementation.
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

from playground.b2_families import build_pool  # noqa: E402
from playground.baselines import run_policy  # noqa: E402
from playground.b2_scoring import aggregate, score_trajectory  # noqa: E402
from playground.semantics import TERMINAL_ACTIONS  # noqa: E402

OUT_DIR = REPO_ROOT / "reports" / "playground" / "b2"


def _replay_policy(actions: List[str]):
    def _policy(env, step):
        if step < len(actions):
            return actions[step]
        return "ESCALATE"
    return _policy


def _bootstrap_ci(values_by_world: Dict[str, List[float]], n_resamples: int = 2000, seed: int = 42):
    """Cluster bootstrap over WORLDS (never episodes/trajectories) --
    resamples whole worlds with replacement, pooling all their values each
    draw. Matches phase8h2l_main_table_build.py's bootstrap_ci pattern."""
    worlds = list(values_by_world.keys())
    if not worlds:
        return None, None, None
    rng = random.Random(seed)
    point = sum(v for vs in values_by_world.values() for v in vs) / \
        sum(len(vs) for vs in values_by_world.values())
    means = []
    for _ in range(n_resamples):
        sample_worlds = [rng.choice(worlds) for _ in worlds]
        vals = [v for w in sample_worlds for v in values_by_world[w]]
        if vals:
            means.append(sum(vals) / len(vals))
    means.sort()
    lo = means[int(0.025 * len(means))] if means else None
    hi = means[int(0.975 * len(means))] if means else None
    return point, lo, hi


def main() -> int:
    traj_path = OUT_DIR / "model_pilot_trajectories.jsonl"
    if not traj_path.exists():
        print(f"ERROR: {traj_path} not found -- run scripts/pg_b2_run.py first.", file=sys.stderr)
        return 1

    pool = {spec.episode_id: (spec, gold) for spec, gold in build_pool()}
    world_of = {eid: spec.world.world_id for eid, (spec, gold) in pool.items()}

    rows = [json.loads(l) for l in traj_path.read_text().splitlines() if l.strip()]
    by_model: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    results = {}
    for model, model_rows in by_model.items():
        per_metric_by_world: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        scores = []
        for r in model_rows:
            spec, gold = pool[r["episode_id"]]
            actions = [t.get("parsed_action") or "ESCALATE" for t in r["turns"] if "parsed_action" in t]
            traj = run_policy(spec, _replay_policy(actions), max_steps=len(actions) + 1)
            score = score_trajectory(spec, gold, traj)
            scores.append(score)
            w = world_of[r["episode_id"]]
            for metric in ("egr", "tgs", "sd", "tda", "premature_commit", "premature_escalate", "escalated"):
                per_metric_by_world[metric][w].append(float(getattr(score, metric)))
            per_metric_by_world["waste_count"][w].append(float(score.waste_count))
            per_metric_by_world["total_cost"][w].append(float(score.total_cost))
            if score.reg_min is not None:
                per_metric_by_world["reg_min"][w].append(float(score.reg_min))

        agg = aggregate(scores)
        cis = {}
        for metric, by_world in per_metric_by_world.items():
            point, lo, hi = _bootstrap_ci(by_world)
            cis[metric] = {"point": point, "ci95_lo": lo, "ci95_hi": hi}

        results[model] = {
            "n_episodes": len(model_rows), "n_worlds": len(set(world_of[r["episode_id"]] for r in model_rows)),
            "aggregate_uncorrected": agg,
            "world_clustered_ci": cis,
        }

    (OUT_DIR / "model_pilot_results.json").write_text(json.dumps(results, indent=2, default=str))

    lines = ["# B2 Model Pilot Summary\n",
             "World-clustered 95% bootstrap CIs (2000 resamples, worlds resampled with replacement). "
             "n_episodes is descriptive only -- CIs are computed at the world level.\n",
             "| Model | n_episodes | n_worlds | EGR | TGS | SD | TDA |",
             "|---|---|---|---|---|---|---|"]
    for model, r in results.items():
        ci = r["world_clustered_ci"]
        def _fmt(m):
            c = ci.get(m, {})
            if c.get("point") is None:
                return "n/a"
            return f"{c['point']:.3f} [{c['ci95_lo']:.3f}, {c['ci95_hi']:.3f}]"
        lines.append(f"| {model} | {r['n_episodes']} | {r['n_worlds']} | {_fmt('egr')} | {_fmt('tgs')} | "
                     f"{_fmt('sd')} | {_fmt('tda')} |")
    (OUT_DIR / "model_pilot_summary.md").write_text("\n".join(lines) + "\n")

    print(f"Wrote {OUT_DIR / 'model_pilot_results.json'}")
    print(f"Wrote {OUT_DIR / 'model_pilot_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
