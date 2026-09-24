#!/usr/bin/env python3
"""pg_b2_run_matched.py — fills the gap pg_b2_run.py left: the 8 matched-
group (M0-M4) episodes that exist only via build_matched_groups(), never
in build_pool(). Appends to the SAME trajectories.jsonl (resumable, same
mechanism as pg_b2_run.py) so pg_b2_score_pilot.py's replay-based scoring
can find them by episode_id. Also writes matched_group_analysis.json
comparing both arms of each group directly, per model.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import eval_model_panel as EMP  # noqa: E402
from llm.openai_compat import OpenAICompatBackend  # noqa: E402

from playground.b2_families import build_matched_groups, build_pool  # noqa: E402
import pg_b2_run as R  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts"))

OUT_DIR = REPO_ROOT / "reports" / "playground" / "b2"


def main() -> int:
    pool_ids = {spec.episode_id for spec, _ in build_pool()}
    groups = build_matched_groups()
    all_matched_specs = {s.episode_id: s for specs in groups.values() for s in specs}
    missing_specs = [s for eid, s in all_matched_specs.items() if eid not in pool_ids]
    print(f"{len(missing_specs)} matched-arm episodes missing from the primary pool; filling them.")

    traj_path = OUT_DIR / "model_pilot_trajectories.jsonl"
    already_done = set()
    for line in traj_path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            already_done.add((row.get("model"), row.get("episode_id")))

    with traj_path.open("a") as f:
        for model_cfg in EMP.PRIMARY_PANEL:
            backend = OpenAICompatBackend(model_cfg.tokenrouter_id)
            for spec in missing_specs:
                if (model_cfg.display_name, spec.episode_id) in already_done:
                    continue
                t0 = time.time()
                result = R.run_one_episode(backend, spec)
                result["model"] = model_cfg.display_name
                result["tokenrouter_id"] = model_cfg.tokenrouter_id
                result["elapsed_s"] = time.time() - t0
                result["matched_only"] = True  # never in build_pool(); flagged for scoring scripts
                f.write(json.dumps(result, default=str) + "\n")
                f.flush()
                print(f"  {model_cfg.display_name} / {spec.episode_id[-10:]}: "
                      f"final={result['final_action']} turns={result['n_turns']} cost={result['total_cost']:.1f}",
                      flush=True)

    # Direct arm-vs-arm comparison per model, per group.
    rows = [json.loads(l) for l in traj_path.read_text().splitlines() if l.strip()]
    by_model_ep = {(r["model"], r["episode_id"]): r for r in rows}
    analysis = {}
    for gname, specs in groups.items():
        analysis[gname] = {}
        for model_cfg in EMP.PRIMARY_PANEL:
            arms = {}
            for s in specs:
                r = by_model_ep.get((model_cfg.display_name, s.episode_id))
                if r:
                    arms[s.variant] = {"final_action": r["final_action"], "n_turns": r["n_turns"],
                                        "total_cost": r["total_cost"]}
            analysis[gname][model_cfg.display_name] = arms

    (OUT_DIR / "matched_group_analysis.json").write_text(json.dumps(analysis, indent=2, default=str))
    print(f"\nWrote {OUT_DIR / 'matched_group_analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
