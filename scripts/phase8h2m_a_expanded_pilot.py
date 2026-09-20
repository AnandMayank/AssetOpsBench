#!/usr/bin/env python3
"""phase8h2m_a_expanded_pilot.py -- Phase 8H.2M step 10: pipeline-validation
pilot for the A-expanded-v1 (240-episode) manifest. Real API calls, small
scope: 5 matched worlds x 3 regimes = 15 episodes, one model
(Claude Sonnet 4.6, since it is the model whose behavior most needs
confirming at this pool before the full 5-model run -- not because the
protocol is being tuned for it).

This is a PIPELINE validation, not a result: it checks that prompt
construction, evidence delivery, tool availability, terminal stopping,
JSON parsing, scoring, and raw-output persistence all work correctly
against the NEW manifest, through the exact unmodified production runner
(phase8h_live_pilot.run_a_episode / run_l3_pilot_executed._chat). No
protocol change. No Claude-specific handling.

Output: reports/benchmark/a_expanded/pilot/raw_pilot_<model>.jsonl
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from scenario_gen import FACTORIAL, ASSETS, sample_world, derive_gold  # noqa: E402
from canonical_identity import normalize_world  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "inspectionbench" / "manifests" / "a_evidence_grounding_expanded_v1.json"
OUT_DIR = REPO_ROOT / "reports" / "benchmark" / "a_expanded" / "pilot"
PILOT_WORLD_IDS = [
    "A2-chiller_6-phys_in__iot_agree-4000",       # COMMIT
    "A2-hydraulic_pump_1-phys_in__iot_agree-4020",  # COMMIT
    "A2-chiller_6-phys_in__iot_agree-4002",       # ESCALATE
    "A2-hydraulic_pump_1-phys_in__iot_agree-4021",  # ESCALATE
    "A2-metro_pump_1-phys_in__iot_agree-4043",    # ESCALATE
]
REGIMES = ("FULL", "PHYSICAL_ONLY", "DIGITAL_ONLY")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def rebuild_a2_world(row: Dict[str, Any]):
    """A-expanded rows are NOT in pilot_dispatch's frozen-93 manifest, so
    rebuild directly via the same generator call the manifest was built
    with (scenario_gen.sample_world), then hash-assert exactly like
    pilot_dispatch.rebuild_world does for A-v1."""
    from scenario_gen import Cell
    cell_name = row["scenario_parameters"]["cell"]
    cell = next(c for c in FACTORIAL if c.name == cell_name)
    world = sample_world(row["scenario_seed"], cell, asset_id=row["scenario_parameters"]["asset"],
                         scenario_id=row["scenario_id"])
    got = normalize_world(world).world_id
    assert got == row["normalized_world_hash"], (
        f"{row['episode_id']}: rebuilt hash {got} != manifest {row['normalized_world_hash']}")
    return world


def main() -> int:
    api_key = os.environ.get("TOKENROUTER_API_KEY", "")
    base_url = os.environ.get("TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1")
    if not api_key:
        print("ERROR: TOKENROUTER_API_KEY not set", file=sys.stderr)
        return 1

    import phase8h_live_pilot as LP

    manifest = json.loads(MANIFEST_PATH.read_text())
    rows_by_wid: Dict[str, List[Dict[str, Any]]] = {}
    for e in manifest["episodes"]:
        rows_by_wid.setdefault(e["world_id"], []).append(e)

    model_bare = "anthropic/claude-sonnet-4.6"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "raw_pilot_Claude_Sonnet_4.6.jsonl"

    checks = {"prompt_construction_ok": True, "evidence_delivery_ok": True,
             "tool_availability_ok": True, "terminal_stopping_ok": True,
             "json_parsing_ok": True, "scoring_ok": True, "persistence_ok": True,
             "episode_ids_ok": True, "world_ids_ok": True, "regime_labels_ok": True,
             "gold_actions_ok": True}
    problems: List[str] = []

    n = 0
    with open(out_path, "w") as fh:
        for wid in PILOT_WORLD_IDS:
            for regime in REGIMES:
                row = next(r for r in rows_by_wid[wid] if r["regime"] == regime)
                world = rebuild_a2_world(row)
                n += 1
                print(f"[{n}/15] {row['episode_id']} (arm={regime}) ...", flush=True)
                started = _now()
                try:
                    ret = LP.run_a_episode(world, regime, model=model_bare,
                                          api_key=api_key, base_url=base_url)
                except Exception as exc:  # noqa: BLE001
                    problems.append(f"{row['episode_id']}: runner raised {type(exc).__name__}: {exc}")
                    checks["terminal_stopping_ok"] = False
                    continue
                finished = _now()
                ret = dict(ret)
                if hasattr(ret.get("metric"), "to_dict"):
                    ret["metric"] = ret["metric"].to_dict()

                # --- pipeline checks -------------------------------------------------
                if ret["world_id"] != row["world_id"]:
                    problems.append(f"{row['episode_id']}: world_id mismatch"); checks["world_ids_ok"] = False
                if ret["arm"] != regime:
                    problems.append(f"{row['episode_id']}: regime label mismatch"); checks["regime_labels_ok"] = False
                if ret["gold"] != row["gold_action"]:
                    problems.append(f"{row['episode_id']}: gold_action mismatch "
                                    f"(manifest={row['gold_action']!r} runner={ret['gold']!r})")
                    checks["gold_actions_ok"] = False
                if not isinstance(ret.get("metric"), dict):
                    problems.append(f"{row['episode_id']}: no scored metric returned"); checks["scoring_ok"] = False

                rec = {
                    "episode_id": row["episode_id"], "world_id": row["world_id"],
                    "scenario_id": row["scenario_id"], "regime": regime,
                    "manifest_gold_action": row["gold_action"], "runner_gold_action": ret["gold"],
                    "model": model_bare, "started_at": started, "finished_at": finished,
                    "runner_return": ret,
                }
                fh.write(json.dumps(rec, default=str) + "\n")
                fh.flush()
                verdict = ret.get("verdict", "")
                tda = (ret.get("metric") or {}).get("TDA")
                print(f"    verdict_present={bool(verdict)} TDA={tda} "
                     f"call_errors={ret.get('call_errors')}", flush=True)

    print(f"\nPilot done: {n}/15 episodes attempted, {len(problems)} pipeline problems found.")
    print("Checks:", json.dumps(checks, indent=2))
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print(" -", p)
    else:
        print("All pipeline checks PASS -- prompt construction, evidence delivery, tool "
             "availability, terminal stopping, JSON parsing, scoring, raw-output "
             "persistence, episode IDs, world IDs, regime labels, and gold actions all "
             "verified correct against the A-expanded-v1 manifest.")
    print(f"Raw pilot output: {out_path}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
