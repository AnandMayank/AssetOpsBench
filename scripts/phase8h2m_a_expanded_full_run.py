#!/usr/bin/env python3
"""phase8h2m_a_expanded_full_run.py -- Phase 8H.2M step 12/13: run ALL FIVE
primary-panel models on the IDENTICAL A-expanded-v1 manifest (240 episodes
each = 1200 model-episode evaluations), through the exact unmodified
production runner (phase8h_live_pilot.run_a_episode). Models run
SEQUENTIALLY (not concurrently) -- concurrent CouchDB writes across models
caused a real ConflictError earlier in this project; sequential execution
is the already-established fix.

No protocol change, no Claude-specific handling, no extra turns, no
tool_calls execution at the terminal step. Raw output persisted per model
under reports/benchmark/v3_full_results/a_expanded_240/. Does not touch
A-v1 (the historical 48-episode pool) or any other dimension's results.
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

from scenario_gen import FACTORIAL, sample_world  # noqa: E402
from canonical_identity import normalize_world  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "inspectionbench" / "manifests" / "a_evidence_grounding_expanded_v1.json"
OUT_DIR = REPO_ROOT / "reports" / "benchmark" / "v3_full_results" / "a_expanded_240"
MAX_RETRIES = 3
RAW_SCHEMA = "phase8h2m_a_expanded_raw/1"

# Same 5 primary-panel models, same BARE ids the raw urllib _chat() path
# needs (see run_ace_unit's documented tokenrouter/-prefix-stripping fix).
MODELS = [
    ("Claude_Sonnet_4.6", "anthropic/claude-sonnet-4.6"),
    ("GPT-5.2", "openai/gpt-5.2"),
    ("DeepSeek_V4_Pro", "deepseek/deepseek-v4-pro"),
    ("Mistral_Medium_3.5", "mistralai/mistral-medium-3-5"),
    ("Qwen3.5-397B-A17B", "qwen/qwen3.5-397b-a17b"),
]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def rebuild_a2_world(row: Dict[str, Any]):
    cell = next(c for c in FACTORIAL if c.name == row["scenario_parameters"]["cell"])
    world = sample_world(row["scenario_seed"], cell, asset_id=row["scenario_parameters"]["asset"],
                         scenario_id=row["scenario_id"])
    got = normalize_world(world).world_id
    assert got == row["normalized_world_hash"], (
        f"{row['episode_id']}: rebuilt hash {got} != manifest {row['normalized_world_hash']}")
    return world


def run_model(model_label: str, model_bare: str, rows: List[Dict[str, Any]],
             api_key: str, base_url: str) -> None:
    import phase8h_live_pilot as LP

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"raw_{model_label}.jsonl"
    print(f"\n=== {model_label} ({model_bare}) -> {out_path} ===", flush=True)

    n_infra_fail = 0
    with open(out_path, "w") as fh:
        for i, row in enumerate(rows, 1):
            eid = row["episode_id"]
            world = rebuild_a2_world(row)
            regime = row["regime"]

            attempt = 0
            last_exc = None
            ret = None
            started = _now()
            while attempt < MAX_RETRIES:
                attempt += 1
                try:
                    ret = LP.run_a_episode(world, regime, model=model_bare,
                                          api_key=api_key, base_url=base_url)
                    last_exc = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    time.sleep(2 * attempt)
            finished = _now()

            if ret is None:
                n_infra_fail += 1
                rec = {
                    "schema": RAW_SCHEMA, "episode_id": eid, "world_id": row["world_id"],
                    "scenario_id": row["scenario_id"], "regime": regime,
                    "model": model_bare, "started_at": started, "finished_at": finished,
                    "attempts": attempt, "infra_failure": True,
                    "infra_failure_reason": f"{type(last_exc).__name__}: {last_exc}",
                    "gold": row.get("gold_action"), "verdict": None, "metric": None,
                    "call_errors": [f"infra_failure: {last_exc}"],
                }
            else:
                ret = dict(ret)
                if hasattr(ret.get("metric"), "to_dict"):
                    ret["metric"] = ret["metric"].to_dict()
                verdict = ret.get("verdict", "")
                rec = {
                    "schema": RAW_SCHEMA, "episode_id": eid, "world_id": row["world_id"],
                    "scenario_id": row["scenario_id"], "regime": regime,
                    "model": model_bare, "started_at": started, "finished_at": finished,
                    "attempts": attempt, "infra_failure": False, "infra_failure_reason": None,
                    "gold": ret.get("gold"), "verdict": verdict,
                    "verdict_present": verdict != "",
                    "metric": ret.get("metric"), "integrity_flags": ret.get("integrity_flags"),
                    "call_errors": ret.get("call_errors") or [],
                }
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            if i % 20 == 0 or i == len(rows):
                tda_so_far = None
                print(f"  [{model_label}] {i}/{len(rows)} done, {n_infra_fail} infra failures so far", flush=True)

    print(f"=== {model_label} done: {len(rows)} episodes, {n_infra_fail} infra failures ===", flush=True)


def main() -> int:
    api_key = os.environ.get("TOKENROUTER_API_KEY", "")
    base_url = os.environ.get("TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1")
    if not api_key:
        print("ERROR: TOKENROUTER_API_KEY not set", file=sys.stderr)
        return 1

    manifest = json.loads(MANIFEST_PATH.read_text())
    rows = manifest["episodes"]
    assert len(rows) == 240

    only = sys.argv[1] if len(sys.argv) > 1 else None
    for label, bare in MODELS:
        if only and label != only:
            continue
        run_model(label, bare, rows, api_key, base_url)

    print("\nAll requested models complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
