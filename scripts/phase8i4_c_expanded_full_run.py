#!/usr/bin/env python3
"""phase8i4_c_expanded_full_run.py -- runs ALL FIVE primary-panel models
on the C-expanded-v1 manifest (8 new scenarios each = 40 model-episode
evaluations), through the exact unmodified production runner
(run_classc_pilot.run()). Models run SEQUENTIALLY -- concurrent CouchDB
writes across models caused a real ConflictError earlier in this project
(same established fix as A/E expansions).

Resume support: skips scenario_ids already present with a non-infra
record in this model's output file, so a re-run after an interruption
never re-spends API calls on completed scenarios (same pattern added to
phase8h2z_e_expanded_full_run.py after the 2026-09-23 session-teardown
incident).

Raw output persisted per model under
reports/benchmark/v3_full_results/c_expanded_17/. Does not touch the
original 9-episode C results or any other dimension's data.
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

from couchdb_executor import CouchDBExecutor  # noqa: E402
from classc_fixtures import FIXTURES, FixtureSession  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "inspectionbench" / "manifests" / "c_procedural_expanded_v1.json"
OUT_DIR = REPO_ROOT / "reports" / "benchmark" / "v3_full_results" / "c_expanded_17"
MAX_RETRIES = 3
RAW_SCHEMA = "phase8i4_c_expanded_raw/1"

MODELS = [
    ("Claude_Sonnet_4.6", "anthropic/claude-sonnet-4.6"),
    ("GPT-5.2", "openai/gpt-5.2"),
    ("DeepSeek_V4_Pro", "deepseek/deepseek-v4-pro"),
    ("Mistral_Medium_3.5", "mistralai/mistral-medium-3-5"),
    ("Qwen3.5-397B-A17B", "qwen/qwen3.5-397b-a17b"),
]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def run_model(model_label: str, model_bare: str, scenario_ids: List[str],
             api_key: str, base_url: str) -> None:
    import run_classc_pilot as C

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"raw_{model_label}.jsonl"
    print(f"\n=== {model_label} ({model_bare}) -> {out_path} ===", flush=True)

    done_sids = set()
    if out_path.exists():
        with open(out_path) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not rec.get("infra_failure"):
                    done_sids.add(rec["scenario_id"])
        if done_sids:
            print(f"Resuming: {len(done_sids)}/{len(scenario_ids)} scenarios already complete, skipping those", flush=True)

    n_infra_fail = 0
    with open(out_path, "a") as fh:
        for i, sid in enumerate(scenario_ids, 1):
            if sid in done_sids:
                continue
            ex = CouchDBExecutor()
            fx = FIXTURES[sid]
            started = _now()
            attempt = 0
            last_exc = None
            ret = None
            while attempt < MAX_RETRIES:
                attempt += 1
                try:
                    ex.reset(sid, "FULL", seed=1)
                    with FixtureSession(ex._robot.db, fx) as sess:
                        problems = sess.verify_applied()
                        if problems:
                            raise RuntimeError(f"fixture not applied: {problems}")
                        ret = C.run(model_bare, api_key, base_url, ex, sid,
                                   skip_reset=True, fixture_verified=not problems)
                    last_exc = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    time.sleep(2 * attempt)
            finished = _now()

            if ret is None:
                n_infra_fail += 1
                rec = {"schema": RAW_SCHEMA, "scenario_id": sid, "model": model_bare,
                      "started_at": started, "finished_at": finished, "attempts": attempt,
                      "infra_failure": True,
                      "infra_failure_reason": f"{type(last_exc).__name__}: {last_exc}"}
                fh.write(json.dumps(rec, default=str) + "\n")
                fh.flush()
                print(f"[{i}/{len(scenario_ids)}] {sid} INFRA FAILURE: {last_exc}", flush=True)
                continue

            rec = {"schema": RAW_SCHEMA, "scenario_id": sid, "model": model_bare,
                  "started_at": started, "finished_at": finished, "attempts": attempt,
                  "infra_failure": False, "infra_failure_reason": None, **ret}
            if hasattr(rec.get("trace"), "to_dict"):
                rec["trace"] = rec["trace"].to_dict()
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            print(f"[{i}/{len(scenario_ids)}] {sid} ok -- verdict={ret['verdict'] or '(empty)'} "
                 f"CC={ret['CC']} ordering_satisfied={ret['ordering_satisfied']}", flush=True)

    print(f"=== {model_label} done: {len(scenario_ids) - n_infra_fail}/{len(scenario_ids)} scenarios, "
         f"{n_infra_fail} infra failures ===", flush=True)


def main() -> int:
    api_key = os.environ.get("TOKENROUTER_API_KEY", "")
    base_url = os.environ.get("TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1")
    if not api_key:
        print("ERROR: TOKENROUTER_API_KEY not set", file=sys.stderr)
        return 1

    manifest = json.loads(MANIFEST_PATH.read_text())
    scenario_ids = manifest["new_8_this_pool"]
    assert len(scenario_ids) == 8

    only = sys.argv[1] if len(sys.argv) > 1 else None
    for label, bare in MODELS:
        if only and label != only:
            continue
        run_model(label, bare, scenario_ids, api_key, base_url)

    print("\nAll requested models complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
