#!/usr/bin/env python3
"""phase8h2l_claude_a_full_validation.py -- Phase 8H.2L: full, fresh,
raw-response-persisting Claude Sonnet 4.6 re-run of all 48 canonical A
episodes, through the REAL, UNMODIFIED phase8h_live_pilot.run_a_episode
(same SYSTEM_PROMPT, same two-turn horizon, same terminal-verdict
requirement, same metric_contract.score_episode scoring). This is a
validation/reproduction run: it does not touch the frozen manifest, the
prior Claude run, other models' results, or any runner/scoring code.

Raw response capture is added by monkey-patching
run_l3_pilot_executed._chat with a thin wrapper that calls the ORIGINAL
_chat unchanged and additionally records the raw payload -- run_a_episode
itself is never edited, so its behavior (including its unmodified
"step2.get('verdict', '')" termination check) is exactly what produced
the original raw_Claude_Sonnet_4.6.jsonl file.

Output: one JSONL row per episode in
reports/benchmark/v3_full_results/frozen93/claude_A_full_validation/
raw_Claude_Sonnet_4.6_validation.jsonl
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pilot_dispatch as PD  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "reports" / "ec" / "phase8h1_pilot_manifest.json"
OUT_DIR = (REPO_ROOT / "reports" / "benchmark" / "v3_full_results" / "frozen93" /
          "claude_A_full_validation")
OUT_PATH = OUT_DIR / "raw_Claude_Sonnet_4.6_validation.jsonl"
MODEL_TOKENROUTER_ID = "tokenrouter/anthropic/claude-sonnet-4.6"  # for the record only
MODEL_BARE = "anthropic/claude-sonnet-4.6"  # what _chat actually needs (raw urllib path)
RAW_SCHEMA = "phase8h2l_claude_a_validation_raw/1"
MAX_RETRIES = 3  # matches phase8h1_run_pilot.py's existing infra-retry convention


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def main() -> int:
    api_key = os.environ.get("TOKENROUTER_API_KEY", "")
    base_url = os.environ.get("TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1")
    if not api_key:
        print("ERROR: TOKENROUTER_API_KEY not set", file=sys.stderr)
        return 1

    import phase8h_live_pilot as LP
    import run_l3_pilot_executed as R

    manifest = PD.load_manifest(MANIFEST_PATH)
    a_rows = [r for r in manifest["episodes"] if r["dimension"] == "A"]
    assert len(a_rows) == 48, f"expected 48 canonical A episodes, found {len(a_rows)}"

    # --- monkey-patch R._chat with a capturing wrapper; calls the REAL,
    # unmodified _chat underneath, so run_a_episode's behavior (including
    # its scoring) is byte-identical to the original run.
    _orig_chat = R._chat
    _capture_buffer: List[Dict[str, Any]] = []

    def _chat_capturing(model, messages, api_key_, base_url_):
        t0 = time.time()
        result, err = _orig_chat(model, messages, api_key_, base_url_)
        _capture_buffer.append({
            "model_requested": model,
            "result": result,
            "err": err,
            "n_messages": len(messages),
            "elapsed_s": round(time.time() - t0, 3),
        })
        return result, err

    R._chat = _chat_capturing

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Running Claude Sonnet 4.6 (bare id {MODEL_BARE!r}) on all 48 canonical A "
         f"episodes -> {OUT_PATH}")

    n_infra_fail = 0
    with open(OUT_PATH, "w") as fh:
        for i, row in enumerate(a_rows, 1):
            eid = row["episode_id"]
            world = PD.rebuild_world(row)
            arm = row["regime"]
            print(f"[{i}/48] {eid} (arm={arm}) ...", flush=True)

            attempt = 0
            last_exc = None
            ret = None
            started = _now()
            while attempt < MAX_RETRIES:
                attempt += 1
                _capture_buffer.clear()
                try:
                    ret = LP.run_a_episode(world, arm, model=MODEL_BARE,
                                          api_key=api_key, base_url=base_url)
                    last_exc = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    print(f"    attempt {attempt} raised {type(exc).__name__}: {exc}; "
                         f"retrying" if attempt < MAX_RETRIES else "    giving up", flush=True)
                    time.sleep(2 * attempt)
            finished = _now()

            step1_cap = _capture_buffer[0] if len(_capture_buffer) >= 1 else None
            step2_cap = _capture_buffer[1] if len(_capture_buffer) >= 2 else None

            if ret is None:
                n_infra_fail += 1
                rec = {
                    "schema": RAW_SCHEMA, "episode_id": eid, "world_id": row["world_id"],
                    "scenario_id": row["scenario_id"], "arm": arm,
                    "model": MODEL_TOKENROUTER_ID, "model_bare_used": MODEL_BARE,
                    "manifest_sha256": manifest.get("_sha256"),
                    "started_at": started, "finished_at": finished, "attempts": attempt,
                    "infra_failure": True,
                    "infra_failure_reason": f"{type(last_exc).__name__}: {last_exc}",
                    "step1_capture": step1_cap, "step2_capture": step2_cap,
                    "verdict": None, "gold": row.get("gold_action"),
                    "metric": None, "call_errors": [f"infra_failure: {last_exc}"],
                }
            else:
                ret = dict(ret)
                if hasattr(ret.get("metric"), "to_dict"):
                    ret["metric"] = ret["metric"].to_dict()
                verdict = ret.get("verdict", "")
                step2_parsed = (step2_cap or {}).get("result") or {}
                rec = {
                    "schema": RAW_SCHEMA, "episode_id": eid, "world_id": row["world_id"],
                    "scenario_id": row["scenario_id"], "arm": arm,
                    "model": MODEL_TOKENROUTER_ID, "model_bare_used": MODEL_BARE,
                    "manifest_sha256": manifest.get("_sha256"),
                    "started_at": started, "finished_at": finished, "attempts": attempt,
                    "infra_failure": False, "infra_failure_reason": None,
                    "step1_capture": step1_cap, "step2_capture": step2_cap,
                    "verdict": verdict,
                    "verdict_present": verdict != "",
                    "terminal_step_tool_calls_present": (
                        verdict == "" and list(step2_parsed.keys()) == ["tool_calls"]
                    ),
                    "gold": ret.get("gold"), "metric": ret.get("metric"),
                    "integrity_flags": ret.get("integrity_flags"),
                    "call_errors": ret.get("call_errors") or [],
                }
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            tda = (rec.get("metric") or {}).get("TDA") if rec.get("metric") else None
            print(f"    verdict_present={rec.get('verdict_present')} "
                 f"tool_calls_at_terminal={rec.get('terminal_step_tool_calls_present')} "
                 f"TDA={tda} infra_failure={rec['infra_failure']}", flush=True)

    R._chat = _orig_chat  # restore, though process exits anyway
    print(f"\nDone. {n_infra_fail} infra failures out of 48. Raw output: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
