#!/usr/bin/env python3
"""phase8j1_e_expanded_v2_full_run.py -- runs the SWAPPED primary-panel
models (Claude Opus 5.5, GPT-6-Astra, DeepSeek V4 Pro 0813, Gemini 3.1
Pro Preview, Qwen3.5-397B-A17B unchanged) on the E-expanded-v2 manifest
(30 sequences x 3 episodes = 90 episodes each = 450 model-episode
evaluations), through the exact unmodified production runner
(run_class_e_pilot.run_sequence). Models run SEQUENTIALLY (not
concurrently) -- concurrent CouchDB writes across models caused a real
ConflictError earlier in this project (same fix already established
for A-expanded / E-expanded-v1).

Combined with E-expanded-v1 (already run for all 5 swapped-panel models,
reports/benchmark/v3_full_results/e_expanded_90/), this gives a
per-model N=180 for E.

No protocol change, no per-model handling except the same disclosed
token/temperature fallbacks already built into run_class_e_pilot._chat.
Raw output persisted per model under
reports/benchmark/v3_full_results/e_expanded_v2_90/. Does not touch
E-v1, E-expanded-v1, or any other dimension's results.
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

from sequence_executor import SequenceExecutor, sample_sequence  # noqa: E402
from couchdb_executor import CouchDBExecutor  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "inspectionbench" / "manifests" / "e_temporal_expanded_v2.json"
OUT_DIR = REPO_ROOT / "reports" / "benchmark" / "v3_full_results" / "e_expanded_v2_90"
MAX_RETRIES = 3
RAW_SCHEMA = "phase8j1_e_expanded_v2_raw/1"

# Swapped primary panel, BARE ids (run_class_e_pilot._chat is the raw
# urllib path, needs the unprefixed id).
MODELS = [
    ("Claude_Opus_5.5", "anthropic/claude-opus-5.5"),
    ("GPT-6-Astra", "openai/gpt-6-astra"),
    ("DeepSeek_V4_Pro_0813", "deepseek/deepseek-v4-pro-0813"),
    ("Gemini_3.1_Pro_Preview", "google/gemini-3.1-pro-preview"),
    ("Qwen3.5-397B-A17B", "qwen/qwen3.5-397b-a17b"),
]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sequences_from_manifest(rows: List[Dict[str, Any]]):
    by_seq: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_seq.setdefault(r["world_id"], []).append(r)
    out = []
    for seq_id, seq_rows in sorted(by_seq.items()):
        seq_rows = sorted(seq_rows, key=lambda r: r["scenario_parameters"]["episode_index"])
        seed = seq_rows[0]["world_seed"]
        world = sample_sequence(seed, n_episodes=3)
        assert world.sequence_id == seq_id, f"{world.sequence_id} != {seq_id}"
        out.append((seq_id, seed, world))
    return out


def run_model(model_label: str, model_bare: str, sequences, api_key: str, base_url: str) -> None:
    import run_class_e_pilot as E

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"raw_{model_label}.jsonl"
    print(f"\n=== {model_label} ({model_bare}) -> {out_path} ===", flush=True)

    done_seq_ids = set()
    if out_path.exists():
        with open(out_path) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not rec.get("infra_failure"):
                    done_seq_ids.add(rec["sequence_id"])
        if done_seq_ids:
            print(f"Resuming: {len(done_seq_ids)}/{len(sequences)} sequences already complete, skipping those", flush=True)

    n_infra_fail = 0
    with open(out_path, "a") as fh:
        for i, (seq_id, seed, world) in enumerate(sequences, 1):
            if seq_id in done_seq_ids:
                continue
            ex = CouchDBExecutor()
            se = SequenceExecutor(ex)
            started = _now()
            attempt = 0
            last_exc = None
            ret = None
            while attempt < MAX_RETRIES:
                attempt += 1
                try:
                    ret = E.run_sequence(model_bare, api_key, base_url, se, world, seed)
                    last_exc = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    time.sleep(2 * attempt)
            finished = _now()

            if ret is None:
                n_infra_fail += 1
                rec = {"schema": RAW_SCHEMA, "sequence_id": seq_id, "world_seed": seed,
                      "model": model_bare, "started_at": started, "finished_at": finished,
                      "attempts": attempt, "infra_failure": True,
                      "infra_failure_reason": f"{type(last_exc).__name__}: {last_exc}",
                      "episodes": None}
                fh.write(json.dumps(rec, default=str) + "\n")
                fh.flush()
                print(f"[{i}/{len(sequences)}] {seq_id} INFRA FAILURE: {last_exc}", flush=True)
                continue

            rec = {"schema": RAW_SCHEMA, "sequence_id": seq_id, "world_seed": seed,
                  "model": model_bare, "started_at": started, "finished_at": finished,
                  "attempts": attempt, "infra_failure": False, "infra_failure_reason": None,
                  "asset": ret["asset"], "episodes": ret["episodes"]}
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()

            n_empty = sum(1 for e in ret["episodes"] if not e["verdict"])
            print(f"[{i}/{len(sequences)}] {seq_id} ok -- {3 - n_empty}/3 valid verdicts", flush=True)

    print(f"=== {model_label} done: {len(sequences) - n_infra_fail}/{len(sequences)} sequences, "
         f"{n_infra_fail} infra failures ===", flush=True)


def main() -> int:
    api_key = os.environ.get("TOKENROUTER_API_KEY", "")
    base_url = os.environ.get("TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1")
    if not api_key:
        print("ERROR: TOKENROUTER_API_KEY not set", file=sys.stderr)
        return 1

    manifest = json.loads(MANIFEST_PATH.read_text())
    rows = manifest["episodes"]
    assert len(rows) == 90
    sequences = sequences_from_manifest(rows)
    assert len(sequences) == 30

    only = sys.argv[1] if len(sys.argv) > 1 else None
    for label, bare in MODELS:
        if only and label != only:
            continue
        run_model(label, bare, sequences, api_key, base_url)

    print("\nAll requested models complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
