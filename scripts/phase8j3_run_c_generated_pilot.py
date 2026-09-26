"""phase8j3_run_c_generated_pilot.py -- runs the swapped primary panel
against the 112-episode cd_generator C pool (inspectionbench/manifests/
cd_4000_final_manifest.json, family C: 7 templates x multiple assets/seeds),
scaling C's per-model N from 17 (the classc_fixtures maximum) to 129
(17 + 112) -- the first time this pool has been run against a real model;
it was previously executed exactly once with a SCRIPTED, non-model tool
sequence (cd_generator.run_scripted_episode) purely to prove the generator
and scorer's own mechanics.

Reuses cd_generator.py's world/fixture setup (reset_from_world + FixtureSession,
byte-for-byte the same call as generate_and_run) and scoring (score_scripted_
episode, required_action_prf1, forbidden_violation, precedence_satisfaction,
ordering_satisfied) completely unmodified -- only run_scripted_episode's
DETERMINISTIC call sequence is replaced with a real 2-turn model interaction
(identical protocol to run_classc_pilot.py's run(): STEP 1 request tools,
STEP 2 give results + get verdict), producing the same ScriptedResult shape
the existing scorer already consumes.

The one new artifact this pool needed was a natural-language prompt per
template (phase8j2_c_generated_prompts.py) -- EpisodeSpec carries only
structured params. Each prompt is a direct parameterization of that
template's own already-validated source scenario (R016, R001, R005, R006/
R007, R017, R018, R024), substituting only asset name/location/gauge units;
the decisive per-episode variable is never disclosed in the prompt, matching
every source prompt's own convention (see phase8j2's docstring for the full
argument).

Models run SEQUENTIALLY -- same CouchDB shared-per-asset-document constraint
as A/C/E-expanded and E-expanded-v2.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from cd_generator import (TEMPLATES, EpisodeSpec, ScriptedResult,  # noqa: E402
                          score_scripted_episode)
from couchdb_executor import CouchDBExecutor  # noqa: E402
from execution_trace import ExecutionTrace, Stage  # noqa: E402
from tool_executor import STATUS_SUCCESS, ToolCall  # noqa: E402
from classc_fixtures import FixtureSession  # noqa: E402
from phase8j2_c_generated_prompts import prompt_for_episode  # noqa: E402
from run_generated_pilot import SYSTEM_PROMPT, _chat  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "inspectionbench" / "manifests" / "cd_4000_final_manifest.json"
OUT_DIR = REPO_ROOT / "reports" / "benchmark" / "v3_full_results" / "c_generated_112"
RAW_SCHEMA = "phase8j3_c_generated_raw/1"

# Swapped primary panel, BARE ids (run_generated_pilot._chat is the raw
# urllib path, needs the unprefixed id).
MODELS = [
    ("Claude_Opus_5.5", "anthropic/claude-opus-5.5"),
    ("GPT-6-Astra", "openai/gpt-6-astra"),
    ("DeepSeek_V4_Pro_0813", "deepseek/deepseek-v4-pro-0813"),
    ("Gemini_3.1_Pro_Preview", "google/gemini-3.1-pro-preview"),
    ("Qwen3.5-397B-A17B", "qwen/qwen3.5-397b-a17b"),
]


def build_spec(tid: str, params: Dict[str, Any]) -> EpisodeSpec:
    fn = TEMPLATES[tid]
    sig = inspect.signature(fn)
    kwargs = {k: v for k, v in params.items() if k in sig.parameters}
    return fn(**kwargs)


def run_episode(model: str, api_key: str, base_url: str, ex: CouchDBExecutor,
                spec: EpisodeSpec) -> Dict[str, Any]:
    """Model-driven replacement for cd_generator.run_scripted_episode, same
    2-turn protocol as run_classc_pilot.run(). Returns a ScriptedResult-shaped
    dict so score_scripted_episode (reused unmodified) can consume it."""
    ex.reset_from_world(spec.world, "FULL", seed=spec.params.get("seed", 1), withheld=[])
    ex.enterprise_override = dict(spec.enterprise)

    with FixtureSession(ex._robot.db, spec.fixture) as sess:
        problems = sess.verify_applied()
        if problems:
            return {"infra_failure": True, "infra_failure_reason": f"fixture not applied: {problems}"}

        trace = ExecutionTrace(spec.scenario_id, "FULL")
        tools = ex.available_tools()
        prompt = prompt_for_episode(spec.template_id, spec.params)
        task = (f"{prompt.strip()}\n\nTools available: {', '.join(tools)}\n"
               f"Allowed verdicts: COMMIT, ESCALATE, ABORT")
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{task}\n\nSTEP 1: request the tools you need."}]
        step1, err1 = _chat(model, msgs, api_key, base_url)

        results, executed_calls = [], []
        for call in (step1.get("tool_calls") or [])[:10]:
            name = str(call.get("tool", "")); args = call.get("args") or {}
            trace.append(Stage.REQUESTED, tool=name, args=args)
            res = ex.execute(ToolCall(name, args))
            if res.executed:
                trace.append(Stage.EXECUTED, tool=name, status=res.status)
                executed_calls.append({"tool": name, "args": args})
            if res.status == STATUS_SUCCESS:
                trace.append(Stage.SUCCEEDED, tool=name)
            if res.delivered:
                trace.append(Stage.OBSERVATION_DELIVERED, tool=name,
                             observation_id=res.observation_id,
                             observation_hash=res.observation_hash, modality=res.modality)
            results.append(res.to_dict())

        msgs += [{"role": "assistant", "content": json.dumps(step1)},
                 {"role": "user", "content": "STEP 2. Tool results:\n"
                  + json.dumps(results, indent=2) + "\n\nNow give your final decision."}]
        step2, err2 = _chat(model, msgs, api_key, base_url)
        verdict = str(step2.get("verdict") or "")
        if verdict:
            trace.append(Stage.DECISION, detail={"verdict": verdict})

        scripted = ScriptedResult(spec=spec, trace_events=[e.to_dict() for e in trace.events],
                                  trace_chain_valid=trace.verify_chain(),
                                  executed_calls=executed_calls, verdict=verdict)
        score = score_scripted_episode(spec, scripted)
        return {"infra_failure": False, "verdict": verdict, "score": score,
               "executed_calls": executed_calls, "reason": step2.get("reason", ""),
               "errors": [e for e in (err1, err2) if e], "apparatus_failure": bool(err2) or not verdict}


def run_model(model_label: str, model_bare: str, episodes: List[Dict[str, Any]],
             api_key: str, base_url: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"raw_{model_label}.jsonl"
    print(f"\n=== {model_label} ({model_bare}) -> {out_path} ===", flush=True)

    done_ids = set()
    if out_path.exists():
        with open(out_path) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not rec.get("infra_failure"):
                    done_ids.add(rec["episode_id"])
        if done_ids:
            print(f"Resuming: {len(done_ids)}/{len(episodes)} episodes already complete, skipping those", flush=True)

    ex = CouchDBExecutor()
    n_infra_fail = 0
    with open(out_path, "a") as fh:
        for i, ep in enumerate(episodes, 1):
            eid = ep["episode_id"]
            if eid in done_ids:
                continue
            spec = build_spec(ep["template_id"], ep["parameterization"])
            attempt, last_exc, ret = 0, None, None
            while attempt < 3:
                attempt += 1
                try:
                    ret = run_episode(model_bare, api_key, base_url, ex, spec)
                    last_exc = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    time.sleep(2 * attempt)

            if ret is None or ret.get("infra_failure"):
                n_infra_fail += 1
                reason = ret.get("infra_failure_reason") if ret else f"{type(last_exc).__name__}: {last_exc}"
                rec = {"schema": RAW_SCHEMA, "episode_id": eid, "template_id": ep["template_id"],
                      "model": model_bare, "infra_failure": True, "infra_failure_reason": reason}
                fh.write(json.dumps(rec, default=str) + "\n")
                fh.flush()
                print(f"[{i}/{len(episodes)}] {eid} INFRA FAILURE: {reason}", flush=True)
                continue

            rec = {"schema": RAW_SCHEMA, "episode_id": eid, "template_id": ep["template_id"],
                  "model": model_bare, "infra_failure": False, **ret}
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            print(f"[{i}/{len(episodes)}] {eid} CC={ret['score']['CC']} "
                 f"ordering_ok={ret['score']['ordering_satisfied']} "
                 f"coverage={ret['score']['required_action_prf1'].get('recall')}", flush=True)

    print(f"=== {model_label} done: {len(episodes) - n_infra_fail}/{len(episodes)} episodes, "
         f"{n_infra_fail} infra failures ===", flush=True)


def main() -> int:
    api_key = os.environ.get("TOKENROUTER_API_KEY", "")
    base_url = os.environ.get("TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1")
    if not api_key:
        print("ERROR: TOKENROUTER_API_KEY not set", file=sys.stderr)
        return 1

    manifest = json.loads(MANIFEST_PATH.read_text())
    episodes = [e for e in manifest["episodes"] if e["family"] == "C"]
    assert len(episodes) == 112

    only = sys.argv[1] if len(sys.argv) > 1 else None
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if limit:
        episodes = episodes[:limit]
    for label, bare in MODELS:
        if only and label != only:
            continue
        run_model(label, bare, episodes, api_key, base_url)

    print("\nAll requested models complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
