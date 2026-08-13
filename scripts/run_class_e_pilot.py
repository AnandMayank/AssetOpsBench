"""run_class_e_pilot.py — Class-E sequential/stale-state pilot (P1).

Class E has never been run against a model. The apparatus (sequence-scoped
execution, hash-chained trace across episode boundaries) is validated by the
ten acceptance tests in ``test_classe_and_composite.py``; this script produces
the first model-behaviour data over it.

Construct: an asset is inspected across 3 episodes. Between episodes the
hidden physical value may drift (stationary / step / linear_drift, drawn
*before* any label — ``sample_sequence`` cannot see one). The conversation
persists across episodes within a sequence, so an agent that answers episode 2
from its episode-0 reading — without re-observing — has the opportunity to do
so; that is the failure this measures.

No new metric. CC is exact match against ``derive_sequence_gold`` per episode.
PROC and CC_grounded reuse ``l3_grounded_scoring`` unchanged, applied to the
per-episode ``ExecutionTrace`` a ``SequenceTrace`` already contains. The
stale-state diagnostic is ``stale_state_reuse``, already implemented and
tested; it is read here, not redefined.

Runs FULL arm only — class E is not an evidence-withholding construct, it is a
temporal one. 20 sequences x 3 episodes, per the frozen protocol table in
L3_FrozenBenchmarkDesign.md.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

from couchdb_executor import CouchDBExecutor              # noqa: E402
from execution_trace import Stage                          # noqa: E402
from l3_grounded_scoring import cc_grounded, proc_from_trace  # noqa: E402
from l3_scoring import normalise_action                     # noqa: E402
from sequence_executor import (                             # noqa: E402
    SequenceExecutor, SequenceTrace, derive_sequence_gold,
    grounded_in_current_episode, sample_sequence, stale_state_reuse,
)
from tool_executor import STATUS_SUCCESS, ToolCall           # noqa: E402

SYSTEM_PROMPT = """\
You are an industrial inspection agent operating a robot with real tools. You \
perform periodic inspections of the same asset over multiple visits.

Each visit, you work in two steps.

STEP 1 — request tools. Reply with EXACTLY:
{"tool_calls": [{"tool": "<name>", "args": {}}, ...]}
Only request tools from the list you are given. A prior visit's results do not
carry forward automatically — if you need the asset's current state, request
the tools that give it to you.

STEP 2 — decide, once you have this visit's results. Reply with EXACTLY:
{"verdict": "COMMIT|ESCALATE|ABORT", "reason": "<one sentence>", "pa": <float or null>, "tool_sequence": ["<tool>", ...]}
"""
TEMPERATURE = 0.0
MAX_TOKENS = 1024
N_EPISODES = 3
FM = "FM-6a"   # physical-read family; same PROC/CC_grounded branch as class A


def _chat(model: str, messages: List[Dict[str, Any]], api_key: str,
         base_url: str) -> Tuple[Dict[str, Any], Optional[str]]:
    body = json.dumps({"model": model, "messages": messages,
                       "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS}).encode()
    req = urllib.request.Request(f"{base_url}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            payload = json.loads(r.read())
    except Exception as exc:  # noqa: BLE001
        return {}, f"call_error: {exc}"
    text = (payload.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    if not text.strip():
        return {}, "no_answer: empty content"
    clean = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    m = re.search(r"\{.*\}", clean, re.DOTALL)
    if not m:
        return {}, f"parse_error: {text[:120]}"
    try:
        return json.loads(m.group(0)), None
    except json.JSONDecodeError as exc:
        return {}, f"parse_error: {exc}"


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                              text=True, timeout=15).stdout.strip()
    except Exception:
        return ""


def run_sequence(model: str, api_key: str, base_url: str,
                 se: SequenceExecutor, seed: int) -> Dict[str, Any]:
    world = sample_sequence(seed=seed, n_episodes=N_EPISODES)
    se.begin_sequence(world)
    seq_trace = SequenceTrace(world.sequence_id, "FULL")

    msgs: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    episodes: List[Dict[str, Any]] = []
    apparatus_failure_any = False

    for k in range(N_EPISODES):
        se.advance_episode("FULL", withheld=[])
        trace = seq_trace.new_episode()
        asset = world.asset

        visit_prompt = (
            f"Visit {k + 1} of {N_EPISODES} for {asset}. Gauge range "
            f"{world.gauge_range[0]}-{world.gauge_range[1]} {world.unit}; expected "
            f"operating band {world.operating_band[0]}-{world.operating_band[1]} "
            f"{world.unit}. Perform this visit's inspection and decide the "
            f"appropriate action.\n\nTools available: "
            f"{', '.join(se.available_tools())}\n"
            f"Allowed verdicts: COMMIT, ESCALATE, ABORT\n\n"
            f"STEP 1: request the tools you need for THIS visit.")
        msgs.append({"role": "user", "content": visit_prompt})
        step1, err1 = _chat(model, msgs, api_key, base_url)
        requested = [c for c in (step1.get("tool_calls") or []) if isinstance(c, dict)]

        results: List[Dict[str, Any]] = []
        images: List[str] = []
        for call in requested[:8]:
            name = str(call.get("tool", ""))
            args = call.get("args") or {}
            trace.append(Stage.REQUESTED, tool=name, args=args)
            res = se.execute(ToolCall(name, args))
            if res.executed:
                trace.append(Stage.EXECUTED, tool=name, status=res.status, error=res.error)
            if res.status == STATUS_SUCCESS:
                trace.append(Stage.SUCCEEDED, tool=name)
            if res.delivered:
                trace.append(Stage.OBSERVATION_DELIVERED, tool=name,
                             observation_id=res.observation_id,
                             observation_hash=res.observation_hash, modality=res.modality)
                if res.image_b64:
                    images.append(res.image_b64)
            results.append(res.to_dict())

        content: List[Dict[str, Any]] = [{
            "type": "text",
            "text": ("Tool results for this visit (produced by the executor, not "
                     "by you):\n" + json.dumps(results, indent=2)
                     + ("\n\nThe captured gauge image follows." if images else "")
                     + "\n\nNow give your decision for THIS visit.")}]
        for b64 in images:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"}})

        msgs.append({"role": "assistant", "content": json.dumps(step1)})
        msgs.append({"role": "user", "content": content})
        step2, err2 = _chat(model, msgs, api_key, base_url)
        msgs.append({"role": "assistant", "content": json.dumps(step2)})

        verdict = str(step2.get("verdict") or step2.get("action") or "")
        if verdict:
            trace.append(Stage.DECISION, detail={"verdict": verdict})

        gold = derive_sequence_gold(world, k)
        apparatus_failure = bool(err2) or not verdict
        apparatus_failure_any = apparatus_failure_any or apparatus_failure

        if apparatus_failure:
            cc = None
            grounding: Dict[str, Any] = {}
            proc = None
        else:
            action_norm = normalise_action(verdict)
            gold_norm = normalise_action(gold["verdict"])
            cc = int(action_norm == gold_norm)
            proc = proc_from_trace(FM, trace)
            grounding = cc_grounded(FM, cc, trace, step2.get("cited_observation_ids", ()) or ())

        episodes.append({
            "episode": k, "value_now": gold["value_now"], "gold": gold["verdict"],
            "band_crossed_by_now": gold["band_crossed_by_now"],
            "verdict": verdict, "reason": step2.get("reason", ""),
            "tools_requested": [c.get("tool") for c in requested],
            "tools_executed": sorted(trace.executed_tools()),
            "CC": cc, "PROC": proc, **grounding,
            "stale_state_reuse": stale_state_reuse(seq_trace, k),
            "reacquired_this_episode": grounded_in_current_episode(seq_trace, k),
            "apparatus_failure": apparatus_failure,
            "errors": [e for e in (err1, err2) if e],
        })

    se.end_sequence()

    return {
        "sequence_id": world.sequence_id, "asset": world.asset,
        "process": world.process.kind, "seed": seed,
        "world": world.to_dict(),
        "episodes": episodes,
        "sequence_trace_chain_valid": seq_trace.verify_chain(),
        "apparatus_failure_any": apparatus_failure_any,
        "trace": seq_trace.to_dict(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="openai/gpt-5.4-mini")
    ap.add_argument("--base-url", default=os.environ.get(
        "TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1"))
    ap.add_argument("--n-sequences", type=int, default=20)
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--json", type=Path,
                    default=REPO_ROOT / "reports" / "v1" / "class_e_pilot.json")
    args = ap.parse_args()

    api_key = os.environ.get("TOKENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: TOKENROUTER_API_KEY not set", file=sys.stderr)
        return 2

    backend = CouchDBExecutor()
    se = SequenceExecutor(backend)

    prov = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "generation": {"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS},
        "n_sequences": args.n_sequences, "n_episodes": N_EPISODES,
        "assetopsbench_sha": _git(REPO_ROOT, "rev-parse", "HEAD"),
        "status": "class-E first model run — apparatus validated, construct previously unmeasured",
    }
    print(f"model={args.model} n_sequences={args.n_sequences} "
         f"episodes={N_EPISODES}\n")

    rows: List[Dict[str, Any]] = []
    for i in range(args.n_sequences):
        seed = args.seed_start + i
        r = run_sequence(args.model, api_key, args.base_url, se, seed)
        rows.append(r)
        cc_vals = [e["CC"] for e in r["episodes"] if e["CC"] is not None]
        stale = sum(1 for e in r["episodes"] if e["stale_state_reuse"])
        print(f"  {r['sequence_id']:14s} {r['asset']:16s} proc={r['process']:12s} "
             f"CC={cc_vals} stale_reuse={stale} chain_valid={r['sequence_trace_chain_valid']} "
             f"apparatus_failure={r['apparatus_failure_any']}")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(
        {"schema": "assetops.class_e_pilot/1", "provenance": prov, "results": rows},
        indent=2) + "\n")
    print(f"\n-> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
