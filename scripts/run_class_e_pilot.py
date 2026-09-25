"""run_class_e_pilot.py — Class-E sequential/stale-state pilot (P1, repaired P3).

Construct: an asset is inspected across 3 episodes. Between episodes the
hidden physical value may drift (stationary / step / linear_drift, drawn
*before* any label — ``sample_sequence`` cannot see one). The conversation
persists across episodes within a sequence, so an agent that answers episode 2
from its episode-0 reading — without re-observing — has the opportunity to do
so; that is the failure this measures.

**Repair (ledger B2, 2026-08-14).** The first run's system prompt stated *"a
prior visit's results do not carry forward automatically"* and told the agent
to request tools *"for THIS visit"* on every turn — both pre-empted the
failure mode by instructing re-observation outright. 0/180 episodes showed
``stale_state_reuse``. Both sentences are removed; nothing tells the agent
whether or how state persists.

That alone was not sufficient: with re-observation free, a model that blindly
re-observes every episode scores perfectly while reasoning about nothing.
``SequenceWorld.battery_budget`` (``sequence_executor.py``) now caps physical
reads across the *whole* sequence at ``n_episodes - 1`` — a world property
fixed at sample time, so which reads succeed depends only on the agent's own
choices, never on gold. The prompt states the constraint in the same
operational terms the rest of the benchmark already uses (a battery shared
across the visit), without stating the exact budget or which episodes need a
fresh read — otherwise this becomes a counting exercise rather than a test of
recognising when re-observation is actually warranted.

No new metric. CC is exact match against ``derive_sequence_gold`` per episode
— **unchanged**, a pure function of world and episode, independent of the
battery budget or any model output. PROC and CC_grounded reuse
``l3_grounded_scoring`` unchanged. ``stale_state_reuse`` and
``grounded_in_current_episode`` are read, not redefined. Two new *diagnostic*
fields, both pure functions of the world (never of what the model chose):
``reobservation_was_necessary`` (did band membership actually change since the
last delivered observation) and ``unnecessary_reobservation`` (the model read
again when it didn't need to). Neither feeds CC.

Runs FULL arm only. Minimal pilot: ``--n-sequences`` restricted to
``stationary`` and ``step`` processes only (``--processes``), because those
are the two conditions the repair's acceptance gate requires — a detectable
one-time change (proxy for "drift" in the gate) and a genuine no-change
control. ``linear_drift`` sequences remain valid and unaffected, just outside
this pilot's two gates.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
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
perform periodic inspections of the same asset over multiple visits in one \
round. The robot's battery is shared across the whole round: repeated sensor \
operations draw on it, and it is not recharged between visits.

Each visit, you work in two steps.

STEP 1 — request tools. Reply with EXACTLY:
{"tool_calls": [{"tool": "<name>", "args": {}}, ...]}
Only request tools from the list you are given.

STEP 2 — decide, once you have this visit's results. Reply with EXACTLY:
{"verdict": "COMMIT|ESCALATE|ABORT", "reason": "<one sentence>", "pa": <float or null>, "tool_sequence": ["<tool>", ...]}
"""


def _sample_filtered_sequence(seed_start: int, index: int, processes: tuple, n_episodes: int):
    """The i-th sequence whose drift process is in ``processes``.

    Scans forward from ``seed_start`` deterministically -- no change to
    ``sample_sequence`` itself, which still draws its process before any
    label and cannot see this filter.
    """
    seed = seed_start
    found = 0
    while True:
        w = sample_sequence(seed=seed, n_episodes=n_episodes)
        if w.process.kind in processes:
            if found == index:
                return w, seed
            found += 1
        seed += 1
TEMPERATURE = 0.0
MAX_TOKENS = 1024
N_EPISODES = 3
FM = "FM-6a"   # physical-read family; same PROC/CC_grounded branch as class A

#: Per-model token-budget override -- see run_l3_pilot_executed.py's
#: identical dict for the verified error and reasoning (this model
#: exhausts MAX_TOKENS on hidden reasoning before producing output).
MODEL_TOKEN_OVERRIDES = {
    "deepseek/deepseek-v4-pro-0813": 16384,
    "tokenrouter/deepseek/deepseek-v4-pro-0813": 16384,
}


def _tokens_for(model: str) -> int:
    return MODEL_TOKEN_OVERRIDES.get(model, MAX_TOKENS)


def _post_chat(model: str, messages: List[Dict[str, Any]], api_key: str, base_url: str,
               *, tokens_param: str = "max_tokens", include_temperature: bool = True) -> Dict[str, Any]:
    payload = {"model": model, "messages": messages, tokens_param: _tokens_for(model)}
    if include_temperature:
        payload["temperature"] = TEMPERATURE
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"{base_url}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def _chat(model: str, messages: List[Dict[str, Any]], api_key: str,
         base_url: str) -> Tuple[Dict[str, Any], Optional[str]]:
    """Two disclosed, model-triggered protocol fallbacks -- see
    run_l3_pilot_executed._chat's identical fix for the verified error
    text and reasoning (max_tokens->max_completion_tokens rename;
    temperature=0 rejected -> omit temperature, a real deviation from the
    rest of the panel's temperature=0 protocol, disclosed not silent)."""
    tokens_param, include_temperature = "max_tokens", True
    for _attempt in range(3):
        try:
            payload = _post_chat(model, messages, api_key, base_url,
                                 tokens_param=tokens_param, include_temperature=include_temperature)
            break
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode(errors="replace")
            if exc.code == 400 and "max_tokens" in body_text and "max_completion_tokens" in body_text and tokens_param == "max_tokens":
                tokens_param = "max_completion_tokens"
                continue
            if exc.code == 400 and "temperature" in body_text and include_temperature:
                include_temperature = False
                continue
            return {}, f"call_error: HTTPError {exc.code}: {body_text[:200]}"
        except Exception as exc:  # noqa: BLE001
            return {}, f"call_error: {exc}"
    else:
        return {}, "call_error: exhausted protocol-fallback retries"
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
                 se: SequenceExecutor, world, seed: int) -> Dict[str, Any]:
    se.begin_sequence(world)
    seq_trace = SequenceTrace(world.sequence_id, "FULL")

    msgs: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    episodes: List[Dict[str, Any]] = []
    apparatus_failure_any = False
    last_grounded_episode: Optional[int] = None   # world-truth bookkeeping only

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
            f"STEP 1: request the tools you need.")
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
                     + "\n\nNow give your decision.")}]
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

        reacquired = grounded_in_current_episode(seq_trace, k)
        # World-truth only: did the band membership actually change since the
        # last episode this agent had a delivered observation? Never a
        # function of what the model did this episode -- only of the world
        # and of *when it last had evidence*, which is itself read from the
        # trace (a fact about delivery, not about correctness).
        if last_grounded_episode is None:
            reobservation_was_necessary = True   # first opportunity is always "necessary"
        else:
            reobservation_was_necessary = (
                world.band_membership_at(k) != world.band_membership_at(last_grounded_episode))
        unnecessary_reobservation = reacquired and not reobservation_was_necessary
        if reacquired:
            last_grounded_episode = k

        episodes.append({
            "episode": k, "value_now": gold["value_now"], "gold": gold["verdict"],
            "band_crossed_by_now": gold["band_crossed_by_now"],
            "verdict": verdict, "reason": step2.get("reason", ""),
            "tools_requested": [c.get("tool") for c in requested],
            "tools_executed": sorted(trace.executed_tools()),
            "CC": cc, "PROC": proc, **grounding,
            "stale_state_reuse": stale_state_reuse(seq_trace, k),
            "reacquired_this_episode": reacquired,
            "reobservation_was_necessary": reobservation_was_necessary,
            "unnecessary_reobservation": unnecessary_reobservation,
            "battery_reads_used": se._physical_reads_used,
            "battery_budget": world.battery_budget,
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
    ap.add_argument("--processes", default="stationary,step",
                    help="drift processes to draw from (comma-separated); "
                         "the minimal pilot's two gates only need these two")
    ap.add_argument("--json", type=Path,
                    default=REPO_ROOT / "reports" / "v1" / "class_e_pilot.json")
    args = ap.parse_args()
    processes = tuple(p.strip() for p in args.processes.split(","))

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
        "processes": list(processes),
        "assetopsbench_sha": _git(REPO_ROOT, "rev-parse", "HEAD"),
        "status": "class-E repaired pilot (ledger B2) -- prompt de-contaminated, "
                 "battery budget makes re-observation costly",
    }
    print(f"model={args.model} n_sequences={args.n_sequences} "
         f"episodes={N_EPISODES} processes={processes}\n")

    rows: List[Dict[str, Any]] = []
    for i in range(args.n_sequences):
        world, seed = _sample_filtered_sequence(args.seed_start, i, processes, N_EPISODES)
        r = run_sequence(args.model, api_key, args.base_url, se, world, seed)
        rows.append(r)
        cc_vals = [e["CC"] for e in r["episodes"] if e["CC"] is not None]
        stale = sum(1 for e in r["episodes"] if e["stale_state_reuse"])
        unnecessary = sum(1 for e in r["episodes"] if e["unnecessary_reobservation"])
        print(f"  {r['sequence_id']:14s} {r['asset']:16s} proc={r['process']:12s} "
             f"CC={cc_vals} stale_reuse={stale} unnecessary_reobs={unnecessary} "
             f"chain_valid={r['sequence_trace_chain_valid']} "
             f"apparatus_failure={r['apparatus_failure_any']}")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(
        {"schema": "assetops.class_e_pilot/1", "provenance": prov, "results": rows},
        indent=2) + "\n")
    print(f"\n-> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
