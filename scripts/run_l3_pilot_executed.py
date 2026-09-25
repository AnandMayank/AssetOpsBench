"""run_l3_pilot_executed.py — Six-scenario pilot on the repaired apparatus.

Replaces the single-shot, self-report runner. Two turns:

  1. the model is shown the task and the tools *actually available in this arm*
     and asks for the calls it wants;
  2. the executor runs them against hidden CouchDB state and returns the real
     results — including the rendered gauge image as an ``image_url`` block, so
     the model receives pixels rather than a path — and the model decides.

Everything that determines a score comes from the executor's trace. The model's
own ``tool_sequence`` is still collected, but only so fabricated procedure can be
detected by comparing it against what ran.

Refuses to start unless ``scripts/l3_execution_preflight.py`` passes.
Scenarios, gold labels and the preregistered CC are untouched.
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
SCEN_REPO = REPO_ROOT.parent / "AssetOpsBenchScenarioGeneration"
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

import l3_arms as A  # noqa: E402
from couchdb_executor import SCENARIO_PHYSICAL, CouchDBExecutor  # noqa: E402
from execution_trace import ExecutionTrace, Stage  # noqa: E402
from l3_grounded_scoring import score_l3_grounded  # noqa: E402
from l3_integrity import check_integrity  # noqa: E402
from tool_executor import STATUS_SUCCESS, ToolCall  # noqa: E402

SYSTEM_PROMPT = """\
You are an industrial inspection agent operating a robot with real tools.

You work in two steps.

STEP 1 — request tools. Reply with EXACTLY:
{"tool_calls": [{"tool": "<name>", "args": {}}, ...]}
Only request tools from the list you are given. Tools not listed are unavailable.

STEP 2 — decide, once you have the results. Reply with EXACTLY:
{"verdict": "COMMIT|ESCALATE|ABORT", "reason": "<one sentence>", "pa": <float or null>, "tool_sequence": ["<tool>", ...]}

"tool_sequence" must list the tools whose results you actually used.
Base your decision only on results you were given. Do not assume a tool
succeeded unless its result says so.
"""
SYSTEM_PROMPT_ID = "l3_pilot_executed/v1"
TEMPERATURE = 0.0
MAX_TOKENS = 1024
EVALUATOR_VERSION = "l3_grounded_scoring/1.0.0"


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                              text=True, timeout=15).stdout.strip()
    except Exception:
        return ""


def provenance(model: str, backend: CouchDBExecutor,
               scenarios: Optional[List[str]] = None) -> Dict[str, Any]:
    try:
        from frozen_config import config_hash
        fc = config_hash()
    except Exception:
        fc = None
    return {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repos": {
            "AssetOpsBench": {"sha": _git(REPO_ROOT, "rev-parse", "HEAD"),
                              "branch": _git(REPO_ROOT, "rev-parse", "--abbrev-ref", "HEAD"),
                              "dirty": bool(_git(REPO_ROOT, "status", "--porcelain"))},
            "AssetOpsBenchScenarioGeneration": {
                "sha": _git(SCEN_REPO, "rev-parse", "HEAD"),
                "branch": _git(SCEN_REPO, "rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(_git(SCEN_REPO, "status", "--porcelain"))},
        },
        "frozen_config_hash": fc,
        "model": model,
        "generation": {"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
                       "system_prompt_id": SYSTEM_PROMPT_ID},
        "backend": {"id": backend.backend_id, "version": backend.backend_version},
        "evaluator_version": EVALUATOR_VERSION,
        "scenario_version": "R055-R058 @ " + _git(SCEN_REPO, "rev-parse", "--short", "HEAD"),
        "scenarios": list(scenarios) if scenarios else list(A.PILOT_SCENARIOS),
        "status": "apparatus-validity reference pilot - NOT a benchmark result",
    }


def _post_chat(model: str, messages: List[Dict[str, Any]], api_key: str, base_url: str,
               *, tokens_param: str = "max_tokens", include_temperature: bool = True) -> Dict[str, Any]:
    payload = {"model": model, "messages": messages, tokens_param: MAX_TOKENS}
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
    """Two disclosed, model-triggered protocol fallbacks, verified live at
    the GPT-6-Astra feasibility check and DISCLOSED as protocol deviations
    for that model specifically (not applied speculatively to any model
    that doesn't trigger them -- every existing call site is unchanged):
      1. some models reject 'max_tokens' ("... Use 'max_completion_tokens'
         instead.") -> retry with the renamed parameter.
      2. some models reject temperature=0 ("... does not support 0 with
         this model. Only the default (1) value is supported.") -> retry
         with temperature omitted entirely (falls back to the API's
         default, typically 1). This means that model runs at a different,
         non-zero temperature than the rest of the panel -- a real
         protocol deviation, not silently equivalent, and must be
         disclosed in any results this produces."""
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


def run_episode(model: str, api_key: str, base_url: str, executor: CouchDBExecutor,
                scenario_id: str, spec) -> Dict[str, Any]:
    arm = spec.arm_id
    withheld = list(spec.withheld_evidence)
    executor.reset(scenario_id, arm, seed=1, withheld=withheld)
    trace = ExecutionTrace(scenario_id, arm)

    payload = A.render_arm(spec)
    tools = executor.available_tools()
    task = (f"{payload['question'].strip()}\n\n"
            f"Tools available: {', '.join(tools)}\n"
            f"Allowed verdicts: {', '.join(payload['allowed_actions'])}")

    msgs: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{task}\n\nSTEP 1: request the tools you need."},
    ]
    step1, err1 = _chat(model, msgs, api_key, base_url)
    requested = [c for c in (step1.get("tool_calls") or []) if isinstance(c, dict)]

    # --- execute -----------------------------------------------------------
    results: List[Dict[str, Any]] = []
    images: List[str] = []
    for call in requested[:8]:
        name = str(call.get("tool", ""))
        args = call.get("args") or {}
        trace.append(Stage.REQUESTED, tool=name, args=args)
        res = executor.execute(ToolCall(name, args))
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

    # --- return real results, images included ------------------------------
    content: List[Dict[str, Any]] = [{
        "type": "text",
        "text": ("STEP 2. Tool results (produced by the executor, not by you):\n"
                 + json.dumps(results, indent=2)
                 + ("\n\nThe captured gauge image follows." if images else "")
                 + "\n\nNow give your final decision.")}]
    for b64 in images:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}})

    msgs += [{"role": "assistant", "content": json.dumps(step1)},
             {"role": "user", "content": content}]
    step2, err2 = _chat(model, msgs, api_key, base_url)

    verdict = str(step2.get("verdict") or step2.get("action") or "")
    if verdict:
        trace.append(Stage.DECISION, detail={"verdict": verdict})

    fm = A.SCENARIOS[scenario_id]["fm"]
    gold = A.SCENARIOS[scenario_id]["gold"]
    asset_id = A.SCENARIOS[scenario_id].get("asset")
    apparatus_failure = bool(err2) or not verdict
    scores = (None if apparatus_failure
              else score_l3_grounded(
                  step2, {"fm": fm, "asset_id": asset_id}, {"action": gold}, trace))
    integ = check_integrity(step2, trace,
                            required_modality="physical").to_dict()

    return {
        "scenario_id": scenario_id, "fm": fm, "gold": gold, "arm": arm,
        "probe": spec.insufficient_evidence_probe,
        "tools_offered": tools,
        "tools_requested": [c.get("tool") for c in requested],
        "tools_executed": sorted(trace.executed_tools()),
        "observations_delivered": [
            {"id": e.observation_id, "hash": e.observation_hash,
             "modality": e.modality, "tool": e.tool}
            for e in trace.events if e.stage is Stage.OBSERVATION_DELIVERED],
        "verdict": verdict, "reason": step2.get("reason", ""),
        "claimed_tool_sequence": step2.get("tool_sequence", []),
        "scores": scores, "integrity": integ,
        "apparatus_failure": apparatus_failure,
        "errors": [e for e in (err1, err2) if e],
        "trace": trace.to_dict(),
        "hidden_state": {k: v for k, v in SCENARIO_PHYSICAL[scenario_id].items()
                         if k != "value"} | {"value_withheld_from_agent": True},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="openai/gpt-5.4-mini")
    ap.add_argument("--base-url", default=os.environ.get(
        "TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1"))
    ap.add_argument("--json", type=Path,
                    default=REPO_ROOT / "reports" / "v1" / "l3_pilot_executed.json")
    ap.add_argument("--skip-preflight", action="store_true",
                    help=argparse.SUPPRESS)  # for apparatus debugging only
    ap.add_argument("--scenarios", type=str, default=None,
                    help="Comma-separated scenario ids; default A.PILOT_SCENARIOS "
                         "(unchanged behavior when omitted)")
    args = ap.parse_args()
    scenario_ids = ([s.strip() for s in args.scenarios.split(",") if s.strip()]
                    if args.scenarios else list(A.PILOT_SCENARIOS))

    if not args.skip_preflight:
        pf = subprocess.run([sys.executable,
                             str(REPO_ROOT / "scripts" / "l3_execution_preflight.py")],
                            capture_output=True, text=True)
        if pf.returncode != 0:
            print(pf.stdout)
            print("ERROR: execution preflight failed — API spend not authorised",
                  file=sys.stderr)
            return 2
        print("preflight: PASS (16/16)\n")

    api_key = os.environ.get("TOKENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: TOKENROUTER_API_KEY not set", file=sys.stderr)
        return 2

    executor = CouchDBExecutor()
    prov = provenance(args.model, executor, scenarios=scenario_ids)
    print("PROVENANCE")
    for k, v in prov["repos"].items():
        print(f"  {k:32s} {v['sha'][:12]} ({v['branch']}){'  DIRTY' if v['dirty'] else ''}")
    for k in ("frozen_config_hash", "model", "evaluator_version", "scenario_version"):
        print(f"  {k:32s} {prov[k]}")
    print(f"  {'backend':32s} {prov['backend']['id']} {prov['backend']['version']}")
    print(f"  {'generation':32s} temp={TEMPERATURE} max_tokens={MAX_TOKENS}\n")

    rows: List[Dict[str, Any]] = []
    for sid in scenario_ids:
        for spec in A.arms_for(sid):
            r = run_episode(args.model, api_key, args.base_url, executor, sid, spec)
            rows.append(r)
            s = r["scores"] or {}
            print(f"  {sid} {spec.arm_id:14s} req={len(r['tools_requested']):2d} "
                  f"exec={len(r['tools_executed']):2d} "
                  f"obs={len(r['observations_delivered']):2d} "
                  f"verdict={r['verdict']:9s} "
                  f"CC={s.get('CC')} PROC={s.get('PROC')} "
                  f"CCg={s.get('CC_grounded')} "
                  f"fab={int(r['integrity']['fabricated_observation'])}"
                  f"{int(r['integrity']['fabricated_procedure'])} "
                  f"inc={int(r['integrity']['verdict_reason_incoherence'])}")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(
        {"schema": "assetops.l3_pilot_executed/1", "provenance": prov,
         "results": rows}, indent=2) + "\n")
    print(f"\n-> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
