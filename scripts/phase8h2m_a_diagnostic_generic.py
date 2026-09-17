#!/usr/bin/env python3
"""phase8h2m_a_diagnostic_generic.py -- model-parameterized version of
phase8h2k_claude_a_diagnostic.py, for diagnosing empty-verdict A rows for
ANY of the 5 primary-panel models (not just Claude). Same additive-only
guarantees: does not touch the frozen-93 manifest or any existing raw/
summary file, writes only to reports/benchmark/claude_a_diag/ (shared
directory, filenames are timestamped so nothing collides).

Usage:
  TOKENROUTER_API_KEY=... uv run python3 scripts/phase8h2m_a_diagnostic_generic.py \
      --label DeepSeek_V4_Pro --bare-model deepseek/deepseek-v4-pro --confirm
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pilot_dispatch as PD  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "reports" / "ec" / "phase8h1_pilot_manifest.json"
FROZEN93_DIR = REPO_ROOT / "reports" / "benchmark" / "v3_full_results" / "frozen93"
OUT_DIR = REPO_ROOT / "reports" / "benchmark" / "claude_a_diag"
TEMPERATURE = 0.0
MAX_TOKENS = 2048


def find_affected_units(raw_path: Path) -> List[str]:
    affected = []
    for line in raw_path.read_text().splitlines():
        rec = json.loads(line)
        if rec["dim"] != "A":
            continue
        rr = rec["runner_return"]
        if rr.get("verdict", "") == "":
            affected.append(rec["episode_id"])
    return affected


def _chat_instrumented(model: str, messages: List[Dict[str, Any]], api_key: str,
                       base_url: str) -> Tuple[Dict[str, Any], Optional[str], Dict[str, Any]]:
    body = json.dumps({"model": model, "messages": messages,
                       "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS}).encode()
    req = urllib.request.Request(f"{base_url}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {api_key}"})
    capture: Dict[str, Any] = {"model_requested": model}
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            payload = json.loads(r.read())
    except Exception as exc:  # noqa: BLE001
        capture["api_error"] = f"{type(exc).__name__}: {exc}"
        return {}, f"call_error: {exc}", capture

    choice = (payload.get("choices") or [{}])[0]
    text = (choice.get("message", {}) or {}).get("content") or ""
    capture["served_model"] = payload.get("model")
    capture["finish_reason"] = choice.get("finish_reason")
    capture["usage"] = payload.get("usage")
    capture["raw_content"] = text
    capture["raw_content_len"] = len(text)

    if not text.strip():
        capture["parse_error"] = "no_answer: empty content"
        return {}, "no_answer: empty content", capture

    clean = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    m = re.search(r"\{.*\}", clean, re.DOTALL)
    if not m:
        capture["parse_error"] = "parse_error: no brace match"
        return {}, f"parse_error: {text[:120]}", capture

    capture["greedy_matched_span"] = m.group(0)
    try:
        parsed = json.loads(m.group(0))
        capture["parsed_json"] = parsed
        capture["parsed_keys"] = list(parsed.keys()) if isinstance(parsed, dict) else None
        return parsed, None, capture
    except json.JSONDecodeError as exc:
        capture["parse_error"] = f"parse_error: {exc}"
        return {}, f"parse_error: {exc}", capture


def run_a_episode_diagnostic(world, arm: str, gold_action: str, *, model: str,
                             api_key: str, base_url: str) -> Dict[str, Any]:
    import run_l3_pilot_executed as R
    from couchdb_executor import CouchDBExecutor
    from execution_trace import ExecutionTrace
    from scenario_gen import render_question
    from tool_executor import ToolCall

    REGIME_WITHHELD = {"FULL": [], "PHYSICAL_ONLY": ["digital"], "DIGITAL_ONLY": ["physical"]}

    ex = CouchDBExecutor()
    ex.reset_from_world(world, arm, seed=1, withheld=REGIME_WITHHELD[arm])
    trace = ExecutionTrace(world.scenario_id, arm)
    tools = ex.available_tools()
    task = (f"{render_question(world).strip()}\n\nTools available: {', '.join(tools)}\n"
           f"Allowed verdicts: COMMIT, ESCALATE, ABORT")
    msgs = [{"role": "system", "content": R.SYSTEM_PROMPT},
           {"role": "user", "content": f"{task}\n\nSTEP 1: request the tools you need."}]
    step1, err1, cap1 = _chat_instrumented(model, msgs, api_key, base_url)
    requested = [c for c in (step1.get("tool_calls") or []) if isinstance(c, dict)]

    results, images = [], []
    for call in requested[:8]:
        name = str(call.get("tool", "")); args = call.get("args") or {}
        res = ex.execute(ToolCall(name, args))
        if res.delivered and res.image_b64:
            images.append(res.image_b64)
        results.append(res.to_dict())

    content = [{"type": "text", "text": ("STEP 2. Tool results (produced by the executor, not "
               "by you):\n" + json.dumps(results, indent=2) +
               ("\n\nThe captured gauge image follows." if images else "") +
               "\n\nNow give your final decision.")}]
    for b64 in images:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    msgs += [{"role": "assistant", "content": json.dumps(step1)}, {"role": "user", "content": content}]
    step2, err2, cap2 = _chat_instrumented(model, msgs, api_key, base_url)

    verdict = step2.get("verdict", "")
    return {
        "world_id": world.scenario_id, "arm": arm, "gold": gold_action,
        "verdict": verdict, "verdict_present": verdict != "",
        "step1_capture": cap1, "step2_capture": cap2,
        "call_errors": [e for e in (err1, err2) if e],
        "n_images_delivered": len(images), "n_tool_calls_requested": len(requested),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="frozen93 file label, e.g. DeepSeek_V4_Pro")
    ap.add_argument("--bare-model", required=True, help="bare model id for the raw urllib _chat path")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    raw_path = FROZEN93_DIR / f"raw_{args.label}.jsonl"
    affected = find_affected_units(raw_path)
    print(f"Affected {args.label} A units (empty verdict): {len(affected)}/48")
    for eid in affected:
        print(f"  {eid}")

    if args.dry_run or not args.confirm:
        print("\n--dry-run (or --confirm not passed): no API calls made.")
        return 0

    api_key = os.environ.get("TOKENROUTER_API_KEY", "")
    base_url = os.environ.get("TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1")
    if not api_key:
        print("ERROR: TOKENROUTER_API_KEY not set", file=sys.stderr)
        return 1

    manifest = PD.load_manifest(MANIFEST_PATH)
    by_episode_id = {r["episode_id"]: r for r in manifest["episodes"]}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"raw_text_capture_{args.label}_{int(time.time())}.jsonl"
    todo = affected[: args.limit] if args.limit else affected
    print(f"\nRunning {len(todo)} diagnostic units against {args.bare_model}, writing to {out_path}")

    with open(out_path, "a") as fh:
        for i, eid in enumerate(todo, 1):
            row = by_episode_id[eid]
            world = PD.rebuild_world(row)
            print(f"[{i}/{len(todo)}] {eid} (arm={row['regime']}) ...", flush=True)
            try:
                result = run_a_episode_diagnostic(
                    world, row["regime"], row["gold_action"],
                    model=args.bare_model, api_key=api_key, base_url=base_url)
            except Exception as exc:  # noqa: BLE001
                result = {"episode_id": eid, "diagnostic_error": f"{type(exc).__name__}: {exc}"}
            result["episode_id"] = eid
            result["world_id"] = row["world_id"]
            result["scenario_id"] = row["scenario_id"]
            fh.write(json.dumps(result, default=str) + "\n")
            fh.flush()
            print(f"    verdict_present={result.get('verdict_present')} "
                 f"finish_reason(step2)={(result.get('step2_capture') or {}).get('finish_reason')}")

    print(f"\nDone. Capture: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
