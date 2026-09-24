#!/usr/bin/env python3
"""pg_b2_run.py — Phase 3: model-facing runner for the B2 pool. Reuses the
existing PRIMARY_PANEL (eval_model_panel.py, the SAME 5 models the frozen
B pilot used) and OpenAICompatBackend (src/llm/openai_compat.py) -- no new
model API is invented.

One turn == one model call == one action. The model NEVER receives world
truth, target verdict, family, matched-group, oracle/frontier information,
or any evaluator-only identifier -- only the task contract, the action
list with costs, remaining budget, and evidence actually delivered so far
(rendered through the SAME substrate.render_reading whitelist env.py's
live path already enforces). Gold is written to a SEPARATE file from the
agent-visible trajectory.

Usage:
    python scripts/pg_b2_run.py --models claude-sonnet-4.6,gpt-5.2 --dry-run
    python scripts/pg_b2_run.py --all-models          # real API calls, costs money
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import eval_model_panel as EMP  # noqa: E402
from llm.openai_compat import OpenAICompatBackend  # noqa: E402

from playground.b2_families import build_pool  # noqa: E402
from playground.env import B2Env  # noqa: E402
from playground.semantics import TERMINAL_ACTIONS  # noqa: E402

OUT_DIR = REPO_ROOT / "reports" / "playground" / "b2"
MAX_HORIZON = 20  # hard turn cap; distinct from the offline PRIMARY_HORIZON used for gold/frontier search

ACTIONS_MENU = (
    "ACQUIRE_IOT", "ACQUIRE_ACOUSTIC", "ACQUIRE_THERMAL", "QUERY_RECORD", "DISPATCH",
    "COMMIT_NORMAL", "COMMIT_FAULT", "ESCALATE",
)

SYSTEM_TASK_CONTRACT = """You are an industrial inspection agent deciding whether an asset's \
condition is NORMAL or requires escalation for a FAULT. You must acquire evidence before \
committing to a verdict, but every action has a cost and your goal is to reach a correct, \
evidence-supported decision as cheaply as possible.

Actions available this episode (unavailable ones will say so if you try them):
{action_list}

Costs (charged when the action succeeds):
{cost_list}

Rules:
- A physical capture (ACQUIRE_ACOUSTIC / ACQUIRE_THERMAL) requires DISPATCH first.
- ACQUIRE_IOT and QUERY_RECORD never require DISPATCH.
- COMMIT_NORMAL / COMMIT_FAULT / ESCALATE end the episode immediately.
- Only commit when you believe the evidence you have actually received supports that verdict.
- If a modality keeps returning "unavailable", stop requesting it and use ESCALATE instead \
of retrying indefinitely.
{budget_line}

Respond with ONLY a JSON object: {{"action": "<ONE_ACTION_FROM_THE_LIST_ABOVE>"}}
No other text."""


def _render_state(env: B2Env, last_observation) -> str:
    lines = ["Evidence delivered so far:"]
    for r in env.delivered.acoustic:
        lines.append(f"  - acoustic: indicator={'positive' if r.positive else 'negative'}")
    for r in env.delivered.iot:
        lines.append(f"  - iot: in_band={r.in_band}")
    for r in env.delivered.thermal:
        lines.append(f"  - thermal: hotspot={r.hotspot_present}")
    if env.delivered.record_queried:
        lines.append("  - enterprise record queried")
    if not (env.delivered.acoustic or env.delivered.iot or env.delivered.thermal or env.delivered.record_queried):
        lines.append("  (none yet)")
    lines.append(f"Dispatched: {env.dispatched}")
    if last_observation is not None:
        lines.append(f"Last action result: status={last_observation.status} payload={last_observation.payload}")
    if env.spec.regime.budget is not None:
        lines.append(f"Remaining budget: {env._remaining_budget()}")
    return "\n".join(lines)


def _system_prompt(env: B2Env) -> str:
    available = sorted(env.spec.regime.available_modalities)
    action_names = []
    if "iot" in available:
        action_names += ["ACQUIRE_IOT"]
    if "acoustic" in available:
        action_names += ["ACQUIRE_ACOUSTIC"]
    if "thermal" in available:
        action_names += ["ACQUIRE_THERMAL"]
    if "record" in available:
        action_names += ["QUERY_RECORD"]
    if "acoustic" in available or "thermal" in available:
        action_names += ["DISPATCH"]
    action_names += list(TERMINAL_ACTIONS)
    costs = env.spec.regime.costs
    cost_list = "\n".join([
        f"  - ACQUIRE_IOT: {costs.iot_read}", f"  - ACQUIRE_ACOUSTIC: {costs.acoustic_capture}",
        f"  - ACQUIRE_THERMAL: {costs.thermal_capture}", f"  - QUERY_RECORD: {costs.query_record}",
        f"  - DISPATCH: {costs.dispatch}", "  - COMMIT_NORMAL / COMMIT_FAULT / ESCALATE: 0 (ends episode)",
    ])
    budget_line = (f"- You have a hard budget of {env.spec.regime.budget}; actions that would exceed it "
                    "will be blocked." if env.spec.regime.budget is not None else "")
    return SYSTEM_TASK_CONTRACT.format(
        action_list="\n".join(f"  - {a}" for a in action_names), cost_list=cost_list, budget_line=budget_line,
    )


def _parse_action(raw_text: Optional[str]) -> Optional[str]:
    if not raw_text:
        return None
    match = re.search(r'\{[^{}]*"action"\s*:\s*"([A-Z_]+)"[^{}]*\}', raw_text)
    if match:
        candidate = match.group(1)
        if candidate in ACTIONS_MENU:
            return candidate
    for action in ACTIONS_MENU:
        if action in raw_text:
            return action
    return None


def run_one_episode(backend: OpenAICompatBackend, spec, *, temperature: float = 0.0,
                     max_horizon: int = MAX_HORIZON) -> Dict[str, Any]:
    env = B2Env()
    obs = env.reset(spec)
    turns: List[Dict[str, Any]] = []
    last_observation = obs
    final_action = None
    for step in range(max_horizon):
        system_prompt = _system_prompt(env)
        state_desc = _render_state(env, last_observation)
        prompt = f"{system_prompt}\n\n{state_desc}\n\nWhat is your next action?"
        try:
            result = backend.generate_with_usage(prompt, temperature=temperature)
            raw_text = result.text
            served_model = result.served_model
        except Exception as exc:  # network/provider error -- record, do not crash the whole run
            turns.append({"turn": step, "error": str(exc)})
            break
        action = _parse_action(raw_text)
        turns.append({"turn": step, "prompt": prompt, "raw_response": raw_text,
                       "parsed_action": action, "served_model": served_model})
        if action is None:
            action = "ESCALATE"  # malformed response -- fail safe, never silently retry forever
        step_result = env.step(action)
        last_observation = step_result.observation
        if step_result.terminated:
            final_action = action
            break
    else:
        step_result = env.step("ESCALATE")
        final_action = "ESCALATE"
        turns.append({"turn": max_horizon, "forced_escalate_horizon_exceeded": True})

    return {
        "episode_id": spec.episode_id, "final_action": final_action, "n_turns": len(turns),
        "turns": turns, "trace": env.trace.to_dict(), "total_cost": env.spent,
        # Minimal replay-free state for pg_b2_score_pilot.py -- avoids needing
        # to reconstruct a live B2Env from the trace to compute EGR/TGS/SD/etc.
        "delivered_summary": {
            "acoustic": [{"positive": r.positive} for r in env.delivered.acoustic],
            "iot": [{"in_band": r.in_band} for r in env.delivered.iot],
            "thermal": [{"hotspot_present": r.hotspot_present} for r in env.delivered.thermal],
            "record_queried": env.delivered.record_queried,
        },
        "dispatched": env.dispatched,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default=None,
                         help="comma-separated display_name substrings; default = all PRIMARY_PANEL models")
    parser.add_argument("--dry-run", action="store_true", help="build prompts/parse logic but never call an API")
    parser.add_argument("--all-models", action="store_true",
                         help="explicit alias for the default (all PRIMARY_PANEL models) -- accepted for clarity")
    parser.add_argument("--max-episodes", type=int, default=None, help="limit episodes per model (debugging)")
    args = parser.parse_args()

    pool = build_pool()
    if args.max_episodes:
        pool = pool[: args.max_episodes]

    models = list(EMP.PRIMARY_PANEL)
    if args.models:
        wanted = [m.strip() for m in args.models.split(",")]
        models = [m for m in models if any(w in m.display_name for w in wanted)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gold_by_id = {spec.episode_id: asdict(gold) for spec, gold in pool}
    (OUT_DIR / "model_pilot_gold_evaluator_only.json").write_text(json.dumps(gold_by_id, indent=2, default=str))

    out_path = OUT_DIR / "model_pilot_trajectories.jsonl"
    # RESUMABLE: if this run is interrupted (session teardown, network
    # error, etc.), a re-run skips (model, episode_id) pairs already
    # present in out_path instead of losing completed work or duplicating
    # spend. Each row is flushed to disk immediately after it completes,
    # not batched to the end -- so a crash loses at most one in-flight call.
    already_done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            already_done.add((row.get("model"), row.get("episode_id")))
        print(f"Resuming: {len(already_done)} (model, episode) pairs already completed in {out_path}")

    n_written = len(already_done)
    with out_path.open("a") as f:
        for model_cfg in models:
            print(f"=== {model_cfg.display_name} ({model_cfg.tokenrouter_id}) ===", flush=True)
            if args.dry_run:
                print("  --dry-run: skipping real API calls", flush=True)
                continue
            backend = OpenAICompatBackend(model_cfg.tokenrouter_id)
            for spec, gold in pool:
                if (model_cfg.display_name, spec.episode_id) in already_done:
                    continue
                t0 = time.time()
                result = run_one_episode(backend, spec)
                result["model"] = model_cfg.display_name
                result["tokenrouter_id"] = model_cfg.tokenrouter_id
                result["elapsed_s"] = time.time() - t0
                f.write(json.dumps(result, default=str) + "\n")
                f.flush()
                n_written += 1
                print(f"  {spec.episode_id[-12:]}: final={result['final_action']} "
                      f"turns={result['n_turns']} cost={result['total_cost']:.1f}", flush=True)

    print(f"\n{out_path} now has {n_written} trajectories total")
    print(f"Wrote {OUT_DIR / 'model_pilot_gold_evaluator_only.json'} (evaluator-only, kept separate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
