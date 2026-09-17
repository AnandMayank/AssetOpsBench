#!/usr/bin/env python3
"""phase8h_live_pilot.py — Phase 8H controlled live-model pilot, 30 real
canonical worlds (within the requested 30-40 range), single model, real
API spend, run only after the golden trace suite and the four
metric-independence tests both pass (both verified before this script is
invoked).

Reuses EXISTING, already-validated runners rather than re-implementing
any execution logic:
  - scripts/phase8d_live_pilot.py:run_c1_episode  (A: world-first, per-arm)
  - scripts/run_l3_pilot_executed.py:run_episode  (B: R011, per-arm)
  - scripts/run_classc_pilot.py:run               (C: R006/R007, FULL only)
  - scripts/run_classd_pilot.py:run               (D: R008/R010, FULL only)
  - scripts/run_class_e_pilot.py:run_sequence     (E: fresh sequences)

Every episode is additionally scored under the FROZEN Phase 8H metric
contract (metric_contract.score_episode) alongside the existing
score_l3_grounded output already embedded in each runner's return value.

Model: openai/gpt-5.4-mini (this project's consistent live-eval default).
Single model only -- no cross-model comparison manufactured.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
EC = REPO_ROOT / "reports" / "ec"

MODEL = "openai/gpt-5.4-mini"
API_KEY = os.environ.get("TOKENROUTER_API_KEY", "")
BASE_URL = os.environ.get("TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1")

from couchdb_executor import CouchDBExecutor  # noqa: E402
from execution_trace import ExecutionTrace  # noqa: E402
from l3_arms import arms_for  # noqa: E402
from l3_integrity import check_integrity  # noqa: E402
from l3_grounded_scoring import REQUIRED_MODALITY  # noqa: E402
from metric_contract import score_episode  # noqa: E402
from scenario_gen import FACTORIAL, sample_world, sample_world_fm7a_contradiction, derive_gold, derive_gold_fm7a  # noqa: E402

_spec = importlib.util.spec_from_file_location("phase8d_live_pilot",
                                                REPO_ROOT / "scripts" / "phase8d_live_pilot.py")
# phase8d_live_pilot.py asserts its OWN frozen-input hash at import time and
# would abort this script too -- import run_c1_episode by exec'ing only the
# function body's dependencies instead: reuse its two-turn tool loop pattern
# via run_l3_pilot_executed, replicated here for the A-family world-first path.
import run_l3_pilot_executed as R  # noqa: E402
import run_classc_pilot as C  # noqa: E402
import run_classd_pilot as D  # noqa: E402
import run_class_e_pilot as E  # noqa: E402
from sequence_executor import sample_sequence, SequenceExecutor  # noqa: E402
from tool_executor import ToolCall, STATUS_SUCCESS  # noqa: E402
from execution_trace import Stage  # noqa: E402

CELL_NAMES = ("phys_in__iot_agree", "phys_in__iot_disagree", "phys_out__iot_agree", "phys_out__iot_disagree")
ASSETS = ("chiller_6", "hydraulic_pump_1", "metro_pump_1", "motor_01")
REGIMES = ("FULL", "PHYSICAL_ONLY", "DIGITAL_ONLY")
REGIME_WITHHELD = {"FULL": [], "PHYSICAL_ONLY": ["digital"], "DIGITAL_ONLY": ["physical"]}


def run_a_episode(world, arm: str, *, model: str | None = None,
                  api_key: str | None = None, base_url: str | None = None) -> Dict[str, Any]:
    """A-family world-first episode -- same two-turn tool-request/execute/
    decide loop as run_l3_pilot_executed.run_episode, adapted for a
    world sampled via sample_world (no l3_arms registration exists for
    these worlds, exactly the gap phase8d_live_pilot.py's run_c1_episode
    already solved -- replicated here verbatim rather than re-derived).

    Phase 8H.1 multi-model pilot: model/api_key/base_url are keyword-only and
    default to this module's globals (MODEL/API_KEY/BASE_URL). Passing them
    explicitly is the ONLY behavioural difference -- every existing call site
    that omits them is byte-for-byte unchanged (regression-tested in
    src/orchestrator/tests/test_run_a_episode_equivalence.py)."""
    _model = MODEL if model is None else model
    _api_key = API_KEY if api_key is None else api_key
    _base_url = BASE_URL if base_url is None else base_url
    ex = CouchDBExecutor()
    ex.reset_from_world(world, arm, seed=1, withheld=REGIME_WITHHELD[arm])
    trace = ExecutionTrace(world.scenario_id, arm)
    tools = ex.available_tools()
    from scenario_gen import render_question
    task = (f"{render_question(world).strip()}\n\nTools available: {', '.join(tools)}\n"
           f"Allowed verdicts: COMMIT, ESCALATE, ABORT")
    msgs = [{"role": "system", "content": R.SYSTEM_PROMPT},
           {"role": "user", "content": f"{task}\n\nSTEP 1: request the tools you need."}]
    step1, err1 = R._chat(_model, msgs, _api_key, _base_url)
    requested = [c for c in (step1.get("tool_calls") or []) if isinstance(c, dict)]

    results, images = [], []
    for call in requested[:8]:
        name = str(call.get("tool", "")); args = call.get("args") or {}
        trace.append(Stage.REQUESTED, tool=name, args=args)
        res = ex.execute(ToolCall(name, args))
        if res.executed:
            trace.append(Stage.EXECUTED, tool=name, status=res.status, error=res.error)
        if res.status == STATUS_SUCCESS:
            trace.append(Stage.SUCCEEDED, tool=name)
        if res.delivered:
            trace.append(Stage.OBSERVATION_DELIVERED, tool=name, observation_id=res.observation_id,
                         observation_hash=res.observation_hash, modality=res.modality,
                         asset_id=res.asset_id)
            if res.image_b64:
                images.append(res.image_b64)
        results.append(res.to_dict())

    content = [{"type": "text", "text": ("STEP 2. Tool results (produced by the executor, not "
               "by you):\n" + json.dumps(results, indent=2) +
               ("\n\nThe captured gauge image follows." if images else "") +
               "\n\nNow give your final decision.")}]
    for b64 in images:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    msgs += [{"role": "assistant", "content": json.dumps(step1)}, {"role": "user", "content": content}]
    step2, err2 = R._chat(_model, msgs, _api_key, _base_url)

    verdict = step2.get("verdict", "")
    reason = step2.get("reason", "")
    tool_sequence = step2.get("tool_sequence") or []
    trace.append(Stage.DECISION, detail={"verdict": verdict, "reason": reason})

    is_contradiction = hasattr(world, "contradiction")
    fm = "FM-7a" if is_contradiction else "FM-6a"
    gold = derive_gold_fm7a(world) if is_contradiction else derive_gold(world)
    resp = {"verdict": verdict, "reason": reason, "tool_sequence": tool_sequence}
    m = score_episode(resp, {"fm": fm, "asset_id": world.asset}, {"action": gold.verdict},
                      trace, f"{world.scenario_id}::{arm}")
    integrity = check_integrity(resp, trace, REQUIRED_MODALITY.get(fm, "physical"))
    return {"world_id": world.scenario_id, "arm": arm, "gold": gold.verdict,
           "verdict": verdict, "metric": m, "integrity_flags": integrity.any_flag,
           "call_errors": [e for e in (err1, err2) if e]}


def main() -> int:
    if not API_KEY:
        print("ERROR: TOKENROUTER_API_KEY not set -- aborting, no fabricated results", file=sys.stderr)
        return 2

    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    world_count = 0

    # --- A: 16 real worlds x 3 regimes = 48 episodes ---
    for i in range(16):
        seed = 3000 + i
        cell_name = CELL_NAMES[i % 4]
        asset = ASSETS[i % 4]
        cell = next(c for c in FACTORIAL if c.name == cell_name)
        label = f"A-{asset}-{cell_name}-{seed}"
        world = sample_world(seed, cell, asset_id=asset, scenario_id=label)
        world_count += 1
        for arm in REGIMES:
            r = run_a_episode(world, arm)
            rows.append({"dim": "A", **r})
            print(f"  A/{label}/{arm:14s} verdict={r['verdict']:9s} gold={r['gold']:9s} "
                 f"TDA={r['metric'].TDA} GSR={r['metric'].GSR}")

    # --- B: R011 (real, l3_arms-registered) + 3 world-first FM-7a worlds x 3 regimes ---
    # run_episode already runs score_l3_grounded internally and returns its
    # full output in r["scores"] -- that IS the frozen GSR conjunct set
    # (cc_grounded), so reuse it directly rather than re-deriving from a
    # reconstructed trace (r["trace"] is trace.to_dict(), a plain dict, not
    # a live ExecutionTrace -- attempting to re-score from it would have
    # been a real bug, caught before any spend).
    ex_b = CouchDBExecutor()
    for spec in arms_for("R011"):
        r = R.run_episode(MODEL, API_KEY, BASE_URL, ex_b, "R011", spec)
        sc = r["scores"]
        rows.append({"dim": "B", "world_id": "R011", "arm": spec.arm_id, "gold": r["gold"],
                    "verdict": r["verdict"], "metric": None,
                    "frozen_TDA": sc.get("CC"), "frozen_GSR": sc.get("CC_grounded"),
                    "frozen_ERA": sc.get("observation_delivered"),
                    "frozen_EAA": sc.get("asset_match"), "frozen_CCR": sc.get("no_forbidden_action"),
                    "frozen_delta": (sc.get("CC") - sc.get("CC_grounded"))
                                    if sc.get("CC") is not None and sc.get("CC_grounded") is not None else None,
                    "integrity_flags": r.get("integrity", {}).get("any_flag", None),
                    "call_errors": r.get("errors", [])})
        print(f"  B/R011/{spec.arm_id:14s} verdict={r['verdict']:9s} gold={r['gold']:9s} "
             f"CC={sc.get('CC')} CCg={sc.get('CC_grounded')}")
    world_count += 1
    for j in range(3):
        seed = 4000 + j
        physical_in_band = (j % 2 == 0)
        asset = ASSETS[j % 4]
        label = f"B-fm7a-{asset}-{seed}"
        world = sample_world_fm7a_contradiction(seed, physical_in_band, asset_id=asset, scenario_id=label)
        world.contradiction = True   # tag so run_a_episode picks derive_gold_fm7a
        world_count += 1
        for arm in REGIMES:
            r = run_a_episode(world, arm)
            rows.append({"dim": "B", **r})
            print(f"  B/{label}/{arm:14s} verdict={r['verdict']:9s} gold={r['gold']:9s} "
                 f"TDA={r['metric'].TDA} GSR={r['metric'].GSR}")

    # --- C: R006, R007 (real, FULL only) ---
    # Phase 8H.1 persistence fix (CCR audit): retain the complete runner
    # result via **r -- previously an explicit whitelist here discarded fm/
    # executed_order/trace, which is exactly what a post-hoc CCR computation
    # over C/D needs (the same class of loss the E field-capture fix already
    # corrected below for E). raw_CC is kept explicit because the runner
    # returns "CC", not "raw_CC"; this does not add "metric"/"frozen_TDA"
    # keys, so paired_metrics.load_canonical_rows' structural C/D branch
    # selection, GSR_defined=False, and paired-Delta eligibility (78/93) are
    # all unaffected. No CCR is computed here.
    ex_c = CouchDBExecutor()
    for sid in ("R006", "R007"):
        world_count += 1
        r = C.run(MODEL, API_KEY, BASE_URL, ex_c, sid)
        rows.append({"dim": "C", "world_id": sid, "arm": "FULL", "metric": None,
                    "raw_CC": r.get("CC"), "integrity_flags": None,
                    "call_errors": r.get("errors", []), **r})
        print(f"  C/{sid}/FULL          verdict={r['verdict']:9s} gold={r['gold']:9s} CC={r['CC']}")

    # --- D: R008, R010 (real, FULL only) ---
    # Same persistence fix as C above -- retains fm/executed_tools/trace.
    ex_d = CouchDBExecutor()
    for sid in ("R008", "R010"):
        world_count += 1
        r = D.run(MODEL, API_KEY, BASE_URL, ex_d, sid)
        rows.append({"dim": "D", "world_id": sid, "arm": "FULL", "metric": None,
                    "raw_CC": r.get("CC"), "integrity_flags": None,
                    "call_errors": r.get("errors", []), **r})
        print(f"  D/{sid}/FULL          verdict={r['verdict']:9s} gold={r['gold']:9s} CC={r['CC']}")

    # --- E: 6 real fresh sequences x 3 episodes ---
    for k in range(6):
        seed = 5000 + k
        world_count += 1
        se = SequenceExecutor(CouchDBExecutor())
        seq_world = sample_sequence(seed, n_episodes=3)
        r = E.run_sequence(MODEL, API_KEY, BASE_URL, se, seq_world, seed)
        for ep in r["episodes"]:
            # Phase 8H.1 Stage 4/E repair: run_sequence() already computes
            # stale_state_reuse/battery_reads_used/battery_budget/
            # reacquired_this_episode/apparatus_failure -- the original 8H
            # row-builder discarded all of them before writing the raw JSON,
            # which is why the 8 GSR=0 E rows could not be classified
            # mechanical-vs-behavioral in Stage 2 (REQUIRES_ADDITIONAL_TRACE_DATA).
            # This is a pure capture fix -- no scoring rule changes.
            rows.append({"dim": "E", "world_id": f"SEQ-{seed}", "arm": f"ep{ep['episode']}",
                        "gold": ep["gold"], "verdict": ep["verdict"], "metric": None,
                        "raw_CC": ep.get("CC"), "raw_CCg": ep.get("CC_grounded"),
                        "integrity_flags": None, "call_errors": ep.get("errors", []),
                        "stale_state_reuse": ep.get("stale_state_reuse"),
                        "reacquired_this_episode": ep.get("reacquired_this_episode"),
                        "reobservation_was_necessary": ep.get("reobservation_was_necessary"),
                        "unnecessary_reobservation": ep.get("unnecessary_reobservation"),
                        "battery_reads_used": ep.get("battery_reads_used"),
                        "battery_budget": ep.get("battery_budget"),
                        "apparatus_failure": ep.get("apparatus_failure")})
            print(f"  E/SEQ-{seed}/ep{ep['episode']}          verdict={ep['verdict']:9s} "
                 f"gold={ep['gold']:9s} CC={ep.get('CC')} CCg={ep.get('CC_grounded')}")

    out_path = EC / "phase8h_pilot_raw_results.json"

    def _serialize(r):
        d = dict(r)
        if isinstance(d.get("metric"), object) and hasattr(d.get("metric"), "to_dict"):
            d["metric"] = d["metric"].to_dict()
        return d

    out_path.write_text(json.dumps({"model": MODEL, "world_count": world_count,
                                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                    "results": [_serialize(r) for r in rows]},
                                   indent=2, default=str))
    print(f"\n-> {out_path}  ({world_count} worlds, {len(rows)} scored episodes, "
         f"{round(time.time()-t0,1)}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
