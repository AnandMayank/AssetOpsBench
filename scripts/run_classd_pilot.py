"""run_classd_pilot.py — Class-D relational/enterprise scenarios (P2).

Structurally identical to ``run_classc_pilot.py`` (same fixture injection, same
two-step executor loop) but scores against gold parsed by ``classd_audit`` and
reports the parent/control contrast explicitly rather than ordering.

R026 is excluded here. Its gold is genuinely composite (``composite_verdict``
handles that, adopted and tested) but running it for real would require the
executor to hold *two* assets' enterprise state independently within one
episode -- technician_present True at chiller_6 and False at motor_01
simultaneously. That is a new execution mechanism, not a scenario or scoring
gap, and adding it is out of scope here.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from classc_fixtures import FIXTURES, FixtureSession               # noqa: E402
from classd_audit import audit                                     # noqa: E402
from couchdb_executor import CouchDBExecutor                        # noqa: E402
from execution_trace import ExecutionTrace, Stage                   # noqa: E402
from l3_integrity import check_integrity                            # noqa: E402
from l3_scoring import normalise_action                             # noqa: E402
from tool_executor import STATUS_SUCCESS, ToolCall                  # noqa: E402
from run_generated_pilot import SYSTEM_PROMPT, _chat                 # noqa: E402

SCEN_ROOT = REPO_ROOT.parent / "AssetOpsBenchScenarioGeneration" / "RobotInspection"

#: Parent/control pairs actually eligible under the current single-asset
#: executor. R026 is excluded (see module docstring).
CLASS_D_RUN = ["R008", "R059", "R010", "R060",
              "R021", "R061", "R022", "R062", "R025", "R063"]


def question(sid: str) -> str:
    n = int(sid.lstrip("R"))
    for c in (SCEN_ROOT / f"scenario_R{n:02d}", SCEN_ROOT / f"scenario_R{n}"):
        if (c / "question.txt").exists():
            return (c / "question.txt").read_text(errors="replace")
    raise FileNotFoundError(sid)


def run(model: str, api_key: str, base_url: str, ex: CouchDBExecutor,
       sid: str) -> Dict[str, Any]:
    a = audit(sid)
    fx = FIXTURES[sid]
    asset = fx.asset
    from couchdb_executor import SCENARIO_PHYSICAL
    SCENARIO_PHYSICAL.setdefault(sid, {"asset": asset, "value": 0.0, "unit": "",
                                       "range": [0, 100], "band": [0, 100],
                                       "source": "class-D default seeded state"})
    ex.reset(sid, "FULL", seed=1)
    ex.enterprise_override = dict(fx.enterprise)
    trace = ExecutionTrace(sid, "FULL")

    tools = ex.available_tools()
    task = (f"{question(sid).strip()}\n\nTools available: {', '.join(tools)}\n"
           f"Allowed verdicts: COMMIT, ESCALATE, ABORT")
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
           {"role": "user", "content": f"{task}\n\nSTEP 1: request the tools you need."}]
    step1, err1 = _chat(model, msgs, api_key, base_url)

    results, executed = [], []
    for call in (step1.get("tool_calls") or [])[:10]:
        name = str(call.get("tool", "")); args = call.get("args") or {}
        trace.append(Stage.REQUESTED, tool=name, args=args)
        res = ex.execute(ToolCall(name, args))
        if res.executed:
            trace.append(Stage.EXECUTED, tool=name, status=res.status)
            executed.append(name)
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

    apparatus_failure = bool(err2) or not verdict
    cc = None if apparatus_failure else int(normalise_action(verdict) == normalise_action(a.gold))
    return {"scenario_id": sid, "fm": a.fm, "gold": a.gold, "gold_source": a.gold_source,
           "verdict": verdict, "precondition": fx.why, "executed_tools": executed,
           "CC": cc, "claimed_tool_sequence": step2.get("tool_sequence", []),
           "reason": step2.get("reason", ""),
           "integrity": check_integrity(step2, trace).to_dict(),
           "apparatus_failure": apparatus_failure,
           "errors": [e for e in (err1, err2) if e],
           "trace": trace.to_dict()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-5.4-mini")
    ap.add_argument("--base-url", default=os.environ.get(
        "TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1"))
    ap.add_argument("--scenarios", default=",".join(CLASS_D_RUN))
    ap.add_argument("--json", type=Path,
                    default=REPO_ROOT / "reports" / "v1" / "classd_pilot.json")
    args = ap.parse_args()

    key = os.environ.get("TOKENROUTER_API_KEY", "")
    if not key:
        print("ERROR: TOKENROUTER_API_KEY not set", file=sys.stderr); return 2
    ex = CouchDBExecutor()
    rows = []
    for sid in [s.strip() for s in args.scenarios.split(",")]:
        fx = FIXTURES[sid]
        with FixtureSession(ex._robot.db, fx) as sess:
            problems = sess.verify_applied()
            if problems:
                print(f"  {sid}: FIXTURE NOT APPLIED -> {problems}"); continue
            r = run(args.model, key, args.base_url, ex, sid)
        rows.append(r)
        print(f"  {sid:6s} {r['fm']:7s} gold={r['gold']:9s} verdict={r['verdict']:9s} CC={r['CC']}")
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({"schema": "assetops.classd_pilot/1",
                                     "model": args.model, "excluded": ["R026"],
                                     "exclusion_reason": ("composite gold, tested; not "
                                                          "executable under the current "
                                                          "single-asset-per-episode executor"),
                                     "results": rows}, indent=2) + "\n")
    print(f"\n-> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
