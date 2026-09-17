"""l3_execution_preflight.py — Blocks API spend until the apparatus is valid (P0-8).

Runs the 15 acceptance properties as live checks — not as skippable unit tests —
plus the pre-existing evidence-dependency audit. Exits non-zero if any fails, so
the pilot runner can refuse to start.

The governing condition, restated so it is visible at the point of enforcement:

    A model must not be able to receive credit for a physical observation that
    was never actually delivered to it.

Unlike the pytest suite, CouchDB being down is a **failure** here rather than a
skip: an apparatus that cannot execute tools cannot be validated.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

from execution_trace import ExecutionTrace, Stage  # noqa: E402
from l3_grounded_scoring import score_l3_grounded  # noqa: E402
from l3_integrity import check_integrity  # noqa: E402
from tool_executor import STATUS_SUCCESS, STATUS_UNAVAILABLE, ToolCall  # noqa: E402

PILOT = ("R009", "R015", "R055", "R056", "R057", "R058")


@dataclass
class Check:
    id: str
    name: str
    passed: bool
    detail: str = ""


def run_checks() -> List[Check]:
    out: List[Check] = []

    def add(cid, name, cond, detail=""):
        out.append(Check(cid, name, bool(cond), detail))
        return bool(cond)

    # --- backend availability is a failure, not a skip --------------------
    try:
        from couchdb_executor import CouchDBExecutor
        ex = CouchDBExecutor()
        live = getattr(ex._robot, "db", None) is not None
    except Exception as exc:  # noqa: BLE001
        add("P0-0", "execution backend available", False, f"{type(exc).__name__}: {exc}")
        return out
    if not add("P0-0", "execution backend available", live,
               "CouchDB reachable" if live else "CouchDB unavailable"):
        return out

    ex.reset("R058", "FULL", seed=1)
    img = ex.execute(ToolCall("capture_image"))
    add("1", "capture_image really executes",
        img.executed and img.status == STATUS_SUCCESS, f"status={img.status}")

    png = base64.b64decode(img.image_b64) if img.image_b64 else b""
    add("2", "capture_image delivers consumable pixels (not a path)",
        png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 2000,
        f"{len(png)} bytes, id={img.observation_id}")

    ex.reset("R056", "FULL", seed=1)
    iot = ex.execute(ToolCall("read_iot"))
    add("3", "read_iot returns actual scenario state",
        iot.status == STATUS_SUCCESS and iot.payload.get("value") == 1.02,
        f"value={iot.payload.get('value')}")

    tr = ExecutionTrace("R058", "FULL")
    tr.append(Stage.REQUESTED, tool="capture_image")
    add("4", "a failed/unexecuted tool is not counted as executed",
        tr.executed_tools() == set() and tr.requested_but_not_executed() == {"capture_image"})

    claim = {"verdict": "COMMIT", "reason": "read it",
             "tool_sequence": ["capture_image", "read_gauge"]}
    s = score_l3_grounded(claim, {"fm": "FM-6a"}, {"action": "COMMIT"},
                          ExecutionTrace("R009", "PHYSICAL_ONLY"))
    add("5", "PROC cannot be earned by self-report", s["PROC"] == 0,
        f"claimed={s['claimed_tools']} executed={s['executed_tools']} PROC={s['PROC']}")
    add("6", "CC_grounded withheld when no observation was delivered",
        s["CC"] == 1 and s["CC_grounded"] == 0,
        f"CC={s['CC']} (preregistered, unchanged) CC_grounded={s['CC_grounded']}")

    fab = check_integrity(
        {"verdict": "COMMIT", "tool_sequence": [],
         "reason": "The physical gauge image shows the flow within the expected band."},
        ExecutionTrace("R056", "PHYSICAL_ONLY"))
    add("7", "fabricated readings detected", fab.fabricated_observation)

    fp = check_integrity({"verdict": "COMMIT", "reason": "ok",
                          "tool_sequence": ["capture_image"]},
                         ExecutionTrace("R009", "PHYSICAL_ONLY"))
    add("8", "fabricated tool execution detected", fp.fabricated_procedure,
        f"claimed-not-executed={fp.claimed_tools_not_executed}")

    inc = check_integrity(
        {"verdict": "COMMIT", "tool_sequence": [],
         "reason": "I cannot verify the gauge, so this should be escalated."},
        ExecutionTrace("R009", "PHYSICAL_ONLY"))
    add("9", "verdict/reason incoherence detected", inc.verdict_reason_incoherence)

    ex.reset("R058", "PHYSICAL_ONLY", seed=1, withheld=["digital"])
    r = ex.execute(ToolCall("read_iot"))
    add("10", "PHYSICAL_ONLY genuinely lacks IoT",
        "read_iot" not in ex.available_tools() and r.status == STATUS_UNAVAILABLE)

    ex.reset("R058", "DIGITAL_ONLY", seed=1, withheld=["physical"])
    r = ex.execute(ToolCall("capture_image"))
    add("11", "DIGITAL_ONLY genuinely lacks the physical image",
        "capture_image" not in ex.available_tools()
        and r.status == STATUS_UNAVAILABLE and r.image_b64 is None)

    ex.reset("R056", "PHYSICAL_ONLY", seed=1, withheld=["digital"])
    leaks = []
    for t in ex.available_tools():
        rr = ex.execute(ToolCall(t, {"attempt_n": 1} if t == "read_gauge" else {}))
        if "1.02" in str(rr.payload):
            leaks.append(t)
    add("12", "withheld value cannot leak through another tool",
        not leaks, f"leaking tools: {leaks}" if leaks else "no leak")

    det = []
    for sid in PILOT:
        ex.reset(sid, "FULL", seed=7)
        a = ex.execute(ToolCall("read_gauge", {"attempt_n": 1})).payload.get("reading")
        ex.reset(sid, "FULL", seed=7)
        b = ex.execute(ToolCall("read_gauge", {"attempt_n": 1})).payload.get("reading")
        det.append(a == b)
    add("13", "all six scenarios reset deterministically", all(det),
        f"{sum(det)}/{len(PILOT)}")

    ok = []
    for sid in PILOT:
        ex.reset(sid, "FULL", seed=1)
        i = ex.execute(ToolCall("capture_image"))
        g = ex.execute(ToolCall("read_gauge", {"attempt_n": 1}))
        ok.append(i.delivered and g.delivered)
    add("14", "six scenarios execute under the repaired executor", all(ok),
        f"{sum(ok)}/{len(PILOT)}")

    ex.reset("R058", "FULL", seed=1)
    tr = ExecutionTrace("R058", "FULL")
    for t in ("capture_image", "read_iot"):
        tr.append(Stage.REQUESTED, tool=t)
        rr = ex.execute(ToolCall(t))
        tr.append(Stage.EXECUTED, tool=t, status=rr.status)
        if rr.delivered:
            tr.append(Stage.OBSERVATION_DELIVERED, tool=t,
                      observation_id=rr.observation_id,
                      observation_hash=rr.observation_hash, modality=rr.modality)
    add("15", "trace reconstructs what the agent actually observed",
        tr.verify_chain() and {"physical", "digital"} <= tr.delivered_modalities(),
        f"modalities={sorted(tr.delivered_modalities())} chain_valid={tr.verify_chain()}")

    # --- ledger B5: preflight must be able to catch B1-shaped defects -------
    # Check 14 only ever exercised PILOT (all family A), so a class-C/D
    # scenario falling back to a fabricated [0,100] placeholder world could
    # never have been caught here. Both gaps closed below.
    from couchdb_executor import SCENARIO_PHYSICAL

    bad = [sid for sid, phys in SCENARIO_PHYSICAL.items()
           if list(phys.get("range", [])) == [0, 100] and list(phys.get("band", [])) == [0, 100]]
    add("16", "no class-C/D scenario carries the [0,100] placeholder signature",
        not bad, f"placeholder-shaped entries: {bad}" if bad else f"{len(SCENARIO_PHYSICAL)} entries clean")

    FAMILY_SAMPLE = {"C": "R006", "D": "R008"}   # a class-A scenario is PILOT; class-E is sequence-scoped
    fam_ok = []
    for fam, sid in FAMILY_SAMPLE.items():
        ex.reset(sid, "FULL", seed=1)
        i = ex.execute(ToolCall("capture_image"))
        g = ex.execute(ToolCall("read_gauge", {"attempt_n": 1}))
        fam_ok.append(i.delivered and g.delivered
                      and 0.0 < g.payload.get("reading", 0.0) < g.payload.get("gauge_range", [0, 0])[1])
    add("17", "at least one class-C and one class-D scenario execute against real, non-degenerate state",
        all(fam_ok), f"{sum(fam_ok)}/{len(FAMILY_SAMPLE)} ({list(FAMILY_SAMPLE)})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path,
                    default=REPO_ROOT / "reports" / "v1" / "l3_execution_preflight.json")
    args = ap.parse_args()

    checks = run_checks()
    width = max(len(c.name) for c in checks) + 2
    print(f"{'ID':5s} {'CHECK':{width}s} RESULT")
    print("-" * (width + 16))
    for c in checks:
        print(f"{c.id:5s} {c.name:{width}s} {'PASS' if c.passed else 'FAIL'}"
              f"{'  ' + c.detail if c.detail else ''}")

    failed = [c for c in checks if not c.passed]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} passed")
    print("API spend " + ("AUTHORISED" if not failed else "NOT AUTHORISED"))

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(
        {"schema": "assetops.l3_execution_preflight/1",
         "acceptance_condition": ("a model must not receive credit for a physical "
                                  "observation that was never delivered to it"),
         "all_pass": not failed,
         "checks": [asdict(c) for c in checks]}, indent=2) + "\n")
    print(f"-> {args.json}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
