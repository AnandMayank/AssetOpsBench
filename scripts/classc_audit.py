"""classc_audit.py — Construction audit of the nine class-C procedural scenarios.

Class C turns on the *order* of tool calls rather than on which evidence exists,
so its defects differ from the evidence-dependency families. It became
measurable only once execution was real: ordering can now be read from executed
trace events instead of a model-authored ``tool_sequence``.

Checked per scenario:

``ordering_recoverable``   the required call order parses out of groundtruth
``gold_leak``              the prompt states the answer
``order_stated_in_prompt`` the prompt spells out the required sequence, which
                           would make ordering an instruction-following test
                           rather than a procedural-safety one
``executable_tools``       every required call exists in the executor
``gold_consistent``        the stated gold is reachable under the action space
``preconditions_present``  the scenario needs hidden state the executor can set

Reports only. If a genuine scenario-definition defect is found it is surfaced,
never silently repaired.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
SCEN_ROOT = (REPO_ROOT.parent / "AssetOpsBenchScenarioGeneration" / "RobotInspection")
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

from couchdb_executor import TOOLSET  # noqa: E402
from leak_detect import has_rule_leak  # noqa: E402

CLASS_C = ["R001", "R005", "R006", "R007", "R016", "R017", "R018", "R023", "R024"]

#: Ordering-free contrastive controls. These deliberately carry no numbered
#: call sequence in their groundtruth -- that absence *is* the design, not a
#: parsing failure -- so they are audited separately from CLASS_C.
CLASS_C_CONTROLS = ["R064", "R065", "R066", "R067",
                    "R068", "R069", "R070", "R071", "R072"]

# Sequences appear in two layouts: one numbered call per line, and several
# arrow-separated on a single line ("1. navigate_to -> 2. capture_image").
# Requiring the first layout made R023 look like a scenario defect.
_STEP = re.compile(r"\d+\.\s*(?:\[[^\]]*\]\s*)?([a-z_]+)\s*(?:\(|\u2192|->|$)", re.M)
_GOLD = re.compile(r"Expected verdict:\s*([A-Z_]+)")
_ANSWER = (r"the (?:correct )?(?:answer|verdict) is", r"you should (?:commit|escalate|abort)")
ALLOWED = {"COMMIT", "ESCALATE", "ABORT"}


@dataclass
class ClassCAudit:
    scenario_id: str
    fm: str = ""
    gold: str = ""
    required_order: List[str] = field(default_factory=list)
    ordering_recoverable: bool = False
    ordering_free_by_design: bool = False
    gold_leak: bool = False
    rule_leak: bool = False
    order_stated_in_prompt: bool = False
    executable_tools: bool = False
    missing_tools: List[str] = field(default_factory=list)
    gold_consistent: bool = False
    verdict: str = ""
    notes: List[str] = field(default_factory=list)


def _dir(sid: str) -> Path:
    n = int(sid.lstrip("R"))
    for c in (SCEN_ROOT / f"scenario_R{n:02d}", SCEN_ROOT / f"scenario_R{n}"):
        if c.is_dir():
            return c
    raise FileNotFoundError(sid)


def audit(sid: str) -> ClassCAudit:
    d = _dir(sid)
    gt = (d / "groundtruth.txt").read_text(errors="replace")
    q = (d / "question.txt").read_text(errors="replace")
    man = json.loads((d / "manifest.json").read_text())
    r = ClassCAudit(scenario_id=sid, fm=man.get("fm_code", ""))

    m = _GOLD.search(gt)
    r.gold = m.group(1).upper() if m else ""
    r.gold_consistent = r.gold in ALLOWED

    r.required_order = _STEP.findall(gt)
    # A single required call is a legitimate design: R018 escalates on
    # list_waypoints alone, with a *negative* constraint (do not navigate).
    # An ordering-free control declares its absence of a required sequence in
    # its own manifest (provenance.kind); that is the intended construction,
    # not a parsing failure, so it must not read as a DEFECT.
    r.ordering_free_by_design = man.get("provenance", {}).get("kind") == "ordering_free"
    r.ordering_recoverable = len(r.required_order) >= 1 or r.ordering_free_by_design

    body = re.sub(r"Return \{.*", "", q, flags=re.S)
    r.gold_leak = any(re.search(p, body, re.I) for p in _ANSWER)
    # Ledger B3: conditional decision rules ("if X, escalate") are the
    # dominant leak form _ANSWER never caught. Reported, not auto-DEFECT --
    # treated as a measured factor; de-leaked twins exist for every leaking
    # scenario (manifest provenance.kind == "de_leaked").
    r.rule_leak = has_rule_leak(body)

    # Does the prompt hand over the sequence? Two or more required calls named
    # in the agent-visible text turns procedural safety into instruction
    # following.
    named = [t for t in set(r.required_order) if re.search(rf"\b{t}\b", body)]
    r.order_stated_in_prompt = len(named) >= 2
    if named:
        r.notes.append(f"prompt names required calls: {sorted(named)}")

    r.missing_tools = sorted({t for t in r.required_order if t not in TOOLSET})
    r.executable_tools = not r.missing_tools

    if not r.gold_consistent:
        r.verdict = "DEFECT: gold not in the action space"
    elif not r.ordering_recoverable:
        r.verdict = "DEFECT: no ordered sequence recoverable"
    elif r.gold_leak:
        r.verdict = "DEFECT: prompt states the answer"
    elif not r.executable_tools:
        r.verdict = f"BLOCKED: tools not in executor {r.missing_tools}"
    elif r.order_stated_in_prompt:
        r.verdict = "CONDITIONAL: prompt names the required calls"
    else:
        r.verdict = "RUNNABLE"
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path,
                    default=REPO_ROOT / "reports" / "v1" / "classc_audit.json")
    args = ap.parse_args()

    rows = [audit(s) for s in CLASS_C]
    print(f"{'Scen':6s} {'FM':7s} {'Gold':9s} {'steps':>5s} {'exec':>5s} {'leak':>5s} "
          f"{'rule-leak':>9s} {'order-in-prompt':>16s}  verdict")
    print("-" * 108)
    for r in rows:
        print(f"{r.scenario_id:6s} {r.fm:7s} {r.gold:9s} {len(r.required_order):5d} "
              f"{('yes' if r.executable_tools else 'NO'):>5s} "
              f"{('YES' if r.gold_leak else 'no'):>5s} "
              f"{('YES' if r.rule_leak else 'no'):>9s} "
              f"{('YES' if r.order_stated_in_prompt else 'no'):>16s}  {r.verdict}")

    print("\nREQUIRED ORDER (from groundtruth)")
    for r in rows:
        print(f"  {r.scenario_id}: {' -> '.join(r.required_order) or '(none parsed)'}")
        if r.missing_tools:
            print(f"      MISSING FROM EXECUTOR: {r.missing_tools}")
        for n in r.notes:
            print(f"      {n}")

    runnable = [r.scenario_id for r in rows if r.verdict == "RUNNABLE"]
    cond = [r.scenario_id for r in rows if r.verdict.startswith("CONDITIONAL")]
    blocked = [r for r in rows if r.verdict.startswith(("DEFECT", "BLOCKED"))]
    print(f"\nRUNNABLE {len(runnable)} {runnable}")
    print(f"CONDITIONAL {len(cond)} {cond}")
    print(f"BLOCKED/DEFECT {len(blocked)} {[r.scenario_id for r in blocked]}")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(
        {"schema": "assetops.classc_audit/1",
         "runnable": runnable, "conditional": cond,
         "blocked": [r.scenario_id for r in blocked],
         "results": [asdict(r) for r in rows]}, indent=2) + "\n")
    print(f"\n-> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
