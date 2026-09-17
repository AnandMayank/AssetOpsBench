"""classd_audit.py — Construction audit of the relational/work-order scenarios.

Class D turns on enterprise state — active work orders, similarity
recommendations, human presence, cross-asset constraints. Checked for
world->gold independence, work-order causality, prompt leakage and executable
tool coverage.

Gold is read from either layout the scenarios use: an ``Expected verdict:`` line,
or the ``verdict`` field of the gold-answer JSON. Requiring only the first made
R021 and R025 look like defects when both state ESCALATE in their gold answer.
"""
from __future__ import annotations
import argparse, json, re, sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
SCEN_ROOT = REPO_ROOT.parent / "AssetOpsBenchScenarioGeneration" / "RobotInspection"
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
from couchdb_executor import TOOLSET  # noqa: E402
from leak_detect import has_rule_leak  # noqa: E402

CLASS_D = ["R008", "R010", "R021", "R022", "R025", "R026"]
ACTION_SPACE = {"COMMIT", "ESCALATE", "ABORT"}
WO_TOOLS = {"get_work_order", "check_wo_similarity", "get_asset_state"}
_STEP = re.compile(r"\d+[-\d]*\.\s*(?:\[[^\]]*\]\s*)?([a-z_]+)\s*(?:\(|→|->|$)", re.M)
_V1 = re.compile(r"Expected verdict:\s*([A-Z_]+)")
_V2 = re.compile(r'"verdict"\s*:\s*"([A-Z_]+)"')
_LEAK = (r"the (?:correct )?(?:answer|verdict) is", r"you should (?:commit|escalate|abort)")


@dataclass
class D:
    scenario_id: str; fm: str = ""; gold: str = ""; gold_source: str = ""
    steps: List[str] = field(default_factory=list)
    wo_causal: bool = False; leak: bool = False; rule_leak: bool = False
    missing_tools: List[str] = field(default_factory=list)
    gold_in_action_space: bool = False
    verdict: str = ""; notes: List[str] = field(default_factory=list)


def _dir(sid: str) -> Path:
    n = int(sid[1:])
    for c in (SCEN_ROOT / f"scenario_R{n:02d}", SCEN_ROOT / f"scenario_R{n}"):
        if c.is_dir():
            return c
    raise FileNotFoundError(sid)


def audit(sid: str) -> D:
    d = _dir(sid)
    gt = (d / "groundtruth.txt").read_text(errors="replace")
    q = re.sub(r"Return \{.*", "", (d / "question.txt").read_text(errors="replace"), flags=re.S)
    man = json.loads((d / "manifest.json").read_text())
    r = D(sid, man.get("fm_code", ""))

    m1, m2 = _V1.search(gt), _V2.search(gt)
    if m1:
        r.gold, r.gold_source = m1.group(1).upper(), "Expected verdict line"
    elif m2:
        r.gold, r.gold_source = m2.group(1).upper(), "gold-answer JSON"
    r.gold_in_action_space = r.gold in ACTION_SPACE

    r.steps = _STEP.findall(gt)
    r.missing_tools = sorted({t for t in r.steps if t not in TOOLSET})
    r.wo_causal = bool(set(r.steps) & WO_TOOLS) or bool(
        re.search(r"work order|human_present|clearance|similarity", gt, re.I))
    r.leak = any(re.search(p, q, re.I) for p in _LEAK)
    # Ledger B3: conditional decision rules are the dominant leak form
    # _LEAK never caught. Measured factor, not auto-DEFECT -- de-leaked
    # twins exist for every leaking D scenario.
    r.rule_leak = has_rule_leak(q)

    if not r.gold:
        r.verdict = "DEFECT: no gold recoverable"
    elif not r.gold_in_action_space:
        r.verdict = f"INTERFACE GAP: gold {r.gold!r} outside {sorted(ACTION_SPACE)}"
        r.notes.append("composite per-asset verdict; the single-verdict action "
                       "interface cannot express it")
    elif r.leak:
        r.verdict = "DEFECT: prompt states the answer"
    elif r.missing_tools:
        r.verdict = f"BLOCKED: {r.missing_tools}"
    else:
        r.verdict = "RUNNABLE"
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path,
                    default=REPO_ROOT / "reports" / "v1" / "classd_audit.json")
    a = ap.parse_args()
    rows = [audit(s) for s in CLASS_D]
    print(f"{'Scen':6s} {'FM':7s} {'Gold':9s} {'source':20s} {'WO-causal':>9s} "
          f"{'exec':>5s} {'leak':>5s} {'rule-leak':>9s}  verdict")
    print("-" * 118)
    for r in rows:
        print(f"{r.scenario_id:6s} {r.fm:7s} {r.gold or '-':9s} {r.gold_source or '-':20s} "
              f"{('yes' if r.wo_causal else 'no'):>9s} "
              f"{('yes' if not r.missing_tools else 'NO'):>5s} "
              f"{('YES' if r.leak else 'no'):>5s} "
              f"{('YES' if r.rule_leak else 'no'):>9s}  {r.verdict}")
        for n in r.notes:
            print(f"        {n}")
    runnable = [r.scenario_id for r in rows if r.verdict == "RUNNABLE"]
    print(f"\nRUNNABLE {len(runnable)} {runnable}")
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps({"schema": "assetops.classd_audit/1",
                                  "runnable": runnable,
                                  "results": [asdict(r) for r in rows]}, indent=2) + "\n")
    print(f"-> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
