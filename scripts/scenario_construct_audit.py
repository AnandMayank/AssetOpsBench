"""scenario_construct_audit.py — Construct validity of scenario *construction*.

The replication moved the bottleneck from apparatus validity to scenario
validity. The provenance audit found that in all six pilot scenarios
``gold ≡ f(hidden value, band)``: the world state was chosen to produce a
desired verdict rather than sampled and then labelled.

This audits every scenario for six construction defects and classifies each.
Nothing is rewritten.

Defects
-------
``gold_conditioned_physical``  the hidden physical value was picked to satisfy gold
``gold_conditioned_iot``       the IoT value was picked relative to gold
``gold_leakage``               prompt or metadata states the answer
``shortcut_implies_gold``      the digital shortcut alone yields gold
``state_conditioned_on_verdict``  construction order runs verdict -> state
``deterministic_reveal``       a hidden variable reveals gold without evidence

Classification
--------------
``VALID``                            no defect that affects a powered comparison
``CONDITIONALLY VALID``              usable for PROC / CC_grounded, not for CC
``INVALID FOR POWERED BENCHMARKING`` gold-conditioned state, or gold reachable
                                     without the evidence the scenario requires

The distinction that decides the class: a scenario whose *shortcut implies gold*
cannot discriminate on CC, because an agent that never consults the required
evidence still scores 1. It may still measure evidence-seeking (PROC) and
grounding (CC_grounded), which is why that case is CONDITIONAL rather than
INVALID. Gold-conditioned state is INVALID for powered use because the world was
built backwards from the label.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
SCEN_ROOT = (REPO_ROOT.parent / "AssetOpsBenchScenarioGeneration" / "RobotInspection")
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

import l3_arms as A  # noqa: E402
from couchdb_executor import SCENARIO_DIGITAL, SCENARIO_PHYSICAL  # noqa: E402

VALID = "VALID"
CONDITIONAL = "CONDITIONALLY VALID"
INVALID = "INVALID FOR POWERED BENCHMARKING"

#: Scenarios whose hidden value was instantiated to match an existing gold.
GOLD_CONDITIONED_VALUE = {"R009", "R015"}

_ANSWER_STATEMENTS = (
    r"the (?:correct )?(?:answer|verdict) is",
    r"you should (?:commit|escalate|abort)",
    r"gold\s*[:=]",
    r"expected verdict",
)


@dataclass
class ConstructAudit:
    scenario_id: str
    fm: str = ""
    gold: Optional[str] = None
    gold_conditioned_physical: bool = False
    gold_conditioned_iot: bool = False
    gold_leakage: bool = False
    shortcut_implies_gold: Optional[bool] = None
    state_conditioned_on_verdict: bool = False
    deterministic_reveal: bool = False
    classification: str = ""
    reasons: List[str] = field(default_factory=list)
    depth: str = "full"      # full | text-only
    notes: List[str] = field(default_factory=list)


def _question(sid: str) -> str:
    n = int(sid.lstrip("Rr"))
    for c in (SCEN_ROOT / f"scenario_R{n:02d}", SCEN_ROOT / f"scenario_R{n}"):
        if (c / "question.txt").exists():
            return (c / "question.txt").read_text(errors="replace")
    return ""


def audit(sid: str) -> ConstructAudit:
    r = ConstructAudit(scenario_id=sid)
    meta = A.SCENARIOS.get(sid)
    text = re.sub(r"Return \{.*", "", _question(sid), flags=re.S)

    # Gold leakage is checkable for every scenario from its agent-visible text.
    r.gold_leakage = bool([p for p in _ANSWER_STATEMENTS if re.search(p, text, re.I)])

    if not meta or sid not in SCENARIO_PHYSICAL:
        r.depth = "text-only"
        r.classification = CONDITIONAL if not r.gold_leakage else INVALID
        r.notes.append("no executable hidden state defined; only the agent-visible "
                       "text could be audited")
        if r.gold_leakage:
            r.reasons.append("prompt text states the answer")
        return r

    r.fm, r.gold = meta["fm"], meta["gold"]
    phys, dig = SCENARIO_PHYSICAL[sid], SCENARIO_DIGITAL[sid]
    lo, hi = phys["band"]

    # Every pilot scenario's world was built to yield its label: the value sits
    # on the side of the band that gold requires. That is construction order
    # verdict -> state, whether or not the number was quoted from groundtruth.
    in_band = lo <= float(phys["value"]) <= hi
    r.state_conditioned_on_verdict = ((r.gold == "COMMIT") == in_band)
    r.gold_conditioned_physical = sid in GOLD_CONDITIONED_VALUE

    d_in_band = lo <= float(dig["value"]) <= hi
    shortcut_verdict = "COMMIT" if d_in_band else "ESCALATE"
    r.shortcut_implies_gold = (shortcut_verdict == r.gold)
    # The IoT value was set relative to the physical value to create or avoid a
    # shortcut, so it is gold-conditioned in the same sense.
    r.gold_conditioned_iot = True

    # A hidden variable deterministically reveals gold when a single scalar
    # settles the verdict with no perceptual step. Here the gauge value does
    # settle it, but only via an observation the agent must acquire, so this is
    # not a reveal; it would be if the value were exposed in metadata.
    r.deterministic_reveal = False

    if r.gold_conditioned_physical:
        r.reasons.append("hidden physical value instantiated to satisfy an "
                         "existing gold label")
    if r.state_conditioned_on_verdict:
        r.reasons.append("world state placed on the band side the verdict requires")
    if r.shortcut_implies_gold:
        r.reasons.append("digital shortcut alone yields gold; CC cannot "
                         "discriminate, only PROC / CC_grounded")
    if r.gold_leakage:
        r.reasons.append("prompt text states the answer")

    if r.gold_leakage or r.gold_conditioned_physical:
        r.classification = INVALID
    elif r.shortcut_implies_gold or r.state_conditioned_on_verdict:
        r.classification = CONDITIONAL
    else:
        r.classification = VALID
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path,
                    default=REPO_ROOT / "reports" / "v1" / "scenario_construct_audit.json")
    ap.add_argument("--all", action="store_true",
                    help="also scan every scenario directory at text-only depth")
    args = ap.parse_args()

    ids = list(A.PILOT_SCENARIOS)
    if args.all:
        ids += sorted({d.name.replace("scenario_R", "R").zfill(4).replace("R", "R", 1)
                       for d in SCEN_ROOT.glob("scenario_R*")} - set(ids))
        ids = list(dict.fromkeys(
            list(A.PILOT_SCENARIOS)
            + [f"R{int(d.name.split('_R')[1]):03d}" for d in sorted(SCEN_ROOT.glob("scenario_R*"))
               if d.name.split("_R")[1].isdigit()]))

    rows = [audit(s) for s in ids]
    pilot = [r for r in rows if r.scenario_id in A.PILOT_SCENARIOS]

    print("PILOT SCENARIOS — full-depth construct audit")
    print(f"{'Scen':6s} {'gold':9s} {'g-phys':>7s} {'g-iot':>6s} {'leak':>5s} "
          f"{'shortcut':>9s} {'state<-v':>9s} classification")
    print("-" * 92)
    for r in pilot:
        print(f"{r.scenario_id:6s} {str(r.gold):9s} "
              f"{('YES' if r.gold_conditioned_physical else 'no'):>7s} "
              f"{('YES' if r.gold_conditioned_iot else 'no'):>6s} "
              f"{('YES' if r.gold_leakage else 'no'):>5s} "
              f"{('->gold' if r.shortcut_implies_gold else 'away'):>9s} "
              f"{('YES' if r.state_conditioned_on_verdict else 'no'):>9s} "
              f"{r.classification}")

    counts: Dict[str, int] = {}
    for r in rows:
        counts[r.classification] = counts.get(r.classification, 0) + 1
    print("\nCLASSIFICATION SUMMARY")
    for k in (VALID, CONDITIONAL, INVALID):
        if counts.get(k):
            print(f"  {k:36s} {counts[k]}")

    print("\nREASONS (pilot)")
    for r in pilot:
        for reason in r.reasons:
            print(f"  {r.scenario_id}: {reason}")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(
        {"schema": "assetops.scenario_construct_audit/1",
         "classes": {"VALID": VALID, "CONDITIONAL": CONDITIONAL, "INVALID": INVALID},
         "results": [asdict(r) for r in rows]}, indent=2) + "\n")
    print(f"\n-> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
