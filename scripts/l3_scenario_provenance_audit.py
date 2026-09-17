"""l3_scenario_provenance_audit.py — Construct-validity audit of the six pilot
scenarios. No API calls; no scenario is modified.

Asks whether the hidden physical state is an independent property of the world
or an artifact of the label, and whether gold is inferable without reading the
evidence. Five questions per scenario:

1. Is the physical state generated independently of gold?
2. Can gold be inferred from scenario construction alone?
3. Was any hidden value chosen specifically to satisfy gold?
4. Does prompt or tool metadata leak gold?
5. For R009/R015 specifically — are the in-band values a genuine scenario
   property or a gold-conditioned construction?

The distinction that matters is not "was the value chosen with gold in mind" —
in a labelled benchmark it always is — but **whether an agent can reach gold
without consulting the evidence the scenario claims is necessary.** A value
picked to be in-band is fine; a prompt that says "the reading is in band" is not.

Reported, never silently repaired.
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
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

import l3_arms as A  # noqa: E402
from couchdb_executor import SCENARIO_DIGITAL, SCENARIO_PHYSICAL  # noqa: E402

#: Which scenarios were authored for this experiment vs pre-existing.
AUTHORED_FOR_PILOT = {"R055", "R056", "R057", "R058"}

#: Where the hidden physical value came from.
VALUE_PROVENANCE = {
    "R009": "instantiated — groundtruth states gold COMMIT but no numeric reading",
    "R015": "instantiated — groundtruth states gold COMMIT but no numeric reading",
    "R055": "stated verbatim in groundtruth (245 bar)",
    "R056": "stated verbatim in groundtruth (1.38 m3/s)",
    "R057": "stated verbatim in groundtruth (92 C)",
    "R058": "stated verbatim in groundtruth (268 bar)",
}

_GOLD_WORDS = (r"\bescalat", r"\bcommit\b", r"\babort\b",
               r"out of (?:the )?(?:expected )?band", r"within (?:the )?(?:expected )?band",
               r"exceeds", r"above the .{0,20}limit")


@dataclass
class ScenarioProvenance:
    scenario_id: str
    fm: str
    gold: str
    hidden_value: float
    band: List[float]
    value_in_band: bool
    gold_consistent_with_band: bool
    value_provenance: str
    authored_for_pilot: bool
    # Q1-Q5
    physical_state_independent_of_gold: bool = False
    gold_inferable_from_construction: bool = False
    value_chosen_to_satisfy_gold: bool = False
    prompt_leaks_gold: bool = False
    leak_evidence: List[str] = field(default_factory=list)
    shortcut_points_at_gold: bool = False
    notes: List[str] = field(default_factory=list)


def audit_one(sid: str) -> ScenarioProvenance:
    meta = A.SCENARIOS[sid]
    phys = SCENARIO_PHYSICAL[sid]
    dig = SCENARIO_DIGITAL[sid]
    lo, hi = phys["band"]
    val = float(phys["value"])
    in_band = lo <= val <= hi
    gold = meta["gold"]

    r = ScenarioProvenance(
        scenario_id=sid, fm=meta["fm"], gold=gold, hidden_value=val,
        band=[lo, hi], value_in_band=in_band,
        gold_consistent_with_band=((gold == "COMMIT") == in_band),
        value_provenance=VALUE_PROVENANCE[sid],
        authored_for_pilot=sid in AUTHORED_FOR_PILOT,
    )

    # Q3 / Q1: the value is gold-conditioned wherever gold determines which side
    # of the band it must fall on — which is every scenario here, by design.
    r.value_chosen_to_satisfy_gold = True
    r.physical_state_independent_of_gold = False
    r.notes.append(
        "gold ≡ f(hidden value, band); the two were fixed together, so the "
        "physical state is not an independent draw")

    # Q4: does the prompt state the answer or the band relation?
    # The response-format line enumerates the action space ("COMMIT|ESCALATE|
    # ABORT"); that is the interface, not a leak, and scanning it flagged all
    # six scenarios on the first pass. Strip it before searching.
    text = re.sub(r"Return \{.*", "", A.question_text(sid), flags=re.S)
    hits = [p for p in _GOLD_WORDS if re.search(p, text, re.I)]
    # The bare operating band is legitimate context the gold decision needs.
    hits = [h for h in hits if h != r"within (?:the )?(?:expected )?band"]
    r.prompt_leaks_gold = bool(hits)
    r.leak_evidence = hits

    # Q2: can gold be reached without the physical read? Only if the digital
    # shortcut happens to agree with gold.
    d_val = float(dig["value"])
    d_in_band = lo <= d_val <= hi
    shortcut_verdict = "COMMIT" if d_in_band else "ESCALATE"
    r.shortcut_points_at_gold = (shortcut_verdict == gold)
    r.gold_inferable_from_construction = r.shortcut_points_at_gold
    if r.shortcut_points_at_gold:
        r.notes.append(
            f"digital shortcut ({dig['signal']}={d_val}) implies {shortcut_verdict}, "
            f"which equals gold — an agent that never reads the gauge still scores "
            f"CC=1; only PROC/CC_grounded discriminate here")
    else:
        r.notes.append(
            f"digital shortcut ({dig['signal']}={d_val}) implies {shortcut_verdict} "
            f"but gold is {gold} — reaching gold requires the physical read")

    if sid in ("R009", "R015"):
        r.notes.append(
            "IN-BAND VALUE IS GOLD-CONDITIONED: the groundtruth gives gold COMMIT "
            "without a numeric reading, so the value was chosen because only an "
            "in-band reading is consistent with that gold. It is not an "
            "independent scenario property.")
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path,
                    default=REPO_ROOT / "reports" / "v1" / "l3_scenario_provenance.json")
    args = ap.parse_args()

    rows = [audit_one(s) for s in A.PILOT_SCENARIOS]

    print(f"{'Scen':6s} {'FM':7s} {'Gold':9s} {'value':>7s} {'band':>13s} "
          f"{'indep':>6s} {'inferable':>10s} {'leak':>5s} shortcut")
    print("-" * 88)
    for r in rows:
        print(f"{r.scenario_id:6s} {r.fm:7s} {r.gold:9s} {r.hidden_value:7g} "
              f"{str(r.band):>13s} "
              f"{'no':>6s} "
              f"{('YES' if r.gold_inferable_from_construction else 'no'):>10s} "
              f"{('YES' if r.prompt_leaks_gold else 'no'):>5s} "
              f"{'->gold' if r.shortcut_points_at_gold else 'away'}")

    inferable = [r.scenario_id for r in rows if r.gold_inferable_from_construction]
    leaks = [r.scenario_id for r in rows if r.prompt_leaks_gold]
    print(f"\ngold reachable without the physical read: {len(inferable)}/6 {inferable}")
    print(f"prompt contains gold-directional language:  {len(leaks)}/6 {leaks}")
    for r in rows:
        if r.leak_evidence:
            print(f"   {r.scenario_id}: {r.leak_evidence}")

    print("\nNOTES")
    for r in rows:
        for n in r.notes:
            print(f"  {r.scenario_id}: {n}")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(
        {"schema": "assetops.l3_scenario_provenance/1",
         "question": ("not whether the value was chosen with gold in mind - in a "
                      "labelled benchmark it always is - but whether an agent can "
                      "reach gold without consulting the evidence the scenario "
                      "claims is necessary"),
         "results": [asdict(r) for r in rows]}, indent=2) + "\n")
    print(f"\n-> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
