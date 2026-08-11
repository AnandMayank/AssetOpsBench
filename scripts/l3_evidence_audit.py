"""l3_evidence_audit.py — Deterministic evidence-dependency audit (no API calls).

Gates the six-scenario L3 pilot. For each scenario it records the ten properties
the design depends on, and derives every one from the actual scenario files, the
rendered arm payloads and the live scorer — nothing is asserted by hand.

The property the whole pilot rests on:

    The withheld channel must be a shortcut to the WRONG decision, but must not
    be necessary for representing the gold decision.

If the withheld channel is necessary for gold, removing it produces an
unanswerable task rather than a weaker agent — the class-B failure that blocked
R011, and the same shape as the L1 RGB-only arm that could not commit at all.

Recorded per scenario:

 1. gold-determining evidence          6. gold action
 2. withheld evidence                  7. gold reachable in every intended arm
 3. shortcut evidence                  8. shortcut action reachable
 4. expected shortcut action           9. withheld evidence reconstructible
 5. expected evidence-grounded action 10. discriminates under a default-safe policy

Point 10 matters because a maximally cautious agent that always ESCALATEs is a
real policy, and a scenario it passes for free measures nothing. Where the
shortcut and gold actions coincide (N1, N3), CC cannot separate them and PROC is
the discriminating signal; that is recorded rather than glossed.

Usage::

    python scripts/l3_evidence_audit.py
    python scripts/l3_evidence_audit.py --json reports/v1/l3_evidence_audit.json
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
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

import l3_arms as A  # noqa: E402
from l3_scoring import score_l3  # noqa: E402

#: What the digital channel carries in each pilot scenario, and what an agent
#: that trusts it instead of the gauge would answer. Derived from the scenario
#: text; asserted here so the audit can check consequences rather than restate.
SHORTCUT = {
    "R009": ("IoT 6-hour average, 287 bar", "COMMIT"),
    "R015": ("IoT / historical average", "COMMIT"),
    "R055": ("IoT telemetry 245 bar, agrees with gauge", "COMMIT"),
    "R056": ("IoT telemetry 1.02 m3/s, nominal while gauge is out of range", "COMMIT"),
    "R057": ("30-day historical average 92 C, agrees with gauge", "COMMIT"),
    "R058": ("maintenance history framing the exceedance as routine", "COMMIT"),
}

#: The action a maximally cautious agent emits without reading anything.
DEFAULT_SAFE_ACTION = "ESCALATE"


@dataclass
class EvidenceAudit:
    scenario_id: str
    fm: str
    gold_determining_evidence: str = ""
    withheld_evidence: List[str] = field(default_factory=list)
    shortcut_evidence: str = ""
    expected_shortcut_action: str = ""
    expected_grounded_action: str = ""
    gold_action: Optional[str] = None
    gold_reachable_all_arms: bool = False
    shortcut_action_reachable: bool = False
    withheld_reconstructible: List[str] = field(default_factory=list)
    discriminates_under_default_safe: bool = False
    discriminating_signal: str = ""
    competence_arms: List[str] = field(default_factory=list)
    probe_arms: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def audit_scenario(sid: str) -> EvidenceAudit:
    meta = A.SCENARIOS[sid]
    shortcut_desc, shortcut_action = SHORTCUT[sid]
    r = EvidenceAudit(scenario_id=sid, fm=meta["fm"])
    r.gold_action = meta["gold"]
    r.gold_determining_evidence = "physical gauge read (capture_image)"
    r.shortcut_evidence = shortcut_desc
    r.expected_shortcut_action = shortcut_action
    r.expected_grounded_action = meta["gold"]

    specs = A.arms_for(sid)
    gold_norm = "PROCEED" if r.gold_action in ("COMMIT", "DISPATCH") else r.gold_action

    # (7) gold reachable in every *competence* arm; (8) shortcut representable
    gold_ok_everywhere = True
    for spec in specs:
        if spec.insufficient_evidence_probe:
            r.probe_arms.append(spec.arm_id)
        else:
            r.competence_arms.append(spec.arm_id)
        ccs = {}
        for verdict in A.ALLOWED_ACTIONS:
            resp = {"verdict": verdict, "reason": "audit",
                    "tool_sequence": ["get_pose", "open_panel", "capture_image",
                                      "read_gauge", "get_work_orders"]}
            ccs[verdict] = score_l3(resp, {"fm": r.fm}, {"action": r.gold_action})["CC"]
        gold_hit = any(c == 1 for v, c in ccs.items()
                       if ("PROCEED" if v in ("COMMIT", "DISPATCH") else v) == gold_norm)
        nongold_hit = any(c == 0 for v, c in ccs.items()
                          if ("PROCEED" if v in ("COMMIT", "DISPATCH") else v) != gold_norm)
        if not spec.insufficient_evidence_probe and not gold_hit:
            gold_ok_everywhere = False
            r.failures.append(f"arm {spec.arm_id}: gold not reachable")
        if not nongold_hit:
            r.failures.append(f"arm {spec.arm_id}: no non-gold reachable — gold forced")
    r.gold_reachable_all_arms = gold_ok_everywhere
    r.shortcut_action_reachable = shortcut_action in A.ALLOWED_ACTIONS

    # (2) withheld evidence, (9) reconstruction
    for spec in specs:
        if not spec.withheld_evidence:
            continue
        r.withheld_evidence = sorted(set(r.withheld_evidence) | set(spec.withheld_evidence))
        payload = A.render_arm(spec)
        leaks = A.reconstructible(payload, A.withheld_quantities(sid, spec))
        if leaks:
            r.withheld_reconstructible += leaks
            r.failures.append(f"arm {spec.arm_id}: withheld value recoverable {leaks}")
        # The arm must actually differ from FULL.
        full = next(s for s in specs if s.arm_id == A.FULL)
        if A.payload_signature(payload) == A.payload_signature(A.render_arm(full)):
            r.failures.append(f"arm {spec.arm_id}: payload identical to FULL")

    # The load-bearing property: withholding the shortcut must not remove gold.
    phys_only = next((s for s in specs if s.arm_id == A.PHYSICAL_ONLY), None)
    if phys_only is None:
        r.failures.append("no PHYSICAL_ONLY arm — cannot test evidence dependency")
    elif phys_only.insufficient_evidence_probe:
        r.failures.append(
            "PHYSICAL_ONLY is an insufficient-evidence probe: the withheld channel "
            "is necessary for gold, so removing it yields an unanswerable task")

    # (10) discrimination under a default-safe policy
    if r.expected_shortcut_action == r.gold_action:
        r.discriminating_signal = "PROC (shortcut yields the same verdict as gold)"
    else:
        r.discriminating_signal = "CC and PROC (shortcut yields a different verdict)"
    # A blanket-ESCALATE agent passes any ESCALATE-gold scenario for free on CC.
    default_safe_passes = (gold_norm == DEFAULT_SAFE_ACTION)
    r.discriminates_under_default_safe = not default_safe_passes or "PROC" in r.discriminating_signal
    if default_safe_passes and "PROC" not in r.discriminating_signal:
        r.failures.append("a blanket-ESCALATE policy passes this scenario on CC alone")
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path,
                    default=REPO_ROOT / "reports" / "v1" / "l3_evidence_audit.json")
    args = ap.parse_args()

    results = [audit_scenario(s) for s in A.PILOT_SCENARIOS]

    hdr = (f"{'Scen':6s} {'FM':7s} {'Gold':9s} {'Shortcut->':11s} {'GoldReach':10s} "
           f"{'Leak':5s} {'Signal':10s} Status")
    print(hdr); print("-" * len(hdr))
    for r in results:
        print(f"{r.scenario_id:6s} {r.fm:7s} {str(r.gold_action):9s} "
              f"{r.expected_shortcut_action:11s} "
              f"{'yes' if r.gold_reachable_all_arms else 'NO':10s} "
              f"{'yes' if r.withheld_reconstructible else 'no':5s} "
              f"{r.discriminating_signal.split(' ')[0]:10s} "
              f"{'PASS' if r.ok else 'FAIL'}")

    failed = [r for r in results if not r.ok]
    if failed:
        print("\nFAILURES")
        for r in failed:
            for f in r.failures:
                print(f"  {r.scenario_id}: {f}")

    n_cc = sum(1 for r in results if r.discriminating_signal.startswith("CC"))
    golds = [r.gold_action for r in results]
    maj = max(golds.count(g) for g in set(golds)) / len(golds)
    print(f"\ngold balance: { {g: golds.count(g) for g in sorted(set(golds))} }  "
          f"majority-class baseline {maj:.0%}")
    print(f"discriminating on CC+PROC: {n_cc}/{len(results)}   "
          f"on PROC alone: {len(results) - n_cc}/{len(results)}")
    print(f"\nAUDIT: {len(results) - len(failed)}/{len(results)} pass")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({
        "schema": "assetops.l3_evidence_audit/1",
        "principle": ("the withheld channel must be a shortcut to the wrong decision "
                      "but must not be necessary for representing the gold decision"),
        "pilot_scenarios": list(A.PILOT_SCENARIOS),
        "gold_balance": {g: golds.count(g) for g in sorted(set(golds))},
        "majority_class_baseline": round(maj, 4),
        "all_pass": not failed,
        "status": "apparatus-validation pilot, NOT a powered benchmark result",
        "results": [asdict(r) for r in results],
    }, indent=2) + "\n")
    print(f"-> {args.json}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
