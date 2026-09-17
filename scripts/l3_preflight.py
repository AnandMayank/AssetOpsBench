"""l3_preflight.py — Apparatus preflight for the L3 evidence-dependency pilot.

Mandated after the L1 failure. The L1 RGB-only arm was run, analysed and
committed before anyone checked whether it could emit the action it was scored
on: with ``H`` pinned at 0.50 the renormalised composite tops out at 0.769
against ``TAU_COMMIT = 0.82``, so that arm could never commit and its 0%
perceive-commit gap measured the threshold rather than the model.

Five checks, none of which spends anything:

A. **scorer** — a scoring branch exists for the FM code, discovered from runner
   source rather than a hardcoded list, so it cannot drift out of date;
B. **arm rendering** — every declared arm renders an agent-input payload;
C. **arm difference** — arms differ at the *rendered payload* level, and a
   withheld quantity cannot be reconstructed from anything still exposed;
D. **gold discrimination** — gold and at least one non-gold action are both
   representable, so the arm neither forces nor forbids the right answer;
E. **capability matrix** — scenario x arm status.

A scenario is READY only if all of A-D hold for it and it has at least two
genuinely differing arms. Arms that make gold unreachable by construction are
labelled INSUFFICIENT_EVIDENCE_PROBE: legitimate (FM-6a and FM-8 are *about*
acting without physical evidence) but a different measurement from competence,
so they are counted separately and never used to reach the readiness gate.

Usage::

    python scripts/l3_preflight.py
    python scripts/l3_preflight.py --json reports/v1/l3_preflight.json
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

import l3_arms  # noqa: E402
from l3_arms import (  # noqa: E402
    ALLOWED_ACTIONS, SCENARIOS, arms_for, payload_signature, reconstructible,
    render_arm, scenario_dir, withheld_quantities,
)

#: Sources scanned for scoring branches. Adding a module here is the only way to
#: extend capability; the supported-FM list is never written down directly.
SCORER_SOURCES = ("run_frm_probe_eval.py", "run_fm7_fm15_probe_eval.py",
                  "l3_scoring.py")

_GOLD_RE = re.compile(r"Expected verdict:\s*([A-Z_]+)", re.I)

READY = "READY"
BLOCKED = "BLOCKED"
INVALID = "INVALID-DESIGN"
PROBE = "INSUFFICIENT_EVIDENCE_PROBE"
NA = "N/A-no-ablation"


@dataclass
class ArmResult:
    arm_id: str
    rendered: bool = False
    differs_from_full: Optional[bool] = None
    reconstructible: List[str] = field(default_factory=list)
    gold_representable: bool = False
    nongold_representable: bool = False
    forces_gold: bool = False
    probe: bool = False
    status: str = BLOCKED
    notes: List[str] = field(default_factory=list)


@dataclass
class ScenarioResult:
    scenario_id: str
    fm: str
    axis: str
    gold: Optional[str] = None
    scorer: bool = False
    scorer_source: str = ""
    arms: List[ArmResult] = field(default_factory=list)
    status: str = BLOCKED
    blockers: List[str] = field(default_factory=list)

    @property
    def measurable_arms(self) -> List[ArmResult]:
        return [a for a in self.arms if a.status == READY]


def discover_scorers() -> Dict[str, str]:
    """FM codes with a real scoring branch -> the module providing it."""
    found: Dict[str, str] = {}
    for name in SCORER_SOURCES:
        p = REPO_ROOT / "src" / "orchestrator" / name
        if not p.exists():
            continue
        src = p.read_text()
        for fm in re.findall(r'fm\s*==\s*"(FM-[0-9a-z]+)"', src):
            found.setdefault(fm, name)
        # FM-7a is the trailing else-branch of run_fm7_fm15_probe_eval.
        if name == "run_fm7_fm15_probe_eval.py" and "FM-7a" in src:
            found.setdefault("FM-7a", name)
    return found


def _score_fn(fm: str, source: str):
    if source == "l3_scoring.py":
        from l3_scoring import score_l3
        return lambda resp, sc, gold: score_l3(resp, sc, gold)
    from run_fm7_fm15_probe_eval import score as score_fm7
    return score_fm7


def gold_action(scenario_id: str) -> Optional[str]:
    d = scenario_dir(scenario_id)
    if d is None:
        return None
    gt = d / "groundtruth.txt"
    if not gt.exists():
        return None
    m = _GOLD_RE.search(gt.read_text(errors="replace"))
    return m.group(1).upper() if m else None


def check_scenario(scenario_id: str, scorers: Dict[str, str]) -> ScenarioResult:
    meta = SCENARIOS[scenario_id]
    res = ScenarioResult(scenario_id=scenario_id, fm=meta["fm"], axis=meta["axis"])

    # --- A. scorer -------------------------------------------------------
    res.scorer_source = scorers.get(res.fm, "")
    res.scorer = bool(res.scorer_source)
    if not res.scorer:
        res.blockers.append(f"no scoring branch for {res.fm} "
                            f"(supported: {sorted(scorers)})")

    res.gold = gold_action(scenario_id)
    if not res.gold:
        res.blockers.append("expected verdict not recoverable from groundtruth.txt")

    specs = arms_for(scenario_id)
    if meta["axis"] == "procedural":
        res.blockers.append("procedural axis: gold depends on tool ordering, not on "
                            "evidence channel — no modality ablation is valid")

    # --- B. rendering, C. difference, D. discrimination ------------------
    signatures: Dict[str, str] = {}
    for spec in specs:
        ar = ArmResult(arm_id=spec.arm_id, probe=spec.insufficient_evidence_probe)
        try:
            payload = render_arm(spec)
            ar.rendered = True
        except Exception as exc:  # noqa: BLE001
            ar.notes.append(f"render failed: {type(exc).__name__}: {exc}")
            res.arms.append(ar)
            continue

        sig = payload_signature(payload)
        signatures[spec.arm_id] = sig
        if spec.arm_id != l3_arms.FULL and l3_arms.FULL in signatures:
            ar.differs_from_full = sig != signatures[l3_arms.FULL]
            if not ar.differs_from_full:
                ar.notes.append("payload identical to FULL — arm is cosmetic")

        leaks = reconstructible(payload, withheld_quantities(scenario_id, spec))
        ar.reconstructible = leaks
        if leaks:
            ar.notes.append(f"withheld value still recoverable: {leaks}")

        if res.scorer and res.gold:
            fn = _score_fn(res.fm, res.scorer_source)
            gold_norm = "PROCEED" if res.gold in ("COMMIT", "DISPATCH") else res.gold
            for verdict in ALLOWED_ACTIONS:
                resp = {"verdict": verdict, "reason": "preflight probe",
                        "tool_sequence": ["get_pose", "open_panel", "capture_image",
                                          "get_work_orders", "get_similar_work_orders",
                                          "read_gauge"]}
                try:
                    cc = fn(resp, {"fm": res.fm}, {"action": res.gold}).get("CC")
                except Exception as exc:  # noqa: BLE001
                    ar.notes.append(f"{verdict} unscoreable: {type(exc).__name__}")
                    continue
                norm = "PROCEED" if verdict in ("COMMIT", "DISPATCH") else verdict
                if norm == gold_norm and cc == 1:
                    ar.gold_representable = True
                if norm != gold_norm and cc == 0:
                    ar.nongold_representable = True
            ar.forces_gold = ar.gold_representable and not ar.nongold_representable

        # --- status ------------------------------------------------------
        if not ar.rendered or ar.reconstructible:
            ar.status = INVALID
        elif ar.differs_from_full is False:
            ar.status = INVALID
        elif meta["axis"] == "procedural":
            ar.status = NA
        elif ar.probe:
            ar.status = PROBE
        elif ar.gold_representable and ar.nongold_representable:
            ar.status = READY
        else:
            ar.status = BLOCKED
            if not ar.gold_representable:
                ar.notes.append("gold action not representable")
            if not ar.nongold_representable:
                ar.notes.append("no non-gold action representable — forces gold")
        res.arms.append(ar)

    # --- scenario verdict ------------------------------------------------
    invalid = [a for a in res.arms if a.status == INVALID]
    if invalid:
        res.status = INVALID
        res.blockers += [f"arm {a.arm_id}: {'; '.join(a.notes)}" for a in invalid]
    elif meta["axis"] == "procedural":
        res.status = NA
    elif res.blockers:
        res.status = BLOCKED
    elif len(res.measurable_arms) >= 2:
        res.status = READY
    else:
        res.status = BLOCKED
        res.blockers.append(
            f"needs >=2 measurable arms, has {len(res.measurable_arms)} "
            f"({[a.arm_id for a in res.arms if a.status == PROBE]} are "
            "insufficient-evidence probes, not competence measurements)")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path,
                    default=REPO_ROOT / "reports" / "v1" / "l3_preflight.json")
    ap.add_argument("--gate", type=int, default=6,
                    help="minimum READY scenarios before API spend is authorised")
    args = ap.parse_args()

    scorers = discover_scorers()
    results = [check_scenario(sid, scorers) for sid in SCENARIOS]

    print(f"scoring branches discovered: {sorted(scorers)}\n")
    hdr = (f"{'Scen':5s} {'FM':6s} {'Gold':9s} {'FULL':6s} {'PHYS':6s} {'DIGI':6s} "
           f"{'ENT-':6s} {'Scor':5s} {'Diff':5s} {'Gold':5s} {'NonG':5s} Status")
    print(hdr)
    print("-" * len(hdr))
    short = {READY: "ok", BLOCKED: "--", INVALID: "XX", PROBE: "probe", NA: "n/a"}
    for r in results:
        by = {a.arm_id: a for a in r.arms}
        def cell(name):
            a = by.get(name)
            return short.get(a.status, "?") if a else "-"
        any_arm = next((a for a in r.arms if a.arm_id == l3_arms.FULL), None)
        diff = "yes" if any(a.differs_from_full for a in r.arms
                            if a.differs_from_full is not None) else "-"
        print(f"{r.scenario_id:5s} {r.fm:6s} {str(r.gold):9s} "
              f"{cell('FULL'):6s} {cell('PHYSICAL_ONLY'):6s} {cell('DIGITAL_ONLY'):6s} "
              f"{cell('NO_ENTERPRISE'):6s} "
              f"{'yes' if r.scorer else 'NO':5s} {diff:5s} "
              f"{'yes' if any_arm and any_arm.gold_representable else '-':5s} "
              f"{'yes' if any_arm and any_arm.nongold_representable else '-':5s} "
              f"{r.status}")

    ready = [r.scenario_id for r in results if r.status == READY]
    blocked = [r for r in results if r.status not in (READY,)]
    if blocked:
        print("\nBLOCKERS / NOTES")
        for r in blocked:
            for b in r.blockers:
                print(f"  {r.scenario_id} [{r.fm}] {r.status}: {b}")

    print(f"\nREADY: {len(ready)}/{len(results)}  {ready}")
    gate_ok = len(ready) >= args.gate
    print(f"gate (>= {args.gate} READY): {'PASS' if gate_ok else 'FAIL'} "
          f"— API spend {'authorised' if gate_ok else 'NOT authorised'}")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({
        "schema": "assetops.l3_preflight/2",
        "action_space": list(ALLOWED_ACTIONS),
        "scoring_branches": scorers,
        "gate": args.gate,
        "ready": ready,
        "gate_pass": gate_ok,
        "arm_manifest": l3_arms.manifest(),
        "results": [asdict(r) for r in results],
    }, indent=2) + "\n")
    print(f"-> {args.json}")
    return 0 if gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
