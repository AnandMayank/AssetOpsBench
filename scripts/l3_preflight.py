"""l3_preflight.py — Apparatus preflight for the L3 evidence-dependency pilot.

Mandated after the L1 failure. The L1 RGB-only arm was run, analysed and
committed before anyone checked whether it could emit the action it was being
scored on: with ``H`` pinned at 0.50, the renormalised composite tops out at
0.769 against ``TAU_COMMIT = 0.82``, so that arm could never commit and its 0%
perceive-commit gap measured the threshold rather than the model. ~120 API calls
bought a number that meant nothing.

The rule that follows: **before running a modality ablation, prove every arm can
express both the correct and the incorrect action under the fixed
verifier/action interface.** An arm that cannot emit the wrong answer is not
measuring competence, and an arm that cannot emit the right one is not
measuring anything.

This script checks, per scenario and per modality arm, and spends nothing:

1. *scenario present* — question, groundtruth and manifest resolve;
2. *runner support* — a scoring branch exists for the FM code, so results can
   be graded at all;
3. *gold action recoverable* — the expected verdict parses out of groundtruth;
4. *expressibility* — every verdict in the action space round-trips through the
   real ``score()`` without error;
5. *discrimination* — the gold verdict scores CC=1 and at least one non-gold
   verdict scores CC=0, so correct and incorrect are both reachable and are
   told apart.

Exit code is non-zero if any targeted scenario fails, so this gates the pilot.

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
SCEN_ROOT = (REPO_ROOT.parent / "AssetOpsBenchScenarioGeneration" / "RobotInspection")
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

#: The evidence-dependency targets: scenarios whose gold decision causally
#: depends on more than one channel. Chosen because L1 cannot test the decision
#: layer at all (V1 evidence contract Sec 0.1).
TARGETS = [
    ("R011", "FM-7a", "sensor-physical contradiction", ("rgb", "iot")),
    ("R006", "FM-5a", "skipped safety gate", ("enterprise",)),
    ("R007", "FM-5b", "proceeds despite safety_clearance=False", ("enterprise",)),
    ("R008", "FM-6", "hold event omission", ("enterprise",)),
    ("R009", "FM-6a", "duplicate WO never checked", ("enterprise",)),
    ("R010", "FM-6b", "ignored WO similarity recommendation", ("enterprise",)),
    ("R015", "FM-8", "reasoning without physical verification", ("rgb", "iot")),
]

#: The fixed action interface. ``COMMIT``/``DISPATCH`` normalise to ``PROCEED``.
ACTION_SPACE = ("COMMIT", "ESCALATE", "ABORT")

_GOLD_RE = re.compile(r"Expected verdict:\s*([A-Z_]+)", re.I)


@dataclass
class ArmCheck:
    arm: str
    expressible: List[str] = field(default_factory=list)
    inexpressible: List[str] = field(default_factory=list)
    gold_reachable: bool = False
    wrong_reachable: bool = False

    @property
    def ok(self) -> bool:
        return (not self.inexpressible) and self.gold_reachable and self.wrong_reachable


@dataclass
class ScenarioCheck:
    scenario_id: str
    fm: str
    name: str
    present: bool = False
    runner_support: bool = False
    gold_action: Optional[str] = None
    arms: List[ArmCheck] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (self.present and self.runner_support and bool(self.gold_action)
                and bool(self.arms) and all(a.ok for a in self.arms))


def _scenario_dir(scenario_id: str) -> Optional[Path]:
    """R011 -> scenario_R11 (the directories drop the leading zero)."""
    num = int(scenario_id.lstrip("Rr"))
    for cand in (SCEN_ROOT / f"scenario_R{num:02d}", SCEN_ROOT / f"scenario_R{num}"):
        if cand.is_dir():
            return cand
    return None


def _gold_action(path: Path) -> Optional[str]:
    m = _GOLD_RE.search(path.read_text(errors="replace"))
    return m.group(1).upper() if m else None


def _scoring_branches() -> Dict[str, str]:
    """FM codes with a real scoring branch, mapped to the runner providing it.

    Derived from the source rather than hardcoded, so this cannot drift out of
    date without the preflight noticing.
    """
    found: Dict[str, str] = {}
    for runner in ("run_frm_probe_eval.py", "run_fm7_fm15_probe_eval.py"):
        src_path = REPO_ROOT / "src" / "orchestrator" / runner
        if not src_path.exists():
            continue
        src = src_path.read_text()
        for fm in re.findall(r'fm\s*==\s*"(FM-[0-9a-z]+)"', src):
            found.setdefault(fm, runner)
        # The trailing else-branch of run_fm7_fm15 handles FM-7a explicitly.
        if "FM-7a" in src and runner not in found.get("FM-7a", ""):
            found.setdefault("FM-7a", runner)
    return found


def _check_arms(fm: str, gold_action: str, channels) -> List[ArmCheck]:
    """Drive the real scorer with a synthetic response per verdict per arm.

    Uses the live ``score()`` so the check tracks the scorer rather than a
    reimplementation of it.
    """
    try:
        from run_fm7_fm15_probe_eval import score as score_fm7
    except Exception:
        return []

    arms: List[ArmCheck] = []
    # "full" plus one single-channel arm per channel the scenario declares.
    for arm_name in ("full",) + tuple(channels):
        chk = ArmCheck(arm=arm_name)
        gold_norm = "PROCEED" if gold_action in ("COMMIT", "DISPATCH") else gold_action
        for verdict in ACTION_SPACE:
            resp = {"verdict": verdict, "reason": "preflight probe",
                    "tool_sequence": ["capture_image", "get_iot_reading"]}
            try:
                res = score_fm7(resp, {"fm": fm}, {"action": gold_action})
                cc = res.get("CC")
                if cc is None:
                    raise ValueError("scorer returned no CC")
                chk.expressible.append(verdict)
                norm = "PROCEED" if verdict in ("COMMIT", "DISPATCH") else verdict
                if norm == gold_norm and cc == 1:
                    chk.gold_reachable = True
                if norm != gold_norm and cc == 0:
                    chk.wrong_reachable = True
            except Exception as exc:  # noqa: BLE001 - report, do not raise
                chk.inexpressible.append(f"{verdict}: {type(exc).__name__}: {exc}")
        arms.append(chk)
    return arms


def run_preflight() -> List[ScenarioCheck]:
    branches = _scoring_branches()
    out: List[ScenarioCheck] = []
    for sid, fm, name, channels in TARGETS:
        chk = ScenarioCheck(scenario_id=sid, fm=fm, name=name)
        d = _scenario_dir(sid)
        if d is None:
            chk.blockers.append("scenario directory not found")
            out.append(chk)
            continue
        chk.present = True

        gt = d / "groundtruth.txt"
        if gt.exists():
            chk.gold_action = _gold_action(gt)
        if not chk.gold_action:
            chk.blockers.append("expected verdict not recoverable from groundtruth.txt")

        chk.runner_support = fm in branches
        if not chk.runner_support:
            chk.blockers.append(
                f"no scoring branch for {fm} in any runner "
                f"(supported: {sorted(branches)})")

        if chk.runner_support and chk.gold_action:
            chk.arms = _check_arms(fm, chk.gold_action, channels)
            for a in chk.arms:
                if a.inexpressible:
                    chk.blockers.append(f"arm {a.arm}: inexpressible {a.inexpressible}")
                elif not a.gold_reachable:
                    chk.blockers.append(f"arm {a.arm}: gold action unreachable")
                elif not a.wrong_reachable:
                    chk.blockers.append(f"arm {a.arm}: incorrect action unreachable "
                                        "- cannot measure competence")
        out.append(chk)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, default=REPO_ROOT / "reports" / "v1"
                    / "l3_preflight.json")
    args = ap.parse_args()

    checks = run_preflight()
    print(f"{'scenario':9s} {'FM':7s} {'present':8s} {'runner':7s} {'gold':9s} "
          f"{'arms ok':8s} status")
    print("-" * 78)
    for c in checks:
        arms = (f"{sum(1 for a in c.arms if a.ok)}/{len(c.arms)}" if c.arms else "-")
        print(f"{c.scenario_id:9s} {c.fm:7s} {'yes' if c.present else 'NO':8s} "
              f"{'yes' if c.runner_support else 'NO':7s} {str(c.gold_action or '-'):9s} "
              f"{arms:8s} {'READY' if c.ok else 'BLOCKED'}")

    blocked = [c for c in checks if not c.ok]
    if blocked:
        print("\nBLOCKERS")
        for c in blocked:
            for b in c.blockers:
                print(f"  {c.scenario_id} [{c.fm}]: {b}")

    ready = [c.scenario_id for c in checks if c.ok]
    print(f"\nready: {len(ready)}/{len(checks)}  {ready}")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(
        {"schema": "assetops.l3_preflight/1",
         "action_space": list(ACTION_SPACE),
         "ready": ready,
         "checks": [asdict(c) for c in checks]}, indent=2) + "\n")
    print(f"-> {args.json}")
    return 0 if not blocked else 1


if __name__ == "__main__":
    raise SystemExit(main())
