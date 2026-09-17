"""fm_coverage_audit.py — Authoritative FM taxonomy and scenario coverage.

Three sources disagree about which failure mode a scenario tests:

* ``scenario_R*/manifest.json`` — ``fm_code`` field
* ``scenario_R*/groundtruth.txt`` — the ``FM flag:`` line
* ``Scenarios/RobotInspection_Scenarios.csv`` — ``fm_code`` column

**manifest.json + groundtruth.txt are authoritative.** They are per-scenario,
they agree with each other wherever both exist, and they are what the scenario
actually implements. The CSV is a summary table that drifted; ``docs/FM_Crosswalk.md``
was built from it and inherited its errors.

Also reports which of FM-1..FM-28 have *zero* scenario coverage, because a
failure mode with no scenario cannot be reported as measured however often it
appears in a taxonomy table.

Usage::

    python scripts/fm_coverage_audit.py
    python scripts/fm_coverage_audit.py --json reports/v7/fm_coverage.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
SCEN_ROOT = (REPO_ROOT.parent / "AssetOpsBenchScenarioGeneration" / "RobotInspection")
CSV_PATH = SCEN_ROOT / "Scenarios" / "RobotInspection_Scenarios.csv"

_FM_FLAG_RE = re.compile(r"FM flag:\s*([A-Za-z0-9\-]+)")
_GOLD_RE = re.compile(r"Expected verdict:\s*([A-Z_]+)", re.I)

#: The declared taxonomy. Sub-codes (FM-5a etc.) are counted against their
#: parent for coverage, but reported separately.
DECLARED_FMS = [f"FM-{i}" for i in range(1, 29)]


def _fm_parent(fm: str) -> str:
    m = re.match(r"(FM-\d+)", fm)
    return m.group(1) if m else fm


def load_sources() -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = defaultdict(dict)

    for d in sorted(SCEN_ROOT.glob("scenario_R*")):
        man_p, gt_p = d / "manifest.json", d / "groundtruth.txt"
        if not man_p.exists():
            continue
        man = json.loads(man_p.read_text())
        sid = man.get("scenario_id", d.name)
        rows[sid]["dir"] = d.name
        rows[sid]["manifest_fm"] = man.get("fm_code")
        rows[sid]["fm_name"] = man.get("fm_name", "")
        comp = man.get("competency", {})
        rows[sid]["competency"] = comp.get("primary", "")
        if gt_p.exists():
            txt = gt_p.read_text(errors="replace")
            m = _FM_FLAG_RE.search(txt)
            rows[sid]["groundtruth_fm"] = m.group(1) if m else None
            g = _GOLD_RE.search(txt)
            rows[sid]["gold"] = g.group(1).upper() if g else None

    if CSV_PATH.exists():
        with CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                rows[r["scenario_id"].strip()]["csv_fm"] = r["fm_code"].strip()
                rows[r["scenario_id"].strip()].setdefault(
                    "competency", r.get("competency_primary", "").strip())
    return dict(rows)


#: Failure modes implemented in the live agent-trial harness rather than as a
#: scenario directory. They are real and runnable (RC001 has archived results in
#: reports/ablations/) but invisible to a directory scan.
HARNESS_IMPLEMENTED = {"FM-12": "agent_rc001_trial.py (RC001, stale state)"}


def audit(rows: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    disagreements = []
    for sid, r in sorted(rows.items()):
        man, gt, cs = r.get("manifest_fm"), r.get("groundtruth_fm"), r.get("csv_fm")
        # A groundtruth with no "FM flag:" line is silent, not contradictory.
        if man and gt and gt.lower() != "none" and man != gt:
            disagreements.append({"scenario_id": sid, "kind": "manifest_vs_groundtruth",
                                  "manifest": man, "groundtruth": gt, "csv": cs})
        elif man and cs and man != cs:
            disagreements.append({"scenario_id": sid, "kind": "csv_wrong",
                                  "authoritative": man, "groundtruth": gt, "csv": cs})

    covered: Dict[str, List[str]] = defaultdict(list)
    no_dir: List[str] = []
    for sid, r in rows.items():
        if not r.get("dir"):
            no_dir.append(sid)
            continue
        fm = r.get("manifest_fm")
        if fm:
            covered[fm].append(sid)

    families: Dict[str, List[str]] = defaultdict(list)
    for fm, sids in covered.items():
        families[_fm_parent(fm)] += sids

    # Two different gaps, conflating which would be misleading:
    #   family_zero  - nothing in the whole FM-n family is implemented
    #   exact_zero   - the bare code has no scenario even though lettered
    #                  sub-codes do. FM-6 "Hold Event Omission" is a named
    #                  failure mode distinct from FM-6a/FM-6b, so it is a real
    #                  gap even though the family looks covered.
    family_zero = [fm for fm in DECLARED_FMS
                   if not families.get(fm) and fm not in HARNESS_IMPLEMENTED]
    exact_zero = [fm for fm in DECLARED_FMS
                  if fm not in covered and families.get(fm)]
    return {
        "n_scenarios": len(rows),
        "n_with_directory": len(rows) - len(no_dir),
        "scenarios_without_directory": sorted(no_dir),
        "disagreements": disagreements,
        "covered_codes": {k: sorted(v) for k, v in sorted(covered.items())},
        "coverage_by_family": {k: sorted(v) for k, v in sorted(families.items())},
        "family_zero_coverage": family_zero,
        "exact_code_zero_coverage": exact_zero,
        "harness_implemented": HARNESS_IMPLEMENTED,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path,
                    default=REPO_ROOT / "reports" / "v7" / "fm_coverage.json")
    args = ap.parse_args()

    rows = load_sources()
    res = audit(rows)

    print(f"scenarios with a manifest: {res['n_scenarios']}\n")
    print("TAXONOMY DISAGREEMENTS (manifest+groundtruth authoritative)")
    print(f"  {'scen':6s} {'authoritative':14s} {'groundtruth':13s} {'csv':10s} kind")
    for d in res["disagreements"]:
        auth = d.get("authoritative") or d.get("manifest")
        print(f"  {d['scenario_id']:6s} {str(auth):14s} {str(d.get('groundtruth')):13s} "
              f"{str(d.get('csv')):10s} {d['kind']}")
    if not res["disagreements"]:
        print("  none")

    if res["scenarios_without_directory"]:
        print(f"\nIN CSV BUT NO SCENARIO DIRECTORY: {res['scenarios_without_directory']}")

    print(f"\nFAMILY ZERO-COVERAGE ({len(res['family_zero_coverage'])} of "
          f"{len(DECLARED_FMS)}) — nothing in the family is implemented")
    for fm in res["family_zero_coverage"]:
        print(f"  {fm}: no scenario, no harness")
    if not res["family_zero_coverage"]:
        print("  none")

    print(f"\nEXACT-CODE ZERO-COVERAGE ({len(res['exact_code_zero_coverage'])}) — "
          "family covered by lettered sub-codes, bare code unimplemented")
    for fm in res["exact_code_zero_coverage"]:
        subs = sorted(c for c in res["covered_codes"] if _fm_parent(c) == fm)
        print(f"  {fm}: a distinct named failure mode with no scenario "
              f"(family carried by {subs})")

    if res["harness_implemented"]:
        print("\nIMPLEMENTED OUTSIDE THE SCENARIO TREE")
        for fm, where in res["harness_implemented"].items():
            print(f"  {fm}: {where}")

    print("\nCOVERAGE BY FAMILY")
    for fm, sids in res["coverage_by_family"].items():
        codes = sorted({rows[s]["manifest_fm"] for s in sids})
        print(f"  {fm:7s} {len(sids):2d} scenario(s)  codes={codes}  {sids}")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(
        {"schema": "assetops.fm_coverage/1",
         "authority": "manifest.json + groundtruth.txt; CSV fm_code is derived and drifted",
         "scenarios": rows, **res}, indent=2, default=str) + "\n")
    print(f"\n-> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
