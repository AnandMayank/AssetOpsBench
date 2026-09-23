#!/usr/bin/env python3
"""phase8i3_build_c_expanded_pool.py -- builds the C-expanded-v1 pool
ENTIRELY OFFLINE. Zero API calls, zero model execution.

C is not seed-generated like A/E -- it is 45 hand-authored fixtures
(classc_fixtures.py) plus their groundtruth in the sibling
AssetOpsBenchScenarioGeneration/RobotInspection repo, of which only 9
are in the frozen-93 manifest (CLASS_C in classc_audit.py). This script
verifies and pools the 8 additional fixtures that are genuine,
independent, already-authored, already-validated C items -- NOT the 11
"ordering-unconstrained control" twins (R059, R060, R064-R072, which
deliberately test a different property and would conflate a validity
check with a capability sample) and NOT the 15 "de-leaked twin" repairs
(R073-R087, patched replacements of specific already-used originals, not
independent new samples).

Candidates verified against BOTH required conditions:
  1. classc_audit.audit(sid) parses a non-empty required_order and a gold
     verdict from groundtruth.txt (real content, not a stub).
  2. sid is in couchdb_executor.SCENARIO_PHYSICAL (ledger B1: a scenario
     without real hidden state must fail loudly, never run against a
     fabricated placeholder world) -- this is a real, wired-up scenario,
     not an orphaned fixture definition.

Outputs (additive, does not touch classc_fixtures.py, classc_audit.py,
the frozen-93 manifest, or the original 9-episode C raw results):
  inspectionbench/manifests/c_procedural_expanded_v1.json
  reports/benchmark/c_expanded/c_expanded_pool_audit.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from classc_audit import audit  # noqa: E402
from classc_fixtures import FIXTURES  # noqa: E402
from couchdb_executor import SCENARIO_PHYSICAL  # noqa: E402

MANIFEST_OUT = REPO_ROOT / "inspectionbench" / "manifests" / "c_procedural_expanded_v1.json"
AUDIT_DIR = REPO_ROOT / "reports" / "benchmark" / "c_expanded"

ORIGINAL_9 = ["R001", "R005", "R006", "R007", "R016", "R017", "R018", "R023", "R024"]
CANDIDATES = ["R008", "R010", "R011", "R012", "R014", "R021", "R022", "R025"]

EXCLUDED_CONTROLS = {
    "R059": "control for R008 (no active WO)", "R060": "control for R010 (no similar WO)",
    "R064": "ordering-unconstrained control for R006", "R065": "ordering-unconstrained control for R007",
    "R066": "ordering-unconstrained (battery threshold, unconstrained)",
    "R067": "ordering-unconstrained (localisation failed, unconstrained)",
    "R068": "ordering-unconstrained control for R001", "R069": "ordering-unconstrained control for R005",
    "R070": "ordering-unconstrained control for R018", "R071": "ordering-unconstrained control for R023",
    "R072": "ordering-unconstrained control for R024",
}
EXCLUDED_TWINS = {f"R{n}": "de-leaked twin of an already-used/already-excluded original"
                  for n in (73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 87)}


def main() -> int:
    rows = []
    audit_rows = []
    for sid in CANDIDATES:
        a = audit(sid)
        fx = FIXTURES[sid]
        in_hidden_state = sid in SCENARIO_PHYSICAL
        has_required_order = len(a.required_order) > 0
        has_gold = bool(a.gold)
        eligible = in_hidden_state and has_required_order and has_gold

        audit_rows.append({
            "scenario_id": sid, "fm": a.fm, "gold": a.gold, "asset": fx.asset,
            "precondition": fx.why, "required_order": "|".join(a.required_order),
            "in_scenario_physical": in_hidden_state, "has_required_order": has_required_order,
            "has_gold": has_gold, "eligible": eligible,
        })
        assert eligible, f"{sid}: failed eligibility check -- {audit_rows[-1]}"

        rows.append({
            "episode_id": f"C2::{sid}::FULL", "dimension": "C", "scenario_id": sid,
            "fm": a.fm, "gold_action": a.gold, "asset": fx.asset,
            "precondition": fx.why, "required_order": a.required_order,
            "pool_version": "c_procedural_expanded_v1",
            "generation_provenance": (
                "Pre-existing, hand-authored fixture (classc_fixtures.py) and "
                "groundtruth (AssetOpsBenchScenarioGeneration/RobotInspection), "
                "not in the frozen-93 manifest's original 9-scenario C set. "
                "Verified via classc_audit.audit() and couchdb_executor."
                "SCENARIO_PHYSICAL membership before inclusion here."
            ),
        })

    manifest = {
        "pool_version": "c_procedural_expanded_v1",
        "dimension": "C",
        "description": (
            "C-expanded-v1: 8 additional, independent, already-authored C-family "
            "fixtures not in the frozen-93 manifest's original 9. Combined with "
            "the original 9 (unchanged, re-used from frozen-93 raw results, not "
            "re-run), this gives 17 canonical C episodes per model -- the maximum "
            "defensible expansion using only already-authored, non-control, "
            "non-twin content (see excluded_controls / excluded_twins below for "
            "why the other 28 of 45 total fixtures were not included)."
        ),
        "original_9_reused_unchanged": ORIGINAL_9,
        "new_8_this_pool": CANDIDATES,
        "combined_total_per_model": len(ORIGINAL_9) + len(CANDIDATES),
        "excluded_controls": EXCLUDED_CONTROLS,
        "excluded_deleaked_twins": EXCLUDED_TWINS,
        "protocol_reference": (
            "IDENTICAL to the original 9: scripts/run_classc_pilot.py::run(), "
            "classc_audit.audit(), classc_fixtures.FixtureSession. No protocol "
            "change of any kind."
        ),
        "episodes": rows,
    }

    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {MANIFEST_OUT} ({len(rows)} new episodes; {len(ORIGINAL_9)} original + "
         f"{len(CANDIDATES)} new = {len(ORIGINAL_9) + len(CANDIDATES)} combined per model)")

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_DIR / "c_expanded_pool_audit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(audit_rows[0].keys()))
        w.writeheader()
        w.writerows(audit_rows)
    print(f"Wrote {AUDIT_DIR}/c_expanded_pool_audit.csv")

    print("\n=== Eligibility audit ===")
    for r in audit_rows:
        print(f"  {r['scenario_id']:6s} fm={r['fm']:8s} gold={r['gold']:10s} "
             f"required_order_len={len(r['required_order'].split('|'))} eligible={r['eligible']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
