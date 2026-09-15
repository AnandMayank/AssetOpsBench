"""eval_eligibility.py — computes, from the actual manifest FILES (never a
hardcoded literal), which of V3's 4,075 canonical episodes are eligible for
real model evaluation under a runner that exists today (Phase 8H.2J).

Non-negotiable #10/#11: scripted generation-time admission artifacts (A/E's
TDA/GSR, C/D-enterprise's `score`/`trace_events` fields in
cd_4000_final_manifest.json beyond the frozen-93 subset) are NOT model
results and must never be counted as eligible just because the manifest
carries a numeric-looking score field.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class EligibilityReport:
    total_canonical: int
    eligible_total: int
    eligible_by_pool: Dict[str, int]
    eligible_by_dimension: Dict[str, int]
    not_eligible_total: int
    not_eligible_reason: str


def compute_eligibility() -> EligibilityReport:
    v3 = json.loads((REPO_ROOT / "reports/benchmark/final_benchmark_manifest_v3.json").read_text())
    total = v3["final_canonical_total"]

    frozen93 = json.loads((REPO_ROOT / "reports/ec/phase8h1_pilot_manifest.json").read_text())
    frozen93_eps = frozen93.get("episodes", frozen93 if isinstance(frozen93, list) else [])

    b_acq = json.loads((REPO_ROOT / "reports/benchmark/b_acquisition_final_manifest.json").read_text())
    d_phys = v3["d_physical_episodes"]

    from collections import Counter
    dims = Counter(e.get("dimension") for e in frozen93_eps)
    by_dim: Dict[str, int] = dict(dims)
    by_dim["B"] = by_dim.get("B", 0) + len(b_acq["episodes"])  # B-legacy(12) + B-Acquisition(66)
    by_dim["D"] = by_dim.get("D", 0) + len(d_phys)             # D-enterprise(6) + D-physical(70)

    by_pool = {
        "frozen_93_A_B_C_D_E": len(frozen93_eps),
        "b_acquisition_66": len(b_acq["episodes"]),
        "d_physical_70": len(d_phys),
    }
    eligible = sum(by_pool.values())

    return EligibilityReport(
        total_canonical=total,
        eligible_total=eligible,
        eligible_by_pool=by_pool,
        eligible_by_dimension=by_dim,
        not_eligible_total=total - eligible,
        not_eligible_reason=(
            "Scripted generation-time admission artifacts only (TDA/GSR for the "
            "A/E pool in final_benchmark_manifest.json's 3,795 combined episodes; "
            "score/trace_events for the C/D-enterprise pool in "
            "cd_4000_final_manifest.json beyond the frozen-93 subset) -- no model "
            "has ever been executed against these episodes. No runner exists today "
            "that dispatches them to a live model; run_classc_pilot.py/"
            "run_classd_pilot.py take a hardcoded scenario-ID list, not this "
            "manifest, and ae_generator.py contains no backend/model call at all."
        ),
    )


if __name__ == "__main__":
    r = compute_eligibility()
    print(json.dumps({
        "total_canonical": r.total_canonical,
        "eligible_total": r.eligible_total,
        "eligible_by_pool": r.eligible_by_pool,
        "eligible_by_dimension": r.eligible_by_dimension,
        "not_eligible_total": r.not_eligible_total,
        "not_eligible_reason": r.not_eligible_reason,
    }, indent=2))
