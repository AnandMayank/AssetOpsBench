"""Regression tests for the Phase 8H.2C C/D bulk generation pipeline
(scripts/phase8h1_cd_bulk_generate.py). Zero model/API calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from cd_generator import ALL_ASSETS  # noqa: E402
from couchdb_executor import CouchDBExecutor  # noqa: E402
import phase8h1_cd_bulk_generate as B  # noqa: E402


def test_full_grid_has_no_duplicate_seeds():
    grid = B.full_grid()
    seeds = [params["seed"] for _, params in grid]
    assert len(seeds) == len(set(seeds)), "seed ranges must be disjoint across templates"


def test_full_grid_size_matches_composition():
    grid = B.full_grid()
    assert len(grid) == 132


def test_seed_ranges_are_disjoint_and_ordered():
    ranges = []
    for tid, base in B.SEED_BASE.items():
        n = len(B._grid(tid, base))
        ranges.append((base, base + n - 1, tid))
    ranges.sort()
    for i in range(1, len(ranges)):
        prev_end = ranges[i - 1][1]
        cur_start = ranges[i][0]
        assert cur_start > prev_end, f"seed ranges overlap: {ranges[i-1]} vs {ranges[i]}"


def test_fingerprint_ignores_seed_but_captures_gold_and_actions():
    ex = CouchDBExecutor()
    from cd_generator import TEMPLATES
    spec1 = TEMPLATES["T-C-PANEL_STUCK"](asset="chiller_6", panel_stuck=True, seed=1)
    spec2 = TEMPLATES["T-C-PANEL_STUCK"](asset="chiller_6", panel_stuck=True, seed=2)
    spec3 = TEMPLATES["T-C-PANEL_STUCK"](asset="chiller_6", panel_stuck=False, seed=1)
    fp1, fp2, fp3 = B.fingerprint("T-C-PANEL_STUCK", spec1), B.fingerprint("T-C-PANEL_STUCK", spec2), B.fingerprint("T-C-PANEL_STUCK", spec3)
    assert fp1 == fp2, "same params, different seed -> same fingerprint (this is the whole point)"
    assert fp1 != fp3, "different gold-changing param -> different fingerprint"


def test_admit_episode_rejects_route_budget_safeguard_violation():
    ex = CouchDBExecutor()
    rec = B.admit_episode(ex, "T-C-ROUTE_BUDGET", {"asset": "chiller_6", "battery_pct": 90.0, "seed": 1})
    assert rec["admitted"] is False
    assert rec["episode_id"] is None
    assert any("R024 safeguard" in r["reason"] for r in rec["rejections"])


def test_admit_episode_produces_full_provenance_on_success():
    ex = CouchDBExecutor()
    rec = B.admit_episode(ex, "T-C-PANEL_STUCK", {"asset": "chiller_6", "panel_stuck": True, "seed": 1})
    assert rec["admitted"] is True
    required_fields = ["episode_id", "template_id", "family", "source_scenario_id", "asset",
                       "world_seed", "world_hash", "canonical_world_id", "parameterization",
                       "generator_version", "schema_version", "gold_terminal_action",
                       "required_actions", "forbidden_actions", "precedence_pairs",
                       "trace_events", "trace_chain_valid", "score", "fingerprint"]
    for f in required_fields:
        assert f in rec, f"missing provenance field: {f}"
    assert rec["source_scenario_id"] == "R001"


def test_bulk_grid_produces_zero_duplicate_fingerprints():
    """The actual claim made in the freeze report -- re-verified here as a
    fast regression rather than only trusted from the one-off script run."""
    ex = CouchDBExecutor()
    from collections import Counter
    grid = B.full_grid()
    fps = []
    for tid, params in grid:
        rec = B.admit_episode(ex, tid, params)
        assert rec["admitted"], f"{tid}/{params}: unexpectedly rejected"
        fps.append(rec["fingerprint"])
    dups = {fp: c for fp, c in Counter(fps).items() if c > 1}
    assert not dups, f"unexpected duplicate fingerprints: {dups}"


def test_all_four_assets_equally_represented_in_full_grid():
    grid = B.full_grid()
    from collections import Counter
    assets = Counter(params["asset"] for _, params in grid)
    assert set(assets) == set(ALL_ASSETS)
    assert len(set(assets.values())) == 1, f"asset counts not balanced: {assets}"
