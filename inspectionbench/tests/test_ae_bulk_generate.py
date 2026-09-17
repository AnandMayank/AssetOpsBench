"""Regression tests for the Phase 8H.2D A/E bulk generation pipeline
(scripts/phase8h1_ae_bulk_generate.py). Zero model/API calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from couchdb_executor import CouchDBExecutor  # noqa: E402
import phase8h1_ae_bulk_generate as B  # noqa: E402


def test_a_strata_covers_full_cell_x_asset_factorial():
    strata = B.a_strata()
    assert len(strata) == 16
    assert len(set(strata)) == 16, "strata must be distinct (cell, asset) pairs"


def test_generate_a_produces_three_regimes_per_world_with_invariant_gold():
    ex = CouchDBExecutor()
    recs = B.generate_a(ex, n_worlds=2)
    assert len(recs) == 6
    by_group = {}
    for r in recs:
        by_group.setdefault(r["matched_group_id"], []).append(r)
    for mg, rows in by_group.items():
        assert len(rows) == 3
        assert len({r["condition_id"] for r in rows}) == 3
        assert len({r["gold_terminal_action"] for r in rows}) == 1


def test_generate_a_admits_scripted_episodes():
    ex = CouchDBExecutor()
    recs = B.generate_a(ex, n_worlds=3)
    assert all(r["admitted"] for r in recs), "scripted-to-gold traces must all pass admission"
    assert all(r["TDA"] == 1 for r in recs)


def test_generate_e_stratifies_by_asset_only_not_process_kind():
    ex = CouchDBExecutor()
    recs = B.generate_e(ex, n_sequences=8)
    assets_seen = {r["asset_id"] for r in recs}
    assert assets_seen <= set(B.ALL_ASSETS)
    assert len(recs) == 8 * 3


def test_generate_e_produces_three_episodes_per_sequence():
    ex = CouchDBExecutor()
    recs = B.generate_e(ex, n_sequences=2)
    by_group = {}
    for r in recs:
        by_group.setdefault(r["matched_group_id"], []).append(r)
    for mg, rows in by_group.items():
        assert len(rows) == 3
        assert len({r["condition_id"] for r in rows}) == 3


def test_generate_e_routes_battery_exhaustion_to_mechanical_not_rejection():
    ex = CouchDBExecutor()
    recs = B.generate_e(ex, n_sequences=6)
    exhausted = [r for r in recs if r.get("mechanical_battery_exhausted")]
    assert exhausted, "expected at least one battery-exhaustion case in 6 sequences x 3 episodes"
    for r in exhausted:
        assert r["admitted"] is True, "mechanical battery exhaustion must not be treated as a rejection"


def test_seed_ranges_disjoint_from_frozen_and_replication_pools():
    a_lo, a_hi = B.A_SEED_BASE, B.A_SEED_BASE + 16 * B.A_DENSITY - 1
    e_lo, e_hi = B.E_SEED_BASE, B.E_SEED_BASE + 4 * B.E_DENSITY - 1
    frozen_original = range(3000, 3016)
    e_replication = range(6000, 6012)
    assert not (a_lo <= 3015 and a_hi >= 3000)
    assert not (e_lo <= 6011 and e_hi >= 6000)
    assert a_hi < e_lo or e_hi < a_lo, "A and E seed ranges must not overlap each other"


def test_fingerprint_excludes_seed_but_distinguishes_condition_and_gold():
    ex = CouchDBExecutor()
    recs = B.generate_a(ex, n_worlds=2)
    admitted = [r for r in recs if r["admitted"]]

    def fp(r):
        import hashlib
        import json
        payload = {"family": r["family"], "asset": r["asset_id"], "gold": r["gold_terminal_action"],
                  "condition_id": r["condition_id"],
                  "params_no_seed": {k: v for k, v in r["parameterization"].items() if k != "seed"}}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]

    same_world_diff_regime = [r for r in admitted if r["world_id"] == admitted[0]["world_id"]]
    fps = {fp(r) for r in same_world_diff_regime}
    assert len(fps) == len(same_world_diff_regime), \
        "different regimes (condition_id) of the SAME world must not fingerprint-collide"
