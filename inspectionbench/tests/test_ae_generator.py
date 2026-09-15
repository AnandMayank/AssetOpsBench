"""Regression tests for the Phase 8H.2D A/E scalable generator
(ae_generator.py). Zero model/API calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

from ae_generator import (  # noqa: E402
    ALL_ASSETS, REGIMES, REGIME_WITHHELD, build_a_world, a_episode_specs,
    run_a_episode_scripted, build_e_world, run_e_sequence_scripted, world_hash,
)
from scenario_gen import FACTORIAL, derive_gold  # noqa: E402
from sequence_executor import derive_sequence_gold  # noqa: E402
from couchdb_executor import CouchDBExecutor  # noqa: E402


def test_a_matched_group_has_three_regimes_with_invariant_gold():
    world = build_a_world(seed=1, cell=FACTORIAL[0], asset_id=ALL_ASSETS[0])
    specs = a_episode_specs(world)
    assert {s.regime for s in specs} == set(REGIMES)
    assert len({s.gold for s in specs}) == 1, "gold must be computed once per world, invariant across regimes"
    assert len({s.matched_group_id for s in specs}) == 1


def test_a_gold_matches_derive_gold_unmodified():
    world = build_a_world(seed=42, cell=FACTORIAL[1], asset_id=ALL_ASSETS[1])
    specs = a_episode_specs(world)
    assert specs[0].gold == derive_gold(world).verdict


def test_a_scripted_episode_scores_tda_1_on_every_regime():
    ex = CouchDBExecutor()
    world = build_a_world(seed=7, cell=FACTORIAL[2], asset_id=ALL_ASSETS[2])
    for spec in a_episode_specs(world):
        r = run_a_episode_scripted(ex, spec)
        assert r.TDA == 1, f"regime {spec.regime} scripted-to-gold trace scored TDA={r.TDA}"
        assert r.trace_chain_valid


def test_a_physical_only_regime_withholds_digital_tools():
    ex = CouchDBExecutor()
    world = build_a_world(seed=8, cell=FACTORIAL[0], asset_id=ALL_ASSETS[0])
    spec = [s for s in a_episode_specs(world) if s.regime == "PHYSICAL_ONLY"][0]
    r = run_a_episode_scripted(ex, spec)
    delivered = {e["tool"] for e in r.trace_events if e["stage"] == "OBSERVATION_DELIVERED"}
    assert "read_iot" not in delivered


def test_e_sequence_has_three_episodes_with_shared_sequence_id():
    ex = CouchDBExecutor()
    world = build_e_world(seed=100)
    results = run_e_sequence_scripted(ex, world)
    assert len(results) == 3
    assert all(r.world.sequence_id == world.sequence_id for r in results)


def test_e_gold_matches_derive_sequence_gold_unmodified():
    ex = CouchDBExecutor()
    world = build_e_world(seed=200)
    results = run_e_sequence_scripted(ex, world)
    for r in results:
        assert r.gold == derive_sequence_gold(world, r.episode)["verdict"]


def test_e_scripted_sequence_scores_tda_1_on_every_episode():
    ex = CouchDBExecutor()
    world = build_e_world(seed=300)
    results = run_e_sequence_scripted(ex, world)
    for r in results:
        assert r.TDA == 1
        assert r.trace_chain_valid


def test_e_asset_parameter_is_respected_when_given():
    ex = CouchDBExecutor()
    world = build_e_world(seed=400, asset_id=ALL_ASSETS[0])
    assert world.asset == ALL_ASSETS[0]


def test_world_hash_is_deterministic_and_seed_sensitive():
    payload_a = {"asset": "chiller_6", "seed": 1, "value": 0.5}
    payload_b = {"asset": "chiller_6", "seed": 2, "value": 0.5}
    assert world_hash(payload_a) == world_hash(dict(payload_a))
    assert world_hash(payload_a) != world_hash(payload_b)


def test_regime_withheld_mapping_is_closed_and_covers_all_regimes():
    assert set(REGIME_WITHHELD) == set(REGIMES)
    assert REGIME_WITHHELD["FULL"] == []
    assert "digital" in REGIME_WITHHELD["PHYSICAL_ONLY"]
    assert "physical" in REGIME_WITHHELD["DIGITAL_ONLY"]
