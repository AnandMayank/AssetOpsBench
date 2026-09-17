"""Regression tests for the Phase 8H.1 P1 E-replication manifest + dry-run gates.
No API calls. Verifies: original pilot untouched, seed disjointness, 3-episode
structure, world-hash determinism, and the dry-run gate chain."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

MANIFEST_PATH = REPO_ROOT / "reports" / "ec" / "phase8h1_e_replication" / "e_replication_manifest.json"
ORIGINAL_MANIFEST = REPO_ROOT / "reports" / "ec" / "phase8h1_pilot_manifest.json"
ORIGINAL_SHA = "d2b48c0b0c9ef19f6c8f6ddd936da098ffd5d0d4023650619f01c1ee043087fe"

MANIFEST = json.loads(MANIFEST_PATH.read_text())


def test_original_pilot_manifest_sha_unchanged():
    got = hashlib.sha256(ORIGINAL_MANIFEST.read_bytes()).hexdigest()
    assert got == ORIGINAL_SHA == MANIFEST["original_pilot_manifest_sha256"]


def test_twelve_new_sequences_thirtysix_episodes():
    assert MANIFEST["n_sequences"] == 12
    assert MANIFEST["n_episodes"] == 36
    assert len(MANIFEST["episodes"]) == 36


def test_new_seeds_disjoint_from_original_six():
    new_seeds = {e["world_seed"] for e in MANIFEST["episodes"]}
    assert new_seeds.isdisjoint(set(MANIFEST["original_E_seeds_excluded"]))
    assert new_seeds.isdisjoint(range(5000, 5006))


def test_every_sequence_has_exactly_three_episodes():
    by_seq = {}
    for e in MANIFEST["episodes"]:
        by_seq.setdefault(e["world_id"], []).append(e)
    assert len(by_seq) == 12
    for wid, rows in by_seq.items():
        idxs = sorted(r["scenario_parameters"]["episode_index"] for r in rows)
        assert idxs == [0, 1, 2], f"{wid}: {idxs}"


def test_world_hashes_are_reproducible_and_distinct():
    from sequence_executor import sample_sequence
    from canonical_identity import normalize_world
    seen = set()
    for e in MANIFEST["episodes"]:
        if e["scenario_parameters"]["episode_index"] != 0:
            continue
        w = sample_sequence(e["world_seed"], n_episodes=3)
        got = normalize_world(w).world_id
        assert got == e["normalized_world_hash"], f"{e['world_id']}: hash drift"
        assert got not in seen, f"{e['world_id']}: duplicate world hash"
        seen.add(got)
    assert len(seen) == 12


def test_gold_derivation_is_a_pure_function_reproducible():
    from sequence_executor import sample_sequence, derive_sequence_gold
    for e in MANIFEST["episodes"]:
        w = sample_sequence(e["world_seed"], n_episodes=3)
        k = e["scenario_parameters"]["episode_index"]
        g = derive_sequence_gold(w, k)["verdict"]
        assert g == e["gold_action"], f"{e['episode_id']}: gold drift {g} != {e['gold_action']}"


def test_dryrun_gates_all_pass_zero_api_calls():
    from phase8h1_e_replication_dryrun_gates import run_all_gates
    res = run_all_gates()
    assert res["all_pass"] is True
    assert res["n_gates_run"] == 4
    assert res["api_calls"] == 0


def test_no_construct_semantics_changed():
    # generator + gold function are the unmodified E primitives, not new ones
    assert MANIFEST["generator"] == "sequence_executor.sample_sequence (unmodified)"
    assert MANIFEST["gold_derivation"] == "sequence_executor.derive_sequence_gold (unmodified)"
    for x in MANIFEST["excluded_construct_changes"]:
        assert isinstance(x, str) and x
