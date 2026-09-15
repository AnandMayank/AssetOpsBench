"""Tests for scripts/phase8h2j_v3_primary_panel_pilot.py's episode-subset
selection -- zero API calls. Guards the exact bug found and fixed during
the primary-panel pilot: sorted(dict.items()) yields (key, value) tuples,
and E's real execution unit is a 3-episode SEQUENCE, not 3 independent
episodes."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pytest  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "reports" / "ec" / "phase8h1_pilot_manifest.json"
pytestmark = pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="frozen-93 manifest not present")

import phase8h2j_v3_primary_panel_pilot as P  # noqa: E402


def test_pick_frozen93_subset_returns_a_c_e_keys():
    subset = P.pick_frozen93_subset(n_per_dim=2)
    assert set(subset.keys()) == {"A", "C", "E"}


def test_pick_frozen93_subset_e_picks_distinct_sequences_not_same_sequence_twice():
    """The exact bug found live: sorting E rows by episode_id and taking
    the first N picks ep0+ep1 of the SAME sequence (E::SEQ-5000::ep0,
    E::SEQ-5000::ep1), which would call run_sequence() twice for
    identical real API work. Each picked row must belong to a distinct
    sequence."""
    subset = P.pick_frozen93_subset(n_per_dim=2)
    seq_ids = [r["episode_id"].split("::")[1] for r in subset["E"]]
    assert len(seq_ids) == len(set(seq_ids)), f"duplicate sequence picked: {seq_ids}"


def test_pick_frozen93_subset_is_deterministic_across_calls():
    a = P.pick_frozen93_subset(n_per_dim=2)
    b = P.pick_frozen93_subset(n_per_dim=2)
    for dim in ("A", "C", "E"):
        assert [r["episode_id"] for r in a[dim]] == [r["episode_id"] for r in b[dim]]


def test_run_b_subset_and_run_d_subset_select_real_episode_dicts_not_dict_keys():
    """The exact bug found live: `[v[0] for v in sorted(by_t.items())]`
    yields the dict KEY (a template_id string), not the episode dict --
    causing a TypeError deep inside run_model. Guard both call sites."""
    import inspect
    src_b = inspect.getsource(P.run_b_subset)
    src_d = inspect.getsource(P.run_d_subset)
    assert "for _, v in sorted" in src_b, "run_b_subset must unpack (key, value), not just value"
    assert "for _, v in sorted" in src_d, "run_d_subset must unpack (key, value), not just value"
