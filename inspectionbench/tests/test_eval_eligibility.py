"""Tests for eval_eligibility.py -- must compute from actual manifest
files, never assume every canonical episode is model-evaluable."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

import eval_eligibility as EE  # noqa: E402


def test_total_canonical_is_4075():
    r = EE.compute_eligibility()
    assert r.total_canonical == 4075


def test_eligible_total_is_229_not_4075():
    """The core non-negotiable: eligibility != canonical count."""
    r = EE.compute_eligibility()
    assert r.eligible_total == 229
    assert r.eligible_total != r.total_canonical


def test_eligible_pools_sum_to_eligible_total():
    r = EE.compute_eligibility()
    assert sum(r.eligible_by_pool.values()) == r.eligible_total


def test_not_eligible_total_is_computed_not_hardcoded():
    r = EE.compute_eligibility()
    assert r.not_eligible_total == r.total_canonical - r.eligible_total
    assert r.not_eligible_total == 3846


def test_not_eligible_reason_names_the_scripted_artifact_problem():
    r = EE.compute_eligibility()
    assert "TDA/GSR" in r.not_eligible_reason
    assert "no model has ever been executed" in r.not_eligible_reason


def test_eligible_by_pool_matches_the_three_known_runnable_manifests():
    r = EE.compute_eligibility()
    assert r.eligible_by_pool == {
        "frozen_93_A_B_C_D_E": 93,
        "b_acquisition_66": 66,
        "d_physical_70": 70,
    }
