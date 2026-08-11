"""Regression tests for the P0-1 dataset repoint and the V7 label gate.

These lock in properties that, when they silently broke, produced results that
looked fine: a catalog where every query is unreadable scores a policy that
always abstains as perfect, and a κ computed on a biased stratum can condemn an
entire corpus.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluation.label_status import (  # noqa: E402
    CONTESTED, PENDING, TRUSTED, LabelStatus,
)
from orchestrator import pmc_dataset  # noqa: E402

FULL_CSV = pmc_dataset.FULL_PERCEPTION_CSV
FULL_PAIRS = pmc_dataset.FULL_PAIRS_CSV
STATUS_CSV = REPO_ROOT / "reports" / "v7" / "label_status.csv"

needs_full = pytest.mark.skipif(
    not (FULL_CSV.exists() and FULL_PAIRS.exists()),
    reason="full release catalog / generated pairs table not present",
)


def _rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------
# P0-1: the catalog must admit both readable and unreadable query scenarios.
# --------------------------------------------------------------------------

@needs_full
def test_paired_queries_have_both_readable_and_unreadable():
    """The dev subset's queries are *all* unreadable, which makes L1 accuracy
    undefined and lets unconditional abstention score perfectly. The release
    catalog must not have that property."""
    catalog = {r["scenario_id"]: r for r in _rows(FULL_CSV)}
    readable = [
        catalog[p["scenario_id"]]["gauge_readable"].strip().lower()
        for p in _rows(FULL_PAIRS)
        if p["scenario_id"] in catalog
    ]
    assert readable, "no paired queries resolved against the catalog"
    n_true, n_false = readable.count("true"), readable.count("false")
    assert n_true > 0 and n_false > 0, (
        f"degenerate readability GT: {n_true} readable / {n_false} unreadable"
    )
    minority = min(n_true, n_false) / len(readable)
    assert minority > 0.10, f"readability GT is {minority:.1%} minority - near-degenerate"


@needs_full
def test_decision_axis_is_not_collapsed():
    """If nearly every scenario resolves to one action, the leaderboard measures
    a class prior rather than a decision."""
    catalog = {r["scenario_id"]: r for r in _rows(FULL_CSV)}
    actions = [
        catalog[p["scenario_id"]]["recommended_action"].strip()
        for p in _rows(FULL_PAIRS) if p["scenario_id"] in catalog
    ]
    majority = max(actions.count(a) for a in set(actions)) / len(actions)
    assert majority < 0.75, (
        f"majority-class baseline would score {majority:.1%} - decision axis collapsed"
    )


def test_pair_builder_reproduces_shipped_table():
    """The reverse-engineered matching rule is only trustworthy if it recovers
    the hand-built pairs.csv exactly, tier and timestamp delta included."""
    r = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "build_pairs.py"), "--verify"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_scenario_carries_instrument_status():
    """Readability and serviceability are independent: a legible dial on a
    decommissioned instrument must still not be committed."""
    sc = pmc_dataset.RealScenario(
        scenario_id="X", category="c", asset="a", location="l", description="d",
        query_image=Path("q.jpg"), reference_image=Path("r.jpg"),
        iot_value=1.0, iot_value_raw="1.0", gauge_readable_gt=True,
        gauge_value_gt=1.0, gauge_range_min=0.0, gauge_range_max=2.0,
        gauge_unit="MPa", recommended_action="CLEAN_GAUGE",
        forbidden_actions=[], trap="",
    )
    assert sc.instrument_status == "unknown"
    assert sc.to_dict()["instrument_status"] == "unknown"


# --------------------------------------------------------------------------
# V7: the label gate must distinguish "found a defect" from "never looked".
# --------------------------------------------------------------------------

@pytest.mark.skipif(not STATUS_CSV.exists(), reason="V7 audit has not been run")
def test_biased_stratum_kappa_does_not_condemn_the_corpus():
    """The conflict stratum is, by construction, where disagreement surfaced.
    Projecting its κ corpus-wide would mark every scenario contested and block
    the benchmark on an estimate that was never valid for it."""
    rows = _rows(STATUS_CSV)
    contested = [r for r in rows if r["status"] == CONTESTED]
    assert len(contested) / len(rows) < 0.05, (
        f"{len(contested)}/{len(rows)} pairs contested - biased stratum is "
        "likely being projected onto the corpus"
    )
    assert {r["status"] for r in rows} <= {TRUSTED, CONTESTED, PENDING}


@pytest.mark.skipif(not STATUS_CSV.exists(), reason="V7 audit has not been run")
def test_contested_labels_are_excluded_from_metrics():
    ls = LabelStatus.load(STATUS_CSV)
    assert ls.registry_present
    contested = [s for (s, f), v in ls.status.items()
                 if f == "gauge_value" and v == CONTESTED]
    assert contested, "expected known gauge_value conflicts"
    assert not set(ls.eligible("gauge_value", contested))


def test_missing_registry_is_not_silently_treated_as_clean():
    """An unaudited corpus must not read as an audited one."""
    ls = LabelStatus.load(Path("/nonexistent/label_status.csv"))
    assert ls.registry_present is False
    assert ls.reportable("gauge_value", ["A", "B"]) is False
    assert "NOT REPORTABLE" in ls.coverage_note("gauge_value", ["A", "B"])


@pytest.mark.skipif(not STATUS_CSV.exists(), reason="V7 audit has not been run")
def test_coverage_note_separates_contested_from_pending():
    ls = LabelStatus.load(STATUS_CSV)
    ids = [r["scenario_id"] for r in _rows(FULL_CSV)] if FULL_CSV.exists() else []
    if not ids:
        pytest.skip("full catalog absent")
    note = ls.coverage_note("gauge_value", ids)
    assert "contested" in note and "pending" in note, note


# --------------------------------------------------------------------------
# Splits: leak safety and the two split-C views.
# --------------------------------------------------------------------------

SPLITS_JSON = REPO_ROOT / "config" / "splits" / "splits.json"


def _splits():
    import json
    if not SPLITS_JSON.exists():
        pytest.skip("run scripts/build_splits.py first")
    return json.loads(SPLITS_JSON.read_text())


def test_no_image_appears_in_two_splits():
    """Reference photographs are reused across scenarios (one serves up to 20),
    so splitting by scenario id leaks the same image into test and regenerated."""
    assert _splits()["leak_safety"]["cross_split_image_leaks"] == 0


def test_splits_are_disjoint():
    m = _splits()["splits"]
    sets = {
        "A": set(m["A_development"]["ids"]),
        "B": set(m["B_pilot"]["ids"]),
        "C": set(m["C_test"]["view_natural_prior"]),
        "D": set(m["D_regenerated"]["ids"]),
    }
    for x, y in [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]:
        assert not sets[x] & sets[y], f"{x} and {y} overlap"


def test_contested_scenarios_are_kept_out_of_the_test_set():
    m = _splits()["splits"]
    dev = set(m["A_development"]["ids"])
    assert dev, "development split should hold the dual-labelled scenarios"
    assert not dev & set(m["C_test"]["view_natural_prior"])


def test_balanced_view_actually_reduces_the_majority_baseline():
    """A 'balanced' view that raises the class prior is not balanced. Capping
    (category, decision) cells uniformly does exactly that here."""
    cov = _splits()["coverage"]
    assert (cov["C_test_balanced"]["majority_class_baseline"]
            <= cov["C_test_natural"]["majority_class_baseline"])


def test_both_split_c_views_are_published():
    """The gap between them is the measurement of prior-exploitation, so
    neither may be dropped."""
    c = _splits()["splits"]["C_test"]
    assert c["view_natural_prior"] and c["view_balanced"]


def test_test_split_has_both_readability_classes():
    r = _splits()["coverage"]["C_test_natural"]["gauge_readable"]
    assert r["true"] > 0 and r["false"] > 0
    assert min(r["true"], r["false"]) / (r["true"] + r["false"]) > 0.2
