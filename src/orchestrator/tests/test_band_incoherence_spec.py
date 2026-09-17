"""Offline tests for the PROPOSED band-comparison incoherence check.

The spec is not wired into the evaluator. These tests establish whether it
*would* behave correctly before anyone proposes adopting it — in particular
that it does not fire on correct reasoning, which is how the previous two
detector bugs manifested.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

from band_incoherence_spec import check_band_incoherence  # noqa: E402


def test_fires_on_the_r056_full_case():
    """The response that motivated the spec."""
    r = check_band_incoherence({
        "verdict": "COMMIT",
        "reason": ("The physical gauge needle is approximately 1.2 m3/s, which is a "
                   "fresh reading from the panel and within the gauge range, though "
                   "slightly above the expected operating band.")},
        band=[0.9, 1.1])
    assert r.band_comparison_incoherence is True and r.trigger == "explicit"


def test_fires_numerically_on_the_r056_physical_only_case():
    """Reading correct, band comparison wrong, no explicit admission."""
    r = check_band_incoherence({
        "verdict": "COMMIT",
        "reason": "The fresh physical gauge reading is 1.371 m3/s, which is within "
                  "the expected 0.9-1.1 m3/s band?"},
        band=[0.9, 1.1])
    assert r.band_comparison_incoherence is True
    assert r.trigger == "numeric" and r.stated_value == 1.371


def test_does_not_fire_when_escalating():
    r = check_band_incoherence({
        "verdict": "ESCALATE",
        "reason": "The gauge reads about 267 bar, well above the expected 160-200 "
                  "bar operating band."},
        band=[160, 200])
    assert r.band_comparison_incoherence is False


def test_does_not_fire_on_a_correct_in_band_commit():
    r = check_band_incoherence({
        "verdict": "COMMIT",
        "reason": "The physical gauge was read at 251.37 bar, within the normal "
                  "220-270 bar band."},
        band=[220, 270])
    assert r.band_comparison_incoherence is False


def test_band_endpoints_restated_do_not_trigger():
    """Reasons routinely restate the band; those numbers are not readings."""
    r = check_band_incoherence({
        "verdict": "COMMIT",
        "reason": "Reading is comfortably inside the expected 0.9-1.1 band."},
        band=[0.9, 1.1])
    assert r.band_comparison_incoherence is False


def test_a_reasoned_override_is_not_incoherence():
    """Proceeding on an out-of-band reading with a stated justification is a
    decision, not a contradiction."""
    r = check_band_incoherence({
        "verdict": "COMMIT",
        "reason": "Reading is above the expected band but this is a transient "
                  "during startup and within tolerance."},
        band=[0.9, 1.1])
    assert r.band_comparison_incoherence is False
    assert r.justified_override


def test_unrelated_numbers_do_not_create_a_comparison():
    r = check_band_incoherence({
        "verdict": "COMMIT",
        "reason": "Battery at 85 percent and pose confirmed; gauge within the "
                  "expected 0.9-1.1 band."},
        band=[0.9, 1.1])
    assert r.band_comparison_incoherence is False


def test_absent_band_falls_back_to_explicit_language_only():
    r = check_band_incoherence({
        "verdict": "COMMIT",
        "reason": "Reading is 1.371 which seems fine."}, band=None)
    assert r.band_comparison_incoherence is False
