"""Tests for the V3 paired-statistics layer.

The exact McNemar implementation is checked against scipy where available, so a
hand-rolled binomial cannot drift from the reference without failing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.paired_stats import (  # noqa: E402
    align, describe, hierarchical_bootstrap_ci, holm_correct, mcnemar,
    min_pairs_for_effect, paired_bootstrap_ci,
)


# --------------------------------------------------------------------------
# McNemar
# --------------------------------------------------------------------------

def test_mcnemar_counts_discordant_pairs():
    a = [True, True, False, False, True]
    b = [True, False, True, False, False]
    r = mcnemar(a, b)
    assert (r.b, r.c, r.n_discordant) == (2, 1, 3)
    assert r.n_pairs == 5


def test_mcnemar_identical_conditions_is_not_significant():
    a = b = [True, False, True, True, False]
    r = mcnemar(a, b)
    assert r.n_discordant == 0 and r.p_value == 1.0
    assert not r.significant and r.delta == 0.0


def test_mcnemar_detects_a_total_flip():
    a = [False] * 12
    b = [True] * 12
    r = mcnemar(a, b)
    assert r.c == 12 and r.b == 0
    assert r.significant and r.delta == pytest.approx(1.0)


def test_mcnemar_matches_scipy_exact():
    scipy_stats = pytest.importorskip("scipy.stats")
    a = [True] * 9 + [False] * 11 + [True] * 5 + [False] * 7
    b = [True] * 9 + [True] * 11 + [False] * 5 + [False] * 7
    r = mcnemar(a, b)
    # Exact McNemar == two-sided binomial on the discordant pairs.
    expected = scipy_stats.binomtest(max(r.b, r.c), r.n_discordant, 0.5).pvalue
    assert r.p_value == pytest.approx(expected, rel=1e-9)


def test_mcnemar_rejects_unpaired_input():
    with pytest.raises(ValueError):
        mcnemar([True, False], [True])
    with pytest.raises(ValueError):
        mcnemar([], [])


def test_concordant_pairs_do_not_change_the_p_value():
    """The information lives entirely in the discordant pairs."""
    a = [True, False]
    b = [False, True]
    p_small = mcnemar(a, b).p_value
    p_padded = mcnemar(a + [True] * 50, b + [True] * 50).p_value
    assert p_small == pytest.approx(p_padded)


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------

def test_bootstrap_ci_brackets_the_point_estimate():
    a = [0.0] * 20 + [1.0] * 5
    b = [1.0] * 20 + [1.0] * 5
    ci = paired_bootstrap_ci(a, b, resamples=2000)
    assert ci.lo <= ci.point <= ci.hi
    assert ci.point == pytest.approx(0.8)
    assert ci.excludes_zero


def test_bootstrap_ci_includes_zero_for_no_effect():
    vals = [0.0, 1.0] * 25
    ci = paired_bootstrap_ci(vals, vals, resamples=2000)
    assert ci.point == 0.0 and ci.lo <= 0 <= ci.hi
    assert not ci.excludes_zero


def test_bootstrap_is_reproducible_under_a_fixed_seed():
    a = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    b = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    x = paired_bootstrap_ci(a, b, resamples=500, seed=7)
    y = paired_bootstrap_ci(a, b, resamples=500, seed=7)
    assert (x.lo, x.hi) == (y.lo, y.hi)


def test_bootstrap_supports_a_custom_statistic():
    """MAE, not just a mean of indicators."""
    a = [1.0, 2.0, 3.0, 4.0]
    b = [2.0, 3.0, 4.0, 5.0]
    ci = paired_bootstrap_ci(a, b, statistic=lambda xs: max(xs), resamples=500)
    assert ci.point == pytest.approx(1.0)


def test_hierarchical_bootstrap_is_wider_than_flat():
    """With variance concentrated between families, ignoring nesting
    understates uncertainty."""
    a, b, fam = {}, {}, {}
    for i in range(20):
        sid = f"s{i}"
        family = "F1" if i < 10 else "F2"
        a[sid], fam[sid] = 0.0, family
        b[sid] = 1.0 if family == "F1" else 0.0
    ids, av, bv = align(a, b)
    flat = paired_bootstrap_ci(av, bv, resamples=3000)
    nested = hierarchical_bootstrap_ci(a, b, fam, resamples=3000)
    assert (nested.hi - nested.lo) > (flat.hi - flat.lo)


# --------------------------------------------------------------------------
# Alignment, correction, power
# --------------------------------------------------------------------------

def test_align_drops_unmatched_and_null_scenarios():
    a = {"s1": 1.0, "s2": 2.0, "s3": None, "s4": 4.0}
    b = {"s1": 1.5, "s3": 3.0, "s4": None, "s5": 5.0}
    ids, av, bv = align(a, b)
    assert ids == ["s1"] and av == [1.0] and bv == [1.5]


def test_holm_is_more_conservative_than_raw_but_ordered():
    out = holm_correct({"x": 0.01, "y": 0.04, "z": 0.20})
    assert out["x"]["p_adjusted"] >= 0.01
    assert out["x"]["p_adjusted"] <= out["y"]["p_adjusted"] <= out["z"]["p_adjusted"]
    assert out["z"]["significant"] is False


def test_min_pairs_is_monotone_in_effect_and_discordance():
    assert min_pairs_for_effect(20) < min_pairs_for_effect(10)
    # More disagreement between conditions => more pairs needed.
    assert min_pairs_for_effect(10, discordance=0.5) > min_pairs_for_effect(10, discordance=0.2)


def test_min_pairs_reproduces_connor_worked_value():
    """pi_d=0.3, delta=0.10, alpha=0.05, power=0.8 -> ~233 pairs."""
    assert min_pairs_for_effect(10, discordance=0.3) == pytest.approx(233, abs=3)


def test_plan_rule_of_thumb_only_holds_at_low_discordance():
    """The Rev-2 plan's '>=100 pairs for 10 pp' assumed conditions largely
    agree; it does not hold at moderate discordance."""
    assert min_pairs_for_effect(10, discordance=0.13) == pytest.approx(100, abs=8)
    assert min_pairs_for_effect(10, discordance=0.3) > 200


def test_effect_cannot_exceed_sqrt_discordance():
    with pytest.raises(ValueError):
        min_pairs_for_effect(60, discordance=0.3)


def test_detectable_effect_inverts_min_pairs():
    from evaluation.paired_stats import detectable_effect_pp
    eff = detectable_effect_pp(184, discordance=0.3)
    assert min_pairs_for_effect(eff, discordance=0.3) <= 184
    # E3's 184 paired scenarios cannot resolve 10 pp at this discordance.
    assert eff > 10


def test_describe_returns_serialisable_output():
    import json
    out = describe("baseline vs informed", [False] * 10, [True] * 10)
    json.dumps(out)
    assert out["mcnemar"]["significant"] is True
