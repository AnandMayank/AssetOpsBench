"""Tests that gold is DERIVED from the world, never supplied to it.

The construct audit found all six pilot scenarios were built backwards from
their labels. These tests make the corrected order structural: if a future edit
threads a verdict into world sampling, a test fails rather than the defect
silently returning.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

from scenario_gen import (  # noqa: E402
    ASSETS, COMMIT, ESCALATE, FACTORIAL, Cell, check_generator_contract,
    derive_gold, generate, sample_world,
)


def test_world_sampler_cannot_receive_a_label():
    """The architectural guarantee, checked by signature inspection."""
    assert check_generator_contract() == []


def test_contract_check_actually_detects_a_violation():
    """A check that never fires proves nothing."""
    def leaky_sampler(seed: int, cell, gold: str = COMMIT):
        return None
    v = check_generator_contract(leaky_sampler)
    assert v and any("label parameters" in x for x in v)


def test_gold_is_a_pure_function_of_the_world():
    for seed in range(20):
        w = sample_world(seed, FACTORIAL[seed % 4])
        assert derive_gold(w).verdict == derive_gold(w).verdict


def test_sampling_is_deterministic_under_seed():
    a = sample_world(11, Cell(True, False), asset_id="chiller_6")
    b = sample_world(11, Cell(True, False), asset_id="chiller_6")
    assert a.to_dict() == b.to_dict()


def test_physical_factor_is_respected_by_the_sampler():
    for seed in range(30):
        for cell in FACTORIAL:
            w = sample_world(seed, cell, asset_id="chiller_6")
            lo, hi = w.operating_band
            inside = lo <= w.physical_value <= hi
            assert inside == cell.physical_in_band, (cell.name, w.physical_value)


def test_telemetry_never_enters_the_safety_rule():
    """Changing only the IoT value must not change gold — the operational point:
    telemetry is a convenience, not the physical fact."""
    w = sample_world(3, Cell(True, True), asset_id="motor_01")
    g1 = derive_gold(w).verdict
    w.iot_value = w.gauge_range[1]      # wildly disagreeing telemetry
    w.history_mean = w.gauge_range[1]
    assert derive_gold(w).verdict == g1


def test_out_of_band_physical_escalates():
    w = sample_world(5, Cell(False, True), asset_id="hydraulic_pump_1")
    w.technician_present = False
    w.active_work_order = False
    assert derive_gold(w).verdict == ESCALATE


def test_in_band_physical_commits_absent_coordination_constraints():
    w = sample_world(5, Cell(True, True), asset_id="hydraulic_pump_1")
    w.technician_present = False
    w.active_work_order = False
    assert derive_gold(w).verdict == COMMIT


def test_coordination_constraint_dominates_an_in_band_reading():
    w = sample_world(7, Cell(True, True), asset_id="chiller_6")
    w.technician_present = True
    g = derive_gold(w)
    assert g.verdict == ESCALATE and "coordination" in g.rule_applied


def test_factorial_pilot_is_balanced_on_world_factors_not_on_gold():
    """Stratification is on the world. Gold balance is an emergent consequence,
    which is the whole distinction from the previous construction."""
    recs = generate(seed=1000, per_cell=3)
    cells = Counter(r["cell"] for r in recs)
    assert len(cells) == 4 and set(cells.values()) == {3}
    # Gold was never an input; verify both verdicts are nonetheless reachable.
    golds = Counter(r["gold"]["verdict"] for r in recs)
    assert golds[COMMIT] > 0 and golds[ESCALATE] > 0


def test_both_verdicts_reachable_within_each_physical_factor_level():
    """Coordination constraints mean an in-band world can still escalate, so
    gold is not a deterministic relabelling of the physical factor alone."""
    recs = generate(seed=4242, per_cell=12)
    in_band = [r for r in recs if r["cell"].startswith("phys_in")]
    verdicts = {r["gold"]["verdict"] for r in in_band}
    assert ESCALATE in verdicts, "coordination path never exercised"


def test_generated_record_lets_a_reviewer_recompute_the_label():
    from scenario_gen import WorldState
    for r in generate(seed=77, per_cell=2):
        w = WorldState(**r["world"])
        assert derive_gold(w).verdict == r["gold"]["verdict"]


def test_every_asset_spec_has_a_band_inside_its_range():
    for spec in ASSETS.values():
        gmin, gmax = spec.gauge_range
        lo, hi = spec.operating_band
        assert gmin <= lo < hi <= gmax
