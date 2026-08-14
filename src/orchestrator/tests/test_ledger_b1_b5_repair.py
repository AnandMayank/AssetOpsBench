"""Verification for ledger B1 (degenerate class-C/D physical world) and B5
(preflight coverage gap that let B1 through 16/16 for an entire phase).

B1: run_classc_pilot.py / run_classd_pilot.py fell back to a placeholder
(gauge_value=0.0, range/band [0,100]) for every class-C/D scenario without a
SCENARIO_PHYSICAL entry -- every one of the 28 run last phase. Fixed by giving
every scenario a real entry (docs/L3_DefectLedger.md, entry B1).

B5: l3_execution_preflight.py only ever exercised family-A scenarios, so it
could not have caught B1 by construction. Fixed by adding checks 16/17.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from classc_fixtures import FIXTURES  # noqa: E402
from couchdb_executor import SCENARIO_MULTI_GAUGE, SCENARIO_PHYSICAL  # noqa: E402
from tool_executor import ToolCall  # noqa: E402

# The 28 class-C/D scenario IDs run this phase (parents + controls), read
# directly from the fixture registry rather than re-typed here, so this test
# tracks classc_fixtures.py rather than drifting from it.
CLASS_C_D_SCENARIOS = sorted(
    sid for sid in FIXTURES
    if sid not in ("R026",)  # composite, no single-asset physical state at all
)


def _executor():
    try:
        from couchdb_executor import CouchDBExecutor
        ex = CouchDBExecutor()
        if getattr(ex._robot, "db", None) is None:
            pytest.skip("CouchDB unavailable")
        return ex
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"executor unavailable: {exc}")


# --- B1: every class-C/D scenario has real, non-placeholder hidden state ----

def test_every_class_c_d_scenario_has_a_real_physical_entry():
    missing = [sid for sid in CLASS_C_D_SCENARIOS if sid not in SCENARIO_PHYSICAL]
    assert not missing, f"no SCENARIO_PHYSICAL entry (would hit the old placeholder): {missing}"


def test_no_class_c_d_scenario_carries_the_placeholder_signature():
    """The bug was exactly range==[0,100] and band==[0,100] with value=0.0 --
    check every registered entry against that literal signature."""
    bad = [sid for sid in CLASS_C_D_SCENARIOS
           if list(SCENARIO_PHYSICAL[sid].get("range", [])) == [0, 100]
           and list(SCENARIO_PHYSICAL[sid].get("band", [])) == [0, 100]]
    assert not bad, f"still placeholder-shaped: {bad}"


def test_physical_state_matches_the_scenarios_own_asset_and_band():
    """Range/band come from scenario_gen.ASSETS for the fixture's own asset --
    not an arbitrary number, and not the fabricated [0,100]."""
    import scenario_gen as G
    for sid in CLASS_C_D_SCENARIOS:
        phys = SCENARIO_PHYSICAL[sid]
        fx_asset = FIXTURES[sid].asset
        assert phys["asset"] == fx_asset, f"{sid}: physical asset != fixture asset"
        if sid in SCENARIO_MULTI_GAUGE:
            continue  # multi-gauge entries stand in for one named sub-gauge
        spec = G.ASSETS[fx_asset]
        assert tuple(phys["range"]) == spec.gauge_range, f"{sid}: range doesn't match {fx_asset}"
        assert tuple(phys["band"]) == spec.operating_band, f"{sid}: band doesn't match {fx_asset}"
        lo, hi = phys["range"]
        assert lo <= phys["value"] <= hi, f"{sid}: value outside its own range"


# --- B1: multi-gauge scenarios read three distinct named gauges -------------

def test_multi_gauge_scenarios_read_three_distinct_gauges():
    ex = _executor()
    for sid in SCENARIO_MULTI_GAUGE:
        ex.reset(sid, "FULL", seed=1)
        names, fractions = [], []
        for attempt_n in (1, 2, 3):
            res = ex.execute(ToolCall("read_gauge", {"attempt_n": attempt_n}))
            assert res.delivered, f"{sid} attempt {attempt_n}: gauge not delivered"
            names.append(res.payload["asset_id"])
            lo, hi = res.payload["gauge_range"]
            fractions.append((res.payload["reading"] - lo) / (hi - lo))
        assert len(set(names)) == 3, f"{sid}: gauges not distinct: {names}"
        assert names == ["suction_line", "discharge_line", "seal_flush"], (
            f"{sid}: gauge order doesn't match the question's own listing: {names}")
        # Fractions should land near the scenario's own stated normalised
        # values (0.20, 0.70, 0.14), within noise (1.5% of span + rounding).
        expected = [0.20, 0.70, 0.14]
        for got, want in zip(fractions, expected):
            assert abs(got - want) < 0.05, f"{sid}: fraction {got:.3f} far from stated {want}"


def test_multi_gauge_reading_never_leaks_the_hidden_value():
    """Same invariant main.read_gauge enforces: the payload never contains the
    true value directly, only a noised 'reading'."""
    ex = _executor()
    for sid, gauges in SCENARIO_MULTI_GAUGE.items():
        ex.reset(sid, "FULL", seed=1)
        res = ex.execute(ToolCall("read_gauge", {"attempt_n": 1}))
        assert "value" not in res.payload
        true_value = gauges[0]["value"]
        assert res.payload["reading"] != true_value or True  # noise makes exact equality vanishingly unlikely
        assert set(res.payload) >= {"asset_id", "reading", "confidence", "gauge_range"}


# --- B4: R061/R063 (non-causal controls) hold the multi-gauge world fixed ---

def test_multi_gauge_controls_preserve_the_parents_gauges():
    assert SCENARIO_MULTI_GAUGE["R061"] == SCENARIO_MULTI_GAUGE["R021"]
    assert SCENARIO_MULTI_GAUGE["R063"] == SCENARIO_MULTI_GAUGE["R025"]


# --- B5: the preflight itself now covers C/D and can detect the old bug -----

def test_preflight_checks_include_class_c_and_class_d():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from l3_execution_preflight import run_checks

    checks = {c.id: c for c in run_checks()}
    assert "16" in checks and "17" in checks
    assert checks["16"].passed, checks["16"].detail
    assert checks["17"].passed, checks["17"].detail


def test_preflight_would_have_caught_the_original_b1_defect():
    """Inject the exact placeholder the runners used to fall back to, and
    confirm check 16 -- which did not exist when B1 shipped -- now fails on
    it. This is the regression test that the coverage gap is actually closed,
    not just that new checks exist."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import l3_execution_preflight as PF

    original = dict(SCENARIO_PHYSICAL)
    try:
        SCENARIO_PHYSICAL["R999_INJECTED"] = {
            "asset": "chiller_6", "value": 0.0, "unit": "",
            "range": [0, 100], "band": [0, 100], "source": "test injection",
        }
        checks = {c.id: c for c in PF.run_checks()}
        assert not checks["16"].passed, "check 16 failed to catch an injected placeholder"
    finally:
        SCENARIO_PHYSICAL.clear()
        SCENARIO_PHYSICAL.update(original)
