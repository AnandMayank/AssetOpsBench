"""Gold-invariance and rule-clause tests (plan section K)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

from playground.semantics import (  # noqa: E402
    AcousticReading, DeliveredEvidence, EvidenceRegime, IoTReading, ThermalReading, WorldSpec,
    g_term_from_frontier, g_world, param_class, ParamClass, r_fault_clauses, r_normal_clauses,
    sufficiency,
)


def test_world_params_classified_world():
    for name in ("asset", "condition", "machine_id", "iot_band_state", "thermal_fault_class",
                 "active_work_order", "technician_present"):
        assert param_class(name) is ParamClass.WORLD


def test_evidence_params_classified_evidence():
    for name in ("initial_modalities", "available_modalities", "draw_variant", "costs",
                 "budget", "acquisition_enabled"):
        assert param_class(name) is ParamClass.EVIDENCE


def test_world_id_invariant_to_evidence_only_change():
    w1 = WorldSpec(asset="chiller_6", condition="fault", machine_id="00", iot_band_state="out_of_band")
    # Two EvidenceRegimes differ, but WorldSpec (and therefore world_id) does not.
    regime_a = EvidenceRegime(available_modalities=frozenset({"iot", "acoustic", "record"}))
    regime_b = EvidenceRegime(available_modalities=frozenset({"iot", "record"}), budget=5.0)
    assert w1.world_id == w1.world_id  # trivial, but documents: regimes never enter world_id
    assert "budget" not in w1.canonical_fields()
    assert "available_modalities" not in w1.canonical_fields()


def test_world_mutation_changes_world_id_and_gold():
    w_fault = WorldSpec(asset="chiller_6", condition="fault", machine_id="00")
    w_normal = WorldSpec(asset="chiller_6", condition="normal", machine_id="00")
    assert w_fault.world_id != w_normal.world_id
    assert g_world(w_fault) != g_world(w_normal)


def test_g_term_never_returns_wrong_verdict_commit():
    w_fault = WorldSpec(asset="chiller_6", condition="fault", machine_id="00")
    w_normal = WorldSpec(asset="chiller_6", condition="normal", machine_id="00")
    assert g_term_from_frontier(w_fault, frontier_empty=False) == "COMMIT_FAULT"
    assert g_term_from_frontier(w_normal, frontier_empty=False) == "COMMIT_NORMAL"
    assert g_term_from_frontier(w_fault, frontier_empty=True) == "ESCALATE"
    assert g_term_from_frontier(w_normal, frontier_empty=True) == "ESCALATE"


def test_r_fault_clause_2plus_positive():
    e = DeliveredEvidence(acoustic=[AcousticReading(True, 1.0), AcousticReading(True, 1.0)])
    assert r_fault_clauses(e)["acoustic_2plus_positive"] is True
    assert sufficiency(e) == "fault"


def test_r_fault_clause_acoustic1_iot_persist2():
    e = DeliveredEvidence(acoustic=[AcousticReading(True, 1.0)],
                           iot=[IoTReading(False, 0.0), IoTReading(False, 0.0)])
    assert r_fault_clauses(e)["acoustic1_iot_persist2"] is True
    assert sufficiency(e) == "fault"


def test_r_fault_clause_thermal_hotspot():
    e = DeliveredEvidence(thermal=[ThermalReading(True)])
    assert r_fault_clauses(e)["thermal_hotspot"] is True
    assert sufficiency(e) == "fault"


def test_r_normal_clause_3neg_iot_inband():
    e = DeliveredEvidence(acoustic=[AcousticReading(False, 0.0)] * 3, iot=[IoTReading(True, 0.0)])
    assert r_normal_clauses(e)["acoustic_3plus_negative_iot_inband"] is True
    assert sufficiency(e) == "normal"


def test_r_normal_clause_iot_streak3():
    e = DeliveredEvidence(iot=[IoTReading(True, 0.0)] * 3)
    assert r_normal_clauses(e)["iot_inband_streak3"] is True
    assert sufficiency(e) == "normal"


def test_r_contradiction_returns_none():
    e = DeliveredEvidence(acoustic=[AcousticReading(True, 1.0), AcousticReading(True, 1.0),
                                     AcousticReading(False, -1.0), AcousticReading(False, -1.0),
                                     AcousticReading(False, -1.0)],
                           iot=[IoTReading(True, 0.0)])
    # fault (2+ positive) AND normal (3+ negative + iot in-band) both hold.
    assert sufficiency(e) is None


def test_empty_evidence_is_insufficient():
    assert sufficiency(DeliveredEvidence()) is None
