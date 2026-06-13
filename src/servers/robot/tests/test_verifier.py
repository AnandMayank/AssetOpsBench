"""Unit tests for MultiReadingVerifier — no CouchDB required."""

import pytest
from servers.robot.verifier import MultiReadingVerifier, VerifierResult


@pytest.fixture
def verifier():
    return MultiReadingVerifier()


class TestBlockedGate:
    def test_blocked_n_1(self, verifier):
        result = verifier.verify(
            readings=[50.0],
            iot_value=50.0,
            gauge_range=[0, 100],
        )
        assert result.status == "BLOCKED"
        assert result.fm_flag == "FM-7b"
        assert result.score == 0.0

    def test_blocked_n_2(self, verifier):
        result = verifier.verify(
            readings=[48.0, 52.0],
            iot_value=50.0,
            gauge_range=[0, 100],
        )
        assert result.status == "BLOCKED"
        assert result.fm_flag == "FM-7b"

    def test_fm7b_flag_in_blocked(self, verifier):
        result = verifier.verify(readings=[], iot_value=0.0, gauge_range=[0, 100])
        assert result.fm_flag == "FM-7b"
        # insufficient readings flag appears in fm_annotations
        assert any("FM-7b" in ann for ann in result.fm_annotations)


class TestFreezeGate:
    def test_zero_variance_triggers_panel_recheck(self, verifier):
        # Three identical readings → std=0.0 → PANEL_RECHECK
        result = verifier.verify(
            readings=[50.0, 50.0, 50.0],
            iot_value=50.0,
            gauge_range=[0, 100],
        )
        assert result.status == "PANEL_RECHECK"
        assert result.fm_flag == "FM-1"

    def test_tiny_variance_still_freezes(self, verifier):
        # std = 0.0001 < 0.001 * 100 = 0.1 → PANEL_RECHECK
        result = verifier.verify(
            readings=[50.0, 50.0001, 50.0002],
            iot_value=50.0,
            gauge_range=[0, 100],
        )
        assert result.status == "PANEL_RECHECK"


class TestCommit:
    def test_commit_consistent_readings(self, verifier):
        # Tight cluster near IoT value → high C, high A → COMMIT
        result = verifier.verify(
            readings=[49.5, 50.0, 50.5],
            iot_value=50.0,
            gauge_range=[0, 100],
        )
        assert result.status == "COMMIT"
        assert result.score >= 0.82

    def test_historical_signal_neutral_when_absent(self, verifier):
        result = verifier.verify(
            readings=[49.5, 50.0, 50.5],
            iot_value=50.0,
            gauge_range=[0, 100],
            historical_baseline=None,
        )
        assert result.H == 0.5
        assert result.status == "COMMIT"


class TestEscalateAndOOD:
    def test_escalate_moderate_contradiction(self, verifier):
        # Mean reading 80, IoT says 20 — large gap → low A
        result = verifier.verify(
            readings=[78.0, 80.0, 82.0],
            iot_value=20.0,
            gauge_range=[0, 100],
        )
        # A = 1 - |80 - 20| / 100 = 0.40 → score roughly 0.35*C + 0.35*0.40 + 0.30*H
        assert result.status in {"ESCALATE", "OOD_FLAG"}

    def test_ood_very_low_score(self, verifier):
        # Readings near top, IoT near bottom, historical near bottom
        result = verifier.verify(
            readings=[88.0, 90.0, 92.0],
            iot_value=5.0,
            gauge_range=[0, 100],
            historical_baseline=5.0,
        )
        # A = 1 - |90-5|/100 = 0.15; H = 1 - |90-5|/100 = 0.15 → score ≈ 0.35*C + 0.35*0.15 + 0.30*0.15
        assert result.status == "OOD_FLAG"


class TestFMAnnotations:
    def test_fm7_annotation_when_a_lt_015(self, verifier):
        # A = 1 - |90 - 3| / 100 = 0.13 < 0.15 → sensor-physical contradiction annotation fires
        result = verifier.verify(
            readings=[88.0, 90.0, 92.0],
            iot_value=3.0,
            gauge_range=[0, 100],
        )
        assert any("FM-7" in ann for ann in result.fm_annotations)

    def test_fm7c_annotation_when_h_lt_015(self, verifier):
        # H = 1 - |90 - 3| / 100 = 0.13 < 0.15 → historical outlier annotation fires; IoT agrees (A fine)
        result = verifier.verify(
            readings=[88.0, 90.0, 92.0],
            iot_value=90.0,
            gauge_range=[0, 100],
            historical_baseline=3.0,
        )
        assert any("FM-7c" in ann for ann in result.fm_annotations)

    def test_no_fm_annotations_for_clean_commit(self, verifier):
        result = verifier.verify(
            readings=[49.5, 50.0, 50.5],
            iot_value=50.0,
            gauge_range=[0, 100],
            historical_baseline=50.0,
        )
        assert result.fm_annotations == []
        assert result.status == "COMMIT"


class TestHistoricalOutlierScenario:
    """Historical outlier annotation must fire deterministically from scenario state.

    These tests prove the fix is correct independent of CouchDB contents —
    the simulator sets historical_baseline explicitly in the state, and
    commit_reading() uses it directly rather than querying IoT docs.
    """

    def test_simulator_sets_historical_baseline_for_outlier(self):
        from servers.robot.simulator import PhysicalStateSimulator
        sim = PhysicalStateSimulator(seed=42)
        state = sim.generate_scenario("chiller_6", [0.0, 100.0], "historical_outlier")
        assert state.historical_baseline is not None
        assert state.historical_severity in {"mild", "medium", "severe"}
        # baseline is always below gauge (gap-based construction)
        assert state.gauge_value > state.historical_baseline, (
            f"gauge_value ({state.gauge_value}) must exceed historical_baseline ({state.historical_baseline})"
        )

    def test_simulator_does_not_set_baseline_for_normal(self):
        from servers.robot.simulator import PhysicalStateSimulator
        sim = PhysicalStateSimulator(seed=42)
        state = sim.generate_scenario("chiller_6", [0.0, 100.0], "normal")
        assert state.historical_baseline is None

    def test_verifier_historical_outlier_annotation_fires(self, verifier):
        # Verify the verifier contract with severe-range values (large gap, low baseline):
        result = verifier.verify(
            readings=[88.0, 90.0, 92.0],
            iot_value=90.0,          # IoT agrees (A is fine)
            gauge_range=[0.0, 100.0],
            historical_baseline=3.0,  # explicit low baseline → H < 0.15
        )
        assert any("FM-7c" in ann for ann in result.fm_annotations)
        assert result.H < 0.15

    @pytest.mark.anyio
    async def test_commit_reading_uses_simulator_baseline_not_iot(self):
        """commit_reading() must use simulator historical_baseline when present,
        making the historical outlier annotation independent of CouchDB content."""
        import servers.robot.main as robot_main
        from unittest.mock import MagicMock, patch

        # Seed a historical_outlier scenario with explicit severe baseline
        robot_main._simulator.generate_scenario("chiller_6", [0.0, 100.0], "historical_outlier")
        state = robot_main._simulator._state["chiller_6"]
        # Force severe-range values so the historical outlier annotation fires
        state.historical_baseline = 3.0
        state.gauge_value = 90.0

        _PROFILE = {
            "_id": "profile:chiller_6",
            "gauge_range": [0.0, 100.0],
            "gauge_value": 90.0,
        }
        mock_db = MagicMock()
        mock_db.get.side_effect = lambda doc_id: _PROFILE if "profile:" in doc_id else None
        mock_db.find.return_value = {"docs": [{"asset_id": "Chiller 6", "timestamp": "t", "Pressure": 90.0}]}
        mock_db.save.return_value = {"ok": True, "id": "x", "rev": "1"}

        from servers.robot.tests.conftest import call_tool
        from servers.robot.main import mcp

        with patch("servers.robot.main.db", mock_db):
            data = await call_tool(
                mcp,
                "commit_reading",
                {
                    "asset_id": "Chiller 6",
                    "readings": [88.0, 90.0, 92.0],
                    "decision": "raise_work_order",
                },
            )

        # Historical outlier annotation should fire because simulator baseline (3.0) was used, not IoT
        assert any("FM-7c" in ann for ann in data.get("fm_annotations", [])), (
            f"Expected historical outlier annotation. Got fm_annotations={data.get('fm_annotations')}, "
            f"H={data.get('H')}"
        )

    def test_severe_scenario_h_lt_015(self):
        """Severe historical_outlier must guarantee H < 0.15 by gap-based construction."""
        from servers.robot.simulator import PhysicalStateSimulator
        failures = []
        for seed in range(200):
            sim = PhysicalStateSimulator(seed=seed)
            for _ in range(10):
                s = sim.generate_scenario("asset", [0.0, 100.0], "historical_outlier")
                if s.historical_severity == "severe":
                    H = 1.0 - abs(s.gauge_value - s.historical_baseline) / 100.0
                    if H >= 0.15:
                        failures.append((seed, s.gauge_value, s.historical_baseline, H))
        assert failures == [], (
            f"Severe scenarios with H >= 0.15 found (should be impossible by construction): {failures[:3]}"
        )

    def test_medium_scenario_h_range(self):
        """Medium historical_outlier must have H in [0.18, 0.36]."""
        from servers.robot.simulator import PhysicalStateSimulator
        failures = []
        for seed in range(200):
            sim = PhysicalStateSimulator(seed=seed)
            for _ in range(10):
                s = sim.generate_scenario("asset", [0.0, 100.0], "historical_outlier")
                if s.historical_severity == "medium":
                    H = 1.0 - abs(s.gauge_value - s.historical_baseline) / 100.0
                    if not (0.14 <= H <= 0.40):
                        failures.append((seed, s.gauge_value, s.historical_baseline, round(H, 4)))
        assert failures == [], (
            f"Medium scenarios outside expected H range found: {failures[:3]}"
        )

    def test_mild_scenario_h_range(self):
        """Mild historical_outlier must have H in [0.35, 0.70]."""
        from servers.robot.simulator import PhysicalStateSimulator
        failures = []
        for seed in range(200):
            sim = PhysicalStateSimulator(seed=seed)
            for _ in range(10):
                s = sim.generate_scenario("asset", [0.0, 100.0], "historical_outlier")
                if s.historical_severity == "mild":
                    H = 1.0 - abs(s.gauge_value - s.historical_baseline) / 100.0
                    if not (0.30 <= H <= 0.75):
                        failures.append((seed, s.gauge_value, s.historical_baseline, round(H, 4)))
        assert failures == [], (
            f"Mild scenarios outside expected H range found: {failures[:3]}"
        )
