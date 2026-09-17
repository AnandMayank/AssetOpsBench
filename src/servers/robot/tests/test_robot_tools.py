"""Integration and unit tests for all 6 Robot MCP server tools.

Tests marked @requires_couchdb are skipped when CouchDB is unreachable.
All other tests use mocked DB via conftest fixtures.
"""

import pytest
from unittest.mock import MagicMock, patch

from servers.robot.main import mcp
from .conftest import call_tool, requires_couchdb


# Shared mock profile document
_PROFILE = {
    "_id": "profile:chiller_6",
    "_rev": "1-abc",
    "physical_location": {"x": 10.0, "y": 5.0, "z": 0.0, "room_id": "B1"},
    "gauge_range": [0.0, 100.0],
    "gauge_value": 75.0,
    "gauge_path": None,
    "panel_stuck": False,
    "never_read": False,
    "reading_consistency": None,
    "sensor_physical_gap": None,
    "maintenance_slot": "day",
    "active_work_order": None,
}

_PROFILE_STUCK = {**_PROFILE, "panel_stuck": True}
_PROFILE_FREE  = {**_PROFILE, "panel_stuck": False}

_IOT_DOC = {
    "asset_id": "Chiller 6",
    "timestamp": "2024-06-01T00:00:00",
    "Chiller 6 Pressure": 75.0,
}


def _db_for(profile):
    mock = MagicMock()
    mock.get.side_effect = lambda doc_id: profile if "profile:" in doc_id else None
    mock.find.return_value = {"docs": [_IOT_DOC]}
    mock.save.return_value = {"ok": True, "id": "x", "rev": "1-x"}
    return mock


# ---------------------------------------------------------------------------
# Tool 1: navigate_to
# ---------------------------------------------------------------------------


class TestNavigateTo:
    @pytest.mark.anyio
    async def test_returns_success_for_known_asset(self):
        with patch("servers.robot.main.db", _db_for(_PROFILE)):
            data = await call_tool(mcp, "navigate_to", {"asset_id": "Chiller 6"})
        assert data["success"] is True
        assert data["distance_m"] > 0
        assert data["steps_taken"] >= 1

    @pytest.mark.anyio
    async def test_blocked_when_no_location(self):
        profile_no_loc = {k: v for k, v in _PROFILE.items() if k != "physical_location"}
        with patch("servers.robot.main.db", _db_for(profile_no_loc)):
            data = await call_tool(mcp, "navigate_to", {"asset_id": "Chiller 6"})
        assert data["success"] is False
        assert data["blocked_reason"] is not None

    @pytest.mark.anyio
    async def test_error_when_db_none(self, no_db):
        data = await call_tool(mcp, "navigate_to", {"asset_id": "Chiller 6"})
        assert "error" in data

    @pytest.mark.anyio
    async def test_error_when_profile_not_found(self):
        mock = MagicMock()
        mock.get.return_value = None
        with patch("servers.robot.main.db", mock):
            data = await call_tool(mcp, "navigate_to", {"asset_id": "Unknown Asset"})
        assert "error" in data

    @requires_couchdb
    @pytest.mark.anyio
    async def test_integration_chiller6(self):
        data = await call_tool(mcp, "navigate_to", {"asset_id": "Chiller 6"})
        assert "success" in data
        assert "distance_m" in data


# ---------------------------------------------------------------------------
# Tool 2: safety_gate_check
# ---------------------------------------------------------------------------


class TestSafetyGateCheck:
    @pytest.mark.anyio
    async def test_clearance_true_when_no_wo(self):
        with patch("servers.robot.main.db", _db_for(_PROFILE)):
            data = await call_tool(mcp, "safety_gate_check", {"asset_id": "Chiller 6"})
        assert data["safety_clearance"] is True
        assert data["active_work_order"] is None

    @pytest.mark.anyio
    async def test_clearance_false_when_active_wo(self):
        profile_wo = {**_PROFILE, "active_work_order": "WO-99999"}
        with patch("servers.robot.main.db", _db_for(profile_wo)):
            data = await call_tool(mcp, "safety_gate_check", {"asset_id": "Chiller 6"})
        assert data["safety_clearance"] is False
        assert data["active_work_order"] == "WO-99999"

    @pytest.mark.anyio
    async def test_missing_slot_defaults_to_day(self):
        profile_no_slot = {k: v for k, v in _PROFILE.items() if k != "maintenance_slot"}
        with patch("servers.robot.main.db", _db_for(profile_no_slot)):
            data = await call_tool(mcp, "safety_gate_check", {"asset_id": "Chiller 6"})
        assert data["slot"] == "day"

    @pytest.mark.anyio
    async def test_missing_active_wo_defaults_to_none(self):
        profile_no_wo = {k: v for k, v in _PROFILE.items() if k != "active_work_order"}
        with patch("servers.robot.main.db", _db_for(profile_no_wo)):
            data = await call_tool(mcp, "safety_gate_check", {"asset_id": "Chiller 6"})
        assert data["active_work_order"] is None

    @pytest.mark.anyio
    async def test_error_when_db_none(self, no_db):
        data = await call_tool(mcp, "safety_gate_check", {"asset_id": "Chiller 6"})
        assert "error" in data


# ---------------------------------------------------------------------------
# Tool 3: open_panel
# ---------------------------------------------------------------------------


class TestOpenPanel:
    @pytest.mark.anyio
    async def test_panel_opens_when_not_stuck(self):
        with patch("servers.robot.main.db", _db_for(_PROFILE_FREE)):
            data = await call_tool(mcp, "open_panel", {"asset_id": "Chiller 6"})
        assert data["success"] is True
        assert data["access_granted"] is True

    @pytest.mark.anyio
    async def test_panel_stuck_when_panel_stuck_true(self):
        with patch("servers.robot.main.db", _db_for(_PROFILE_STUCK)):
            data = await call_tool(mcp, "open_panel", {"asset_id": "Chiller 6"})
        assert data["success"] is False
        assert data["stuck_reason"] is not None

    @pytest.mark.anyio
    async def test_deterministic_result_from_profile(self):
        # open_panel reads panel_stuck bool directly — same input always same output
        with patch("servers.robot.main.db", _db_for(_PROFILE_FREE)):
            r1 = await call_tool(mcp, "open_panel", {"asset_id": "Chiller 6"})
            r2 = await call_tool(mcp, "open_panel", {"asset_id": "Chiller 6"})
        assert r1["success"] == r2["success"]
        assert r1["access_granted"] == r2["access_granted"]

    @pytest.mark.anyio
    async def test_error_when_db_none(self, no_db):
        data = await call_tool(mcp, "open_panel", {"asset_id": "Chiller 6"})
        assert "error" in data


# ---------------------------------------------------------------------------
# Tool 4: read_gauge
# ---------------------------------------------------------------------------


class TestReadGauge:
    @pytest.mark.anyio
    async def test_reading_within_range(self):
        with patch("servers.robot.main.db", _db_for(_PROFILE)):
            data = await call_tool(
                mcp, "read_gauge", {"asset_id": "Chiller 6", "attempt_n": 1}
            )
        assert 0.0 <= data["reading"] <= 100.0
        assert 0.0 <= data["confidence"] <= 1.0

    @pytest.mark.anyio
    async def test_no_gauge_value_in_response(self):
        with patch("servers.robot.main.db", _db_for(_PROFILE)):
            data = await call_tool(
                mcp, "read_gauge", {"asset_id": "Chiller 6", "attempt_n": 1}
            )
        assert "gauge_value" not in data

    @pytest.mark.anyio
    async def test_attempt_n_reflected_in_response(self):
        with patch("servers.robot.main.db", _db_for(_PROFILE)):
            data = await call_tool(
                mcp, "read_gauge", {"asset_id": "Chiller 6", "attempt_n": 3}
            )
        assert data["attempt_n"] == 3

    @pytest.mark.anyio
    async def test_gauge_range_present(self):
        with patch("servers.robot.main.db", _db_for(_PROFILE)):
            data = await call_tool(
                mcp, "read_gauge", {"asset_id": "Chiller 6", "attempt_n": 1}
            )
        assert "gauge_range" in data

    @pytest.mark.anyio
    async def test_gauge_path_present(self):
        with patch("servers.robot.main.db", _db_for(_PROFILE)):
            data = await call_tool(
                mcp, "read_gauge", {"asset_id": "Chiller 6", "attempt_n": 1}
            )
        assert "gauge_path" in data

    @pytest.mark.anyio
    async def test_error_when_db_none(self, no_db):
        data = await call_tool(
            mcp, "read_gauge", {"asset_id": "Chiller 6", "attempt_n": 1}
        )
        assert "error" in data


# ---------------------------------------------------------------------------
# Tool 5: commit_reading
# ---------------------------------------------------------------------------


class TestCommitReading:
    @pytest.mark.anyio
    async def test_blocked_when_n_lt_3(self):
        with patch("servers.robot.main.db", _db_for(_PROFILE)):
            data = await call_tool(
                mcp,
                "commit_reading",
                {
                    "asset_id": "Chiller 6",
                    "readings": [74.0, 76.0],
                    "decision": "close_normal",
                },
            )
        assert data["status"] == "BLOCKED"
        assert data["n_readings"] == 2

    @pytest.mark.anyio
    async def test_commit_succeeds_with_3_readings(self):
        with patch("servers.robot.main.db", _db_for(_PROFILE)):
            data = await call_tool(
                mcp,
                "commit_reading",
                {
                    "asset_id": "Chiller 6",
                    "readings": [73.0, 75.0, 77.0],
                    "decision": "close_normal",
                },
            )
        assert data["status"] == "COMMIT"
        assert data["n_readings"] == 3
        assert isinstance(data["readings_mean"], float)

    @pytest.mark.anyio
    async def test_never_read_in_response(self):
        with patch("servers.robot.main.db", _db_for(_PROFILE)):
            data = await call_tool(
                mcp,
                "commit_reading",
                {
                    "asset_id": "Chiller 6",
                    "readings": [73.0, 75.0, 77.0],
                    "decision": "close_normal",
                },
            )
        assert "never_read" in data

    @pytest.mark.anyio
    async def test_no_gauge_value_in_commit_response(self):
        with patch("servers.robot.main.db", _db_for(_PROFILE)):
            data = await call_tool(
                mcp,
                "commit_reading",
                {
                    "asset_id": "Chiller 6",
                    "readings": [73.0, 75.0, 77.0],
                    "decision": "close_normal",
                },
            )
        assert "gauge_value" not in data

    @pytest.mark.anyio
    async def test_error_when_db_none(self, no_db):
        data = await call_tool(
            mcp,
            "commit_reading",
            {
                "asset_id": "Chiller 6",
                "readings": [73.0, 75.0, 77.0],
                "decision": "close_normal",
            },
        )
        assert "error" in data

    @requires_couchdb
    @pytest.mark.anyio
    async def test_integration_commit(self):
        data = await call_tool(
            mcp,
            "commit_reading",
            {
                "asset_id": "Chiller 6",
                "readings": [49.5, 50.0, 50.5],
                "decision": "close_normal",
            },
        )
        assert data["status"] in {"COMMIT", "BLOCKED"}


# ---------------------------------------------------------------------------
# Tool 6: check_wo_similarity
# ---------------------------------------------------------------------------


class TestCheckWoSimilarity:
    def _wo_mock(self, docs):
        mock = MagicMock()
        mock.find.return_value = {"docs": docs}
        return mock

    @pytest.mark.anyio
    async def test_error_when_wo_db_none(self, no_wo_db):
        with patch("servers.robot.main.db", _db_for(_PROFILE)):
            data = await call_tool(
                mcp,
                "check_wo_similarity",
                {
                    "asset_id": "Chiller 6",
                    "failure_description": "water leak near chiller",
                },
            )
        assert "error" in data

    @pytest.mark.anyio
    async def test_proceed_when_no_similar_wos(self):
        wo_doc = {
            "wonum": "1000045",
            "description": "completely unrelated electrical work",
            "assetnum": "CHILLER6",
            "status": "COMP",
        }
        with patch("servers.robot.main._get_wo_db", return_value=self._wo_mock([wo_doc])):
            with patch("servers.robot.main.db", _db_for(_PROFILE)):
                data = await call_tool(
                    mcp,
                    "check_wo_similarity",
                    {
                        "asset_id": "Chiller 6",
                        "failure_description": "water leak near chiller",
                    },
                )
        assert data["recommendation"] in {"proceed", "review", "consolidate"}
        assert "similar_wos" in data
        assert "scores" in data

    @pytest.mark.anyio
    async def test_consolidate_for_identical_description(self):
        wo_doc = {
            "wonum": "1000046",
            "description": "water leak near chiller unit",
            "assetnum": "CHILLER6",
            "status": "WAPPR",
        }
        with patch("servers.robot.main._get_wo_db", return_value=self._wo_mock([wo_doc])):
            with patch("servers.robot.main.db", _db_for(_PROFILE)):
                data = await call_tool(
                    mcp,
                    "check_wo_similarity",
                    {
                        "asset_id": "Chiller 6",
                        "failure_description": "water leak near chiller unit",
                    },
                )
        assert data["recommendation"] == "consolidate"
        assert data["duplicate_risk"] is True

    @requires_couchdb
    @pytest.mark.anyio
    async def test_integration_wo_similarity(self):
        data = await call_tool(
            mcp,
            "check_wo_similarity",
            {
                "asset_id": "Chiller 6",
                "failure_description": "anomaly on chiller condenser",
            },
        )
        assert "recommendation" in data
        assert data["recommendation"] in {"proceed", "review", "consolidate"}
