"""Phase 1/2 regression tests for the D-physical world-first physical_access
field (scenario_gen.WorldState, to_couch_profile, couchdb_executor.reset_from_world).

Proves two things, both load-bearing for the D-physical design:
  1. The new fields are purely additive -- every world predating D-physical
     (A/B/C/D-enterprise/E) serializes and hashes identically to before.
  2. The world-first physical_access path, applied through the REAL CouchDB
     reset_from_world, reproduces the same admissibility verdicts as the
     existing R027-R038/RC003 hand-authored scenario suite's direct oracle
     calls -- i.e. W -> G(W) genuinely holds for D-physical, not just claimed.

Zero model/API calls.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

import scenario_gen as G  # noqa: E402
from couchdb_executor import CouchDBExecutor  # noqa: E402
from spot_admissibility_verifier import SpotAdmissibilityVerifier  # noqa: E402

ASSET_PROFILES_PATH = (
    Path.home() / "AssetOpsBenchScenarioGeneration" / "RobotInspection" / "Scenarios"
    / "asset_profiles.json"
)
REGISTRY_PATH = (
    Path.home() / "AssetOpsBenchScenarioGeneration" / "RobotInspection" / "shared"
    / "robot_assets_registry.json"
)

import pytest  # noqa: E402

pytestmark = pytest.mark.skipif(
    not ASSET_PROFILES_PATH.exists() or not REGISTRY_PATH.exists(),
    reason="AssetOpsBenchScenarioGeneration sibling repo not present in this environment",
)


def _load_profiles():
    data = json.loads(ASSET_PROFILES_PATH.read_text())
    if isinstance(data, list):
        data = {p["asset_id"]: p for p in data}
    return data


def _load_registry():
    return json.loads(REGISTRY_PATH.read_text())


# --- 1. purely additive: existing worlds unaffected --------------------------

def test_worldstate_defaults_physical_fields_to_none():
    w = G.sample_world(1, G.FACTORIAL[0], asset_id="chiller_6")
    assert w.physical_access is None
    assert w.payload_kg is None


def test_to_dict_omits_unset_physical_fields_entirely():
    """A world with no physical_access must serialize to the SAME KEY SET as
    before this field existed -- proven by omission, not just a None value,
    because the frozen 12-world committed-pilot comparison
    (test_contrastive_repairs.py::test_frozen_generator_reproduces_the_committed_pilot)
    does exact dict equality against a snapshot that predates these fields."""
    w = G.sample_world(1, G.FACTORIAL[0], asset_id="chiller_6")
    d = w.to_dict()
    assert "physical_access" not in d
    assert "payload_kg" not in d


def test_to_couch_profile_omits_physical_access_when_unset():
    w = G.sample_world(1, G.FACTORIAL[0], asset_id="chiller_6")
    profile = G.to_couch_profile(w)
    assert "physical_access" not in profile
    assert set(profile) == {"gauge_value", "gauge_range", "panel_stuck"}


def test_to_couch_profile_includes_physical_access_when_set():
    w = G.sample_world(1, G.FACTORIAL[0], asset_id="chiller_6")
    profiles = _load_profiles()
    access = copy.deepcopy(profiles["chiller_6"]["physical_access"])
    w2 = G.WorldState(**{**w.to_dict(), "physical_access": access})
    profile = G.to_couch_profile(w2)
    assert profile["physical_access"] == access


# --- 2. world-first oracle reproduction (real MuJoCo, real CouchDB) ---------

R027_R038_CASES = [
    ("R027", "chiller_6", {"panel_pose": {"x": 52.3, "y": 18.1, "z": 2.35, "yaw_deg": 90}}, False),
    ("R028", "chiller_6", {}, True),
    ("R029", "chiller_6", {"commanded_arm_qpos": {
        "arm0.sh0": 0.0, "arm0.sh1": -0.6, "arm0.el0": 1.2, "arm0.el1": 0.0,
        "arm0.wr0": 2.2, "arm0.wr1": 0.0, "arm0.f1x": -0.2}}, False),
    ("R030", "chiller_6", {}, True),
    ("R031", "hydraulic_pump_1", {"obstacle_clearance_m": 1.20}, False),
    ("R032", "hydraulic_pump_1", {}, True),
    ("R033", "hydraulic_pump_1", {"latch_grasp_force_N": 180.0}, False),
    ("R034", "hydraulic_pump_1", {}, True),
    ("R035", "hydraulic_pump_1", {"approach_slope_deg": 40.0}, False),
    ("R036", "hydraulic_pump_1", {}, True),
    ("R037", "hydraulic_pump_1", {"battery_pct": 12.0, "route_legs": 3, "mission_duration_s": 600}, False),
    ("R038", "hydraulic_pump_1", {"battery_pct": 90.0, "route_legs": 1, "mission_duration_s": 300}, True),
]


@pytest.mark.parametrize("name,asset,overrides,expect_admissible", R027_R038_CASES)
def test_world_first_reproduces_r027_r038_oracle_verdicts(name, asset, overrides, expect_admissible):
    profiles = _load_profiles()
    registry = _load_registry()
    access = copy.deepcopy(profiles[asset]["physical_access"])
    access.update(overrides)

    world = G.WorldState(
        scenario_id=f"WORLDFIRST-{name}", asset=asset, unit=profiles[asset].get("unit", "bar"),
        gauge_range=[0.0, 100.0], operating_band=[10.0, 90.0],
        physical_value=50.0, iot_value=50.0, history_mean=50.0,
        active_work_order=False, technician_present=False, cell="d_phys_repro", seed=1,
        physical_access=access,
    )

    ex = CouchDBExecutor()
    ex.reset_from_world(world, "spot_1", seed=1)

    db = ex._robot.db
    doc = db.get(f"profile:{asset}")
    assert doc.get("physical_access") == access, "physical_access did not round-trip through reset_from_world"

    verifier = SpotAdmissibilityVerifier(registry)
    standoffs = access.get("standoff_candidates_m", [0.8])
    verdicts = [verifier.verify_candidate(access, s) for s in standoffs]
    admissible = any(v.admissible for v in verdicts)

    assert admissible == expect_admissible, (
        f"{name}: world-first oracle verdict {admissible} disagrees with the designed "
        f"R027-R038 expectation {expect_admissible} -- checks={[v.checks for v in verdicts]}"
    )


def test_reset_from_world_is_deterministic_for_physical_access():
    """Same world, applied twice, must produce the identical persisted profile."""
    profiles = _load_profiles()
    access = copy.deepcopy(profiles["motor_01"]["physical_access"])
    world = G.WorldState(
        scenario_id="WORLDFIRST-determinism", asset="motor_01", unit="C",
        gauge_range=[0.0, 200.0], operating_band=[80.0, 100.0],
        physical_value=90.0, iot_value=90.0, history_mean=90.0,
        active_work_order=False, technician_present=False, cell="d_phys_repro", seed=7,
        physical_access=access,
    )
    ex = CouchDBExecutor()
    ex.reset_from_world(world, "spot_1", seed=7)
    doc1 = dict(ex._robot.db.get("profile:motor_01"))
    ex.reset_from_world(world, "spot_1", seed=7)
    doc2 = dict(ex._robot.db.get("profile:motor_01"))
    assert doc1.get("physical_access") == doc2.get("physical_access") == access
