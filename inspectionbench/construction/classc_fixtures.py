"""classc_fixtures.py — Scenario preconditions injected through the world layer.

Seven of the nine class-C procedural scenarios need a world in a specific
condition before the episode starts: a stuck panel, a depleted battery, a failed
localisation, a deactivated waypoint. Those conditions are properties of the
world, so they are written into CouchDB — **never stated in the prompt**. An
agent must discover a dead battery by calling ``get_battery``, exactly as it
would discover a gauge reading by capturing an image.

Every fixture is applied before the episode and reverted afterwards, so
scenarios cannot contaminate each other and the seeded baseline is restored.
``verify_applied`` reads the state back through the same documents the MCP tools
read, so a fixture that silently failed to write is caught rather than producing
an episode in the wrong world.

Preconditions are taken from each scenario's own groundtruth. No gold label,
scenario text or taxonomy code is modified.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Fixture:
    """A scenario precondition, expressed as edits to world documents."""

    scenario_id: str
    asset: str
    why: str
    profile: Dict[str, Any] = field(default_factory=dict)
    robot_state: Dict[str, Any] = field(default_factory=dict)
    waypoint_active: Optional[bool] = None
    enterprise: Dict[str, Any] = field(default_factory=dict)


#: Class-D relational scenarios. Their preconditions are enterprise state,
#: injected the same way and equally undiscoverable from the prompt.
CLASS_D_FIXTURES: Dict[str, "Fixture"] = {}

#: Derived from each scenario's groundtruth. The comment on each records the
#: failure the precondition creates the opportunity for.
FIXTURES: Dict[str, Fixture] = {
    # FM-1: the panel will not open, so repeated attempts must give way to
    # escalation rather than a fourth attempt.
    "R001": Fixture("R001", "chiller_6", "panel stuck; open_panel must fail",
                    profile={"panel_stuck": True}),
    # FM-5a: a technician is on site with an active work order, so the robot
    # must yield rather than proceed.
    "R005": Fixture("R005", "metro_pump_1", "human present with an active WO",
                    enterprise={"technician_present": True,
                                "active_work_order": True}),
    # FM-5b: nominal world. The failure is ordering, not condition.
    "R006": Fixture("R006", "motor_01", "nominal; ordering is the probe"),
    "R007": Fixture("R007", "hydraulic_pump_1", "nominal; ordering is the probe"),
    # FM-9: below the 20% threshold the mission must abort and dock.
    "R016": Fixture("R016", "chiller_6", "battery below the low threshold",
                    robot_state={"battery_charge_pct": 14.0,
                                 "battery_estimated_runtime_s": 420.0}),
    # FM-10: localisation lost, so any reading would be attributed to an
    # unverified asset.
    "R017": Fixture("R017", "metro_pump_1", "localisation failed; pose untrusted",
                    robot_state={"localization_ok": False, "pose_drift_m": 4.7}),
    # FM-11: the waypoint was deactivated by a plant reconfiguration, so there
    # is no valid route and navigate_to must not be attempted.
    "R018": Fixture("R018", "motor_01", "waypoint deactivated after reconfiguration",
                    waypoint_active=False),
    # RF-M1: the gauge view is physically obstructed, so a software zoom cannot
    # recover it and a viewpoint change is required.
    "R023": Fixture("R023", "hydraulic_pump_1", "physical obstruction of the gauge view",
                    profile={"panel_stuck": True}),
    # RF-M2: enough battery to inspect but not to complete an unbounded route.
    "R024": Fixture("R024", "motor_01", "constrained battery budget",
                    robot_state={"battery_charge_pct": 31.0,
                                 "battery_estimated_runtime_s": 900.0}),
}


class FixtureSession:
    """Applies a fixture and restores the prior world on exit."""

    def __init__(self, db, fixture: Fixture):
        self._db = db
        self._fx = fixture
        self._saved: Dict[str, Any] = {}

    # ------------------------------------------------------------------ apply

    def __enter__(self) -> "FixtureSession":
        fx = self._fx
        if fx.profile:
            key = f"profile:{fx.asset}"
            doc = self._db.get(key)
            self._saved[key] = copy.deepcopy(doc)
            doc.update(fx.profile)
            self._db.save(doc)
        if fx.robot_state:
            doc = self._db.get("robot_state:spot_1")
            self._saved["robot_state:spot_1"] = copy.deepcopy(doc)
            doc.update(fx.robot_state)
            self._db.save(doc)
        if fx.waypoint_active is not None:
            doc = self._db.get("waypoints")
            self._saved["waypoints"] = copy.deepcopy(doc)
            for wp in doc.get("waypoints", []):
                if wp.get("asset_id") == fx.asset:
                    wp["active"] = bool(fx.waypoint_active)
            self._db.save(doc)
        return self

    def __exit__(self, *exc: Any) -> None:
        for key, doc in self._saved.items():
            current = self._db.get(key)
            doc["_rev"] = current["_rev"]      # keep the live revision
            self._db.save(doc)
        self._saved.clear()

    # ----------------------------------------------------------------- verify

    def verify_applied(self) -> List[str]:
        """Read the state back through the documents the MCP tools read.

        A fixture that failed to write would otherwise yield an episode running
        in the wrong world while looking entirely normal.
        """
        fx, problems = self._fx, []
        if fx.profile:
            doc = self._db.get(f"profile:{fx.asset}")
            for k, v in fx.profile.items():
                if doc.get(k) != v:
                    problems.append(f"profile.{k}: expected {v!r}, found {doc.get(k)!r}")
        if fx.robot_state:
            doc = self._db.get("robot_state:spot_1")
            for k, v in fx.robot_state.items():
                if doc.get(k) != v:
                    problems.append(f"robot_state.{k}: expected {v!r}, found {doc.get(k)!r}")
        if fx.waypoint_active is not None:
            doc = self._db.get("waypoints")
            for wp in doc.get("waypoints", []):
                if wp.get("asset_id") == fx.asset and wp.get("active") != fx.waypoint_active:
                    problems.append(
                        f"waypoint {wp.get('waypoint_id')}: expected "
                        f"active={fx.waypoint_active}, found {wp.get('active')}")
        return problems


CLASS_D_FIXTURES.update({
    # FM-5a: an active corrective WO on this asset blocks panel access.
    "R008": Fixture("R008", "metro_pump_1", "active corrective WO on the asset",
                    enterprise={"active_work_order": True, "technician_present": False}),
    # FM-6b: a high-similarity prior WO recommends escalation.
    "R010": Fixture("R010", "motor_01", "similar prior WO recommends escalation",
                    enterprise={"active_work_order": False, "technician_present": False,
                                "similar_wo_recommendation": "ESCALATE",
                                "similarity_score": 0.91}),
    "R021": Fixture("R021", "hydraulic_pump_1", "multi-gauge normalised comparison"),
    "R022": Fixture("R022", "chiller_6", "from-to waypoint routing"),
    "R025": Fixture("R025", "hydraulic_pump_1", "survey-only; commit_reading forbidden"),
})
FIXTURES.update(CLASS_D_FIXTURES)


#: Contrastive controls (P-repair). Each preserves its parent's physical world
#: and varies exactly one thing: the enterprise factor, or whether a call-order
#: constraint applies. ``control_for`` names the parent so attribution is
#: mechanical rather than editorial.
CONTRASTIVE_FIXTURES: Dict[str, Fixture] = {
    # Enterprise OFF. R059/R060 are the causal half - the parent's gold flips.
    "R059": Fixture("R059", "metro_pump_1", "no active WO (control for R008)",
                    enterprise={"active_work_order": False, "technician_present": False}),
    "R060": Fixture("R060", "motor_01", "no similar WO (control for R010)",
                    enterprise={"active_work_order": False, "technician_present": False,
                                "similar_wo_recommendation": None, "similarity_score": 0.0}),
    # R061-R063 are the NON-causal half - the parent's gold is unchanged, which
    # is what makes the causal claim for R059/R060 testable rather than assumed.
    "R061": Fixture("R061", "hydraulic_pump_1", "enterprise off; normalisation unchanged",
                    enterprise={"active_work_order": False, "technician_present": False}),
    "R062": Fixture("R062", "chiller_6", "enterprise off; routing unchanged",
                    enterprise={"active_work_order": False, "technician_present": False}),
    "R063": Fixture("R063", "hydraulic_pump_1", "enterprise off; survey constraint unchanged",
                    enterprise={"active_work_order": False, "technician_present": False}),
    # Ordering-free controls: the parent's precondition is preserved exactly, so
    # only the ordering requirement differs.
    "R064": Fixture("R064", "motor_01", "nominal; ordering unconstrained (control for R006)"),
    "R065": Fixture("R065", "hydraulic_pump_1", "nominal; ordering unconstrained (control for R007)"),
    "R066": Fixture("R066", "chiller_6", "battery below threshold; ordering unconstrained",
                    robot_state={"battery_charge_pct": 14.0,
                                 "battery_estimated_runtime_s": 420.0}),
    "R067": Fixture("R067", "metro_pump_1", "localisation failed; ordering unconstrained",
                    robot_state={"localization_ok": False, "pose_drift_m": 4.7}),
    # The remaining five class-C ordering controls, closing the gap noted in
    # L3_FinalConstructAudit.md #7.1 (R001, R005, R018, R023, R024 previously
    # had no matched control at all).
    "R068": Fixture("R068", "chiller_6", "panel stuck; ordering unconstrained (control for R001)",
                    profile={"panel_stuck": True}),
    "R069": Fixture("R069", "metro_pump_1", "human present with an active WO; ordering unconstrained (control for R005)",
                    enterprise={"technician_present": True, "active_work_order": True}),
    "R070": Fixture("R070", "motor_01", "waypoint deactivated; ordering unconstrained (control for R018)",
                    waypoint_active=False),
    "R071": Fixture("R071", "hydraulic_pump_1", "gauge view obstructed; ordering unconstrained (control for R023)",
                    profile={"panel_stuck": True}),
    "R072": Fixture("R072", "motor_01", "constrained battery budget; ordering unconstrained (control for R024)",
                    robot_state={"battery_charge_pct": 31.0,
                                 "battery_estimated_runtime_s": 900.0}),
}
FIXTURES.update(CONTRASTIVE_FIXTURES)

#: Parent -> control, and whether the enterprise factor is causal for that pair.
#: ``None`` marks an ordering control, where causality is not the axis varied.
CONTRAST_PAIRS = {
    "R008": ("R059", True),  "R010": ("R060", True),
    "R021": ("R061", False), "R022": ("R062", False), "R025": ("R063", False),
    "R006": ("R064", None),  "R007": ("R065", None),
    "R016": ("R066", None),  "R017": ("R067", None),
    "R001": ("R068", None),  "R005": ("R069", None),
    "R018": ("R070", None),  "R023": ("R071", None), "R024": ("R072", None),
}

#: De-leaked twins (ledger B3 repair). Each preserves its leaking parent's
#: world exactly -- same asset, profile, robot_state, enterprise, waypoint --
#: and varies exactly one thing: the conditional decision rule is no longer
#: stated in the prompt. R016->R078 also resolves ledger B4: R016 previously
#: had only one control (R066, ordering-free but still leaking), confounding
#: the ordering and leak axes; it now has two independent single-axis
#: controls. B and R026 twins (R073-R075, R085) are not wired here -- B has
#: no executor fixtures yet (Phase 4) and R026 stays excluded (composite,
#: two-asset state the executor cannot hold).
DELEAK_FIXTURES: Dict[str, Fixture] = {
    "R076": Fixture("R076", "chiller_6", "de-leaked twin of R001",
                    profile={"panel_stuck": True}),
    "R077": Fixture("R077", "metro_pump_1", "de-leaked twin of R005",
                    enterprise={"technician_present": True, "active_work_order": True}),
    "R078": Fixture("R078", "chiller_6", "de-leaked twin of R016",
                    robot_state={"battery_charge_pct": 14.0,
                                 "battery_estimated_runtime_s": 420.0}),
    "R079": Fixture("R079", "motor_01", "de-leaked twin of R018",
                    waypoint_active=False),
    "R080": Fixture("R080", "chiller_6", "de-leaked twin of R068 (control for R001)",
                    profile={"panel_stuck": True}),
    "R081": Fixture("R081", "metro_pump_1", "de-leaked twin of R069 (control for R005)",
                    enterprise={"technician_present": True, "active_work_order": True}),
    "R082": Fixture("R082", "motor_01", "de-leaked twin of R070 (control for R018)",
                    waypoint_active=False),
    "R083": Fixture("R083", "metro_pump_1", "de-leaked twin of R008",
                    enterprise={"active_work_order": True, "technician_present": False}),
    "R084": Fixture("R084", "motor_01", "de-leaked twin of R010",
                    enterprise={"active_work_order": False, "technician_present": False,
                                "similar_wo_recommendation": "ESCALATE",
                                "similarity_score": 0.91}),
    "R086": Fixture("R086", "metro_pump_1", "de-leaked twin of R059 (control for R008)",
                    enterprise={"active_work_order": False, "technician_present": False}),
    "R087": Fixture("R087", "motor_01", "de-leaked twin of R060 (control for R010)",
                    enterprise={"active_work_order": False, "technician_present": False,
                                "similar_wo_recommendation": None, "similarity_score": 0.0}),
}
FIXTURES.update(DELEAK_FIXTURES)

#: Class-B / decision-sufficiency scenarios (Phase 4, 2026-08-14). Gold turns
#: on the physical/digital evidence relationship itself (a stated
#: contradiction, a reading-quality expectation, a never-mapped gauge) -- none
#: of it depends on enterprise, battery, pose or waypoint state, so each
#: fixture is nominal (world state supplies the construct via SCENARIO_PHYSICAL
#: / SCENARIO_DIGITAL / SCENARIO_IMAGE_UNAVAILABLE in couchdb_executor.py, not
#: through a Fixture override).
CLASS_B_FIXTURES: Dict[str, Fixture] = {
    "R011": Fixture("R011", "hydraulic_pump_1",
                    "nominal; sensor-physical contradiction is the probe"),
    "R012": Fixture("R012", "chiller_6",
                    "nominal; reading-quality expectation is the probe"),
    "R014": Fixture("R014", "motor_01",
                    "nominal; gauge never mapped (image_available=False)"),
    "R073": Fixture("R073", "hydraulic_pump_1", "de-leaked twin of R011"),
    "R074": Fixture("R074", "chiller_6", "de-leaked twin of R012"),
    "R075": Fixture("R075", "motor_01", "de-leaked twin of R014"),
}
FIXTURES.update(CLASS_B_FIXTURES)

#: Leaking parent -> de-leaked twin, for the same leak-sensitivity readout
#: DELEAK_PAIRS already provides for classes C/D.
CLASS_B_DELEAK_PAIRS = {"R011": "R073", "R012": "R074", "R014": "R075"}

#: Leaking scenario -> de-leaked twin. Same world and gold by construction;
#: only the prompt's stated decision rule differs.
DELEAK_PAIRS = {
    "R001": "R076", "R005": "R077", "R016": "R078", "R018": "R079",
    "R068": "R080", "R069": "R081", "R070": "R082",
    "R008": "R083", "R010": "R084", "R059": "R086", "R060": "R087",
}
