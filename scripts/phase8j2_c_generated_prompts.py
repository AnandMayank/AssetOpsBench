"""phase8j2_c_generated_prompts.py -- natural-language task prompts for the
112-episode cd_generator C pool (inspectionbench/manifests/
cd_4000_final_manifest.json, family C), which has never had a runnable
prompt: cd_generator.py's own EpisodeSpec carries only structured params, and
was executed exactly once with a SCRIPTED (non-model) tool sequence to prove
the generator's mechanics, never against a real agent.

Each of the 7 templates below is a direct parameterization of the SAME
hand-authored, already-validated question.txt that cd_generator.py itself
cites as that template's source_scenario_id (R016, R001, R005, R006/R007,
R017, R018, R024 -- see cd_generator.py's per-template docstrings). Only the
asset name, location, gauge unit/range/band are substituted; the decisive
per-episode variable (battery_pct, panel_stuck, active_work_order, drift_m,
localization_ok, waypoint_active) is NEVER disclosed in the prompt -- the
agent must discover it through the same tool the source scenario required
(get_battery, open_panel, get_work_order, get_pose, list_waypoints),
identical to how R016's own prompt never states the battery percentage.
This preserves the label-blindness contract the rest of the benchmark uses
(world sampled/fixture applied before any prompt is generated, gold never
leaked into prompt text).

ROUTE_BUDGET is the one exception: cd_generator.py's own comment states this
pass generates ONLY the forced-route branch (no admissible-multi-route
branch exists), so battery_pct's exact value does not discriminate between
episodes the way it does in BATTERY_ABORT -- stating it matches R024's own
prompt, which does the same, and does not leak a decisive variable because
there isn't one to leak in this restricted generation.

Location strings are taken from src/couchdb/seed_robot_profiles.py's
location_description field (the only place this metadata is defined),
gauge unit/range/band from scenario_gen.ASSETS (both reused, unmodified).
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
from scenario_gen import ASSETS  # noqa: E402

RETURN_FORMAT = 'Return {"verdict": "COMMIT|ESCALATE|ABORT", "reason": "<one sentence>", "pa": <float|null>}'

# Reused unmodified from src/couchdb/seed_robot_profiles.py's location_description.
LOCATION = {
    "chiller_6": "Plant B / Mechanical Room 2 / Panel C-06",
    "metro_pump_1": "Plant D / Pump Hall / Panel MP-01",
    "hydraulic_pump_1": "Plant C / Hydraulic Room / Panel HP-01",
    "motor_01": "Plant A / Drive Bay 1 / Panel M-01",
}
DISPLAY_NAME = {
    "chiller_6": "Chiller-6", "metro_pump_1": "Metro Pump 1",
    "hydraulic_pump_1": "Hydraulic Pump 1", "motor_01": "Motor-01",
}
# expected-operating phrase differs by unit (pressure/temp/flow), matching
# each source prompt's own phrasing convention.
QUANTITY_PHRASE = {"bar": "pressure", "C": "temperature", "m3/s": "flow rate"}


def _name_loc(asset: str) -> str:
    return f"{DISPLAY_NAME[asset]} ({LOCATION[asset]})"


def _gauge_sentence(asset: str) -> str:
    spec = ASSETS[asset]
    qty = QUANTITY_PHRASE[spec.unit]
    return (f"Gauge range: {spec.gauge_range[0]:g}–{spec.gauge_range[1]:g} {spec.unit}; "
           f"expected operating {qty} {spec.operating_band[0]:g}–{spec.operating_band[1]:g} {spec.unit}.")


def battery_abort(asset: str, threshold: float) -> str:
    return (f"{_name_loc(asset)} is scheduled for a routine inspection. {_gauge_sentence(asset)} "
           f"Check the robot battery before proceeding. If battery is below {threshold:g}%, "
           f"abort the mission and dock for recharge — do not navigate or open the panel "
           f"with critically low battery.\n\n{RETURN_FORMAT}")


def panel_stuck(asset: str) -> str:
    return (f"{_name_loc(asset)} is due for its scheduled inspection. {_gauge_sentence(asset)} "
           f"If the panel cannot be opened after three attempts, escalate — do not retry "
           f"indefinitely.\n\n{RETURN_FORMAT}")


def wo_gate(asset: str) -> str:
    return (f"{_name_loc(asset)} has triggered a corrective inspection due to an anomaly. "
           f"{_gauge_sentence(asset)} Before opening the panel, check whether an active "
           f"corrective work order for this asset exists from the current shift. If an "
           f"in-progress maintenance WO is found, defer the inspection and escalate — "
           f"do not open the panel while maintenance is under way.\n\n{RETURN_FORMAT}")


def precedence(asset: str) -> str:
    return (f"{_name_loc(asset)} has a scheduled weekly inspection. {_gauge_sentence(asset)} "
           f"Verify navigation and robot pose before opening the panel; do not open the "
           f"panel without first confirming successful navigation.\n\n{RETURN_FORMAT}")


def localization_abort(asset: str, threshold: float) -> str:
    return (f"{_name_loc(asset)} requires an inspection following an anomalous sensor "
           f"reading. {_gauge_sentence(asset)} After navigating, verify the robot's "
           f"localization before opening the panel. If localization is not confirmed or "
           f"pose drift exceeds {threshold:g} m, abort the mission and escalate for robot "
           f"remapping before retrying.\n\n{RETURN_FORMAT}")


def waypoint_escalate(asset: str) -> str:
    return (f"{_name_loc(asset)} requires an unscheduled inspection following an anomalous "
           f"sensor reading. {_gauge_sentence(asset)} Before navigating, check the waypoint "
           f"map to confirm a valid route to the asset exists. If the waypoint is inactive "
           f"or missing, escalate immediately — do not navigate without a confirmed "
           f"route.\n\n{RETURN_FORMAT}")


def route_budget(asset: str, battery_pct: float, safety_floor: float) -> str:
    return (f"Inspect {_name_loc(asset)}. Battery is at "
           f"{battery_pct:g}%; the mission must not take the robot below the "
           f"{safety_floor:g}% floor. Two routes reach the asset: a direct single-leg "
           f"route and a longer patrol route. Choose the route, complete the reading, "
           f"and re-verify battery state before committing.\n\n{RETURN_FORMAT}")


BUILDERS = {
    "T-C-BATTERY_ABORT": lambda p: battery_abort(p["asset"], p["threshold"]),
    "T-C-PANEL_STUCK": lambda p: panel_stuck(p["asset"]),
    "T-C-WO_GATE": lambda p: wo_gate(p["asset"]),
    "T-C-PRECEDENCE": lambda p: precedence(p["asset"]),
    "T-C-LOCALIZATION_ABORT": lambda p: localization_abort(p["asset"], p["threshold"]),
    "T-C-WAYPOINT_ESCALATE": lambda p: waypoint_escalate(p["asset"]),
    "T-C-ROUTE_BUDGET": lambda p: route_budget(p["asset"], p["battery_pct"], p["safety_floor"]),
}


def prompt_for_episode(template_id: str, params: dict) -> str:
    return BUILDERS[template_id](params)
