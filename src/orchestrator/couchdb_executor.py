"""couchdb_executor.py — Stage-1 execution backend over the live MCP tools (P0-1).

Implements ``ToolExecutor`` by calling the tool functions in
``src/servers/robot/main.py``, which read hidden state from CouchDB. The
important property is inherited rather than re-implemented: ``read_gauge``
resolves the true ``gauge_value``, adds noise and never returns the truth, and
the invariant is asserted in that module's source. The model receives a noisy
observation it could not have manufactured.

Two things the MCP layer does not provide, added here:

**Per-scenario hidden state.** The seeded profiles carry ``gauge_value=0.0`` for
every asset, so all six pilot scenarios would read the same. ``reset()`` writes
the scenario's hidden physical value into the profile before the episode and is
idempotent, giving deterministic reset.

**Actual pixels.** ``capture_image`` in the MCP server returns a ``gauge_path``,
and every pilot profile has ``gauge_path=None`` — nothing was ever deliverable.
P0-7 requires an observation the model consumes, so this backend renders the
dial at the hidden value and returns base64 PNG. The needle position *is* the
ground truth, so reading it is a perceptual act rather than a lookup.

Hidden physical values are derived from each scenario's existing groundtruth and
operating band, not invented: where the groundtruth states the reading it is
used verbatim; where it does not (R009, R015) a value inside the stated band is
chosen, because the scenario's gold is COMMIT and only an in-band reading is
consistent with that. No gold label, band or scenario text is changed.

**Repair (ledger B1, 2026-08-14).** ``run_classc_pilot.py`` / ``run_classd_pilot.py``
previously fell back to a placeholder (``gauge_value=0.0``, range/band
``[0,100]``) for every class-C/D scenario ID absent from ``SCENARIO_PHYSICAL`` —
which was every one of the 28 run this phase, since this table only ever covered
the six original pilot scenarios. Every scenario now has a real entry, in-band at
the midpoint of its asset's operating band where the groundtruth does not state a
value and gold does not depend on the gauge reading (panel-stuck, battery,
localisation, waypoint, work-order and similarity scenarios — the gauge must read
*normal* there so it cannot spuriously drive the decision), or the groundtruth's
own stated reading otherwise.

**Repair (ledger B1, multi-gauge).** R021/R025 and their controls R061/R063
require *three independently-valued named gauges* on one asset
(``suction_line``, ``discharge_line``, ``seal_flush``), which
``read_gauge(asset_id, attempt_n)`` cannot express — it has no gauge-name
parameter and reads a single profile value. Rather than change that tool
signature (``src/servers/robot/main.py`` is off-limits for this repair),
``SCENARIO_MULTI_GAUGE`` maps ``attempt_n`` to a named gauge and this module
synthesises the noisy reading itself, at the exact normalised fractions each
scenario's own groundtruth states (0.20 / 0.70 / 0.14 of span) — not invented.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "servers" / "robot"))
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

from scenario_gen import to_enterprise_payload as G_to_enterprise  # noqa: E402
from scenario_gen import to_iot_payload as G_to_iot  # noqa: E402
from tool_executor import (  # noqa: E402
    STATUS_FAILED, STATUS_SUCCESS, STATUS_UNAVAILABLE, TOOL_MODALITY,
    ToolCall, ToolResult, mask_tools, observation_id, png_to_b64, render_gauge,
)

#: Hidden physical state per pilot scenario: (asset, true value, unit, band).
#: Values are taken from each scenario's groundtruth where stated; R009 and R015
#: state only that gold is COMMIT, so an in-band value is used.
SCENARIO_PHYSICAL: Dict[str, Dict[str, Any]] = {
    "R009": {"asset": "chiller_6", "value": 245.0, "unit": "bar",
             "range": [0, 400], "band": [220, 270], "source": "in-band (gold COMMIT)"},
    "R015": {"asset": "chiller_6", "value": 200.0, "unit": "bar",
             "range": [0, 400], "band": [180, 220], "source": "in-band (gold COMMIT)"},
    "R055": {"asset": "chiller_6", "value": 245.0, "unit": "bar",
             "range": [0, 400], "band": [220, 270], "source": "groundtruth"},
    "R056": {"asset": "metro_pump_1", "value": 1.38, "unit": "m3/s",
             "range": [0, 1.5], "band": [0.9, 1.1], "source": "groundtruth"},
    "R057": {"asset": "motor_01", "value": 92.0, "unit": "C",
             "range": [0, 200], "band": [80, 100], "source": "groundtruth"},
    "R058": {"asset": "hydraulic_pump_1", "value": 268.0, "unit": "bar",
             "range": [0, 350], "band": [160, 200], "source": "groundtruth"},

    # --- class-C / class-D (ledger B1 repair, 2026-08-14) -------------------
    # In-band midpoint of the asset's own operating band. Gold in every one of
    # these scenarios turns on something other than the gauge value (panel,
    # battery, localisation, waypoint, work order, similarity) — a legible,
    # normal-looking reading is the neutral choice so it cannot spuriously
    # drive the decision either way. Values for R006/R007/R023/R024 and their
    # controls, whose gold *does* require an in-band legible reading, are
    # equally satisfied by the same midpoint.
    "R001": {"asset": "chiller_6", "value": 245.0, "unit": "bar",
             "range": [0, 400], "band": [220, 270], "source": "in-band midpoint; gold turns on panel_stuck"},
    "R005": {"asset": "metro_pump_1", "value": 1.0, "unit": "m3/s",
             "range": [0, 1.5], "band": [0.9, 1.1], "source": "in-band midpoint; gold turns on active WO"},
    "R006": {"asset": "motor_01", "value": 90.0, "unit": "C",
             "range": [0, 200], "band": [80, 100], "source": "in-band midpoint; gold requires legible in-band reading"},
    "R007": {"asset": "hydraulic_pump_1", "value": 180.0, "unit": "bar",
             "range": [0, 350], "band": [160, 200], "source": "in-band midpoint; gold requires legible in-band reading"},
    "R008": {"asset": "metro_pump_1", "value": 1.0, "unit": "m3/s",
             "range": [0, 1.5], "band": [0.9, 1.1], "source": "in-band midpoint; gold turns on active WO"},
    "R010": {"asset": "motor_01", "value": 90.0, "unit": "C",
             "range": [0, 200], "band": [80, 100], "source": "in-band midpoint; gold turns on WO similarity"},
    "R016": {"asset": "chiller_6", "value": 245.0, "unit": "bar",
             "range": [0, 400], "band": [220, 270], "source": "in-band midpoint; gold turns on battery"},
    "R017": {"asset": "metro_pump_1", "value": 1.0, "unit": "m3/s",
             "range": [0, 1.5], "band": [0.9, 1.1], "source": "in-band midpoint; gold turns on localisation"},
    "R018": {"asset": "motor_01", "value": 90.0, "unit": "C",
             "range": [0, 200], "band": [80, 100], "source": "in-band midpoint; gold turns on waypoint active"},
    "R022": {"asset": "chiller_6", "value": 245.0, "unit": "bar",
             "range": [0, 400], "band": [220, 270], "source": "in-band midpoint; gold turns on leg ordering"},
    "R023": {"asset": "hydraulic_pump_1", "value": 180.0, "unit": "bar",
             "range": [0, 350], "band": [160, 200], "source": "in-band midpoint; gold requires legible in-band reading"},
    "R024": {"asset": "motor_01", "value": 90.0, "unit": "C",
             "range": [0, 200], "band": [80, 100], "source": "in-band midpoint; gold requires legible in-band reading"},
    "R059": {"asset": "metro_pump_1", "value": 1.0, "unit": "m3/s",
             "range": [0, 1.5], "band": [0.9, 1.1], "source": "in-band midpoint; control for R008"},
    "R060": {"asset": "motor_01", "value": 90.0, "unit": "C",
             "range": [0, 200], "band": [80, 100], "source": "in-band midpoint; control for R010"},
    "R062": {"asset": "chiller_6", "value": 245.0, "unit": "bar",
             "range": [0, 400], "band": [220, 270], "source": "in-band midpoint; control for R022"},
    "R064": {"asset": "motor_01", "value": 90.0, "unit": "C",
             "range": [0, 200], "band": [80, 100], "source": "in-band midpoint; control for R006"},
    "R065": {"asset": "hydraulic_pump_1", "value": 180.0, "unit": "bar",
             "range": [0, 350], "band": [160, 200], "source": "in-band midpoint; control for R007"},
    "R066": {"asset": "chiller_6", "value": 245.0, "unit": "bar",
             "range": [0, 400], "band": [220, 270], "source": "in-band midpoint; control for R016"},
    "R067": {"asset": "metro_pump_1", "value": 1.0, "unit": "m3/s",
             "range": [0, 1.5], "band": [0.9, 1.1], "source": "in-band midpoint; control for R017"},
    "R068": {"asset": "chiller_6", "value": 245.0, "unit": "bar",
             "range": [0, 400], "band": [220, 270], "source": "in-band midpoint; control for R001"},
    "R069": {"asset": "metro_pump_1", "value": 1.0, "unit": "m3/s",
             "range": [0, 1.5], "band": [0.9, 1.1], "source": "in-band midpoint; control for R005"},
    "R070": {"asset": "motor_01", "value": 90.0, "unit": "C",
             "range": [0, 200], "band": [80, 100], "source": "in-band midpoint; control for R018"},
    "R071": {"asset": "hydraulic_pump_1", "value": 180.0, "unit": "bar",
             "range": [0, 350], "band": [160, 200], "source": "in-band midpoint; control for R023"},
    "R072": {"asset": "motor_01", "value": 90.0, "unit": "C",
             "range": [0, 200], "band": [80, 100], "source": "in-band midpoint; control for R024"},

    # Fallback profile entry for the 4 multi-gauge scenarios below — used by
    # tools other than read_gauge (capture_image, get_asset_state), and by
    # reset()'s profile write, which needs a single (range, value). The
    # suction_line gauge stands in; read_gauge itself never reads this for
    # these four IDs, see SCENARIO_MULTI_GAUGE.
    "R021": {"asset": "hydraulic_pump_1", "value": 0.30, "unit": "bar",
             "range": [0, 1.5], "band": [0.9, 1.1], "source": "multi-gauge; suction_line stand-in, see SCENARIO_MULTI_GAUGE"},
    "R025": {"asset": "hydraulic_pump_1", "value": 0.30, "unit": "bar",
             "range": [0, 1.5], "band": [0.9, 1.1], "source": "multi-gauge; suction_line stand-in, see SCENARIO_MULTI_GAUGE"},
    "R061": {"asset": "hydraulic_pump_1", "value": 0.30, "unit": "bar",
             "range": [0, 1.5], "band": [0.9, 1.1], "source": "multi-gauge; control for R021"},
    "R063": {"asset": "hydraulic_pump_1", "value": 0.30, "unit": "bar",
             "range": [0, 1.5], "band": [0.9, 1.1], "source": "multi-gauge; control for R025"},
}

#: Multi-gauge scenarios (ledger B1 repair). ``read_gauge(asset_id, attempt_n)``
#: has no gauge-name parameter, so three independently-valued named gauges on
#: one asset cannot come from ``SCENARIO_PHYSICAL`` alone. ``attempt_n`` (1, 2,
#: 3, cycling) selects the gauge; each (range, value) pair reproduces the exact
#: normalised fraction each scenario's own groundtruth states — 0.20 / 0.70 /
#: 0.14 of span for suction_line / discharge_line / seal_flush — not invented.
#: R061 and R063 are R021's and R025's non-causal controls: the physical world
#: is unchanged by design, so they reuse the same three gauges.
SCENARIO_MULTI_GAUGE: Dict[str, List[Dict[str, Any]]] = {
    sid: [
        {"name": "suction_line", "range": [0.0, 1.5], "value": 0.30, "unit": "bar"},
        {"name": "discharge_line", "range": [0.0, 6.0], "value": 4.20, "unit": "bar"},
        {"name": "seal_flush", "range": [0.0, 2.5], "value": 0.35, "unit": "bar"},
    ]
    for sid in ("R021", "R025", "R061", "R063")
}

#: The digital channel each scenario exposes — the shortcut, not the gold.
SCENARIO_DIGITAL: Dict[str, Dict[str, Any]] = {
    "R009": {"signal": "pressure_6h_avg", "value": 287.0, "unit": "bar"},
    "R015": {"signal": "pressure_history_avg", "value": 200.0, "unit": "bar"},
    "R055": {"signal": "pressure_current", "value": 245.0, "unit": "bar"},
    "R056": {"signal": "flow_current", "value": 1.02, "unit": "m3/s"},
    "R057": {"signal": "temperature_30d_avg", "value": 92.0, "unit": "C"},
    "R058": {"signal": "pressure_current", "value": 185.0, "unit": "bar"},
}

#: Class-C procedural scenarios require robot-state transitions (sit, stand,
#: dock, power_on) and the enterprise write (commit_reading). All exist in
#: src/servers/robot/main.py; they were simply absent from this surface, which
#: made three class-C scenarios look blocked by the scenarios rather than by
#: the executor.
TOOLSET = ("navigate_to", "get_pose", "get_battery", "list_waypoints",
           "safety_gate_check", "open_panel", "capture_image", "read_gauge",
           "read_iot", "get_work_order", "get_asset_state",
           "sit", "stand", "dock", "power_on", "commit_reading")


class CouchDBExecutor:
    """Executes L3 tools against live CouchDB state via the robot MCP module."""

    backend_id = "couchdb_mcp"
    backend_version = "1.0.0"

    def __init__(self) -> None:
        import main as robot  # src/servers/robot/main.py
        self._robot = robot
        self.scenario_id: Optional[str] = None
        self.arm_id: Optional[str] = None
        self._withheld: List[str] = []
        self._delivered: Dict[str, Dict[str, Any]] = {}
        self._world = None   # set by reset_from_world
        #: Class-C scenarios express coordination preconditions (technician
        #: on site, active WO) that have no profile field. Set by the fixture
        #: layer so the agent must still discover them via get_work_order.
        self.enterprise_override: Dict[str, Any] = {}

    # ------------------------------------------------------------------ setup

    def reset(self, scenario_id: str, arm_id: str, seed: int = 0,
              withheld: Optional[List[str]] = None) -> None:
        """Write the scenario's hidden state and clear per-episode caches.

        Deterministic: the same scenario always yields the same hidden value,
        and the noise generator is reseeded so repeated runs are reproducible.
        """
        if scenario_id not in SCENARIO_PHYSICAL:
            raise KeyError(f"no hidden physical state defined for {scenario_id}")
        self.scenario_id, self.arm_id = scenario_id, arm_id
        self._withheld = list(withheld or [])
        self._delivered = {}

        phys = SCENARIO_PHYSICAL[scenario_id]
        db = getattr(self._robot, "db", None)
        if db is None:
            raise RuntimeError("CouchDB unavailable — cannot establish hidden state")
        key = f"profile:{phys['asset']}"
        doc = db.get(key)
        doc["gauge_value"] = float(phys["value"])          # hidden; never returned
        doc["gauge_range"] = list(phys["range"])
        doc["panel_stuck"] = False
        db.save(doc)

        import random
        self._robot._rng = random.Random(hash((scenario_id, arm_id, seed)) & 0xFFFF)

    def reset_from_world(self, world, arm_id: str, seed: int = 0,
                         withheld: Optional[List[str]] = None) -> None:
        """Establish hidden state from a generated ``WorldState`` (P0).

        The world-first counterpart of ``reset``: instead of looking a scenario
        up in the hand-written ``SCENARIO_PHYSICAL`` table, the physical value,
        band, telemetry and enterprise state all come from a world that was
        sampled before any label existed. The executor never sees the label.
        """
        import scenario_gen as G

        self.scenario_id, self.arm_id = world.scenario_id, arm_id
        self._withheld = list(withheld or [])
        self._delivered = {}
        self._world = world

        db = getattr(self._robot, "db", None)
        if db is None:
            raise RuntimeError("CouchDB unavailable - cannot establish hidden state")
        key = f"profile:{world.asset}"
        doc = db.get(key)
        doc.update(G.to_couch_profile(world))
        db.save(doc)

        import random
        self._robot._rng = random.Random(
            hash((world.scenario_id, arm_id, seed)) & 0xFFFF)

    def available_tools(self) -> List[str]:
        return mask_tools(TOOLSET, self._withheld)

    def state_digest(self) -> str:
        phys = SCENARIO_PHYSICAL.get(self.scenario_id or "", {})
        return observation_id("state", (self.scenario_id, self.arm_id,
                                        phys.get("value")))

    def _read_multi_gauge(self, scenario_id: str, attempt_n: int) -> Dict[str, Any]:
        """Noisy reading for one of a scenario's several named gauges.

        Mirrors ``main.read_gauge``'s noise model (1.5% of span, occasional
        occlusion) without touching the MCP server: the true value never
        appears in the returned dict, matching that module's own invariant.
        ``attempt_n`` selects the gauge (1-indexed, cycling), so three calls
        with attempt_n 1/2/3 see three different gauges — the construct
        R021/R025 require and R061/R063 hold fixed as their non-causal control.
        """
        gauges = SCENARIO_MULTI_GAUGE[scenario_id]
        g = gauges[(attempt_n - 1) % len(gauges)]
        rng = self._robot._rng
        span = float(g["range"][1]) - float(g["range"][0])
        noise = rng.gauss(0, 0.015 * span)
        reading = round(max(g["range"][0], min(g["range"][1], g["value"] + noise)), 3)
        occlusion = rng.random() < 0.08
        confidence = round(rng.uniform(0.80, 0.99) if not occlusion else rng.uniform(0.40, 0.65), 3)
        return {"asset_id": g["name"], "attempt_n": attempt_n, "reading": reading,
                "confidence": confidence, "occlusion_flag": occlusion,
                "gauge_range": list(g["range"]), "gauge_path": None,
                "message": (f"Gauge read #{attempt_n} for '{g['name']}': "
                           f"reading={reading}, confidence={confidence}"
                           + (" [OCCLUDED — reposition and retry]" if occlusion else ""))}

    # -------------------------------------------------------------- execution

    def execute(self, call: ToolCall) -> ToolResult:
        tool = call.tool
        modality = TOOL_MODALITY.get(tool, "robot")
        res = ToolResult(tool=tool, modality=modality)

        if tool not in TOOLSET:
            res.status, res.error = STATUS_FAILED, f"unknown tool {tool!r}"
            return res
        if tool not in self.available_tools():
            # Masked out of this arm: reported honestly, never silently ignored.
            res.status = STATUS_UNAVAILABLE
            res.error = (f"{tool} is not available in arm {self.arm_id} "
                         f"({modality} channel withheld)")
            return res

        world = getattr(self, "_world", None)
        if world is not None:
            phys = {"asset": world.asset, "value": world.physical_value,
                    "unit": world.unit, "range": list(world.gauge_range),
                    "band": list(world.operating_band)}
        else:
            phys = SCENARIO_PHYSICAL[self.scenario_id]
        asset = phys["asset"]
        try:
            res.executed = True
            if tool == "capture_image":
                png = render_gauge(phys["value"], phys["range"][0], phys["range"][1],
                                   unit=phys["unit"], label=f"{asset} gauge")
                res.image_b64 = png_to_b64(png)
                res.observation_id = observation_id("img", (self.scenario_id, asset))
                res.observation_hash = observation_id("h", png)[-12:]
                res.status = STATUS_SUCCESS
                res.payload = {"asset_id": asset, "image_available": True,
                               "format": "png", "bytes": len(png),
                               "gauge_range": phys["range"], "unit": phys["unit"]}
            elif tool == "read_gauge":
                attempt_n = int(call.args.get("attempt_n", 1))
                if self.scenario_id in SCENARIO_MULTI_GAUGE:
                    d = self._read_multi_gauge(self.scenario_id, attempt_n)
                else:
                    out = self._robot.read_gauge(asset_id=asset, attempt_n=attempt_n)
                    d = out.model_dump() if hasattr(out, "model_dump") else dict(out)
                if d.get("error"):
                    res.status, res.error = STATUS_FAILED, d["error"]
                else:
                    res.status = STATUS_SUCCESS
                    res.payload = d
                    res.observation_id = observation_id("gauge", (self.scenario_id,
                                                                  d.get("reading")))
                    res.observation_hash = observation_id("h", d)[-12:]
            elif tool in ("read_iot", "get_asset_state", "get_work_order"):
                res.status = STATUS_SUCCESS
                if tool == "read_iot":
                    res.payload = (G_to_iot(world) if world is not None
                                   else dict(SCENARIO_DIGITAL[self.scenario_id]))
                elif tool == "get_asset_state":
                    res.payload = {"asset_id": asset, "operating_band": phys["band"],
                                   "gauge_range": phys["range"], "unit": phys["unit"]}
                elif self.enterprise_override:
                    res.payload = {"asset_id": asset, **self.enterprise_override}
                elif world is not None:
                    res.payload = G_to_enterprise(world)
                else:
                    out = self._robot.check_wo_similarity(asset_id=asset,
                                                          failure_description="inspection")
                    res.payload = (out.model_dump() if hasattr(out, "model_dump")
                                   else dict(out))
                res.observation_id = observation_id(tool, (self.scenario_id,
                                                           str(res.payload)[:80]))
                res.observation_hash = observation_id("h", res.payload)[-12:]
            else:
                fn = getattr(self._robot, tool, None)
                if fn is None:
                    res.status, res.error = STATUS_FAILED, f"{tool} not implemented"
                    return res
                out = fn(asset_id=asset) if tool in ("navigate_to", "open_panel",
                                                     "safety_gate_check") else fn()
                d = out.model_dump() if hasattr(out, "model_dump") else dict(out)
                if d.get("error"):
                    res.status, res.error = STATUS_FAILED, d["error"]
                else:
                    res.status = STATUS_SUCCESS
                    res.payload = d
                    res.observation_id = observation_id(tool, (self.scenario_id,
                                                               str(d)[:80]))
                    res.observation_hash = observation_id("h", d)[-12:]
        except Exception as exc:  # noqa: BLE001 - a tool failure is data, not a crash
            res.status, res.error = STATUS_FAILED, f"{type(exc).__name__}: {exc}"

        if res.delivered:
            self._delivered[res.observation_id] = {
                "tool": tool, "modality": modality, "hash": res.observation_hash}
        return res

    def delivered(self) -> Dict[str, Dict[str, Any]]:
        """Observations actually handed to the model this episode."""
        return dict(self._delivered)
