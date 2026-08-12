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
                out = self._robot.read_gauge(asset_id=asset,
                                             attempt_n=int(call.args.get("attempt_n", 1)))
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
