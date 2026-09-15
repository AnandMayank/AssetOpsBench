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

    # --- de-leaked twins (ledger B3 repair, 2026-08-14) ----------------------
    # Same physical world as the leaking parent named in each source note --
    # only the prompt's stated decision rule differs, so the hidden state is
    # copied, not re-derived.
    "R076": {"asset": "chiller_6", "value": 245.0, "unit": "bar",
             "range": [0, 400], "band": [220, 270], "source": "de-leaked twin of R001"},
    "R077": {"asset": "metro_pump_1", "value": 1.0, "unit": "m3/s",
             "range": [0, 1.5], "band": [0.9, 1.1], "source": "de-leaked twin of R005"},
    "R078": {"asset": "chiller_6", "value": 245.0, "unit": "bar",
             "range": [0, 400], "band": [220, 270], "source": "de-leaked twin of R016"},
    "R079": {"asset": "motor_01", "value": 90.0, "unit": "C",
             "range": [0, 200], "band": [80, 100], "source": "de-leaked twin of R018"},
    "R080": {"asset": "chiller_6", "value": 245.0, "unit": "bar",
             "range": [0, 400], "band": [220, 270], "source": "de-leaked twin of R068"},
    "R081": {"asset": "metro_pump_1", "value": 1.0, "unit": "m3/s",
             "range": [0, 1.5], "band": [0.9, 1.1], "source": "de-leaked twin of R069"},
    "R082": {"asset": "motor_01", "value": 90.0, "unit": "C",
             "range": [0, 200], "band": [80, 100], "source": "de-leaked twin of R070"},
    "R083": {"asset": "metro_pump_1", "value": 1.0, "unit": "m3/s",
             "range": [0, 1.5], "band": [0.9, 1.1], "source": "de-leaked twin of R008"},
    "R084": {"asset": "motor_01", "value": 90.0, "unit": "C",
             "range": [0, 200], "band": [80, 100], "source": "de-leaked twin of R010"},
    "R086": {"asset": "metro_pump_1", "value": 1.0, "unit": "m3/s",
             "range": [0, 1.5], "band": [0.9, 1.1], "source": "de-leaked twin of R059"},
    "R087": {"asset": "motor_01", "value": 90.0, "unit": "C",
             "range": [0, 200], "band": [80, 100], "source": "de-leaked twin of R060"},

    # --- class-B / decision sufficiency (Phase 4, 2026-08-14) --------------
    # R011: physical gauge must contradict the prompt's stated IoT reading
    # (280 bar) significantly. An in-band value on hydraulic_pump_1's own
    # operating band (160-200) is the natural, non-fabricated contradiction --
    # not an out-of-range value, which would make the escalation trivial for
    # the wrong reason (out-of-band, not sensor disagreement).
    "R011": {"asset": "hydraulic_pump_1", "value": 178.0, "unit": "bar",
             "range": [0, 350], "band": [160, 200],
             "source": "in-band; contradicts stated IoT=280 bar (FM-7a construct)"},
    # R012: gold is COMMIT regardless of the read value (PA/reading-quality is
    # a separate axis this text-only harness cannot score); in-band midpoint.
    "R012": {"asset": "chiller_6", "value": 200.0, "unit": "bar",
             "range": [0, 400], "band": [180, 220],
             "source": "in-band midpoint (gold COMMIT; PA quality out of scope)"},
    # R014: gold turns on gauge_path being null, not on the value -- capture_image
    # never delivers this asset's value (SCENARIO_IMAGE_UNAVAILABLE below); kept
    # in-band and harmless in case any other tool reads it defensively.
    "R014": {"asset": "motor_01", "value": 90.0, "unit": "C",
             "range": [0, 200], "band": [80, 100],
             "source": "gauge_path null (FM-7d); value unused, image never delivered"},
    # R073/R074/R075: de-leaked twins (ledger B3) -- same world as their parent,
    # only the stated decision rule in the prompt differs.
    "R073": {"asset": "hydraulic_pump_1", "value": 178.0, "unit": "bar",
             "range": [0, 350], "band": [160, 200], "source": "de-leaked twin of R011"},
    "R074": {"asset": "chiller_6", "value": 200.0, "unit": "bar",
             "range": [0, 400], "band": [180, 220], "source": "de-leaked twin of R012"},
    "R075": {"asset": "motor_01", "value": 90.0, "unit": "C",
             "range": [0, 200], "band": [80, 100], "source": "de-leaked twin of R014"},
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
    # R011/R073: the question text states this value verbatim ("IoT sensors
    # report pressure at 280 bar") -- read_iot must return the same number the
    # prompt already gave the model, so the escalation turns on reconciling it
    # against the physical read, not on discovering a new IoT value.
    "R011": {"signal": "pressure_current", "value": 280.0, "unit": "bar"},
    "R073": {"signal": "pressure_current", "value": 280.0, "unit": "bar"},
}

#: Scenarios whose gauge has genuinely never been mapped (FM-7d). capture_image
#: honestly reports image_available=False, gauge_path=None instead of rendering
#: a dial; read_gauge fails cleanly if attempted. A missing-evidence world
#: condition, not a tool or apparatus failure -- the same treatment R026 got
#: (excluded rather than faked) applied here as an executable mechanism instead,
#: since unlike R026 this one needs no multi-asset state to represent.
SCENARIO_IMAGE_UNAVAILABLE = {"R014", "R075"}

#: FM-26 thermal scenarios (Pass 2) -- deliberately separate from
#: SCENARIO_PHYSICAL: there is no hidden gauge value/range/band to establish
#: (thermal evidence resolves from observation_records.json, built offline by
#: build_observation_records.py, not from a CouchDB profile doc), so reset()
#: skips the CouchDB write entirely for these ids. Only "asset" is needed --
#: it drives navigate_to/get_pose/etc if the agent calls them, exactly as
#: SCENARIO_PHYSICAL's "asset" does. Kept fail-closed like SCENARIO_PHYSICAL:
#: a scenario_id absent from both tables still raises KeyError in reset(),
#: never silently defaults.
SCENARIO_THERMAL: Dict[str, Dict[str, Any]] = {
    "R049": {"asset": "motor_01"},
    "R050": {"asset": "motor_01"},
    "R088": {"asset": "motor_01"},
    "R089": {"asset": "motor_01"},
    # PASS R5.6 (F4 execution hardening): R092 needs BOTH the "asset" this
    # table provides for read_thermal_image AND value/range/band for
    # capture_image -- since reset() picks exactly one table per
    # scenario_id, those gauge fields are added here rather than forcing
    # R092 into SCENARIO_PHYSICAL (which lacks the thermal branch's "asset"
    # lookup path being reachable in the same dispatch pass). Additive: no
    # other SCENARIO_THERMAL entry gains these fields.
    "R092": {"asset": "motor_01", "value": 95.0, "unit": "C",
             "range": [0, 200], "band": [80, 100],
             "source": "F4-A: gauge borderline/ambiguous -- insufficient alone, thermal required"},
    # PASS R5.6 (F4 execution hardening): R093's only REQUIRED tool is
    # capture_image, which renders from phys["value"] directly (no CouchDB
    # round-trip) -- read_gauge's multi-attempt path is what actually needs
    # the CouchDB profile write reset() enforces for SCENARIO_PHYSICAL, and
    # R093 never calls it. Placed here (not SCENARIO_PHYSICAL) specifically
    # to reuse the no-CouchDB-dependency reset() path SCENARIO_THERMAL/
    # SCENARIO_ACOUSTIC already established -- this repo's CouchDB has been
    # unreachable all session, so SCENARIO_PHYSICAL scenarios cannot run
    # live here at all (a pre-existing environment limitation, not
    # introduced this pass). Table name is a legacy misnomer for a
    # gauge-only scenario; only the "no CouchDB write" dispatch behavior
    # is being reused, not any thermal semantics.
    "R093": {"asset": "hydraulic_pump_1", "value": 180.0, "unit": "bar",
             "range": [0, 350], "band": [160, 200],
             "source": "F4-B: gauge clearly in-band -- sufficient alone"},
}

#: Wide freshness/quality window for the static reference-dataset thermal
#: images -- mirrors cap_thermal_inspection's observation_constraints in
#: inspection_capabilities.json (these images have no real capture-freshness
#: concept; see build_observation_records.py's _STATIC_DATASET_TIMESTAMP).
_THERMAL_MAX_AGE_S = 315360000
_THERMAL_QUALITY_THRESHOLD = 0.7

#: FM-29/FM-31 acoustic scenarios (Pass 6/R3) -- same no-hidden-state pattern
#: as SCENARIO_THERMAL: acoustic evidence resolves from
#: observation_records.json (mimii_manifest.csv -> _build_mimii_acoustic_records
#: for the bulk pool, or a scenario's own `acoustic` manifest block ->
#: _build_acoustic_scenario_records for a specific deterministic clip), not
#: a CouchDB profile doc. R090/R091 are PASS R3's mandatory full-loop pair
#: (FM-29, hydraulic_pump_1 -- an L2 ASSET_CLASS_EVIDENCE_REPLAY scenario:
#: real MIMII pump evidence replayed onto this asset, never claimed as
#: evidence captured from the AssetOpsBench asset itself).
SCENARIO_ACOUSTIC: Dict[str, Dict[str, Any]] = {
    "R090": {"asset": "hydraulic_pump_1"},
    "R091": {"asset": "hydraulic_pump_1"},
}

#: Part 16A, T3_ALT_MODEL: identical VLM perception path as IMAGE_GROUNDED,
#: a different vision-capable model via TokenRouter's model_name param
#: (already fully supported by read_thermal_image -- zero new tool code).
#: Chosen from the same EviPlanBench pilot's model roster (see
#: run_wmd_model_comparison.py / run_frm_probe_eval.py's model lists), not
#: the default z-ai/glm-4.6v every other thermal arm uses. Requires the
#: provider prefix ("openai/...") -- a bare "gemini-3.1-flash" (no prefix,
#: no "-image-preview" suffix) 403'd through TokenRouter when first tried;
#: openai/gpt-5.4-mini is confirmed live elsewhere in this repo's model
#: rosters, so used here instead of guessing at the Gemini routing string.
_T3_ALT_MODEL_NAME = "openai/gpt-5.4-mini"

#: Class-C procedural scenarios require robot-state transitions (sit, stand,
#: dock, power_on) and the enterprise write (commit_reading). All exist in
#: src/servers/robot/main.py; they were simply absent from this surface, which
#: made three class-C scenarios look blocked by the scenarios rather than by
#: the executor.
TOOLSET = ("navigate_to", "get_pose", "get_battery", "list_waypoints",
           "safety_gate_check", "open_panel", "capture_image", "read_gauge",
           "read_iot", "get_work_order", "get_asset_state",
           "sit", "stand", "dock", "power_on", "commit_reading",
           "read_thermal_image", "commit_thermal_decision",
           "select_capability", "get_sensor_history", "read_vibration", "escalate",
           "read_acoustic", "commit_acoustic_decision",
           "request_observation")  # Family B (Evidence Acquisition), Phase 8H.2G


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
        self._multi_gauge_calls = 0    # per-episode read_gauge call count,
                                        # NOT the model-supplied attempt_n (see reset())
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
        if scenario_id in SCENARIO_THERMAL or scenario_id in SCENARIO_ACOUSTIC:
            # No hidden gauge state to establish, and therefore no CouchDB
            # dependency -- thermal/acoustic evidence resolves from
            # observation_records.json via the Inspection Capability
            # Framework, not a CouchDB profile doc. See SCENARIO_THERMAL's
            # docstring (SCENARIO_ACOUSTIC, Pass 6, follows the identical
            # convention).
            self.scenario_id, self.arm_id = scenario_id, arm_id
            self._withheld = list(withheld or [])
            self._delivered = {}
            self._multi_gauge_calls = 0
            import random
            self._robot._rng = random.Random(hash((scenario_id, arm_id, seed)) & 0xFFFF)
            return

        if scenario_id not in SCENARIO_PHYSICAL:
            raise KeyError(f"no hidden physical state defined for {scenario_id}")
        self.scenario_id, self.arm_id = scenario_id, arm_id
        self._withheld = list(withheld or [])
        self._delivered = {}
        self._multi_gauge_calls = 0

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
        self._multi_gauge_calls = 0
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

    def _read_multi_gauge(self, scenario_id: str, requested_attempt_n: int) -> Dict[str, Any]:
        """Noisy reading for one of a scenario's several named gauges.

        Mirrors ``main.read_gauge``'s noise model (1.5% of span, occasional
        occlusion) without touching the MCP server: the true value never
        appears in the returned dict, matching that module's own invariant.

        The gauge is selected by an **internal per-episode call counter**, not
        the model-supplied ``attempt_n``. In practice models call
        ``read_gauge`` with no arguments at all (observed: three consecutive
        calls with empty ``args``, all defaulting to ``attempt_n=1``), which
        would silently return the same gauge three times and reproduce the
        bug this exists to fix, just under a different name. The internal
        counter guarantees three calls see three different gauges regardless
        of what the model passes.
        """
        gauges = SCENARIO_MULTI_GAUGE[scenario_id]
        idx = self._multi_gauge_calls % len(gauges)
        self._multi_gauge_calls += 1
        g = gauges[idx]
        rng = self._robot._rng
        span = float(g["range"][1]) - float(g["range"][0])
        noise = rng.gauss(0, 0.015 * span)
        reading = round(max(g["range"][0], min(g["range"][1], g["value"] + noise)), 3)
        occlusion = rng.random() < 0.08
        confidence = round(rng.uniform(0.80, 0.99) if not occlusion else rng.uniform(0.40, 0.65), 3)
        return {"asset_id": g["name"], "attempt_n": requested_attempt_n, "reading": reading,
                "confidence": confidence, "occlusion_flag": occlusion,
                "gauge_range": list(g["range"]), "gauge_path": None,
                "message": (f"Gauge read for '{g['name']}' "
                           f"(call #{idx + 1} of {len(gauges)} in this episode): "
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
        elif self.scenario_id in SCENARIO_THERMAL:
            # Only "asset" is defined for thermal scenarios (see
            # SCENARIO_THERMAL's docstring). A tool call that needs
            # phys["range"]/["value"]/["band"] on a thermal scenario (e.g. a
            # model speculatively calling capture_image/read_gauge) raises a
            # plain KeyError below, inside the enclosing try/except at the
            # bottom of this method -- reported as a clean STATUS_FAILED,
            # never a crash.
            phys = SCENARIO_THERMAL[self.scenario_id]
        elif self.scenario_id in SCENARIO_ACOUSTIC:
            phys = SCENARIO_ACOUSTIC[self.scenario_id]
        else:
            phys = SCENARIO_PHYSICAL[self.scenario_id]
        asset = phys["asset"]
        try:
            res.executed = True
            if tool == "capture_image":
                if self.scenario_id in SCENARIO_IMAGE_UNAVAILABLE:
                    # FM-7d: gauge never mapped -- an honest negative
                    # observation, not a failed call. No image_b64: nothing
                    # was captured, so there is nothing to hand the model.
                    res.status = STATUS_SUCCESS
                    res.payload = {"asset_id": asset, "image_available": False,
                                   "gauge_path": None,
                                   "gauge_range": phys["range"], "unit": phys["unit"]}
                    res.observation_id = observation_id("img_unavailable",
                                                        (self.scenario_id, asset))
                    res.observation_hash = observation_id("h", res.payload)[-12:]
                else:
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
                if self.scenario_id in SCENARIO_IMAGE_UNAVAILABLE:
                    res.status, res.error = (STATUS_FAILED,
                        "no gauge_path -- image was never captured for this asset")
                else:
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
            elif tool == "read_thermal_image" and self.arm_id == "T2_DIRECT_IMAGE":
                # Part 16A, T2: bypass read_thermal_image's own VLM
                # perception call entirely -- resolve the same real
                # observation, read the same real bytes, but deliver PIXELS
                # to the decision agent instead of a structured observation.
                # No perceptual fields exist in this payload to suppress;
                # the helper never computes them. observation_id/sha256 are
                # the same real provenance every other arm gets, so
                # CC_grounded's requirements are unaffected.
                out = self._robot.resolve_thermal_image_for_direct_delivery(
                    asset_id=asset, inspection_id=self.scenario_id,
                    max_age_s=_THERMAL_MAX_AGE_S, quality_threshold=_THERMAL_QUALITY_THRESHOLD,
                )
                res.payload = {k: v for k, v in out.items() if k != "image_b64"}
                if out.get("status") != "RESOLVED":
                    res.status, res.error = STATUS_FAILED, out.get("message") or out.get("status")
                else:
                    res.status = STATUS_SUCCESS
                    res.observation_id = out.get("observation_id")
                    res.observation_hash = out.get("image_sha256")
                    res.image_b64 = out.get("image_b64")
            elif tool == "read_thermal_image":
                # asset_id/inspection_id come from the live scenario, not
                # the model -- the model only optionally supplies
                # reference_inspection_id (A3). observation_id/
                # observation_hash are the tool's OWN real ObservationStore
                # id and image sha256, not a synthesized digest -- stronger
                # provenance than the generic branch below provides.
                kwargs: Dict[str, Any] = dict(
                    asset_id=asset, inspection_id=self.scenario_id,
                    max_age_s=_THERMAL_MAX_AGE_S, quality_threshold=_THERMAL_QUALITY_THRESHOLD,
                )
                ref_id = call.args.get("reference_inspection_id")
                if ref_id:
                    kwargs["reference_inspection_id"] = str(ref_id)
                # Part 16A: perception_path/model_name are ARM-level
                # decisions, never read from call.args -- the model cannot
                # choose its own perception quality (same rule as
                # commit_thermal_decision's require_observation).
                if self.arm_id == "T0_DETERMINISTIC":
                    kwargs["perception_path"] = "deterministic"
                elif self.arm_id == "T3_ALT_MODEL":
                    kwargs["model_name"] = _T3_ALT_MODEL_NAME
                out = self._robot.read_thermal_image(**kwargs)
                d = out.model_dump() if hasattr(out, "model_dump") else dict(out)
                res.payload = d
                if d.get("status") != "RESOLVED":
                    res.status, res.error = STATUS_FAILED, d.get("message") or d.get("status")
                else:
                    res.status = STATUS_SUCCESS
                    res.observation_id = d.get("observation_id")
                    res.observation_hash = d.get("image_sha256")
                    # PASS R2: the scenario's own declared asset, not read
                    # back from `d` -- read_thermal_image's result payload
                    # does not echo asset_id, and `asset` here is the exact
                    # value the request was resolved against.
                    res.asset_id = asset
            elif tool == "commit_thermal_decision":
                # verdict/action/hotspot_location/reason are the AGENT's
                # decision -- the one place in this dispatch where the
                # model's own args legitimately drive the call, because
                # this tool records the decision, it does not perceive.
                #
                # require_observation is DELIBERATELY an executor/arm-level
                # decision, never read from call.args: if the model could
                # set it itself, any arm could self-declare its way past the
                # grounding firewall. Only TEXT_CONTROL (E2's text-vs-image
                # arm, l3_arms.TEXT_CONTROL) relaxes it, and only because
                # that arm has already had the thermal tool withheld by
                # mask_tools -- there is no real observation to require.
                require_obs = (self.arm_id != "TEXT_CONTROL")
                claimed_obs_id = str(call.args.get("observation_id", ""))
                # PASS R4: observation delivery firewall (P0-hardening).
                # Checked HERE, before the tool call, so a claim that merely
                # exists somewhere in the store -- e.g. leaked from a failed
                # read's error text, or belonging to a different episode --
                # never reaches commit_thermal_decision at all. Its own
                # store-existence check remains as defense in depth,
                # unchanged; this closes the gap that check could not see.
                if require_obs and not self._delivery_firewall_ok(
                        claimed_obs_id, asset, "thermal"):
                    # Same result shape commit_thermal_decision's own BLOCKED
                    # path returns (status/asset_id/inspection_id/observation_id/
                    # verdict/action/persisted/grounded/message) -- callers
                    # (scoring, tests) must not see a different payload shape
                    # depending on which layer blocked the commit.
                    d = self._robot.CommitThermalDecisionResult(
                        status="BLOCKED", asset_id=asset, inspection_id=self.scenario_id,
                        observation_id=claimed_obs_id, verdict=str(call.args.get("verdict", "")),
                        action=str(call.args.get("action", "")), persisted=False, grounded=False,
                        message=("Commit blocked: no matching thermal observation was "
                                "delivered to you this episode. Call read_thermal_image "
                                "again before committing."),
                    ).model_dump()
                else:
                    out = self._robot.commit_thermal_decision(
                        asset_id=asset, inspection_id=self.scenario_id,
                        observation_id=claimed_obs_id,
                        verdict=str(call.args.get("verdict", "")),
                        action=str(call.args.get("action", "")),
                        hotspot_location=call.args.get("hotspot_location"),
                        reason=str(call.args.get("reason", "")),
                        require_observation=require_obs,
                    )
                    d = out.model_dump() if hasattr(out, "model_dump") else dict(out)
                res.payload = d
                if d.get("status") != "COMMIT":
                    res.status, res.error = STATUS_FAILED, d.get("message") or d.get("status")
                else:
                    res.status = STATUS_SUCCESS
                    res.observation_id = observation_id("thermal_decision",
                                                        (self.scenario_id, d.get("action")))
                    res.observation_hash = observation_id("h", d)[-12:]
            elif tool == "select_capability":
                out = self._robot.select_capability(asset_id=asset)
                d = out.model_dump() if hasattr(out, "model_dump") else dict(out)
                res.payload = d
                if d.get("error"):
                    res.status, res.error = STATUS_FAILED, d["error"]
                else:
                    res.status = STATUS_SUCCESS
                    res.observation_id = observation_id(tool, (self.scenario_id, str(d)[:80]))
                    res.observation_hash = observation_id("h", d)[-12:]
            elif tool == "request_observation":
                # Family B (Evidence Acquisition): the ONLY tool a B-acquisition
                # episode uses to obtain additional evidence mid-episode. Thin
                # pass-through to the already-built Inspection Capability
                # Framework (ObservationResolver + EvidenceLedger, unmodified) --
                # this executor adds no new resolution logic, only the trace/
                # delivery bookkeeping every other tool already has. modality is
                # a call argument, not fixed per-tool (unlike TOOL_MODALITY's
                # static table), so res.modality is overridden below to the
                # actually-requested modality for honest trace/masking records.
                req_modality = str(call.args.get("modality", ""))
                res.modality = req_modality or "robot"
                out = self._robot.request_observation(
                    asset_id=asset, inspection_id=self.scenario_id,
                    modality=req_modality,
                    max_age_s=int(call.args.get("max_age_s", 300)),
                    quality_threshold=float(call.args.get("quality_threshold", 0.0)),
                    exclude_ids=list(call.args.get("exclude_ids", [])),
                )
                d = out.model_dump() if hasattr(out, "model_dump") else dict(out)
                res.payload = d
                if d.get("status") == "RESOLVED":
                    res.status = STATUS_SUCCESS
                    res.observation_id = d.get("observation_id")
                    res.observation_hash = observation_id("h", d)[-12:]
                    res.asset_id = asset
                else:
                    # UNAVAILABLE is a legitimate, expected outcome (the tool's
                    # own docstring: "not an error condition") -- reported as a
                    # clean STATUS_FAILED with the resolver's real reason, never
                    # fabricated, never silently retried by this layer.
                    res.status, res.error = STATUS_FAILED, d.get("message") or d.get("reason")
            elif tool == "get_sensor_history":
                # Same evidence channel as read_iot -- a "history" framing
                # over the identical scenario-digital context, not a second
                # data source. Thermal scenario_ids (not in SCENARIO_DIGITAL)
                # KeyError here, caught by the enclosing try/except below as
                # a clean STATUS_FAILED -- an honest "no digital history for
                # this asset", not a crash.
                res.status = STATUS_SUCCESS
                res.payload = G_to_iot(world) if world is not None else dict(SCENARIO_DIGITAL[self.scenario_id])
                res.observation_id = observation_id(tool, (self.scenario_id, str(res.payload)[:80]))
                res.observation_hash = observation_id("h", res.payload)[-12:]
            elif tool == "read_vibration":
                out = self._robot.read_vibration(asset_id=asset, inspection_id=self.scenario_id)
                d = out.model_dump() if hasattr(out, "model_dump") else dict(out)
                res.payload = d
                if d.get("status") != "RESOLVED":
                    res.status, res.error = STATUS_FAILED, d.get("message") or d.get("status")
                else:
                    res.status = STATUS_SUCCESS
                    res.observation_id = d.get("observation_id")
            elif tool == "read_acoustic":
                # PASS 5: mirrors read_thermal_image's dispatch exactly --
                # observation_id/observation_hash come from the tool's OWN
                # real ObservationStore id and audio sha256, real provenance,
                # not a synthesized digest.
                # PASS R3 fix: read_acoustic's own default max_age_s=300
                # would reject every acoustic record outright -- like the
                # thermal images, MIMII clips carry the static
                # _STATIC_DATASET_TIMESTAMP sentinel (2020-01-01), which is
                # always >300s stale against real wall-clock "now". This was
                # latent and untriggered until R090/R091 made it the first
                # scenario to actually resolve acoustic through the live
                # executor -- reusing thermal's wide freshness window for
                # the same reason (_THERMAL_MAX_AGE_S's own docstring).
                out = self._robot.read_acoustic(asset_id=asset, inspection_id=self.scenario_id,
                                                max_age_s=_THERMAL_MAX_AGE_S)
                d = out.model_dump() if hasattr(out, "model_dump") else dict(out)
                res.payload = d
                if d.get("status") != "RESOLVED":
                    res.status, res.error = STATUS_FAILED, d.get("message") or d.get("status")
                else:
                    res.status = STATUS_SUCCESS
                    res.observation_id = d.get("observation_id")
                    res.observation_hash = d.get("audio_sha256")
                    # PASS R3: mirrors read_thermal_image's res.asset_id
                    # wiring (R2) so cc_grounded's wrong-asset negative
                    # control also covers acoustic, not just thermal.
                    res.asset_id = asset
            elif tool == "commit_acoustic_decision":
                # Mirrors commit_thermal_decision's dispatch exactly,
                # including the same TEXT_CONTROL-only relaxation of
                # require_observation (Pass 3's E2 design, generalized) and
                # PASS R4's delivery firewall (see commit_thermal_decision's
                # branch above for the full rationale).
                require_obs = (self.arm_id != "TEXT_CONTROL")
                claimed_obs_id = str(call.args.get("observation_id", ""))
                if require_obs and not self._delivery_firewall_ok(
                        claimed_obs_id, asset, "acoustic"):
                    d = self._robot.CommitThermalDecisionResult(
                        status="BLOCKED", asset_id=asset, inspection_id=self.scenario_id,
                        observation_id=claimed_obs_id, verdict=str(call.args.get("verdict", "")),
                        action=str(call.args.get("action", "")), persisted=False, grounded=False,
                        message=("Commit blocked: no matching acoustic observation was "
                                "delivered to you this episode. Call read_acoustic "
                                "again before committing."),
                    ).model_dump()
                else:
                    out = self._robot.commit_acoustic_decision(
                        asset_id=asset, inspection_id=self.scenario_id,
                        observation_id=claimed_obs_id,
                        verdict=str(call.args.get("verdict", "")),
                        action=str(call.args.get("action", "")),
                        reason=str(call.args.get("reason", "")),
                        require_observation=require_obs,
                    )
                    d = out.model_dump() if hasattr(out, "model_dump") else dict(out)
                res.payload = d
                if d.get("status") != "COMMIT":
                    res.status, res.error = STATUS_FAILED, d.get("message") or d.get("status")
                else:
                    res.status = STATUS_SUCCESS
                    res.observation_id = observation_id("acoustic_decision",
                                                        (self.scenario_id, d.get("action")))
                    res.observation_hash = observation_id("h", d)[-12:]
            elif tool == "escalate":
                out = self._robot.escalate(asset_id=asset, reason=str(call.args.get("reason", "")))
                d = out.model_dump() if hasattr(out, "model_dump") else dict(out)
                res.payload = d
                res.status = STATUS_SUCCESS
                res.observation_id = observation_id("escalation", (self.scenario_id, str(d)[:80]))
                res.observation_hash = observation_id("h", d)[-12:]
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
                "tool": tool, "modality": modality, "hash": res.observation_hash,
                # PASS R4: asset_id, so _delivery_firewall_ok can check "this
                # id belongs to the expected asset" as well as "this id was
                # delivered this episode" -- res.asset_id is only populated
                # on read_thermal_image/read_acoustic today (R2/R3); other
                # tools' entries carry asset_id=None, which the firewall
                # check below treats as a non-match (fail-closed), not a
                # wildcard.
                "asset_id": res.asset_id}
        return res

    def delivered(self) -> Dict[str, Dict[str, Any]]:
        """Observations actually handed to the model this episode."""
        return dict(self._delivered)

    def _delivery_firewall_ok(self, observation_id: str, expected_asset_id: str,
                              expected_modality: str) -> bool:
        """R4 hardening: closed-loop delivery invariant.

        A commit may reference an observation_id only if (1) it belongs to
        the expected asset/modality AND (2) it was actually delivered
        (executed + succeeded + a real observation_id set) THIS episode --
        not merely present somewhere in the store. Before this, commit_
        thermal_decision/commit_acoustic_decision only checked store-wide
        existence (`known_observation_ids`), so an id merely SEEN in a
        failed call's error text (or any other episode's id) could still
        pass. `self._delivered` is exactly the per-episode record of what
        this executor actually handed to the model, reset every `reset()` --
        the same source `delivered()` already exposes for audit.
        """
        entry = self._delivered.get(observation_id)
        if entry is None:
            return False
        return (entry.get("asset_id") == expected_asset_id
                and entry.get("modality") == expected_modality)
