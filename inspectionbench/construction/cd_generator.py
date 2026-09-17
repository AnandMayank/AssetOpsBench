"""cd_generator.py — Phase 8H.2A proof-of-pattern C/D parameterized-scenario
generator, for exactly 3 templates: T-C-BATTERY_ABORT, T-C-PANEL_STUCK,
T-C-WO_GATE.

Architecture (mirrors A/E exactly, no new pattern invented):

    template params
        -> WorldState (reused unmodified from scenario_gen)
        -> deterministic gold RULE, a pure function of params (never of the
           agent's trace, never regex-scraped from prose)
        -> fixture spec (Fixture, reused unmodified from classc_fixtures)
        -> ex.reset_from_world(world, ...) THEN FixtureSession(...) THEN
           verify -- the same reset-before-fixture order the Phase 8H.2
           blocker repair established, because reset_from_world's own
           to_couch_profile() also unconditionally writes panel_stuck=False
           and would otherwise clobber a just-applied fixture exactly like
           the R001 bug did.
        -> a SCRIPTED (non-model) tool-call sequence executed through the
           REAL CouchDBExecutor, producing a REAL ExecutionTrace with a real
           hash chain -- proves the mechanics without any model/API call.
        -> frozen scoring (ordering_satisfied, normalise_action equality)
           PLUS the three schema-freeze-plan procedural metrics
           (required-action P/R/F1, forbidden-action violation rate,
           precedence satisfaction), computed here for the first time from a
           REAL trace rather than only designed on paper.

No gold value is ever hand-authored per episode. Every gold value here is
`derive_gold_for_template(template_id, params)`, a pure function.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

from scenario_gen import ASSETS, WorldState  # noqa: E402 -- UNCHANGED, reused
from classc_fixtures import Fixture, FixtureSession  # noqa: E402 -- UNCHANGED, reused
from couchdb_executor import CouchDBExecutor  # noqa: E402 -- UNCHANGED, reused
from execution_trace import ExecutionTrace, Stage  # noqa: E402 -- UNCHANGED, reused
from tool_executor import ToolCall, STATUS_SUCCESS  # noqa: E402 -- UNCHANGED, reused
from l3_scoring import normalise_action  # noqa: E402 -- UNCHANGED, reused
from ordering import ordering_satisfied  # noqa: E402 -- UNCHANGED, reused

COMMIT, ESCALATE, ABORT = "COMMIT", "ESCALATE", "ABORT"
ALL_ASSETS = tuple(sorted(ASSETS))  # ("chiller_6", "hydraulic_pump_1", "metro_pump_1", "motor_01")

# ---------------------------------------------------------------------------
# Required-action record: action_requirement is uniformly "attempt", per the
# frozen gold schema (phase8h1_c_gold_schema_freeze_plan.md §1/§9). No
# mode=success/fail is used anywhere in this module.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RequiredAction:
    tool: str
    multiplicity_required: int = 1
    multiplicity_maximum: Optional[int] = None  # None = no source-grounded bound
    action_requirement: str = "attempt"

    def __post_init__(self):
        if self.action_requirement != "attempt":
            raise ValueError(
                f"schema mismatch: action_requirement={self.action_requirement!r} "
                "-- the frozen schema (Gate freeze plan) permits only 'attempt'. "
                "STOP -- do not silently map a success/fail mode.")


@dataclass(frozen=True)
class EpisodeSpec:
    template_id: str
    scenario_id: str            # synthetic, unique per generated instance
    asset: str
    world: WorldState
    fixture: Fixture
    enterprise: Dict[str, Any]
    gold_terminal_action: str
    required_actions: Tuple[RequiredAction, ...]
    forbidden_actions: Tuple[str, ...]
    precedence_pairs: Tuple[Tuple[str, str], ...]  # (a, b): a precedes b when both executed
    params: Dict[str, Any]       # the exact generating parameters (for reproduction)
    source_scenario_id: Optional[str] = None  # e.g. "R016", for the reproduction gate


def world_hash(spec_like: Dict[str, Any]) -> str:
    """Deterministic hash of a JSON-serializable dict, sorted keys. Same
    construction family as tool_executor.observation_id -- unsalted,
    reproducible, not a security hash."""
    blob = json.dumps(spec_like, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Template 1 — T-C-BATTERY_ABORT (source: R016)
# ---------------------------------------------------------------------------
BATTERY_THRESHOLD = 20.0  # frozen, from R016's groundtruth ("below 20% threshold")


def build_battery_abort(asset: str, battery_pct: float, seed: int,
                        rep: int = 0) -> EpisodeSpec:
    if asset not in ALL_ASSETS:
        raise ValueError(f"asset {asset!r} not in {ALL_ASSETS}")
    spec = ASSETS[asset]
    # Physical value/band are not the construct under test here; held at the
    # midpoint of the operating band so the gauge channel is never the
    # decisive variable -- consistent with R016's own "in-band midpoint;
    # gold turns on <non-gauge factor>" convention used throughout Family C.
    mid = (spec.operating_band[0] + spec.operating_band[1]) / 2
    world = WorldState(
        scenario_id=f"GEN-BATTERY_ABORT-{asset}-{seed}-{rep}", asset=asset, unit=spec.unit,
        gauge_range=list(spec.gauge_range), operating_band=list(spec.operating_band),
        physical_value=mid, iot_value=mid, history_mean=mid,
        active_work_order=False, technician_present=False,
        cell="cd_gen_battery_abort", seed=seed)

    abort = battery_pct < BATTERY_THRESHOLD
    gold = ABORT if abort else COMMIT
    if abort:
        required = (RequiredAction("list_waypoints"), RequiredAction("get_battery"),
                   RequiredAction("sit"), RequiredAction("dock"))
        forbidden = ("navigate_to", "open_panel", "capture_image")
        precedence: Tuple[Tuple[str, str], ...] = ()
    else:
        required = (RequiredAction("list_waypoints"), RequiredAction("get_battery"),
                   RequiredAction("navigate_to"), RequiredAction("get_pose"),
                   RequiredAction("open_panel"), RequiredAction("capture_image"))
        forbidden = ()
        precedence = ()

    fixture = Fixture(world.scenario_id, asset,
                      why=f"battery_pct={battery_pct} (generated, threshold={BATTERY_THRESHOLD})",
                      robot_state={"battery_charge_pct": float(battery_pct),
                                  "battery_estimated_runtime_s": round(5400.0 * battery_pct / 85.0, 1)})
    return EpisodeSpec(
        template_id="T-C-BATTERY_ABORT", scenario_id=world.scenario_id, asset=asset, world=world,
        fixture=fixture, enterprise={}, gold_terminal_action=gold, required_actions=required,
        forbidden_actions=forbidden, precedence_pairs=precedence,
        params={"asset": asset, "battery_pct": battery_pct, "threshold": BATTERY_THRESHOLD,
               "seed": seed, "rep": rep},
        source_scenario_id="R016" if (asset == "hydraulic_pump_1" and battery_pct == 9.4) else None)


# ---------------------------------------------------------------------------
# Template 2 — T-C-PANEL_STUCK (source: R001)
# ---------------------------------------------------------------------------
def build_panel_stuck(asset: str, panel_stuck: bool, seed: int, rep: int = 0) -> EpisodeSpec:
    if asset not in ALL_ASSETS:
        raise ValueError(f"asset {asset!r} not in {ALL_ASSETS}")
    spec = ASSETS[asset]
    mid = (spec.operating_band[0] + spec.operating_band[1]) / 2
    world = WorldState(
        scenario_id=f"GEN-PANEL_STUCK-{asset}-{seed}-{rep}", asset=asset, unit=spec.unit,
        gauge_range=list(spec.gauge_range), operating_band=list(spec.operating_band),
        physical_value=mid, iot_value=mid, history_mean=mid,
        active_work_order=False, technician_present=False,
        cell="cd_gen_panel_stuck", seed=seed)

    gold = ESCALATE if panel_stuck else COMMIT
    base = (RequiredAction("list_waypoints"), RequiredAction("get_battery"),
           RequiredAction("navigate_to"), RequiredAction("get_pose"))
    if panel_stuck:
        required = base + (RequiredAction("open_panel", multiplicity_required=3),)
    else:
        required = base + (RequiredAction("open_panel", multiplicity_required=1),
                          RequiredAction("capture_image", multiplicity_required=1))
    fixture = Fixture(world.scenario_id, asset, why=f"panel_stuck={panel_stuck} (generated)",
                      profile={"panel_stuck": bool(panel_stuck)})
    return EpisodeSpec(
        template_id="T-C-PANEL_STUCK", scenario_id=world.scenario_id, asset=asset, world=world,
        fixture=fixture, enterprise={}, gold_terminal_action=gold, required_actions=required,
        forbidden_actions=(), precedence_pairs=(),
        params={"asset": asset, "panel_stuck": panel_stuck, "seed": seed, "rep": rep},
        source_scenario_id="R001" if (asset == "chiller_6" and panel_stuck is True) else None)


# ---------------------------------------------------------------------------
# Template 3 — T-C-WO_GATE (source: R005)
# ---------------------------------------------------------------------------
def build_wo_gate(asset: str, active_work_order: bool, seed: int, rep: int = 0) -> EpisodeSpec:
    if asset not in ALL_ASSETS:
        raise ValueError(f"asset {asset!r} not in {ALL_ASSETS}")
    spec = ASSETS[asset]
    mid = (spec.operating_band[0] + spec.operating_band[1]) / 2
    world = WorldState(
        scenario_id=f"GEN-WO_GATE-{asset}-{seed}-{rep}", asset=asset, unit=spec.unit,
        gauge_range=list(spec.gauge_range), operating_band=list(spec.operating_band),
        physical_value=mid, iot_value=mid, history_mean=mid,
        active_work_order=False, technician_present=False,  # WorldState's own coordination
                                                             # fields stay False: this template's
                                                             # construct is the C-family PROCEDURAL
                                                             # get_work_order check, kept distinct
                                                             # from A's world-level coordination rule
        cell="cd_gen_wo_gate", seed=seed)

    gold = ESCALATE if active_work_order else COMMIT
    base = (RequiredAction("list_waypoints"), RequiredAction("get_battery"),
           RequiredAction("navigate_to"), RequiredAction("get_work_order"))
    precedence = (("get_work_order", "open_panel"),)  # kept even when vacuous on the
                                                      # ESCALATE branch -- per §2 of the
                                                      # schema freeze plan, vacuity is a
                                                      # metric-denominator fact, not a
                                                      # reason to delete the source relation
    if active_work_order:
        required = base
        forbidden = ("open_panel",)
    else:
        required = base + (RequiredAction("get_pose"), RequiredAction("open_panel"),
                          RequiredAction("capture_image"))
        forbidden = ()

    fixture = Fixture(world.scenario_id, asset, why=f"active_work_order={active_work_order} (generated)")
    enterprise = {"active_work_order": bool(active_work_order), "technician_present": False}
    return EpisodeSpec(
        template_id="T-C-WO_GATE", scenario_id=world.scenario_id, asset=asset, world=world,
        fixture=fixture, enterprise=enterprise, gold_terminal_action=gold, required_actions=required,
        forbidden_actions=forbidden, precedence_pairs=precedence,
        params={"asset": asset, "active_work_order": active_work_order, "seed": seed, "rep": rep},
        source_scenario_id="R005" if (asset == "chiller_6" and active_work_order is True) else None)


TEMPLATES: Dict[str, Callable[..., EpisodeSpec]] = {
    "T-C-BATTERY_ABORT": build_battery_abort,
    "T-C-PANEL_STUCK": build_panel_stuck,
    "T-C-WO_GATE": build_wo_gate,
}


# ---------------------------------------------------------------------------
# Apply + verify (reset-before-fixture order, per the Phase 8H.2 blocker fix)
# ---------------------------------------------------------------------------
def apply_fixture_and_check(ex: CouchDBExecutor, spec: EpisodeSpec) -> Tuple[bool, List[str], Dict[str, Any]]:
    """reset_from_world FIRST, then read back the world state BEFORE any
    fixture is applied, to prove reset_from_world's own clobber-risk fields
    (panel_stuck) start at their nominal value -- so a later pass that finds
    the fixture value is proof the fixture (not a leftover) produced it."""
    ex.reset_from_world(spec.world, "FULL", seed=spec.params.get("seed", 1), withheld=[])
    ex.enterprise_override = dict(spec.enterprise)
    doc = ex._robot.db.get(f"profile:{spec.asset}")
    pre_fixture_state = {"panel_stuck": doc.get("panel_stuck")}
    return True, [], pre_fixture_state


# ---------------------------------------------------------------------------
# Scripted (non-model) episode execution through the REAL executor
# ---------------------------------------------------------------------------
@dataclass
class ScriptedResult:
    spec: EpisodeSpec
    trace_events: List[Dict[str, Any]]
    trace_chain_valid: bool
    executed_calls: List[Dict[str, Any]]  # [{"tool": str, "args": dict}], in order
    verdict: str  # SCRIPTED to equal gold, to test the scorer's correctness,
                  # never a model's decision


def _scripted_call_sequence(spec: EpisodeSpec) -> List[Dict[str, Any]]:
    """Deterministic scripted calls satisfying spec.required_actions exactly
    at their required multiplicity, in the order the required_actions tuple
    already encodes (which is itself the precedence-respecting order used
    when building each spec above)."""
    calls = []
    for ra in spec.required_actions:
        for i in range(ra.multiplicity_required):
            args: Dict[str, Any] = {}
            if ra.tool == "open_panel" and spec.template_id == "T-C-PANEL_STUCK":
                args = {}  # asset resolved by the executor from world state
            calls.append({"tool": ra.tool, "args": args})
    return calls


def run_scripted_episode(ex: CouchDBExecutor, spec: EpisodeSpec) -> ScriptedResult:
    trace = ExecutionTrace(spec.scenario_id, "FULL")
    executed_calls: List[Dict[str, Any]] = []
    for call in _scripted_call_sequence(spec):
        name, args = call["tool"], call["args"]
        trace.append(Stage.REQUESTED, tool=name, args=args)
        res = ex.execute(ToolCall(name, args))
        if res.executed:
            trace.append(Stage.EXECUTED, tool=name, status=res.status, error=res.error)
            executed_calls.append(call)
        if res.status == STATUS_SUCCESS:
            trace.append(Stage.SUCCEEDED, tool=name)
        if res.delivered:
            trace.append(Stage.OBSERVATION_DELIVERED, tool=name, observation_id=res.observation_id,
                         observation_hash=res.observation_hash, modality=res.modality,
                         asset_id=res.asset_id)
    verdict = spec.gold_terminal_action  # SCRIPTED to equal gold: this proves the
                                         # SCORER is correct, never proves anything
                                         # about model behavior (no model was called)
    trace.append(Stage.DECISION, detail={"verdict": verdict})
    return ScriptedResult(spec=spec, trace_events=[e.to_dict() for e in trace.events],
                          trace_chain_valid=trace.verify_chain(), executed_calls=executed_calls,
                          verdict=verdict)


# ---------------------------------------------------------------------------
# Procedural scoring — required-action P/R/F1, forbidden-action violation
# rate, precedence satisfaction. Definitions per
# phase8h1_c_gold_schema_freeze_plan.md §5. Computed here for the first time
# against a REAL trace, not just designed on paper.
# ---------------------------------------------------------------------------
def _executed_tool_multiset(result: ScriptedResult) -> List[str]:
    return [e["tool"] for e in result.trace_events if e["stage"] == "EXECUTED"]


def required_action_prf1(spec: EpisodeSpec, result: ScriptedResult) -> Dict[str, Any]:
    executed = _executed_tool_multiset(result)
    forbidden = set(spec.forbidden_actions)
    executed_nonforbidden = [t for t in executed if t not in forbidden]

    from collections import Counter
    exec_counts = Counter(executed_nonforbidden)
    matched = 0
    for ra in spec.required_actions:
        matched += min(exec_counts.get(ra.tool, 0), ra.multiplicity_required)
    precision_den = len(executed_nonforbidden)
    recall_den = sum(ra.multiplicity_required for ra in spec.required_actions)
    precision = (matched / precision_den) if precision_den else None
    recall = (matched / recall_den) if recall_den else None
    f1 = (2 * precision * recall / (precision + recall)
         if (precision is not None and recall is not None and (precision + recall) > 0) else
         (0.0 if (precision == 0 or recall == 0) else None))
    return {"matched": matched, "precision_den": precision_den, "recall_den": recall_den,
           "precision": precision, "recall": recall, "f1": f1}


def forbidden_violation(spec: EpisodeSpec, result: ScriptedResult) -> Dict[str, Any]:
    executed = set(_executed_tool_multiset(result))
    violated = sorted(executed & set(spec.forbidden_actions))
    return {"forbidden_set": list(spec.forbidden_actions), "violated": violated,
           "violation": bool(violated)}


def precedence_satisfaction(spec: EpisodeSpec, result: ScriptedResult) -> Dict[str, Any]:
    executed = _executed_tool_multiset(result)
    pos = {}
    for i, t in enumerate(executed):
        pos.setdefault(t, []).append(i)
    rows = []
    for a, b in spec.precedence_pairs:
        a_idx, b_idx = pos.get(a), pos.get(b)
        if not a_idx or not b_idx:
            rows.append({"pair": [a, b], "evaluable": False, "satisfied": None,
                        "reason": "vacuous -- one or both actions not executed in this trace"})
            continue
        satisfied = min(a_idx) < min(b_idx)
        rows.append({"pair": [a, b], "evaluable": True, "satisfied": satisfied})
    evaluable = [r for r in rows if r["evaluable"]]
    rate = (sum(1 for r in evaluable if r["satisfied"]) / len(evaluable)) if evaluable else None
    return {"pairs": rows, "n_evaluable": len(evaluable), "satisfaction_rate": rate}


def score_scripted_episode(spec: EpisodeSpec, result: ScriptedResult) -> Dict[str, Any]:
    cc = int(normalise_action(result.verdict) == normalise_action(spec.gold_terminal_action))
    required_order = [ra.tool for ra in spec.required_actions]
    executed_order = _executed_tool_multiset(result)
    ordering_ok = ordering_satisfied(required_order, executed_order)
    return {"CC": cc, "ordering_satisfied": ordering_ok,
           "required_action_prf1": required_action_prf1(spec, result),
           "forbidden_violation": forbidden_violation(spec, result),
           "precedence_satisfaction": precedence_satisfaction(spec, result)}


# ---------------------------------------------------------------------------
# Top-level orchestration: the full generate -> apply -> execute -> score
# flow, properly scoped (FixtureSession opened and closed exactly once, no
# leakage into the next episode).
# ---------------------------------------------------------------------------
@dataclass
class FullEpisodeResult:
    spec: EpisodeSpec
    pre_fixture_state: Dict[str, Any]
    fixture_verified: bool
    fixture_problems: List[str]
    post_fixture_state: Dict[str, Any]
    scripted: ScriptedResult
    score: Dict[str, Any]
    world_hash: str


def generate_and_run(ex: CouchDBExecutor, spec: EpisodeSpec) -> FullEpisodeResult:
    ex.reset_from_world(spec.world, "FULL", seed=spec.params.get("seed", 1), withheld=[])
    ex.enterprise_override = dict(spec.enterprise)
    pre_doc = ex._robot.db.get(f"profile:{spec.asset}")
    pre_fixture_state = {"panel_stuck": pre_doc.get("panel_stuck")}

    with FixtureSession(ex._robot.db, spec.fixture) as sess:
        problems = sess.verify_applied()
        post_doc = ex._robot.db.get(f"profile:{spec.asset}")
        post_fixture_state = {"panel_stuck": post_doc.get("panel_stuck")}
        if spec.fixture.robot_state:
            rs = ex._robot.db.get("robot_state:spot_1")
            post_fixture_state.update({k: rs.get(k) for k in spec.fixture.robot_state})
        result = run_scripted_episode(ex, spec)
        score = score_scripted_episode(spec, result)

    return FullEpisodeResult(
        spec=spec, pre_fixture_state=pre_fixture_state, fixture_verified=not problems,
        fixture_problems=problems, post_fixture_state=post_fixture_state, scripted=result,
        score=score, world_hash=world_hash(spec.params))


def spec_dict(spec: EpisodeSpec) -> Dict[str, Any]:
    """JSON-serializable view of an EpisodeSpec's gold-relevant fields."""
    return {
        "template_id": spec.template_id, "scenario_id": spec.scenario_id, "asset": spec.asset,
        "params": spec.params, "source_scenario_id": spec.source_scenario_id,
        "gold_terminal_action": spec.gold_terminal_action,
        "required_actions": [{"tool": ra.tool, "multiplicity_required": ra.multiplicity_required,
                             "multiplicity_maximum": ra.multiplicity_maximum,
                             "action_requirement": ra.action_requirement}
                            for ra in spec.required_actions],
        "forbidden_actions": list(spec.forbidden_actions),
        "precedence_pairs": [list(p) for p in spec.precedence_pairs],
        "world_hash": world_hash(spec.params),
    }


# ---------------------------------------------------------------------------
# Template 4 — T-C-PRECEDENCE (source: R006, R007)
# Locked interpretation (R006/R007 safeguard): terminal gold is
# unconditionally COMMIT on this world path. Procedural violations
# (get_pose not preceding open_panel) are captured ONLY by the precedence
# metric, never by gold, never by a continuous hazard-weighted CC.
# ---------------------------------------------------------------------------
def build_precedence(asset: str, seed: int, rep: int = 0) -> EpisodeSpec:
    if asset not in ALL_ASSETS:
        raise ValueError(f"asset {asset!r} not in {ALL_ASSETS}")
    aspec = ASSETS[asset]
    mid = (aspec.operating_band[0] + aspec.operating_band[1]) / 2
    world = WorldState(
        scenario_id=f"GEN-PRECEDENCE-{asset}-{seed}-{rep}", asset=asset, unit=aspec.unit,
        gauge_range=list(aspec.gauge_range), operating_band=list(aspec.operating_band),
        physical_value=mid, iot_value=mid, history_mean=mid,
        active_work_order=False, technician_present=False,
        cell="cd_gen_precedence", seed=seed)
    required = (RequiredAction("list_waypoints"), RequiredAction("get_battery"),
               RequiredAction("navigate_to"), RequiredAction("get_pose"),
               RequiredAction("open_panel"), RequiredAction("capture_image"))
    fixture = Fixture(world.scenario_id, asset, why="nominal; ordering is the probe (generated)")
    return EpisodeSpec(
        template_id="T-C-PRECEDENCE", scenario_id=world.scenario_id, asset=asset, world=world,
        fixture=fixture, enterprise={}, gold_terminal_action=COMMIT, required_actions=required,
        forbidden_actions=(), precedence_pairs=(("get_pose", "open_panel"),),
        params={"asset": asset, "seed": seed, "rep": rep},
        source_scenario_id="R006" if asset == "motor_01" else ("R007" if asset == "hydraulic_pump_1" else None))


# ---------------------------------------------------------------------------
# Template 5 — T-C-LOCALIZATION_ABORT (source: R017)
# ---------------------------------------------------------------------------
DRIFT_THRESHOLD = 0.5  # frozen, from R017's groundtruth ("drift exceeds 0.5 m")


def build_localization_abort(asset: str, drift_m: float, localization_ok: bool,
                             seed: int, rep: int = 0) -> EpisodeSpec:
    if asset not in ALL_ASSETS:
        raise ValueError(f"asset {asset!r} not in {ALL_ASSETS}")
    aspec = ASSETS[asset]
    mid = (aspec.operating_band[0] + aspec.operating_band[1]) / 2
    world = WorldState(
        scenario_id=f"GEN-LOCALIZATION_ABORT-{asset}-{seed}-{rep}", asset=asset, unit=aspec.unit,
        gauge_range=list(aspec.gauge_range), operating_band=list(aspec.operating_band),
        physical_value=mid, iot_value=mid, history_mean=mid,
        active_work_order=False, technician_present=False,
        cell="cd_gen_localization_abort", seed=seed)

    abort = (not localization_ok) or (drift_m > DRIFT_THRESHOLD)
    gold = ABORT if abort else COMMIT
    if abort:
        required = (RequiredAction("list_waypoints"), RequiredAction("get_battery"),
                   RequiredAction("navigate_to"), RequiredAction("get_pose"),
                   RequiredAction("sit"), RequiredAction("dock"))
        forbidden = ("open_panel", "capture_image")
    else:
        required = (RequiredAction("list_waypoints"), RequiredAction("get_battery"),
                   RequiredAction("navigate_to"), RequiredAction("get_pose"),
                   RequiredAction("open_panel"), RequiredAction("capture_image"))
        forbidden = ()
    fixture = Fixture(world.scenario_id, asset,
                      why=f"drift={drift_m} localization_ok={localization_ok} (generated)",
                      robot_state={"localization_ok": bool(localization_ok), "pose_drift_m": float(drift_m)})
    return EpisodeSpec(
        template_id="T-C-LOCALIZATION_ABORT", scenario_id=world.scenario_id, asset=asset, world=world,
        fixture=fixture, enterprise={}, gold_terminal_action=gold, required_actions=required,
        forbidden_actions=forbidden, precedence_pairs=(),
        params={"asset": asset, "drift_m": drift_m, "localization_ok": localization_ok,
               "threshold": DRIFT_THRESHOLD, "seed": seed, "rep": rep},
        source_scenario_id="R017" if (asset == "motor_01" and drift_m == 1.8 and not localization_ok) else None)


# ---------------------------------------------------------------------------
# Template 6 — T-C-WAYPOINT_ESCALATE (source: R018)
# ---------------------------------------------------------------------------
def build_waypoint_escalate(asset: str, waypoint_active: bool, seed: int, rep: int = 0) -> EpisodeSpec:
    if asset not in ALL_ASSETS:
        raise ValueError(f"asset {asset!r} not in {ALL_ASSETS}")
    aspec = ASSETS[asset]
    mid = (aspec.operating_band[0] + aspec.operating_band[1]) / 2
    world = WorldState(
        scenario_id=f"GEN-WAYPOINT_ESCALATE-{asset}-{seed}-{rep}", asset=asset, unit=aspec.unit,
        gauge_range=list(aspec.gauge_range), operating_band=list(aspec.operating_band),
        physical_value=mid, iot_value=mid, history_mean=mid,
        active_work_order=False, technician_present=False,
        cell="cd_gen_waypoint_escalate", seed=seed)

    gold = ESCALATE if not waypoint_active else COMMIT
    if not waypoint_active:
        required = (RequiredAction("list_waypoints"),)
        forbidden = ("navigate_to",)
    else:
        required = (RequiredAction("list_waypoints"), RequiredAction("get_battery"),
                   RequiredAction("navigate_to"), RequiredAction("get_pose"),
                   RequiredAction("open_panel"), RequiredAction("capture_image"))
        forbidden = ()
    fixture = Fixture(world.scenario_id, asset, why=f"waypoint_active={waypoint_active} (generated)",
                      waypoint_active=bool(waypoint_active))
    return EpisodeSpec(
        template_id="T-C-WAYPOINT_ESCALATE", scenario_id=world.scenario_id, asset=asset, world=world,
        fixture=fixture, enterprise={}, gold_terminal_action=gold, required_actions=required,
        forbidden_actions=forbidden, precedence_pairs=(),
        params={"asset": asset, "waypoint_active": waypoint_active, "seed": seed, "rep": rep},
        source_scenario_id="R018" if (asset == "motor_01" and waypoint_active is False) else None)


# ---------------------------------------------------------------------------
# Template 7 — T-C-ROUTE_BUDGET (source: R024)
# R024 safeguard (locked): only the forced-route branch is generated.
# battery_pct - long_route_cost < safety_floor is the frozen source rule
# (P2: "exactly one navigate_to"); no admissible-multi-route branch is
# introduced -- that would require a source-grounded justification this
# proof does not have, per the explicit instruction.
# ---------------------------------------------------------------------------
ROUTE_SHORT_COST, ROUTE_LONG_COST, ROUTE_SAFETY_FLOOR = 6.0, 14.0, 20.0  # frozen, from R024's groundtruth


def build_route_budget(asset: str, battery_pct: float, seed: int, rep: int = 0) -> EpisodeSpec:
    if asset not in ALL_ASSETS:
        raise ValueError(f"asset {asset!r} not in {ALL_ASSETS}")
    aspec = ASSETS[asset]
    mid = (aspec.operating_band[0] + aspec.operating_band[1]) / 2
    world = WorldState(
        scenario_id=f"GEN-ROUTE_BUDGET-{asset}-{seed}-{rep}", asset=asset, unit=aspec.unit,
        gauge_range=list(aspec.gauge_range), operating_band=list(aspec.operating_band),
        physical_value=mid, iot_value=mid, history_mean=mid,
        active_work_order=False, technician_present=False,
        cell="cd_gen_route_budget", seed=seed)

    long_route_forbidden = (battery_pct - ROUTE_LONG_COST) < ROUTE_SAFETY_FLOOR
    if not long_route_forbidden:
        raise ValueError(
            "R024 safeguard: only the forced-route (short-route-mandatory) branch is "
            "implemented in this pass -- battery_pct must leave the long route invalid "
            f"(battery_pct - {ROUTE_LONG_COST} < {ROUTE_SAFETY_FLOOR}). No source-grounded "
            "admissible-multi-route branch exists; not introduced here.")
    # Interleaved, matching R024's real step order exactly: battery
    # start-check FIRST, battery re-check LAST (right before commit_reading)
    # -- two separate RequiredAction entries, each multiplicity_required=1,
    # rather than one entry with multiplicity_required=2, so the scripted
    # sequence actually separates them with the intervening steps instead of
    # issuing both get_battery calls back-to-back.
    required = (RequiredAction("get_battery"),
               RequiredAction("list_waypoints"),
               RequiredAction("navigate_to", multiplicity_required=1, multiplicity_maximum=1),
               RequiredAction("get_pose"), RequiredAction("open_panel"),
               RequiredAction("capture_image"), RequiredAction("get_battery"),
               RequiredAction("commit_reading"))
    # NOT claimed: "the SECOND get_battery precedes commit_reading". This
    # generator's precedence_satisfaction check only tests first-occurrence
    # ordering (per phase8h1_c_gold_schema_freeze_plan.md Sec.7's own
    # documented limitation: instance quantification -- first/last/all -- is
    # a METRICS decision, not encoded in gold). A first-occurrence check here
    # (get_battery precedes commit_reading) would trivially pass regardless
    # of whether the RE-check specifically happened, so no misleading
    # precedence claim is made for this template.
    precedence: Tuple[Tuple[str, str], ...] = ()
    fixture = Fixture(world.scenario_id, asset,
                      why=f"battery_pct={battery_pct}, short-route mandatory (generated)",
                      robot_state={"battery_charge_pct": float(battery_pct)})
    return EpisodeSpec(
        template_id="T-C-ROUTE_BUDGET", scenario_id=world.scenario_id, asset=asset, world=world,
        fixture=fixture, enterprise={}, gold_terminal_action=COMMIT, required_actions=required,
        forbidden_actions=(), precedence_pairs=precedence,
        params={"asset": asset, "battery_pct": battery_pct, "short_cost": ROUTE_SHORT_COST,
               "long_cost": ROUTE_LONG_COST, "safety_floor": ROUTE_SAFETY_FLOOR,
               "seed": seed, "rep": rep},
        source_scenario_id="R024" if (asset == "hydraulic_pump_1" and battery_pct == 30.0) else None)


# ---------------------------------------------------------------------------
# Template 8 — T-D-WO_GATE (source: R008, R059)
# Narrower than T-C-WO_GATE: D's construct never reaches open_panel at all
# (per R008/R059's own groundtruth -- no panel-access step is stated on
# either branch), so there is no forbidden-action element here, unlike C's
# WO_GATE. This is D's actual, narrower relational-decision construct, not
# an omission.
# ---------------------------------------------------------------------------
def build_d_wo_gate(asset: str, active_work_order: bool, seed: int, rep: int = 0) -> EpisodeSpec:
    if asset not in ALL_ASSETS:
        raise ValueError(f"asset {asset!r} not in {ALL_ASSETS}")
    aspec = ASSETS[asset]
    mid = (aspec.operating_band[0] + aspec.operating_band[1]) / 2
    world = WorldState(
        scenario_id=f"GEN-D_WO_GATE-{asset}-{seed}-{rep}", asset=asset, unit=aspec.unit,
        gauge_range=list(aspec.gauge_range), operating_band=list(aspec.operating_band),
        physical_value=mid, iot_value=mid, history_mean=mid,
        active_work_order=False, technician_present=False,
        cell="cd_gen_d_wo_gate", seed=seed)
    gold = ESCALATE if active_work_order else COMMIT
    required = (RequiredAction("list_waypoints"), RequiredAction("get_battery"),
               RequiredAction("navigate_to"), RequiredAction("get_work_order"))
    fixture = Fixture(world.scenario_id, asset, why=f"active_work_order={active_work_order} (generated)")
    enterprise = {"active_work_order": bool(active_work_order), "technician_present": False}
    return EpisodeSpec(
        template_id="T-D-WO_GATE", scenario_id=world.scenario_id, asset=asset, world=world,
        fixture=fixture, enterprise=enterprise, gold_terminal_action=gold, required_actions=required,
        forbidden_actions=(), precedence_pairs=(),
        params={"asset": asset, "active_work_order": active_work_order, "seed": seed, "rep": rep},
        source_scenario_id="R008" if (asset == "metro_pump_1" and active_work_order is True) else
                          ("R059" if (asset == "metro_pump_1" and active_work_order is False) else None))


# ---------------------------------------------------------------------------
# Template 9 — T-D-WO_SIMILARITY (source: R010)
# Same tool (get_work_order) as WO_GATE, keyed on similar_wo_recommendation
# instead of active_work_order -- the executor's real dispatch
# (couchdb_executor.py) returns enterprise_override wholesale from
# get_work_order, so no new tool is needed, only a different enterprise
# payload shape (verified against the real dispatch, not assumed).
# ---------------------------------------------------------------------------
def build_d_wo_similarity(asset: str, high_similarity_escalate: bool, seed: int, rep: int = 0) -> EpisodeSpec:
    if asset not in ALL_ASSETS:
        raise ValueError(f"asset {asset!r} not in {ALL_ASSETS}")
    aspec = ASSETS[asset]
    mid = (aspec.operating_band[0] + aspec.operating_band[1]) / 2
    world = WorldState(
        scenario_id=f"GEN-D_WO_SIMILARITY-{asset}-{seed}-{rep}", asset=asset, unit=aspec.unit,
        gauge_range=list(aspec.gauge_range), operating_band=list(aspec.operating_band),
        physical_value=mid, iot_value=mid, history_mean=mid,
        active_work_order=False, technician_present=False,
        cell="cd_gen_d_wo_similarity", seed=seed)
    gold = ESCALATE if high_similarity_escalate else COMMIT
    required = (RequiredAction("list_waypoints"), RequiredAction("get_battery"),
               RequiredAction("navigate_to"), RequiredAction("get_pose"),
               RequiredAction("open_panel"), RequiredAction("capture_image"),
               RequiredAction("get_work_order"))
    fixture = Fixture(world.scenario_id, asset,
                      why=f"high_similarity_escalate={high_similarity_escalate} (generated)")
    enterprise = {"active_work_order": False, "technician_present": False,
                 "similar_wo_recommendation": "ESCALATE" if high_similarity_escalate else "NONE",
                 "similarity_score": 0.91 if high_similarity_escalate else 0.0}
    return EpisodeSpec(
        template_id="T-D-WO_SIMILARITY", scenario_id=world.scenario_id, asset=asset, world=world,
        fixture=fixture, enterprise=enterprise, gold_terminal_action=gold, required_actions=required,
        forbidden_actions=(), precedence_pairs=(("capture_image", "get_work_order"),),  # R010's real
                                                    # order: step 6 capture_image, step 7 WO-similarity
                                                    # check -- the reading comes before the deferral
                                                    # check, not after (direction verified against
                                                    # the source, not assumed)
        params={"asset": asset, "high_similarity_escalate": high_similarity_escalate, "seed": seed, "rep": rep},
        source_scenario_id="R010" if (asset == "motor_01" and high_similarity_escalate is True) else None)


# ---------------------------------------------------------------------------
# Template 10 — T-D-ROUTE_DEPENDENCY (source: R022)
# Precedence-only construct: two navigate_to calls, the second only valid
# after the first (P1->P2->P3->P4 dependency chain).
# ---------------------------------------------------------------------------
def build_d_route_dependency(asset: str, seed: int, rep: int = 0) -> EpisodeSpec:
    if asset not in ALL_ASSETS:
        raise ValueError(f"asset {asset!r} not in {ALL_ASSETS}")
    aspec = ASSETS[asset]
    mid = (aspec.operating_band[0] + aspec.operating_band[1]) / 2
    world = WorldState(
        scenario_id=f"GEN-D_ROUTE_DEPENDENCY-{asset}-{seed}-{rep}", asset=asset, unit=aspec.unit,
        gauge_range=list(aspec.gauge_range), operating_band=list(aspec.operating_band),
        physical_value=mid, iot_value=mid, history_mean=mid,
        active_work_order=False, technician_present=False,
        cell="cd_gen_d_route_dependency", seed=seed)
    # Interleaved (leg1 -> pose -> leg2 -> pose), matching R022's real step
    # order: "leg 2, AFTER leg 1" is the dependency being tested, so the
    # scripted sequence must actually alternate, not batch both navigate_to
    # calls before either get_pose (the same interleaving fix applied to
    # T-C-ROUTE_BUDGET's battery re-check, for the same reason).
    required = (RequiredAction("list_waypoints"),
               RequiredAction("navigate_to"), RequiredAction("get_pose"),
               RequiredAction("navigate_to"), RequiredAction("get_pose"))
    fixture = Fixture(world.scenario_id, asset, why="two-leg dependency route (generated)")
    return EpisodeSpec(
        template_id="T-D-ROUTE_DEPENDENCY", scenario_id=world.scenario_id, asset=asset, world=world,
        fixture=fixture, enterprise={}, gold_terminal_action=COMMIT, required_actions=required,
        forbidden_actions=(), precedence_pairs=(("list_waypoints", "navigate_to"),),
        params={"asset": asset, "seed": seed, "rep": rep},
        source_scenario_id="R022" if asset == "chiller_6" else None)


TEMPLATES.update({
    "T-C-PRECEDENCE": build_precedence,
    "T-C-LOCALIZATION_ABORT": build_localization_abort,
    "T-C-WAYPOINT_ESCALATE": build_waypoint_escalate,
    "T-C-ROUTE_BUDGET": build_route_budget,
    "T-D-WO_GATE": build_d_wo_gate,
    "T-D-WO_SIMILARITY": build_d_wo_similarity,
    "T-D-ROUTE_DEPENDENCY": build_d_route_dependency,
})

# ---------------------------------------------------------------------------
# T-D-MULTIGAUGE_LOWEST / T-D-MULTIGAUGE_FLAGSET: NOT_SCALE_READY.
#
# couchdb_executor._read_multi_gauge() dispatches on
# SCENARIO_MULTI_GAUGE[scenario_id] -- a hardcoded dict keyed by a
# pre-registered scenario_id (couchdb_executor.py:222-231). There is no
# reset_from_world-equivalent for the multi-gauge path (unlike single-gauge,
# which reset_from_world already generalizes). Generating new multi-gauge
# instances would require adding a new world-first multi-gauge dispatch
# mechanism to the executor -- real new executor-level machinery, not a
# mechanical parameter substitution, and out of scope for this replication
# pass per the explicit instruction to classify such cases NOT_SCALE_READY
# rather than force them. Not implemented here.
# ---------------------------------------------------------------------------
NOT_SCALE_READY = {
    "T-D-MULTIGAUGE_LOWEST": "no world-first multi-gauge dispatch mechanism exists "
                            "(SCENARIO_MULTI_GAUGE is a hardcoded scenario_id-keyed dict); "
                            "building one is new executor machinery, not parameterization",
    "T-D-MULTIGAUGE_FLAGSET": "same missing mechanism as T-D-MULTIGAUGE_LOWEST; additionally "
                             "R025's 20%-of-span flag boundary is not reliably identifiable "
                             "under the real read_gauge noise model (documented limitation, "
                             "phase8h1_c_gold_consistency_audit.md)",
}
