"""dphys_generator.py — D-physical constraint-integration generator
(Phase 8H.2E, "D-PHYS" subfamily).

Mirrors cd_generator.py's architecture exactly, applied to a NEW construct:
whether an agent can integrate several simultaneously-active physical
constraints (reach, joint/collision feasibility, clearance, grasp/payload,
stability, energy) to determine feasibility and identify the binding
constraint -- distinct from every other family (see
reports/paper/d_candidate_taxonomy.md Sec. 4 for the formal distinctness
proof).

    template params (asset, constraint(s), margin condition)
        -> WorldState.physical_access (scenario_gen, additive field, UNCHANGED
           schema for every other field)
        -> ex.reset_from_world(world, ...) -- writes physical_access into the
           REAL CouchDB profile doc via to_couch_profile (both UNCHANGED
           beyond the new optional field)
        -> SpotAdmissibilityVerifier.verify_candidate -- the REAL, already-
           tested MuJoCo-backed oracle (UNCHANGED) -- is the ORACLE, called
           here ONLY to compute gold, never exposed to the agent
        -> RelationalGold, a typed dataclass derived from the oracle's own
           `checks`/`violated_constraints` output (reused, not re-derived)
        -> a SCRIPTED (non-model) tool-call sequence through the REAL
           CouchDBExecutor/ExecutionTrace, exactly like cd_generator.py
        -> dphys_scoring.score_dphys_episode (CSA, CS-P/R/F1, LCA, + reused
           CC/PAC/UDR/ODR)

check_admissibility / check_cdc are NEVER called through the agent-facing
CouchDBExecutor.execute() path in this module -- the oracle is invoked
directly by the generator, exactly the same "evaluator computes gold before
the episode starts" pattern cd_generator.py already uses for derive_gold.
This is enforced structurally by
tests/test_d_physical_oracle_evaluator_only.py.

Zero model/API calls anywhere in this module.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

from scenario_gen import ASSETS, WorldState  # noqa: E402 -- UNCHANGED, reused
from couchdb_executor import CouchDBExecutor  # noqa: E402 -- UNCHANGED, reused
from execution_trace import ExecutionTrace, Stage  # noqa: E402 -- UNCHANGED, reused
from tool_executor import ToolCall, STATUS_SUCCESS  # noqa: E402 -- UNCHANGED, reused
from spot_admissibility_verifier import SpotAdmissibilityVerifier  # noqa: E402 -- UNCHANGED, reused

DISPATCH, ESCALATE = "DISPATCH", "ESCALATE"
ALL_ASSETS = tuple(sorted(ASSETS))

CONSTRAINT_NAMES = ("reach", "joint_and_collision", "clearance", "grasp_payload",
                    "stability", "energy")
SATISFIED, VIOLATED = "SATISFIED", "VIOLATED"
MULTIPLE = "MULTIPLE"  # sentinel: >=2 constraints simultaneously violated, no single binding one

_SG_ROOT = (Path.home() / "AssetOpsBenchScenarioGeneration" / "RobotInspection")
ASSET_PROFILES_PATH = _SG_ROOT / "Scenarios" / "asset_profiles.json"
REGISTRY_PATH = _SG_ROOT / "shared" / "robot_assets_registry.json"


def load_asset_profiles() -> Dict[str, Any]:
    data = json.loads(ASSET_PROFILES_PATH.read_text())
    if isinstance(data, list):
        data = {p["asset_id"]: p for p in data}
    return data


def load_registry() -> Dict[str, Any]:
    return json.loads(REGISTRY_PATH.read_text())


def _base_access(profiles: Dict[str, Any], asset: str) -> Dict[str, Any]:
    return copy.deepcopy(profiles[asset]["physical_access"])


def _clean_probe_standoff(asset: str, access: Dict[str, Any],
                          verifier: "SpotAdmissibilityVerifier") -> float:
    """The standoff to probe for single-binding/coupled templates: the first
    candidate where the UNMODIFIED access is fully admissible (all 6 checks
    SATISFIED). Required because not every asset's smallest
    standoff_candidates_m entry is baseline-clean -- verified directly:
    motor_01's smallest standoff (0.5) already fails `clearance` at baseline
    (its declared obstacle_clearance_m exceeds it), so blindly using index 0
    contaminates every single-binding override built on top of it with a
    second, unintended VIOLATED constraint. Falls back to index 0 with a
    loud assertion failure rather than silently returning a contaminated
    probe if no candidate is baseline-clean for some future asset."""
    for s in access.get("standoff_candidates_m", [0.8]):
        r = verifier.verify_candidate(access, s)
        if all(r.checks.values()):
            return s
    raise ValueError(
        f"{asset}: no standoff_candidates_m entry is baseline-clean (all 6 "
        "constraints SATISFIED) -- single-binding/coupled templates require "
        "one to isolate the tested constraint(s) honestly; this asset's "
        "geometry needs review before it can be used in these templates")


# ---------------------------------------------------------------------------
# Typed relational gold contract (Phase 4). Every field's source is a direct
# read of the oracle's own output -- never independently re-derived.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RelationalGold:
    constraints: Dict[str, str]              # all 6 CONSTRAINT_NAMES -> SATISFIED|VIOLATED
    admissible: bool                          # source: AdmissibilityVerdict.admissible
    limiting_constraint: Optional[str]        # name | None (all satisfied) | "MULTIPLE"
    selected_standoff: Optional[float]        # source: AdmissibilityVerdict.selected_candidate
    allowed_terminal_actions: Tuple[str, ...] = (DISPATCH, ESCALATE)

    def to_dict(self) -> Dict[str, Any]:
        return {"constraints": dict(self.constraints), "admissible": self.admissible,
               "limiting_constraint": self.limiting_constraint,
               "selected_standoff": self.selected_standoff,
               "allowed_terminal_actions": list(self.allowed_terminal_actions)}


def _relational_gold_from_verdict(verdict) -> RelationalGold:
    """`verdict` is a spot_admissibility_verifier.AdmissibilityVerdict (real
    oracle output, unmodified). Every field below is read directly from it."""
    constraints = {name: (SATISFIED if verdict.checks.get(name, True) else VIOLATED)
                  for name in CONSTRAINT_NAMES}
    violated = [n for n, v in constraints.items() if v == VIOLATED]
    if not violated:
        limiting = None
    elif len(violated) == 1:
        limiting = violated[0]
    else:
        limiting = MULTIPLE
    return RelationalGold(constraints=constraints, admissible=verdict.admissible,
                          limiting_constraint=limiting,
                          selected_standoff=verdict.selected_candidate)


def gold_for_access(registry: Dict[str, Any], access: Dict[str, Any],
                    standoff_m: float, payload_kg: float = 0.0) -> Tuple[RelationalGold, Any]:
    """Single-candidate gold: run the oracle once and derive RelationalGold.
    Returns (gold, raw_verdict) -- the raw verdict is kept only for episode
    provenance/debugging, never surfaced to the agent."""
    verifier = SpotAdmissibilityVerifier(registry)
    verdict = verifier.verify_candidate(access, standoff_m, payload_kg=payload_kg)
    return _relational_gold_from_verdict(verdict), verdict


def gold_for_candidates(registry: Dict[str, Any], access: Dict[str, Any],
                        standoffs: List[float], payload_kg: float = 0.0
                        ) -> Tuple[RelationalGold, List[Any]]:
    """Multi-candidate gold (D-PHYS-CAP-X): admissible iff ANY candidate is
    admissible; selected_standoff is the first admissible one (matches
    SpotAdmissibilityVerifier.select_admissible's own first-admissible
    policy, verified against RC003 in test_d_physical_world_first.py's
    sibling oracle-agreement tests). Per-candidate detail is preserved in
    `all_verdicts` for provenance."""
    verifier = SpotAdmissibilityVerifier(registry)
    verdicts = [verifier.verify_candidate(access, s, payload_kg=payload_kg) for s in standoffs]
    admissible_idxs = [i for i, v in enumerate(verdicts) if v.admissible]
    if admissible_idxs:
        chosen = verdicts[admissible_idxs[0]]
        gold = _relational_gold_from_verdict(chosen)
    else:
        # No candidate admissible: report the union of violated constraints
        # across all candidates so CS-F1 has a meaningful target, but
        # limiting_constraint is MULTIPLE-or-single per the union, and
        # selected_standoff is None (nothing to select).
        constraints = {name: SATISFIED for name in CONSTRAINT_NAMES}
        for v in verdicts:
            for name, ok in v.checks.items():
                if not ok:
                    constraints[name] = VIOLATED
        violated = [n for n, v in constraints.items() if v == VIOLATED]
        limiting = violated[0] if len(violated) == 1 else (MULTIPLE if violated else None)
        gold = RelationalGold(constraints=constraints, admissible=False,
                              limiting_constraint=limiting, selected_standoff=None)
    return gold, verdicts


# ---------------------------------------------------------------------------
# Episode spec (mirrors cd_generator.EpisodeSpec)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DPhysEpisodeSpec:
    template_id: str
    scenario_id: str
    asset: str
    world: WorldState
    gold: RelationalGold
    gold_terminal_action: str          # DISPATCH | ESCALATE, derived from gold.admissible
    params: Dict[str, Any]
    constraint_stratum: str            # e.g. "single:reach", "dual:reach+stability", "capx"
    source_precedent: Optional[str] = None  # e.g. "R027" where applicable


def _make_world(scenario_id: str, asset: str, access: Dict[str, Any], seed: int) -> WorldState:
    profiles = load_asset_profiles()
    unit = {"chiller_6": "bar", "hydraulic_pump_1": "bar", "motor_01": "C",
           "metro_pump_1": "m3/s"}.get(asset, "bar")
    return WorldState(
        scenario_id=scenario_id, asset=asset, unit=unit,
        gauge_range=[0.0, 100.0], operating_band=[10.0, 90.0],
        physical_value=50.0, iot_value=50.0, history_mean=50.0,
        active_work_order=False, technician_present=False,
        cell="d_phys", seed=seed, physical_access=access,
    )


# ---------------------------------------------------------------------------
# Template builders
# ---------------------------------------------------------------------------
def _reach_only_override(access: Dict[str, Any], verifier: "SpotAdmissibilityVerifier") -> Dict[str, Any]:
    """Isolate `reach` from `joint_and_collision`.

    Pushing panel z out of workspace bounds (the naive override) ALSO fails
    joint_and_collision, because check 2 derives its arm pose from the SAME
    (standoff, z) via derive_arm_qpos -- an unreachable z has no valid derived
    pose either. Verified directly: {'panel_pose.z': 2.35} yields
    checks={'reach': False, 'joint_and_collision': False, ...} on every
    asset. This is a REAL physical coupling in the analytic model (the same
    reason (reach, joint_and_collision) is excluded from the dual-binding
    stratum in d_scaling_analysis.csv as "near-redundant proxies for the same
    obstacle-avoidance geometry"), not a generator bug.

    Fix: push z to 1.7 (still inside the declared workspace z-range [0,1.8],
    so the coarse in-bounds check still passes) while required_reach(standoff,
    1.7) exceeds arm_reach_m (verified: 1.10m > 0.985m) -- this alone fails
    the reach check. Pair it with an EXPLICIT commanded_arm_qpos derived from
    the asset's own comfortable baseline (a pose already proven in-limits),
    which check 2 uses verbatim instead of re-deriving one for z=1.7. This
    commanded_arm_qpos is oracle-internal (never agent-visible, per
    tests/test_d_physical_oracle_evaluator_only.py) so using an
    otherwise-unrelated valid pose purely to isolate the construct leaks
    nothing to the agent."""
    baseline = copy.deepcopy(access)
    baseline_standoff = baseline["standoff_candidates_m"][0]
    valid_pose = verifier._derive_arm_qpos(baseline_standoff, baseline["panel_pose"]["z"])
    return {"panel_pose": {**access["panel_pose"], "z": 1.7}, "commanded_arm_qpos": valid_pose}


MARGIN_OVERRIDES = {
    # constraint -> {margin_label: override_fn(access, verifier) -> override_dict}
    # each override pushes exactly the ONE named constraint past/near its
    # threshold, leaving the other 5 at the asset's comfortable default --
    # verified empirically per constraint in test_dphys_generator.py, not
    # merely assumed from the formulas.
    "reach": {
        "inadmissible": _reach_only_override,
        "comfortable": lambda access, verifier: {},
    },
    "joint_and_collision": {
        "inadmissible": lambda access, verifier: {"commanded_arm_qpos": {
            "arm0.sh0": 0.0, "arm0.sh1": -0.6, "arm0.el0": 1.2, "arm0.el1": 0.0,
            "arm0.wr0": 2.2, "arm0.wr1": 0.0, "arm0.f1x": -0.2}},
        "comfortable": lambda access, verifier: {},
    },
    "clearance": {
        "inadmissible": lambda access, verifier: {"obstacle_clearance_m": 1.20},
        "comfortable": lambda access, verifier: {},
    },
    "grasp_payload": {
        "inadmissible": lambda access, verifier: {"latch_grasp_force_N": 180.0},
        "comfortable": lambda access, verifier: {},
    },
    "stability": {
        "inadmissible": lambda access, verifier: {"approach_slope_deg": 40.0},
        "comfortable": lambda access, verifier: {},
    },
    "energy": {
        "inadmissible": lambda access, verifier: {"battery_pct": 12.0, "route_legs": 3, "mission_duration_s": 600},
        "comfortable": lambda access, verifier: {"battery_pct": 90.0, "route_legs": 1, "mission_duration_s": 300},
    },
}

_SOURCE_PRECEDENT = {
    "reach": ("R027", "R028"), "joint_and_collision": ("R029", "R030"),
    "clearance": ("R031", "R032"), "grasp_payload": ("R033", "R034"),
    "stability": ("R035", "R036"), "energy": ("R037", "R038"),
}


def build_reach_or_energy(asset: str, constraint: str, margin: str, seed: int
                          ) -> DPhysEpisodeSpec:
    """T-D-PHYS-REACH / T-D-PHYS-ENERGY: single-binding constraint recognition.
    `constraint` in CONSTRAINT_NAMES; `margin` in {"inadmissible", "comfortable"}."""
    assert constraint in CONSTRAINT_NAMES, constraint
    assert margin in ("inadmissible", "comfortable"), margin
    profiles, registry = load_asset_profiles(), load_registry()
    verifier = SpotAdmissibilityVerifier(registry)
    base_access = _base_access(profiles, asset)
    standoff = _clean_probe_standoff(asset, base_access, verifier)
    access = _base_access(profiles, asset)
    access.update(MARGIN_OVERRIDES[constraint][margin](access, verifier))
    scenario_id = f"GEN-DPHYS-{constraint}-{margin}-{asset}-{seed:06d}"
    world = _make_world(scenario_id, asset, access, seed)
    gold, _ = gold_for_access(registry, access, standoff)
    template_id = "T-D-PHYS-REACH" if constraint in ("reach", "joint_and_collision", "clearance") \
        else "T-D-PHYS-ENERGY"
    precedent = _SOURCE_PRECEDENT[constraint][0 if margin == "inadmissible" else 1]
    return DPhysEpisodeSpec(
        template_id=template_id, scenario_id=scenario_id, asset=asset, world=world,
        gold=gold, gold_terminal_action=(DISPATCH if gold.admissible else ESCALATE),
        params={"constraint": constraint, "margin": margin, "asset": asset, "seed": seed,
               "standoff": standoff},
        constraint_stratum=f"single:{constraint}", source_precedent=precedent,
    )


REALIZABLE_PAIRS = (
    ("reach", "clearance"), ("reach", "stability"), ("clearance", "stability"),
    ("grasp_payload", "energy"), ("energy", "reach"), ("joint_and_collision", "stability"),
)
REALIZABLE_TRIPLE = ("reach", "clearance", "stability")


def build_coupled(asset: str, coupling: Tuple[str, ...], seed: int) -> DPhysEpisodeSpec:
    """T-D-PHYS-COUPLED: multi-constraint attribution + limiting-constraint ID.
    `coupling` is one of REALIZABLE_PAIRS or REALIZABLE_TRIPLE. Pushes ALL
    named constraints toward violation simultaneously via their respective
    MARGIN_OVERRIDES["inadmissible"] override, leaving the rest comfortable."""
    assert coupling in REALIZABLE_PAIRS or coupling == REALIZABLE_TRIPLE, coupling
    profiles, registry = load_asset_profiles(), load_registry()
    verifier = SpotAdmissibilityVerifier(registry)
    base_access = _base_access(profiles, asset)
    standoff = _clean_probe_standoff(asset, base_access, verifier)
    access = _base_access(profiles, asset)
    for c in coupling:
        access.update(MARGIN_OVERRIDES[c]["inadmissible"](access, verifier))
    tag = "+".join(coupling)
    scenario_id = f"GEN-DPHYS-COUPLED-{tag}-{asset}-{seed:06d}"
    world = _make_world(scenario_id, asset, access, seed)
    gold, _ = gold_for_access(registry, access, standoff)
    return DPhysEpisodeSpec(
        template_id="T-D-PHYS-COUPLED", scenario_id=scenario_id, asset=asset, world=world,
        gold=gold, gold_terminal_action=(DISPATCH if gold.admissible else ESCALATE),
        params={"coupling": list(coupling), "asset": asset, "seed": seed, "standoff": standoff},
        constraint_stratum=f"{'triple' if len(coupling) == 3 else 'dual'}:{tag}",
        source_precedent=None,
    )


CANDIDATE_CONFIGS = ("comfortable_baseline", "mixed_one_admissible", "all_inadmissible")


def build_capx(asset: str, config: str, seed: int) -> DPhysEpisodeSpec:
    """T-D-PHYS-CAP-X: candidate inspection-position selection over the
    asset's real standoff_candidates_m list.

    Note on config semantics (a genuine finding, verified empirically, not a
    shortfall silently forced through): with these assets' real geometry
    (panel z, arm_reach_m=0.985), the LARGEST standoff_candidates_m entry
    always fails reach/joint_and_collision for every one of the 4 assets --
    it is simply too far. So a literal "every candidate simultaneously
    admissible" config does not exist without fabricating unrealistic
    geometry. "comfortable_baseline" therefore means "unmodified asset
    geometry, at least one candidate cleanly admissible" (verified: exactly
    1 for chiller_6/motor_01, exactly 2 for hydraulic_pump_1/metro_pump_1 at
    baseline) rather than claiming all three clear."""
    assert config in CANDIDATE_CONFIGS, config
    profiles, registry = load_asset_profiles(), load_registry()
    access = _base_access(profiles, asset)
    standoffs = list(access.get("standoff_candidates_m", [0.8]))

    if config == "comfortable_baseline":
        pass  # unmodified geometry; >=1 candidate admissible, verified per-asset above
    elif config == "all_inadmissible":
        access.update({"obstacle_clearance_m": 1.20})  # exceeds every standoff
    else:  # mixed_one_admissible: raise clearance just enough to exclude the
        # smallest admissible-at-baseline candidate(s), leaving exactly one.
        # Verified empirically per-asset (chiller_6/motor_01 already have
        # exactly one admissible at baseline, so no override is needed;
        # hydraulic_pump_1/metro_pump_1 have two at baseline, so
        # obstacle_clearance_m=0.65 excludes their smallest while the
        # baseline reach/joint failure already excludes their largest).
        MIXED_CLEARANCE_OVERRIDE = {"hydraulic_pump_1": 0.65, "metro_pump_1": 0.65}
        if asset in MIXED_CLEARANCE_OVERRIDE:
            access.update({"obstacle_clearance_m": MIXED_CLEARANCE_OVERRIDE[asset]})

    scenario_id = f"GEN-DPHYS-CAPX-{config}-{asset}-{seed:06d}"
    world = _make_world(scenario_id, asset, access, seed)
    gold, _ = gold_for_candidates(registry, access, standoffs)
    return DPhysEpisodeSpec(
        template_id="T-D-PHYS-CAP-X", scenario_id=scenario_id, asset=asset, world=world,
        gold=gold, gold_terminal_action=(DISPATCH if gold.admissible else ESCALATE),
        params={"config": config, "asset": asset, "seed": seed, "standoffs": standoffs},
        constraint_stratum=f"capx:{config}", source_precedent="RC003",
    )


TEMPLATES = {
    "T-D-PHYS-REACH": build_reach_or_energy,
    "T-D-PHYS-ENERGY": build_reach_or_energy,
    "T-D-PHYS-COUPLED": build_coupled,
    "T-D-PHYS-CAP-X": build_capx,
}


# ---------------------------------------------------------------------------
# Scripted execution (mirrors cd_generator.run_scripted_episode)
# ---------------------------------------------------------------------------
AGENT_VISIBLE_SCRIPT = ["list_waypoints", "get_battery", "navigate_to", "get_pose"]


def run_scripted_episode(ex: CouchDBExecutor, spec: DPhysEpisodeSpec) -> Dict[str, Any]:
    """Run the fixed, non-model tool sequence a real agent would call to
    gather the AGENT-VISIBLE primitives (never the oracle verdict) before its
    terminal decision. The verdict itself is scripted to equal gold -- this
    validates the scorer/trace mechanics, never a model."""
    ex.reset_from_world(spec.world, "spot_1", seed=spec.world.seed)
    trace = ExecutionTrace(spec.scenario_id, "FULL")
    available = set(ex.available_tools())
    for name in AGENT_VISIBLE_SCRIPT:
        if name not in available:
            continue
        trace.append(Stage.REQUESTED, tool=name, args={})
        res = ex.execute(ToolCall(name, {}))
        if res.executed:
            trace.append(Stage.EXECUTED, tool=name, status=res.status, error=res.error)
        if res.status == STATUS_SUCCESS:
            trace.append(Stage.SUCCEEDED, tool=name)
        if res.delivered:
            trace.append(Stage.OBSERVATION_DELIVERED, tool=name, observation_id=res.observation_id,
                         observation_hash=res.observation_hash, modality=res.modality)

    verdict = spec.gold_terminal_action  # scripted to equal gold
    trace.append(Stage.DECISION, detail={"verdict": verdict})
    return {
        "trace_events": [e.to_dict() for e in trace.events],
        "trace_chain_valid": trace.verify_chain(),
        "verdict": verdict,
        "predicted_constraints": dict(spec.gold.constraints),   # scripted = gold, proves the scorer
        "predicted_limiting_constraint": spec.gold.limiting_constraint,
        "predicted_selected_standoff": spec.gold.selected_standoff,
    }
