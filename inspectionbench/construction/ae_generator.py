"""ae_generator.py — Phase 8H.2D scalable A/E canonical-episode generator.

Mirrors cd_generator.py's architecture exactly, applied to Family A and E,
which (unlike C/D) already have production, continuously-stochastic
generators (`scenario_gen.sample_world`/`derive_gold`,
`sequence_executor.sample_sequence`/`derive_sequence_gold`) — reused here
UNCHANGED. No new world/gold semantics are introduced.

    seed + cell/process + asset
        -> WorldState / SequenceWorld (scenario_gen / sequence_executor, unmodified)
        -> deterministic gold (derive_gold / derive_sequence_gold, unmodified)
        -> reset_from_world (A) / begin_sequence+advance_episode (E), unmodified
        -> SCRIPTED (non-model) tool-call sequence through the REAL
           CouchDBExecutor / SequenceExecutor
        -> frozen scoring (metric_contract.score_episode), unmodified

Zero model/API calls anywhere in this module.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

from scenario_gen import ASSETS, Cell, FACTORIAL, WorldState, derive_gold  # noqa: E402 -- UNCHANGED
from sequence_executor import (  # noqa: E402 -- UNCHANGED
    PROCESSES, SequenceExecutor, SequenceWorld, sample_sequence, derive_sequence_gold,
)
from couchdb_executor import CouchDBExecutor  # noqa: E402 -- UNCHANGED
from execution_trace import ExecutionTrace, Stage  # noqa: E402 -- UNCHANGED
from tool_executor import ToolCall, STATUS_SUCCESS  # noqa: E402 -- UNCHANGED
from metric_contract import score_episode  # noqa: E402 -- UNCHANGED
from l3_integrity import check_integrity  # noqa: E402 -- UNCHANGED
from l3_grounded_scoring import REQUIRED_MODALITY  # noqa: E402 -- UNCHANGED

ALL_ASSETS = tuple(sorted(ASSETS))
REGIMES = ("FULL", "PHYSICAL_ONLY", "DIGITAL_ONLY")
REGIME_WITHHELD = {"FULL": [], "PHYSICAL_ONLY": ["digital"], "DIGITAL_ONLY": ["physical"]}
FM_LABEL_A = "FM-6a"


def world_hash(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Family A
# ---------------------------------------------------------------------------
A_SCRIPT_FULL = ["list_waypoints", "get_battery", "navigate_to", "get_pose",
                 "open_panel", "capture_image", "read_gauge", "read_iot"]


@dataclass
class AEpisodeSpec:
    world: WorldState
    regime: str
    gold: str
    matched_group_id: str  # shared across the 3 regime variants of ONE world


def build_a_world(seed: int, cell: Cell, asset_id: str) -> WorldState:
    scenario_id = f"GENA-{cell.name}-{asset_id}-{seed:06d}"
    from scenario_gen import sample_world
    return sample_world(seed, cell, asset_id=asset_id, scenario_id=scenario_id)


def a_episode_specs(world: WorldState) -> List[AEpisodeSpec]:
    gold = derive_gold(world).verdict
    matched_group_id = f"MG-A-{world.scenario_id}"
    return [AEpisodeSpec(world=world, regime=r, gold=gold, matched_group_id=matched_group_id)
           for r in REGIMES]


@dataclass
class AResult:
    spec: AEpisodeSpec
    fixture_verified: bool
    trace_events: List[Dict[str, Any]]
    trace_chain_valid: bool
    TDA: Optional[int]
    GSR: Optional[int]
    delta: Optional[int]
    integrity_any_flag: Any
    world_hash: str


def run_a_episode_scripted(ex: CouchDBExecutor, spec: AEpisodeSpec) -> AResult:
    world, regime, gold = spec.world, spec.regime, spec.gold
    withheld = REGIME_WITHHELD[regime]
    ex.reset_from_world(world, regime, seed=world.seed, withheld=withheld)
    trace = ExecutionTrace(world.scenario_id, regime)
    available = set(ex.available_tools())

    for name in A_SCRIPT_FULL:
        if name not in available:
            continue  # honestly reflects the regime's real masking, not skipped silently
        trace.append(Stage.REQUESTED, tool=name, args={})
        res = ex.execute(ToolCall(name, {}))
        if res.executed:
            trace.append(Stage.EXECUTED, tool=name, status=res.status, error=res.error)
        if res.status == STATUS_SUCCESS:
            trace.append(Stage.SUCCEEDED, tool=name)
        if res.delivered:
            trace.append(Stage.OBSERVATION_DELIVERED, tool=name, observation_id=res.observation_id,
                         observation_hash=res.observation_hash, modality=res.modality, asset_id=res.asset_id)

    verdict = gold  # SCRIPTED to equal gold -- proves the scorer, not model behavior
    trace.append(Stage.DECISION, detail={"verdict": verdict})
    resp = {"verdict": verdict, "reason": "scripted canonical-episode generation", "tool_sequence": []}
    sc = {"fm": FM_LABEL_A, "asset_id": world.asset}
    m = score_episode(resp, sc, {"action": gold}, trace, f"{world.scenario_id}::{regime}")
    integrity = check_integrity(resp, trace, REQUIRED_MODALITY.get(FM_LABEL_A, "physical"))
    payload = {"scenario_id": world.scenario_id, "regime": regime, "asset": world.asset,
              "physical_value": world.physical_value, "iot_value": world.iot_value,
              "active_work_order": world.active_work_order, "technician_present": world.technician_present,
              "cell": world.cell, "seed": world.seed}
    return AResult(spec=spec, fixture_verified=True, trace_events=[e.to_dict() for e in trace.events],
                  trace_chain_valid=trace.verify_chain(), TDA=m.TDA, GSR=m.GSR, delta=m.delta,
                  integrity_any_flag=integrity.any_flag, world_hash=world_hash(payload))


# ---------------------------------------------------------------------------
# Family E
# ---------------------------------------------------------------------------
@dataclass
class EResult:
    world: SequenceWorld
    episode: int
    gold: str
    trace_events: List[Dict[str, Any]]
    trace_chain_valid: bool
    TDA: Optional[int]
    world_hash: str
    reads_used_before: int
    read_attempted: bool
    read_delivered: bool


def build_e_world(seed: int, asset_id: Optional[str] = None, n_episodes: int = 3) -> SequenceWorld:
    return sample_sequence(seed, n_episodes=n_episodes, asset_id=asset_id)


def run_e_sequence_scripted(inner: CouchDBExecutor, world: SequenceWorld) -> List[EResult]:
    """SCRIPTED policy: always attempt read_gauge in episode 0 (establish
    baseline) and in every subsequent episode (tests the real battery-budget
    mechanic honestly -- the budget is a world property, exhausting it is a
    genuine trace fact, not something this generator fabricates)."""
    se = SequenceExecutor(inner)
    se.begin_sequence(world)
    results: List[EResult] = []
    for k in range(world.n_episodes):
        se.advance_episode("FULL", withheld=[])
        trace = ExecutionTrace(f"{world.sequence_id}#{k}", "FULL")
        reads_before = se._physical_reads_used
        trace.append(Stage.REQUESTED, tool="read_gauge", args={})
        res = se.execute(ToolCall("read_gauge", {}))
        read_delivered = bool(res.delivered)
        if res.executed:
            trace.append(Stage.EXECUTED, tool="read_gauge", status=res.status, error=res.error)
        if res.status == STATUS_SUCCESS:
            trace.append(Stage.SUCCEEDED, tool="read_gauge")
        if res.delivered:
            trace.append(Stage.OBSERVATION_DELIVERED, tool="read_gauge", observation_id=res.observation_id,
                         observation_hash=res.observation_hash, modality=res.modality)
        gold = derive_sequence_gold(world, k)["verdict"]
        trace.append(Stage.DECISION, detail={"verdict": gold})
        m = score_episode({"verdict": gold, "reason": "scripted", "tool_sequence": []},
                          {"fm": "FM-6a", "asset_id": world.asset}, {"action": gold}, trace,
                          f"{world.sequence_id}#{k}")
        payload = {"sequence_id": world.sequence_id, "episode": k, "asset": world.asset,
                  "process_kind": world.process.kind, "process_rate": world.process.rate,
                  "process_step_at": world.process.step_at, "process_step_delta": world.process.step_delta,
                  "base_value": world.base_value, "seed": world.seed}
        results.append(EResult(world=world, episode=k, gold=gold, trace_events=[e.to_dict() for e in trace.events],
                              trace_chain_valid=trace.verify_chain(), TDA=m.TDA, world_hash=world_hash(payload),
                              reads_used_before=reads_before, read_attempted=True, read_delivered=read_delivered))
    se.end_sequence()
    return results
