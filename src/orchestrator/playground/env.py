"""env.py — B2Env: the live, multi-step evidence-acquisition environment
(plan section D/J). Wraps real resolver draws (`substrate.resolve_next`)
and the real `ExecutionTrace`/`Stage` machinery. Reuses, never
reimplements: ExecutionTrace, Stage, ObservationResolver (via substrate),
ToolResult.delivered semantics, `tool_executor.STATUS_*` constants.

No environment-side commit gate (plan section B.5): a COMMIT/ESCALATE
call always ends the episode; whether it was grounded is an evaluator-
side judgment recorded on the DECISION trace event, never something the
environment blocks. This is deliberate -- it keeps the correct final
action always the agent's own doing, never a safety net's (plan finding
about audit #9).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional

_ORCH_SRC = Path(__file__).resolve().parents[1]
if str(_ORCH_SRC) not in sys.path:
    sys.path.insert(0, str(_ORCH_SRC))

from execution_trace import ExecutionTrace, Stage  # noqa: E402
from tool_executor import STATUS_SUCCESS, STATUS_UNAVAILABLE  # noqa: E402

from . import substrate  # noqa: E402
from .semantics import (  # noqa: E402
    ACTION_COST_FIELD, AcousticReading, DeliveredEvidence, EpisodeSpec, IoTReading,
    ThermalReading, TERMINAL_ACTIONS, action_cost, commit_is_grounded, g_world, sufficiency,
)

__all__ = ["Observation", "StepResult", "B2Env"]

_MODALITY_OF_ACTION = {
    "ACQUIRE_IOT": "iot", "REINSPECT_IOT": "iot",
    "ACQUIRE_ACOUSTIC": "acoustic", "ACQUIRE_THERMAL": "thermal",
}
_STORE_MODALITY = {"iot": "iot_timeseries", "acoustic": "acoustic", "thermal": "thermal"}
_ACQUIRE_ACTIONS = frozenset({"ACQUIRE_IOT", "REINSPECT_IOT", "ACQUIRE_ACOUSTIC", "ACQUIRE_THERMAL"})
_ALL_ACTIONS = _ACQUIRE_ACTIONS | {"QUERY_RECORD", "DISPATCH"} | frozenset(TERMINAL_ACTIONS)


@dataclass
class Observation:
    status: str                        # "delivered" | "unavailable" | "blocked" | "ok"
    payload: Dict[str, Any] = field(default_factory=dict)
    remaining_budget: Optional[float] = None
    cost_charged: float = 0.0
    episode_done: bool = False


@dataclass
class StepResult:
    observation: Observation
    delivered: DeliveredEvidence
    terminated: bool


class B2Env:
    """One episode's live state machine. `reset` initializes from an
    EpisodeSpec (building the WorldPool once, reading the hidden label --
    an evaluator-only step, plan section C); `step` executes exactly one
    agent action against the real substrate and returns only whitelisted,
    agent-visible content."""

    def __init__(self) -> None:
        self.spec: Optional[EpisodeSpec] = None
        self.trace: Optional[ExecutionTrace] = None
        self.delivered = DeliveredEvidence()
        self.dispatched = False
        self.spent = 0.0
        self._drawn: Dict[str, FrozenSet[str]] = {"iot": frozenset(), "acoustic": frozenset(), "thermal": frozenset()}
        self._pool: Optional[substrate.WorldPool] = None
        self._done = False
        self._step_index = 0

    # ------------------------------------------------------------------ reset

    def reset(self, spec: EpisodeSpec) -> Observation:
        self.spec = spec
        self.trace = ExecutionTrace(scenario_id=spec.episode_id, arm_id=spec.regime.draw_variant)
        self.delivered = DeliveredEvidence()
        self.dispatched = False
        self.spent = 0.0
        self._drawn = {"iot": frozenset(), "acoustic": frozenset(), "thermal": frozenset()}
        self._done = False
        self._step_index = 0
        self._pool = substrate.build_world_pool(
            asset=spec.world.asset, condition=spec.world.condition, machine_id=spec.world.machine_id,
            iot_band_state=spec.world.iot_band_state, thermal_fault_class=spec.world.thermal_fault_class,
            max_draws=10,
        )
        for modality in spec.regime.initial_modalities:
            self._acquire(modality, is_initial=True)
        return Observation(status="ok", payload={"initial_evidence": self._render_delivered_summary()},
                            remaining_budget=self._remaining_budget())

    # ------------------------------------------------------------------- step

    def step(self, action: str) -> StepResult:
        if self._done:
            raise RuntimeError("step() called after episode terminated")
        if action not in _ALL_ACTIONS:
            raise ValueError(f"unknown B2 action: {action!r}")

        self.trace.append(Stage.REQUESTED, step=self._step_index, tool=action)

        if action in TERMINAL_ACTIONS:
            obs = self._terminal(action)
            self._done = True
            return StepResult(observation=obs, delivered=self.delivered, terminated=True)

        if action == "DISPATCH":
            obs = self._dispatch()
        elif action == "QUERY_RECORD":
            obs = self._query_record()
        elif action in _ACQUIRE_ACTIONS:
            obs = self._acquire(_MODALITY_OF_ACTION[action])
        else:  # pragma: no cover -- guarded by _ALL_ACTIONS above
            raise AssertionError(action)

        self._step_index += 1
        return StepResult(observation=obs, delivered=self.delivered, terminated=False)

    # ------------------------------------------------------------- internals

    def _remaining_budget(self) -> Optional[float]:
        if self.spec.regime.budget is None:
            return None
        return self.spec.regime.budget - self.spent

    def _charge(self, action_name: str) -> Optional[float]:
        cost = action_cost(action_name, self.spec.regime.costs)
        if self.spec.regime.budget is not None and self.spent + cost > self.spec.regime.budget:
            return None  # would exceed budget -- caller reports BLOCKED, nothing charged
        self.spent += cost
        return cost

    def _tool_available(self, modality: Optional[str]) -> bool:
        if modality is None:
            return True
        if not self.spec.regime.acquisition_enabled:
            return False
        return modality in self.spec.regime.available_modalities

    def _dispatch(self) -> Observation:
        available = ("acoustic" in self.spec.regime.available_modalities or
                     "thermal" in self.spec.regime.available_modalities) and \
                    self.spec.regime.acquisition_enabled
        if not available:
            self.trace.append(Stage.EXECUTED, step=self._step_index, tool="DISPATCH", status=STATUS_UNAVAILABLE)
            return Observation(status="unavailable", payload={"reason": "robot dispatch not available"})
        cost = self._charge("DISPATCH")
        if cost is None:
            return Observation(status="blocked", payload={"reason": "over budget"},
                                remaining_budget=self._remaining_budget())
        self.dispatched = True
        self.trace.append(Stage.EXECUTED, step=self._step_index, tool="DISPATCH", status=STATUS_SUCCESS)
        self.trace.append(Stage.SUCCEEDED, step=self._step_index, tool="DISPATCH", detail={"cost": cost})
        return Observation(status="ok", payload={"dispatched": True}, cost_charged=cost,
                            remaining_budget=self._remaining_budget())

    def _query_record(self) -> Observation:
        if not self._tool_available("record") and "record" not in self.spec.regime.available_modalities:
            self.trace.append(Stage.EXECUTED, step=self._step_index, tool="QUERY_RECORD", status=STATUS_UNAVAILABLE)
            return Observation(status="unavailable", payload={"reason": "record lookup not available"})
        cost = self._charge("QUERY_RECORD")
        if cost is None:
            return Observation(status="blocked", payload={"reason": "over budget"},
                                remaining_budget=self._remaining_budget())
        self.delivered.record_queried = True
        self.trace.append(Stage.EXECUTED, step=self._step_index, tool="QUERY_RECORD", status=STATUS_SUCCESS)
        self.trace.append(Stage.SUCCEEDED, step=self._step_index, tool="QUERY_RECORD", detail={"cost": cost})
        payload = {"active_work_order": self.spec.world.active_work_order,
                   "technician_present": self.spec.world.technician_present}
        obs_id = f"PGOBS::{self.spec.episode_id}::record::{self._step_index}"
        self.trace.append(Stage.OBSERVATION_DELIVERED, step=self._step_index, tool="QUERY_RECORD",
                           observation_id=obs_id, modality="enterprise")
        return Observation(status="delivered", payload=payload, cost_charged=cost,
                            remaining_budget=self._remaining_budget())

    def _acquire(self, modality: str, *, is_initial: bool = False) -> Observation:
        action_name = {"iot": "ACQUIRE_IOT", "acoustic": "ACQUIRE_ACOUSTIC", "thermal": "ACQUIRE_THERMAL"}[modality]
        if not self._tool_available(modality):
            self.trace.append(Stage.EXECUTED, step=self._step_index, tool=action_name, status=STATUS_UNAVAILABLE)
            return Observation(status="unavailable", payload={"reason": f"{modality} not available this regime"})
        if modality in ("acoustic", "thermal") and not self.dispatched and not is_initial:
            self.trace.append(Stage.EXECUTED, step=self._step_index, tool=action_name, status="failed")
            return Observation(status="blocked", payload={"reason": "requires DISPATCH first"})

        cost = 0.0 if is_initial else self._charge(action_name)
        if cost is None:
            return Observation(status="blocked", payload={"reason": "over budget"},
                                remaining_budget=self._remaining_budget())

        pool_ids = {"iot": self._pool.iot_ids, "acoustic": self._pool.acoustic_ids,
                    "thermal": self._pool.thermal_ids}[modality]
        store_modality = _STORE_MODALITY[modality]
        record = substrate.resolve_next(self.spec.world.asset, store_modality, list(pool_ids),
                                         self._drawn[modality], inspection_id=self.spec.episode_id)
        if record is None:
            self.trace.append(Stage.EXECUTED, step=self._step_index, tool=action_name, status=STATUS_UNAVAILABLE)
            return Observation(status="unavailable", payload={"reason": f"no further {modality} evidence"},
                                cost_charged=cost or 0.0, remaining_budget=self._remaining_budget())

        self._drawn[modality] = self._drawn[modality] | {record.observation_id}
        minted_id = f"PGOBS::{self.spec.episode_id}::{modality}::{len(self._drawn[modality])}"
        rendered = substrate.render_reading(record, minted_id)
        self._append_delivered(modality, rendered)

        self.trace.append(Stage.EXECUTED, step=self._step_index, tool=action_name, status=STATUS_SUCCESS,
                           modality=modality, asset_id=self.spec.world.asset)
        self.trace.append(Stage.SUCCEEDED, step=self._step_index, tool=action_name)
        self.trace.append(Stage.OBSERVATION_DELIVERED, step=self._step_index, tool=action_name,
                           observation_id=minted_id, modality=modality, asset_id=self.spec.world.asset,
                           detail={"rendered": rendered})
        return Observation(status="delivered", payload=rendered, cost_charged=cost or 0.0,
                            remaining_budget=self._remaining_budget())

    def _append_delivered(self, modality: str, rendered: Dict[str, Any]) -> None:
        if modality == "acoustic":
            self.delivered.acoustic.append(AcousticReading(positive=rendered["indicator"] == "positive",
                                                             indicator_value=rendered["indicator_value"]))
        elif modality == "iot":
            self.delivered.iot.append(IoTReading(in_band=rendered["in_band"],
                                                   value=rendered.get("value_relative") or 0.0))
        elif modality == "thermal":
            self.delivered.thermal.append(ThermalReading(hotspot_present=rendered["hotspot"]))

    def _terminal(self, action: str) -> Observation:
        delivered_ids = set(self.trace.delivered_observations().keys())
        grounded = commit_is_grounded(action, self.delivered) if action != "ESCALATE" else None
        r = sufficiency(self.delivered)
        self.trace.append(Stage.DECISION, step=self._step_index, tool=action, status=STATUS_SUCCESS,
                           detail={"claimed_action": action, "sufficient_at_decision": r,
                                   "grounded": grounded, "world_truth": g_world(self.spec.world),
                                   "delivered_observation_count": len(delivered_ids),
                                   "spent": self.spent, "dispatched": self.dispatched})
        return Observation(status="ok", payload={"final_action": action}, episode_done=True,
                            remaining_budget=self._remaining_budget())

    def _render_delivered_summary(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for r in self.delivered.acoustic:
            out.append({"modality": "acoustic", "indicator": "positive" if r.positive else "negative"})
        for r in self.delivered.iot:
            out.append({"modality": "iot", "in_band": r.in_band})
        for r in self.delivered.thermal:
            out.append({"modality": "thermal", "hotspot": r.hotspot_present})
        return out
