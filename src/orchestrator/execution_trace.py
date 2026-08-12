"""execution_trace.py — Append-only execution trace for L3 (P0-2).

The L3 pilot scored PROC from a `tool_sequence` field the model wrote, so a
model that said "I called capture_image" while stating it could not capture
anything received PROC=1. The fix is to record what the *executor* did and to
keep the stages that were previously collapsed distinct:

    REQUESTED             the model asked for a tool
    EXECUTED              the executor ran it
    SUCCEEDED             it returned without error
    OBSERVATION_DELIVERED an observation reached the model's context
    OBSERVATION_USED      the decision cites that observation
    DECISION              a verdict was emitted
    ACTION_EXECUTED       the action changed environment state

``requested=True, executed=False`` must never yield PROC=1, which is the whole
point of separating the first two.

The trace is append-only: events carry a monotonic index and a hash chain, so a
later step cannot rewrite an earlier one and an evaluator can verify the record
was not edited after the fact. Observations are stored by id and content hash,
so "was this observation actually delivered" is answerable without trusting any
model-authored text.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Stage(str, Enum):
    REQUESTED = "REQUESTED"
    EXECUTED = "EXECUTED"
    SUCCEEDED = "SUCCEEDED"
    OBSERVATION_DELIVERED = "OBSERVATION_DELIVERED"
    OBSERVATION_USED = "OBSERVATION_USED"
    DECISION = "DECISION"
    ACTION_EXECUTED = "ACTION_EXECUTED"


def content_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass(frozen=True)
class TraceEvent:
    step: int
    stage: Stage
    tool: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    status: Optional[str] = None          # success | failed | unavailable
    error: Optional[str] = None
    observation_id: Optional[str] = None
    observation_hash: Optional[str] = None
    modality: Optional[str] = None        # physical | digital | enterprise | robot
    detail: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    prev_hash: str = ""
    event_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["stage"] = self.stage.value
        return d


class ExecutionTrace:
    """Append-only, hash-chained record of one episode."""

    def __init__(self, scenario_id: str, arm_id: str):
        self.scenario_id = scenario_id
        self.arm_id = arm_id
        self._events: List[TraceEvent] = []

    # ------------------------------------------------------------------ write

    def append(self, stage: Stage, **kw: Any) -> TraceEvent:
        prev = self._events[-1].event_hash if self._events else ""
        base = {"step": len(self._events), "stage": stage,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "prev_hash": prev, **kw}
        ev = TraceEvent(**base)
        payload = {k: v for k, v in ev.to_dict().items() if k != "event_hash"}
        ev = TraceEvent(**{**base, "event_hash": content_hash(payload)})
        self._events.append(ev)
        return ev

    # ------------------------------------------------------------------- read

    @property
    def events(self) -> List[TraceEvent]:
        return list(self._events)

    def stages(self, tool: str) -> set:
        return {e.stage for e in self._events if e.tool == tool}

    def executed_tools(self) -> set:
        """Tools the executor actually ran — never what the model claimed."""
        return {e.tool for e in self._events
                if e.stage is Stage.EXECUTED and e.tool}

    def succeeded_tools(self) -> set:
        return {e.tool for e in self._events
                if e.stage is Stage.SUCCEEDED and e.tool}

    def delivered_observations(self) -> Dict[str, TraceEvent]:
        return {e.observation_id: e for e in self._events
                if e.stage is Stage.OBSERVATION_DELIVERED and e.observation_id}

    def delivered_modalities(self) -> set:
        return {e.modality for e in self._events
                if e.stage is Stage.OBSERVATION_DELIVERED and e.modality}

    def requested_but_not_executed(self) -> set:
        req = {e.tool for e in self._events if e.stage is Stage.REQUESTED and e.tool}
        return req - self.executed_tools()

    def verify_chain(self) -> bool:
        """True if no event was altered or reordered after being appended."""
        prev = ""
        for e in self._events:
            if e.prev_hash != prev:
                return False
            payload = {k: v for k, v in e.to_dict().items() if k != "event_hash"}
            if e.event_hash != content_hash(payload):
                return False
            prev = e.event_hash
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "assetops.execution_trace/1",
            "scenario_id": self.scenario_id,
            "arm_id": self.arm_id,
            "chain_valid": self.verify_chain(),
            "events": [e.to_dict() for e in self._events],
        }
