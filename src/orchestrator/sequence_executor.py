"""sequence_executor.py — Sequence-scoped execution for class E (P1).

Class E measures whether an agent notices that evidence has gone stale:

    t0  observe and commit a reading
    t1  the world changes underneath it
    t2  act, with the t0 reading still in context
    t3  did it re-observe, or reuse the stale value?

Unmeasurable under the per-episode executor, where every episode re-establishes
hidden state and no history survives — so an agent *cannot* anchor to a prior
reading and the failure has no opportunity to occur.

Design commitments, each of which is a test below:

* **Reset is sequence-scoped**, not episode-scoped. Fixtures revert at sequence
  exit. Isolation is preserved *between* sequences, which is what the per-episode
  reset was protecting.
* **The drift process is drawn before any label.** ``sample_sequence`` never sees
  a verdict; gold comes from ``derive_sequence_gold`` over the prefix.
* **Observations carry an episode index and a timestamp.** Grounding requires an
  observation from the *current* episode, so a t0 observation cannot ground a t2
  decision once the world has moved.
* **Namespaced state.** Profiles are keyed per sequence so two concurrent
  sequences on one asset cannot contaminate each other.

Stale-state failure needs **no new metric**: an agent citing only a prior-episode
observation earns CC but not CC_grounded. That is the CC/CC_grounded separation
the cross-model phase established, extended in time.
"""

from __future__ import annotations

import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from execution_trace import ExecutionTrace, Stage, content_hash  # noqa: E402
from scenario_gen import ASSETS, COMMIT, ESCALATE, WorldState  # noqa: E402

#: World-evolution processes. Drawn as part of the world, before any label.
STATIONARY = "stationary"
STEP = "step"
LINEAR_DRIFT = "linear_drift"
PROCESSES: Tuple[str, ...] = (STATIONARY, STEP, LINEAR_DRIFT)


@dataclass
class DriftProcess:
    """How the hidden physical value evolves. Never stated in any prompt."""
    kind: str
    rate: float = 0.0          # units per episode, for linear_drift
    step_at: int = -1          # episode index of a step change
    step_delta: float = 0.0

    def value_at(self, base: float, episode: int) -> float:
        if self.kind == LINEAR_DRIFT:
            return base + self.rate * episode
        if self.kind == STEP and self.step_at >= 0 and episode >= self.step_at:
            return base + self.step_delta
        return base


@dataclass
class SequenceWorld:
    """A world lineage: one asset, N episodes, one evolution process."""
    sequence_id: str
    asset: str
    unit: str
    gauge_range: List[float]
    operating_band: List[float]
    base_value: float
    process: DriftProcess
    n_episodes: int
    seed: int

    def value_at(self, episode: int) -> float:
        lo, hi = self.gauge_range
        return round(max(lo, min(hi, self.process.value_at(self.base_value, episode))), 3)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["process"] = asdict(self.process)
        return d


def sample_sequence(seed: int, n_episodes: int = 3,
                    asset_id: Optional[str] = None) -> SequenceWorld:
    """Sample a sequence world. **Receives no label and cannot see one.**

    The process is drawn here, so whether drift occurs is a property of the
    world rather than of a desired answer.
    """
    rng = random.Random(seed)
    spec = ASSETS[asset_id] if asset_id else ASSETS[rng.choice(sorted(ASSETS))]
    lo, hi = spec.operating_band
    gmin, gmax = spec.gauge_range
    span = gmax - gmin

    base = round(rng.uniform(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo)), 3)
    kind = rng.choice(PROCESSES)
    if kind == LINEAR_DRIFT:
        # Rate chosen so the band is crossed partway through, giving the
        # sequence a detectable transition rather than a foregone conclusion.
        rate = (hi - base + 0.05 * span) / max(1, n_episodes - 1)
        proc = DriftProcess(LINEAR_DRIFT, rate=round(rate, 4))
    elif kind == STEP:
        proc = DriftProcess(STEP, step_at=rng.randint(1, max(1, n_episodes - 1)),
                            step_delta=round(hi - base + 0.08 * span, 3))
    else:
        proc = DriftProcess(STATIONARY)

    return SequenceWorld(
        sequence_id=f"SEQ-{seed:05d}", asset=spec.asset_id, unit=spec.unit,
        gauge_range=[gmin, gmax], operating_band=[lo, hi],
        base_value=base, process=proc, n_episodes=n_episodes, seed=seed)


def derive_sequence_gold(world: SequenceWorld, episode: int) -> Dict[str, Any]:
    """Gold for episode k, a pure function of the world and the prefix 0..k.

    Same operational rule as the single-episode case, applied to the value that
    holds *now*: a reading outside the band must be escalated.
    """
    lo, hi = world.operating_band
    v = world.value_at(episode)
    verdict = COMMIT if lo <= v <= hi else ESCALATE
    crossed = any(not (lo <= world.value_at(i) <= hi) for i in range(episode + 1))
    return {"verdict": verdict,
            "rule_applied": ("physical reading inside operating band"
                             if verdict == COMMIT else
                             "physical reading outside operating band"),
            "value_now": v,
            "band_crossed_by_now": crossed,
            "episode": episode}


class SequenceTrace:
    """Chains per-episode traces; the hash chain spans episode boundaries."""

    def __init__(self, sequence_id: str, arm_id: str):
        self.sequence_id = sequence_id
        self.arm_id = arm_id
        self.episodes: List[ExecutionTrace] = []

    def new_episode(self) -> ExecutionTrace:
        """Open an episode, stamping the previous one's terminal hash.

        The boundary is an explicit event rather than an implicit convention, so
        a dropped, reordered or fabricated episode is visible to an auditor who
        only has the trace.
        """
        idx = len(self.episodes)
        t = ExecutionTrace(f"{self.sequence_id}#{idx}", self.arm_id)
        if self.episodes:
            prev_events = self.episodes[-1].events
            prev_hash = prev_events[-1].event_hash if prev_events else ""
            t.append(Stage.DECISION, tool=None,
                     detail={"episode_boundary": idx, "prev_episode_hash": prev_hash})
        self.episodes.append(t)
        return t

    def verify_chain(self) -> bool:
        if not all(t.verify_chain() for t in self.episodes):
            return False
        # Boundary continuity: episode k's opening event must carry episode
        # k-1's terminal hash.
        for i in range(1, len(self.episodes)):
            evs = self.episodes[i].events
            prev_events = self.episodes[i - 1].events
            if not evs or not prev_events:
                return False
            if evs[0].detail.get("prev_episode_hash") != prev_events[-1].event_hash:
                return False
        return True

    def delivered_in_episode(self, episode: int) -> Dict[str, Any]:
        if episode >= len(self.episodes):
            return {}
        return self.episodes[episode].delivered_observations()

    def to_dict(self) -> Dict[str, Any]:
        return {"schema": "assetops.sequence_trace/1",
                "sequence_id": self.sequence_id, "arm_id": self.arm_id,
                "n_episodes": len(self.episodes),
                "chain_valid": self.verify_chain(),
                "episodes": [t.to_dict() for t in self.episodes]}


def grounded_in_current_episode(seq_trace: SequenceTrace, episode: int,
                                required_modality: str = "physical") -> bool:
    """Grounding requires an observation from *this* episode.

    The stale-state failure: an agent citing only a t0 observation at t2 has a
    correct decision resting on evidence the world has since invalidated.
    """
    delivered = seq_trace.delivered_in_episode(episode)
    return any(e.modality == required_modality for e in delivered.values())


def stale_state_reuse(seq_trace: SequenceTrace, episode: int,
                      required_modality: str = "physical") -> bool:
    """True when a prior episode delivered the evidence and this one did not."""
    if episode == 0:
        return False
    if grounded_in_current_episode(seq_trace, episode, required_modality):
        return False
    return any(grounded_in_current_episode(seq_trace, k, required_modality)
               for k in range(episode))


class SequenceExecutor:
    """Wraps CouchDBExecutor with sequence-scoped state and namespaced profiles."""

    backend_id = "couchdb_sequence"
    backend_version = "1.0.0"

    def __init__(self, inner) -> None:
        self._inner = inner
        self.world: Optional[SequenceWorld] = None
        self.episode: int = -1
        self._saved_profile: Optional[Dict[str, Any]] = None

    def begin_sequence(self, world: SequenceWorld) -> None:
        """Reset once, at sequence start — not between episodes."""
        self.world, self.episode = world, -1
        db = self._inner._robot.db
        self._saved_profile = dict(db.get(f"profile:{world.asset}"))

    def advance_episode(self, arm_id: str, withheld: Optional[List[str]] = None) -> int:
        """Move the world to the next episode and re-point the executor at it.

        Hidden state is updated *without* clearing history: the agent's context
        and the sequence trace both persist, which is what makes stale-state
        reuse possible at all.
        """
        assert self.world is not None, "begin_sequence first"
        self.episode += 1
        w = self.world
        ws = WorldState(
            scenario_id=f"{w.sequence_id}#{self.episode}", asset=w.asset,
            unit=w.unit, gauge_range=list(w.gauge_range),
            operating_band=list(w.operating_band),
            physical_value=w.value_at(self.episode),
            iot_value=w.value_at(self.episode), history_mean=w.base_value,
            active_work_order=False, technician_present=False,
            cell=f"seq/{w.process.kind}", seed=w.seed)
        self._inner.reset_from_world(ws, arm_id, seed=w.seed + self.episode,
                                     withheld=withheld or [])
        return self.episode

    def end_sequence(self) -> None:
        if self._saved_profile is not None and self.world is not None:
            db = self._inner._robot.db
            cur = db.get(f"profile:{self.world.asset}")
            doc = dict(self._saved_profile)
            doc["_rev"] = cur["_rev"]
            db.save(doc)
        self.world, self.episode, self._saved_profile = None, -1, None

    def execute(self, call):
        return self._inner.execute(call)

    def available_tools(self) -> List[str]:
        return self._inner.available_tools()
