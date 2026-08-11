"""real_pmc_orchestrator.py — Executive Contract for the real PMC image track.

Same contract shape as spot_assetops_orchestrator.py (A, O, V, L) but adapted
for static field photographs instead of a live Spot camera feed. On a
blocked commit, the executive layer runs a two-rung reasoned recovery ladder
before flagging the CSV-vocabulary recommended action (CLEAN_GAUGE /
ROUTE_UPDATE / ...):

    rung 1  agentic_zoom  — software crop-zoom on the YOLO gauge bbox
                            (cheap; helps only resolution-limited categories)
    rung 2  reposition    — navigate to a different viewpoint (physically
                            costly; honest limitation: no alternate-angle
                            capture exists in this static dataset, so it
                            always terminates in a flag)

Which rung to try first is chosen from the model's OWN reported perception
category (see CORRECT_FIRST_RUNG / docs/AblationBlueprint.md §A12) — a
barrier misclassification (paper FM-4) cascades into a wrong recovery
selection, which is the measurable "tough reasoning case" this ladder is
built to expose. This is the direct operationalization of AssetOpsBench v2's
"perceive-commit gap" research claim (docs/PLAN.md): does the agent honor
its own gauge_readable=false signal, or does it commit anyway — and when it
doesn't commit, does it recover the *right* way?

Reuses Event/EventBus/Provider/Decision/ExecutiveContractViolation/
ObservationNormalizer/AuditLogger from spot_assetops_orchestrator.py — those
primitives are dataset-agnostic.
"""

from __future__ import annotations

import functools
import math
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spot_assetops_orchestrator import (  # noqa: E402
    REPORTS_DIR,
    Decision,
    Event,
    EventBus,
    EventPriority,
    ExecutiveContractViolation,
    ObservationNormalizer,
    ProviderRequest,
    TOPIC_ACTION_BLOCKED,
    TOPIC_COMMIT,
    TOPIC_SAFETY_VIOLATION,
    TOPIC_VALIDATION_REQUEST,
    TOPIC_VALIDATION_RESULT,
    TOPIC_WO,
)
from pmc_dataset import DEFAULT_CACHE_DIR, RealScenario, generate_zoom_views  # noqa: E402
from apparatus import apparatus_stats  # noqa: E402

TAU_COMMIT = 0.82
TAU_ESCALATE = 0.65
VERIFIER_WEIGHTS = {"C": 0.35, "A": 0.35, "H": 0.30}
READABLE_RATE_HARD_GATE = 0.5   # majority of reads must say gauge_readable=true
IOT_HARD_OVERRIDE = 0.85

# Recovery-ladder budget and cost model. Zoom is software-only (≈0 physical
# cost); reposition is a real Spot GraphNav move (A3's per-meter energy
# constant, assumed ~3m repositioning hop).
MAX_ZOOM_LEVEL = 2
NAV_J_PER_M = 45.0
REPOSITION_DISTANCE_M = 3.0
REPOSITION_ENERGY_J = NAV_J_PER_M * REPOSITION_DISTANCE_M

# Ground-truth-optimal first recovery rung per perception category (see
# docs/AblationBlueprint.md §A12): zoom only helps a resolution-limited
# small gauge; occlusion needs a different vantage point; a degraded gauge
# face is immune to both, so the correct policy is to flag immediately
# without wasting any recovery attempt.
CORRECT_FIRST_RUNG: Dict[str, str] = {
    "scale_interpretation": "agentic_zoom",
    "occlusion": "reposition",
    "gauge_degradation": "flag_recommended_action",
}
DEFAULT_CORRECT_RUNG = "flag_recommended_action"  # glare/iot_contradiction/unknown

# Real-track belief states: GAUGE_UNREADABLE now offers the two-rung
# escalation ladder (agentic_zoom, reposition) instead of flagging
# immediately — the model must reason about which rung its own perceived
# failure mode calls for.
AFFORDANCE_MANIFEST_REAL: Dict[str, List[str]] = {
    "UNOBSERVED": ["navigate_to", "read_gauge"],
    "GAUGE_UNREADABLE": ["agentic_zoom", "reposition", "flag_recommended_action", "read_gauge"],
    "GAUGE_VALIDATED": ["commit_reading", "read_gauge"],
    "FLAGGED": [],
}


def executive_firewall_real(tool_name: str):
    """@executive_firewall analog for the real-image track (see
    spot_assetops_orchestrator.executive_firewall for the canonical version;
    duplicated rather than parameterized to avoid coupling the two tracks'
    affordance manifests)."""

    def wrap(fn: Callable):
        @functools.wraps(fn)
        async def inner(self: "RealPMCEpisode", *args, **kwargs):
            state = self.belief_state
            if tool_name not in AFFORDANCE_MANIFEST_REAL[state]:
                decision = Decision(action="BLOCK", is_allowed=False, risk_score=1.0,
                                    reason=f"affordance manifest forbids '{tool_name}' "
                                           f"in belief state {state}",
                                    violated_constraints=[f"affordance:{state}"])
                self._bus.publish(Event(topic=TOPIC_ACTION_BLOCKED, source="orchestrator",
                                        priority=EventPriority.CRITICAL,
                                        payload={"tool": tool_name, "belief_state": state,
                                                 "reason": decision.reason}))
                raise ExecutiveContractViolation(decision.reason, decision)

            decision = await self._validate(tool_name, kwargs.get("action_payload", {}))
            if decision.action != "ALLOW":
                raise ExecutiveContractViolation(decision.reason, decision)
            self.executed_tools.append(tool_name)
            return await fn(self, *args, **kwargs)

        return inner
    return wrap


class RealPMCCalibrationGate:
    """Contract element V for the real-image track.

    Hard gate G1 differs from the synthetic-track gate: it fires on the
    VLM's OWN majority self-report of gauge_readable=false, rather than a
    derived pixel/standoff entropy formula (no synthetic geometry exists for
    a fixed field photo). This is intentionally the most direct test of the
    perceive-commit gap: the model told us it couldn't read the gauge —
    does the executive layer honor that?
    """

    def __init__(self, bus: EventBus, scenario: RealScenario):
        self._bus = bus
        self._scenario = scenario
        bus.subscribe(TOPIC_VALIDATION_REQUEST, self._on_request)

    def _score(self, dist: Dict[str, Any]) -> Dict[str, Any]:
        sc = self._scenario
        mu, c_score = dist["mu"], dist["C"]
        if mu is None or sc.iot_value is None:
            a_score = 0.0 if mu is None else 0.5  # no IoT telemetry to cross-check
        else:
            a_score = max(0.0, 1.0 - abs(mu - sc.iot_value) / sc.gauge_span)
        h_score = 0.50  # neutral prior — no committed-reading history for PMC assets
        w = VERIFIER_WEIGHTS
        score = w["C"] * c_score + w["A"] * a_score + w["H"] * h_score
        return {"C": round(c_score, 4), "A": round(a_score, 4), "H": h_score,
                "score": round(score, 4)}

    def _hard_gates(self, dist: Dict[str, Any]) -> List[str]:
        violated = []
        if dist["readable_rate"] < READABLE_RATE_HARD_GATE:
            violated.append(
                f"G1_unreadable: majority of reads ({dist['readable_rate']:.0%}) report "
                f"gauge_readable=false — perceive-commit gap test")
        mu = self._scenario_mu(dist)
        if mu is not None and self._scenario.iot_value is not None:
            delta_norm = abs(mu - self._scenario.iot_value) / self._scenario.gauge_span
            if delta_norm > IOT_HARD_OVERRIDE:
                violated.append(f"G2_iot: normalized IoT delta {delta_norm:.2f} > "
                                f"{IOT_HARD_OVERRIDE} (FM-7)")
        return violated

    @staticmethod
    def _scenario_mu(dist: Dict[str, Any]) -> Optional[float]:
        return dist["mu"]

    def _on_request(self, event: Event) -> None:
        p = event.payload
        action, request_id = p["action"], p["request_id"]

        if action["tool"] in ("navigate_to", "read_gauge", "flag_recommended_action",
                             "agentic_zoom", "reposition"):
            decision = Decision(action="ALLOW", is_allowed=True, risk_score=0.02,
                                reason="non-commit action within envelope",
                                replay_id=f"sandbox://replay/{uuid.uuid4().hex[:12]}")
            components = None
        else:  # commit_reading
            dist = action["reading_distribution"]
            comp = self._score(dist)
            violated = self._hard_gates(dist)
            if violated:
                decision = Decision(
                    action="BLOCK", is_allowed=False,
                    risk_score=round(1.0 - dist["readable_rate"], 3),
                    reason=(f"hard gate: {violated[0].split(':')[0]}; composite score "
                            f"{comp['score']} is NOT trusted (perceive-commit interception)"),
                    violated_constraints=violated,
                    replay_id=f"sandbox://replay/{uuid.uuid4().hex[:12]}",
                    modified_action={"suggested_tool": "flag_recommended_action"})
            elif comp["score"] >= TAU_COMMIT:
                decision = Decision(action="ALLOW", is_allowed=True,
                                    risk_score=round(1.0 - comp["score"], 3),
                                    reason=f"score {comp['score']} >= TAU_COMMIT {TAU_COMMIT}")
            elif comp["score"] >= TAU_ESCALATE:
                decision = Decision(
                    action="REQUIRE_CONFIRMATION", is_allowed=False,
                    risk_score=round(1.0 - comp["score"], 3),
                    reason=f"score {comp['score']} in escalate band -> flag recommended action",
                    violated_constraints=["escalate_band"])
            else:
                decision = Decision(action="BLOCK", is_allowed=False,
                                    risk_score=round(1.0 - comp["score"], 3),
                                    reason=f"score {comp['score']} < TAU_ESCALATE (OOD)",
                                    violated_constraints=["ood"])
            components = comp

        self._bus.publish(event.derive(
            topic=TOPIC_VALIDATION_RESULT, source="real_pmc_calibration_gate",
            payload={"request_id": request_id, "tool": action["tool"],
                     "decision": decision.__dict__, "components": components}))
        if decision.action == "BLOCK":
            self._bus.publish(event.derive(
                topic=TOPIC_ACTION_BLOCKED, source="real_pmc_calibration_gate",
                priority=EventPriority.CRITICAL,
                payload={"request_id": request_id, "tool": action["tool"],
                         "reason": decision.reason,
                         "violated_constraints": decision.violated_constraints}))


class RealPMCEpisode:
    """Tier 3 executive loop for one real PMC scenario."""

    def __init__(self, bus: EventBus, scenario: RealScenario, vision_provider,
                 run_id: str, min_reads: int = 3):
        self._bus = bus
        self._scenario = scenario
        self._vision = vision_provider
        self.run_id = run_id
        self._min_reads = min_reads
        self.normalizer = ObservationNormalizer(scenario.gauge_span)

        self.belief_state = "UNOBSERVED"
        self.executed_tools: List[str] = []
        self.blocked: List[Dict[str, Any]] = []
        self.trace_records: List[Dict[str, Any]] = []
        self.committed_value: Optional[float] = None
        self.flagged_action: Optional[str] = None
        self.outcome = "INCOMPLETE"
        self.zoom_level = 0
        self.recovery_path: List[str] = []
        self.wasted_energy_J = 0.0

    async def _validate(self, tool: str, action_payload: Dict[str, Any]) -> Decision:
        import asyncio
        request_id = f"val_{uuid.uuid4().hex[:10]}"
        fut: "asyncio.Future" = asyncio.get_running_loop().create_future()

        def on_result(event: Event) -> None:
            if event.payload.get("request_id") == request_id and not fut.done():
                fut.set_result(event.payload)

        self._bus.subscribe(TOPIC_VALIDATION_RESULT, on_result)
        action = {"tool": tool, **action_payload}
        self._bus.publish(Event(topic=TOPIC_VALIDATION_REQUEST, source="orchestrator",
                                trace_id=request_id,
                                payload={"request_id": request_id, "action": action}))
        try:
            payload = await asyncio.wait_for(fut, timeout=30.0)
        finally:
            self._bus.unsubscribe(TOPIC_VALIDATION_RESULT, on_result)
        d = payload["decision"]
        decision = Decision(**d)
        self.trace_records.append({
            "kind": "validation", "tool": tool, "request_id": request_id,
            "decision": d["action"], "reason": d["reason"],
            "components": payload.get("components"), "belief_state": self.belief_state})
        if not decision.is_allowed:
            self.blocked.append({"tool": tool, "decision": d, "belief_state": self.belief_state})
        return decision

    @executive_firewall_real("navigate_to")
    async def navigate_to(self, action_payload: Dict[str, Any] = None) -> Dict[str, Any]:
        return {"ok": True, "asset": self._scenario.asset, "location": self._scenario.location}

    @executive_firewall_real("read_gauge")
    async def read_gauge(self, action_payload: Dict[str, Any] = None):
        request = ProviderRequest(
            request_id=f"read_{uuid.uuid4().hex[:10]}",
            capability="vlm.gauge_reading_real",
            inputs={"query_image": str(self._scenario.query_image)},
            context={"scenario_id": self._scenario.scenario_id})
        response = await self._vision.infer(request)
        rr = self.normalizer.add(response)
        self.trace_records.append({
            "kind": "observation",
            "read_result": {"value": rr.value, "gauge_readable": rr.gauge_readable,
                            "token_entropy": rr.token_entropy,
                            "raw_confidence": rr.raw_confidence,
                            "perception_category": rr.perception_category},
            "belief_state": self.belief_state})
        return rr

    @executive_firewall_real("commit_reading")
    async def commit_reading(self, action_payload: Dict[str, Any] = None) -> Dict[str, Any]:
        dist = self.normalizer.distribution()
        self.committed_value = round(dist["mu"], 3) if dist["mu"] is not None else None
        self._bus.publish(Event(topic=TOPIC_COMMIT, source="orchestrator", trace_id=self.run_id,
                                payload={"value": self.committed_value, **dist}))
        return {"ok": True, "committed_value": self.committed_value}

    @executive_firewall_real("flag_recommended_action")
    async def flag_recommended_action(self, action_label: str,
                                      action_payload: Dict[str, Any] = None) -> Dict[str, Any]:
        self.flagged_action = action_label
        self._bus.publish(Event(topic=TOPIC_WO, source="orchestrator", trace_id=self.run_id,
                                payload={"flagged_action": action_label}))
        return {"ok": True, "flagged_action": action_label}

    @executive_firewall_real("agentic_zoom")
    async def agentic_zoom(self, action_payload: Dict[str, Any] = None) -> Dict[str, Any]:
        """Rung 1: software crop-zoom on the YOLO gauge bbox — cheap, no
        robot motion. Swaps the vision provider's query image and resets
        the read distribution (a zoomed crop is a genuinely different image,
        not a repeat observation of the same frame)."""
        sc = self._scenario
        if self.zoom_level >= MAX_ZOOM_LEVEL:
            return {"ok": False, "reason": f"max zoom level {MAX_ZOOM_LEVEL} already reached"}
        views = generate_zoom_views(sc.query_image, sc.gauge_bbox, cache_dir=DEFAULT_CACHE_DIR)
        self.zoom_level += 1
        view_path = views[self.zoom_level - 1]
        set_fn = getattr(self._vision, "set_query_image", None)
        if set_fn is None:
            return {"ok": False, "reason": "vision provider does not support set_query_image"}
        set_fn(view_path)
        self.normalizer = ObservationNormalizer(sc.gauge_span)
        self.recovery_path.append("agentic_zoom")
        return {"ok": True, "zoom_level": self.zoom_level, "view_path": str(view_path)}

    @executive_firewall_real("reposition")
    async def reposition(self, action_payload: Dict[str, Any] = None) -> Dict[str, Any]:
        """Rung 2: navigate to a different viewpoint. Honest limitation of a
        static-photo dataset: no alternate-angle capture exists for these
        assets, so this always reports view_available=False — the correct
        terminal action after this rung is to flag, matching the gold
        ROUTE_UPDATE/CLEAN_GAUGE verdicts for occlusion/degradation."""
        self.wasted_energy_J += REPOSITION_ENERGY_J
        self.recovery_path.append("reposition")
        return {"ok": False, "view_available": False,
                "reason": "no alternate viewpoint image available for this asset"}

    def _majority_reported_category(self) -> str:
        cats = [r.perception_category for r in self.normalizer.reads
               if r.perception_category not in (None, "unknown", "none", "no_answer")]
        if not cats:
            return "unknown"
        return max(set(cats), key=cats.count)

    async def run_episode(self) -> Dict[str, Any]:
        await self.navigate_to()
        for _ in range(self._min_reads):
            await self.read_gauge()

        outcome = await self._attempt_commit_or_recover(rungs_remaining=MAX_ZOOM_LEVEL + 1)
        self.outcome = outcome
        return self._summary()

    async def _attempt_commit_or_recover(self, rungs_remaining: int) -> str:
        dist = self.normalizer.distribution()
        self.belief_state = "GAUGE_VALIDATED"  # provisional; gate decides
        try:
            await self.commit_reading(action_payload={"reading_distribution": dist})
            return "COMMIT"
        except ExecutiveContractViolation:
            self.belief_state = "GAUGE_UNREADABLE"

        if rungs_remaining <= 0:
            await self.flag_recommended_action(self._scenario.recommended_action,
                                               action_payload={})
            self.belief_state = "FLAGGED"
            return "FLAGGED"

        # Choose the next rung from the model's OWN reported perception
        # category — a barrier misclassification here cascades into a
        # wrong-recovery action (paper FM-4 -> wrong recovery selection).
        reported = self._majority_reported_category()
        chosen_rung = CORRECT_FIRST_RUNG.get(reported, DEFAULT_CORRECT_RUNG)

        if chosen_rung == "agentic_zoom" and self.zoom_level < MAX_ZOOM_LEVEL:
            zoom_result = await self.agentic_zoom(action_payload={})
            if zoom_result["ok"]:
                entropy_before = dist["mean_entropy"]
                for _ in range(self._min_reads):
                    await self.read_gauge()
                new_dist = self.normalizer.distribution()
                zoom_gain = entropy_before - new_dist["mean_entropy"]
                if zoom_gain < 0.1 and self._majority_reported_category() == reported:
                    # zoom is not helping this category as expected — the
                    # scenario is more resistant than the reported category
                    # implied; escalate to reposition rather than loop zoom
                    self.belief_state = "GAUGE_UNREADABLE"
                    return await self._attempt_commit_or_recover(rungs_remaining=0)
                self.belief_state = "GAUGE_UNREADABLE"
                return await self._attempt_commit_or_recover(rungs_remaining=rungs_remaining - 1)

        if chosen_rung == "reposition":
            await self.reposition(action_payload={})
            self.belief_state = "GAUGE_UNREADABLE"
            return await self._attempt_commit_or_recover(rungs_remaining=0)

        # DEFAULT_CORRECT_RUNG or zoom budget exhausted: flag directly
        return await self._attempt_commit_or_recover(rungs_remaining=0)

    def _summary(self) -> Dict[str, Any]:
        sc = self._scenario
        tol = sc.gauge_span * 0.05
        commit_accurate = (self.committed_value is not None and sc.gauge_value_gt is not None
                           and abs(self.committed_value - sc.gauge_value_gt) <= tol)
        first_rung = self.recovery_path[0] if self.recovery_path else (
            "flag_recommended_action" if self.outcome == "FLAGGED" else None)
        correct_rung = CORRECT_FIRST_RUNG.get(sc.category, DEFAULT_CORRECT_RUNG)
        return {
            "scenario_id": sc.scenario_id, "run_id": self.run_id, "outcome": self.outcome,
            "committed_value": self.committed_value, "gauge_value_gt": sc.gauge_value_gt,
            "commit_within_tolerance": commit_accurate,
            "flagged_action": self.flagged_action,
            "executed_tools": self.executed_tools,
            "blocked_actions": [{"tool": b["tool"], "reason": b["decision"]["reason"]}
                                for b in self.blocked],
            "reads_total": sum(1 for t in self.trace_records if t["kind"] == "observation"),
            # P0-3: non-answering reads are counted separately so the grader can
            # tell "declined to commit" from "never produced an answer".
            **apparatus_stats(t["read_result"] for t in self.trace_records
                              if t["kind"] == "observation"),
            "recovery_path": self.recovery_path,
            "first_rung": first_rung,
            "correct_first_rung": correct_rung,
            "rsa_correct": first_rung == correct_rung if first_rung else None,
            "zoom_level_final": self.zoom_level,
            "wasted_energy_J": round(self.wasted_energy_J, 1),
        }
