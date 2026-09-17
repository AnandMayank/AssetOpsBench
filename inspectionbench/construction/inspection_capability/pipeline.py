"""InspectionPipeline — wires capability selection, observation resolution,
evidence accumulation, and (for capabilities that require physical
dispatch) the existing rosclaw/`SpotAdmissibilityVerifier` gate into the
requested stage sequence:

    Scenario -> Capability Selection -> Observation Requests ->
    Observation Resolver -> Evidence Accumulation ->
    ROSClaw DigitalTwinFirewall -> Tool Execution -> Enterprise Action

`check_physical_admissibility()` is a thin, unmodified pass-through to the
**existing** `SpotAdmissibilityVerifier.verify_candidate()` (via the
existing `SkillCacheManager`) — it is a separate, sequential gate, never
merged into a `CapabilityVerdict`. Tool Execution and Enterprise Action
stages call the **existing**, unmodified MCP tools in
`src/servers/robot/main.py`; this module does not reimplement them, it
documents where in the sequence they belong and what verdict feeds them.

Scope note on `success_criteria` inputs: `ObservationResolver` confirms
evidence *exists* and is fresh/quality-sufficient (e.g. "3 gauge images were
captured within the last 5 minutes") — it does not parse the numeric gauge
value out of an image, or know which IoT JSON field is the pressure reading
for a given asset (confirmed heterogeneous: `motor_01.json` and
`chiller_6.json` use different sensor field names). Those are
capability-specific perception/parsing steps outside this framework's
charter, so `run_capability()` accepts `extra_inputs` for values only the
caller (the VLM/`read_gauge` step, the trajectory tracker) can supply —
see `_AUTO_DERIVED_KEYS` below for exactly which inputs the pipeline
computes on its own from resolved evidence, and which it always leaves to
the caller.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from .capability_registry import Capability, CapabilityRegistry, CapabilitySelector
from .evidence_accumulator import EvidenceLedger
from .observation_resolver import ObservationRequest, ObservationResolver
from .rule_engine import RuleEngineError, evaluate_success_criteria

__all__ = ["CapabilityVerdict", "InspectionPipeline", "build_scenario_context"]

# Inputs the pipeline always computes itself from resolved evidence (and
# will overwrite even if present in `extra_inputs`), vs. everything else in
# a capability's success_criteria variables/rules, which must come from
# `extra_inputs` because the framework has no generic way to derive it.
_AUTO_DERIVED_KEYS = {"n_readings", "iot_available", "hist_baseline_exists"}


@dataclass
class CapabilityVerdict:
    capability_id: str
    inspection_id: str
    asset_id: str
    verdict: str
    evidence_summary: dict[str, dict[str, int]] = field(default_factory=dict)
    error: Optional[str] = None


def build_scenario_context(manifest: dict[str, Any], asset_registry: dict[str, Any]) -> dict[str, Any]:
    """Flattens a scenario `manifest.json` + `robot_assets_registry.json`
    into the structural signal dict `CapabilitySelector` matches against.
    Field mapping is intentionally minimal — only what the example
    capabilities' `trigger` blocks actually check.

    `active_payload_modalities` answers a question distinct from every other
    field here: not "what does this scenario's evidence look like" but "what
    CAN this Spot's currently-mounted sensor payloads even produce" — derived
    from `robot.sensor_payloads` × `robot.active_payloads` in
    `robot_assets_registry.json`. A capability whose required modality isn't
    in this set gets excluded at Capability Selection, before any
    ObservationRequest is ever issued (payload absence is known upfront;
    evidence absence is what ObservationResolver checks at runtime — see
    docs/InspectionCapabilityFramework_Blueprint.md)."""
    asset_id = manifest.get("asset_id")
    asset_entry = next(
        (a for a in asset_registry.get("inspection_assets", []) if a.get("asset_id") == asset_id),
        {},
    )
    visual = manifest.get("visual") or {}

    robot = asset_registry.get("robot", {})
    active_payload_ids = set(robot.get("active_payloads", []))
    active_payload_modalities: set[str] = set()
    for payload in robot.get("sensor_payloads", []):
        if payload.get("payload_id") in active_payload_ids:
            active_payload_modalities.update(payload.get("provides_modalities", []))

    return {
        "asset_id": asset_id,
        "asset_type": asset_entry.get("asset_type"),
        "competency_primary": (manifest.get("competency") or {}).get("primary"),
        "gauge_type_present": bool(asset_entry.get("gauge_type")),
        "iot_present": bool(manifest.get("iot")),
        "visual_category": visual.get("category"),
        "hazard_class": asset_entry.get("hazard_class", 0),
        "active_payload_modalities": active_payload_modalities,
    }


class InspectionPipeline:
    def __init__(
        self,
        capability_registry: CapabilityRegistry,
        resolver: ObservationResolver,
        skill_cache_manager: Optional[Any] = None,
    ):
        self._registry = capability_registry
        self._selector = CapabilitySelector(capability_registry)
        self._resolver = resolver
        # Optional: the EXISTING SkillCacheManager (owns SpotAdmissibilityVerifier
        # / rosclaw's DigitalTwinFirewall). Not constructed here — passed in,
        # so this module never depends on skill_cache.py's CouchDB/MuJoCo
        # setup unless a caller actually needs check_physical_admissibility().
        self._skill_cache_manager = skill_cache_manager

    # --- Stage 2: Capability Selection --------------------------------
    def select_capabilities(self, scenario_context: dict[str, Any]) -> list[str]:
        return self._selector.select(scenario_context)

    # --- Stages 3-5: Observation Requests -> Resolver -> Evidence Accumulation
    def run_capability(
        self,
        capability_id: str,
        asset_id: str,
        inspection_id: str,
        extra_inputs: Optional[dict[str, Any]] = None,
    ) -> CapabilityVerdict:
        capability = self._registry.get(capability_id)
        if capability is None:
            return CapabilityVerdict(capability_id, inspection_id, asset_id, "INSUFFICIENT_EVIDENCE",
                                      error=f"unknown capability_id: {capability_id!r}")

        ledger = EvidenceLedger(inspection_id, capability_id)
        self._accumulate(capability, capability.required_evidence, asset_id, inspection_id, ledger, blocking=True)
        self._accumulate(capability, capability.optional_evidence, asset_id, inspection_id, ledger, blocking=False)

        if not ledger.all_satisfied(capability.required_evidence):
            return CapabilityVerdict(
                capability_id, inspection_id, asset_id, "INSUFFICIENT_EVIDENCE",
                evidence_summary=ledger.summary(),
                error=f"unresolved required evidence: {ledger.unresolved(capability.required_evidence)}",
            )

        inputs = self._build_rule_inputs(capability, ledger, extra_inputs or {})
        try:
            verdict = evaluate_success_criteria(capability.success_criteria, inputs)
        except RuleEngineError as exc:
            return CapabilityVerdict(capability_id, inspection_id, asset_id, "INSUFFICIENT_EVIDENCE",
                                      evidence_summary=ledger.summary(), error=str(exc))
        return CapabilityVerdict(capability_id, inspection_id, asset_id, verdict, evidence_summary=ledger.summary())

    def _accumulate(
        self,
        capability: Capability,
        evidence_specs: list[dict[str, Any]],
        asset_id: str,
        inspection_id: str,
        ledger: EvidenceLedger,
        *,
        blocking: bool,
    ) -> None:
        constraints = capability.observation_constraints
        overrides = constraints.get("per_modality_overrides", {})
        recovery = capability.recovery_policy
        strict = bool(constraints.get("require_inspection_consistency", False))
        # Capability-level default. "retry_with_relaxed_constraints" is the
        # fallback for any value without dedicated handling below (currently
        # also covers "proceed_degraded" and "request_alternate_modality" —
        # neither has a distinct code path yet, see
        # docs/InspectionCapabilityFramework_Blueprint.md §9-10 for why
        # "request_alternate_modality" doesn't currently need one).
        capability_on_unavailable = recovery.get("on_unavailable", "retry_with_relaxed_constraints")

        for spec in evidence_specs:
            modality = spec["modality"]
            min_count = spec.get("min_count", 1)
            modality_overrides = overrides.get(modality, {})
            max_age_s = modality_overrides.get("max_age_s", constraints.get("default_max_age_s", 300))
            quality_threshold = modality_overrides.get(
                "quality_threshold", constraints.get("default_quality_threshold", 0.0)
            )
            # A spec can override the capability-wide strategy for its own
            # modality — e.g. cap_sensor_physical_reconciliation wants
            # forward_fill for iot_timeseries (a stale plant sensor reading
            # is benign) but NOT for rgb_gauge (a stale gauge photo could
            # show an outdated physical reading — that's exactly the kind
            # of silent staleness FM-3 exists to catch).
            on_unavailable = spec.get("on_unavailable", capability_on_unavailable)

            consumed_ids: set[str] = set()
            # "escalate" means no retries at all, regardless of a declared
            # max_retries — an explicit named branch rather than a side
            # effect of that number happening to be 0.
            if blocking and on_unavailable != "escalate":
                retries_left = recovery.get("max_retries", 0)
            else:
                retries_left = 0

            # Each iteration excludes previously-consumed observation_ids so
            # a min_count>1 request accumulates distinct observations rather
            # than re-resolving the same "best" record forever. A retry
            # (relaxing max_age/quality per relax_order) only fires when a
            # request comes back UNAVAILABLE — accumulating further distinct
            # RESOLVED records towards min_count is not itself a "failure"
            # requiring relaxation.
            while len(ledger.resolved(modality)) < min_count:
                request = ObservationRequest(
                    asset_id=asset_id,
                    inspection_id=inspection_id,
                    modality=modality,
                    max_age_s=max_age_s,
                    quality_threshold=quality_threshold,
                    strict_inspection_consistency=strict,
                    exclude_ids=frozenset(consumed_ids),
                )
                result = self._resolver.resolve(request)
                ledger.record(modality, spec, result)

                if result.status == "RESOLVED":
                    consumed_ids.add(result.record.observation_id)  # type: ignore[union-attr]
                    continue

                # UNAVAILABLE
                if retries_left > 0:
                    retries_left -= 1
                    max_age_s, quality_threshold = self._relax(
                        recovery.get("relax_order", []), max_age_s, quality_threshold
                    )
                    continue

                if on_unavailable == "forward_fill":
                    # Last resort, tried exactly once: reuse the most recent
                    # real observation regardless of staleness (Ego-METAS's
                    # placeholder mechanism, arXiv:2606.02246v1 — see
                    # observation_resolver.py's ObservationRequest.ignore_freshness
                    # docstring). Still subject to asset/inspection-consistency
                    # and quality checks; only recency is relaxed, and the
                    # result carries is_forward_filled=True so
                    # _build_rule_inputs can expose it to success_criteria.
                    fallback_request = ObservationRequest(
                        asset_id=asset_id,
                        inspection_id=inspection_id,
                        modality=modality,
                        max_age_s=max_age_s,
                        quality_threshold=quality_threshold,
                        strict_inspection_consistency=strict,
                        exclude_ids=frozenset(consumed_ids),
                        ignore_freshness=True,
                    )
                    fallback_result = self._resolver.resolve(fallback_request)
                    ledger.record(modality, spec, fallback_result)
                    if fallback_result.status == "RESOLVED":
                        consumed_ids.add(fallback_result.record.observation_id)  # type: ignore[union-attr]
                        continue

                break

    @staticmethod
    def _relax(relax_order: list[str], max_age_s: int, quality_threshold: float) -> tuple[int, float]:
        for field_name in relax_order:
            if field_name == "max_age":
                return max_age_s * 2, quality_threshold
            if field_name == "quality_threshold":
                return max_age_s, max(0.0, quality_threshold - 0.2)
        return max_age_s, quality_threshold

    def _build_rule_inputs(
        self, capability: Capability, ledger: EvidenceLedger, extra_inputs: dict[str, Any]
    ) -> dict[str, Any]:
        inputs: dict[str, Any] = dict(extra_inputs)

        rgb_gauge = ledger.resolved("rgb_gauge")
        inputs["n_readings"] = len(rgb_gauge)
        inputs["iot_available"] = ledger.is_satisfied("iot_timeseries")
        inputs["hist_baseline_exists"] = ledger.is_satisfied("workorder_history")

        # fault_class is derivable directly from a resolved thermal
        # observation's precomputed sensor_metadata (see
        # build_observation_records.py) — auto-populate it when present,
        # but let extra_inputs override for live (non-precomputed) captures.
        if "fault_class" not in extra_inputs:
            thermal = ledger.resolved("thermal")
            if thermal:
                inputs["fault_class"] = thermal[-1].sensor_metadata.get("fault_class")

        # Generalized presence flag: "<modality>_available" for every
        # modality this capability declares (required or optional) — this
        # is what lets a capability like cap_leak_detection's rule-DSL ask
        # "was acoustic evidence present at all" (acoustic_available)
        # without pipeline.py needing a hardcoded case for every new modality.
        for spec in capability.required_evidence + capability.optional_evidence:
            modality = spec["modality"]
            inputs[f"{modality}_available"] = ledger.is_satisfied(modality)
            # Generalized staleness flag: "<modality>_forward_filled" — true
            # iff this modality's resolved evidence includes at least one
            # record admitted only via recovery_policy.on_unavailable=
            # "forward_fill" (see _accumulate). Lets a capability's
            # success_criteria explicitly downgrade confidence on stale
            # evidence instead of treating it as indistinguishable from
            # fresh — see cap_leak_detection's *_STALE verdict states.
            inputs[f"{modality}_forward_filled"] = ledger.has_forward_filled(modality)

        for key in _AUTO_DERIVED_KEYS:
            # Re-assert after the extra_inputs copy above: these three keys
            # always reflect actual resolved evidence, never a caller value.
            if key == "n_readings":
                inputs[key] = len(rgb_gauge)
            elif key == "iot_available":
                inputs[key] = ledger.is_satisfied("iot_timeseries")
            elif key == "hist_baseline_exists":
                inputs[key] = ledger.is_satisfied("workorder_history")
        return inputs

    @property
    def resolver(self) -> ObservationResolver:
        """The ObservationResolver this pipeline was built on — exposed so
        callers (e.g. a `request_observation` MCP tool) can resolve a single
        piece of evidence directly, without going through a capability's
        required/optional evidence list."""
        return self._resolver

    # --- Stage 6: ROSClaw DigitalTwinFirewall (existing, unmodified) ---
    def check_physical_admissibility(self, asset_id: str, standoff_m: float, **kwargs: Any) -> Any:
        """Thin pass-through to the EXISTING `SpotAdmissibilityVerifier`
        (rosclaw's MuJoCo DigitalTwinFirewall), owned by the
        `SkillCacheManager` passed into `__init__`. A separate, sequential
        gate from capability evidence sufficiency — a capability can be
        evidence-sufficient and still be blocked here on physical grounds,
        and vice versa. Not called unless the caller actually needs
        physical dispatch for this asset/standoff."""
        if self._skill_cache_manager is None:
            raise RuntimeError(
                "InspectionPipeline was constructed without a skill_cache_manager; "
                "physical admissibility checks require the existing SpotAdmissibilityVerifier"
            )
        return self._skill_cache_manager.verifier.verify_candidate(asset_id, standoff_m, **kwargs)
