"""Inspection Capability Framework.

Replaces an ASPIRE-style manipulation skill library with a framework built
for embodied *inspection*: capabilities declare what evidence they need
(`inspection_capabilities.json`), an `ObservationResolver` mediates every
piece of raw sensor evidence an agent ever sees (no direct file access),
and evidence is treated as asynchronous — different modalities (RGB,
thermal, IoT, depth) may arrive at different times, or not exist at all for
a given episode, and capabilities must degrade gracefully rather than
assume synchronized multimodal input.

Physical admissibility (joint/collision/reach/executive safety gate) stays
exactly where it already lives — rosclaw's DigitalTwinFirewall, wrapped by
the existing `orchestrator.spot_admissibility_verifier.SpotAdmissibilityVerifier`
— as a separate, sequential gate. This package does not modify or merge
with it; see `InspectionPipeline.check_physical_admissibility`.

See `docs/InspectionCapabilityFramework_Blueprint.md` for the full design.
"""
from __future__ import annotations

from .capability_registry import Capability, CapabilityRegistry, CapabilitySelector
from .evidence_accumulator import EvidenceLedger
from .observation_resolver import ObservationRequest, ObservationResolver, ObservationResult
from .observation_store import ObservationRecord, ObservationStore
from .pipeline import CapabilityVerdict, InspectionPipeline, build_scenario_context
from .rule_engine import RuleEngineError, evaluate_expression, evaluate_success_criteria

__all__ = [
    "Capability",
    "CapabilityRegistry",
    "CapabilitySelector",
    "EvidenceLedger",
    "ObservationRequest",
    "ObservationResolver",
    "ObservationResult",
    "ObservationRecord",
    "ObservationStore",
    "CapabilityVerdict",
    "InspectionPipeline",
    "build_scenario_context",
    "RuleEngineError",
    "evaluate_expression",
    "evaluate_success_criteria",
]
