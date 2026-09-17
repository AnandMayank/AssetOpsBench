"""Loads/validates `inspection_capabilities.json` and selects which
capabilities apply to a given scenario, via a small structural signal match
against each capability's `trigger` block — deterministic rule matching,
not an LLM decision (mirrors the determinism of the old
`spot_assetops_orchestrator.py`'s `AFFORDANCE_MANIFEST` state table, applied
to capability selection instead of tool-affordance gating).

`trigger.manifest_signals` entries are small structural checks evaluated
against a flattened `scenario_context` dict (built by
`pipeline.build_scenario_context()`), supporting three forms:
  "<key> present"     -> bool(scenario_context.get(key))
  "<key>==<value>"     -> str(scenario_context.get(key)) == value
  "<key>><number>"     -> scenario_context.get(key, 0) > number
This is a small hand-rolled matcher, not `rule_engine.py`'s expression
evaluator — trigger signals operate on flat scenario metadata, not on
resolved evidence with derived variables/thresholds, so the richer DSL
would be overkill here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

__all__ = ["Capability", "CapabilityRegistry", "CapabilitySelector"]

_REQUIRED_CAPABILITY_FIELDS = (
    "capability_id",
    "display_name",
    "supported_fm_families",
    "required_evidence",
    "optional_evidence",
    "recovery_policy",
    "success_criteria",
    "observation_constraints",
)


@dataclass
class Capability:
    capability_id: str
    display_name: str
    description: str
    supported_fm_families: list[str]
    trigger: dict[str, Any]
    required_evidence: list[dict[str, Any]]
    optional_evidence: list[dict[str, Any]]
    recovery_policy: dict[str, Any]
    success_criteria: dict[str, Any]
    observation_constraints: dict[str, Any]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Capability":
        missing = [f for f in _REQUIRED_CAPABILITY_FIELDS if f not in d]
        if missing:
            raise ValueError(f"capability {d.get('capability_id', '<unknown>')!r} missing fields: {missing}")
        return cls(
            capability_id=d["capability_id"],
            display_name=d["display_name"],
            description=d.get("description", ""),
            supported_fm_families=list(d["supported_fm_families"]),
            trigger=d.get("trigger", {}),
            required_evidence=list(d["required_evidence"]),
            optional_evidence=list(d["optional_evidence"]),
            recovery_policy=d["recovery_policy"],
            success_criteria=d["success_criteria"],
            observation_constraints=d["observation_constraints"],
        )


class CapabilityRegistry:
    def __init__(self, path: Path):
        payload = json.loads(path.read_text())
        self._capabilities: dict[str, Capability] = {}
        for raw in payload.get("capabilities", []):
            cap = Capability.from_dict(raw)
            if cap.capability_id in self._capabilities:
                raise ValueError(f"duplicate capability_id: {cap.capability_id!r}")
            self._capabilities[cap.capability_id] = cap

    def get(self, capability_id: str) -> Optional[Capability]:
        return self._capabilities.get(capability_id)

    def all(self) -> list[Capability]:
        return list(self._capabilities.values())


def _check_signal(signal: str, scenario_context: dict[str, Any]) -> bool:
    if "==" in signal:
        key, _, value = signal.partition("==")
        return str(scenario_context.get(key.strip())) == value.strip()
    if ">" in signal:
        key, _, value = signal.partition(">")
        try:
            threshold = float(value.strip())
        except ValueError:
            return False
        actual = scenario_context.get(key.strip(), 0)
        try:
            return float(actual) > threshold
        except (TypeError, ValueError):
            return False
    if signal.endswith(" present"):
        key = signal[: -len(" present")].strip()
        return bool(scenario_context.get(key))
    # Unrecognized signal form — fail closed (does not match) rather than
    # silently matching everything.
    return False


class CapabilitySelector:
    """Deterministic trigger match: a capability is selected iff every
    populated field of its `trigger` block matches `scenario_context`
    (empty/absent trigger fields impose no constraint — a capability with no
    `trigger` at all matches every scenario, which is intentionally
    permissive rather than intentionally restrictive)."""

    def __init__(self, registry: CapabilityRegistry):
        self._registry = registry

    def select(self, scenario_context: dict[str, Any]) -> list[str]:
        selected = []
        for cap in self._registry.all():
            if self._matches(cap.trigger, scenario_context):
                selected.append(cap.capability_id)
        return selected

    @staticmethod
    def _matches(trigger: dict[str, Any], scenario_context: dict[str, Any]) -> bool:
        asset_types = trigger.get("asset_types")
        if asset_types and scenario_context.get("asset_type") not in asset_types:
            return False

        competency_primary = trigger.get("competency_primary")
        if competency_primary and scenario_context.get("competency_primary") not in competency_primary:
            return False

        for signal in trigger.get("manifest_signals", []):
            if not _check_signal(signal, scenario_context):
                return False

        # Payload-awareness: distinct from manifest_signals because this asks
        # "can the currently-mounted sensor payloads even produce this
        # modality" rather than "does this scenario's evidence look a
        # certain way." Two semantics: _all (hard gate -- every listed
        # modality must be available) and _any (soft gate -- at least one
        # must be, for capabilities designed to degrade gracefully across
        # alternative sensors, e.g. leak detection from acoustic OR gas
        # alone). scenario_context["active_payload_modalities"] is built by
        # pipeline.build_scenario_context() from robot_assets_registry.json's
        # sensor_payloads x active_payloads.
        # Coerce defensively: pipeline.build_scenario_context() supplies a
        # set, but a manually-built scenario_context (e.g. the
        # select_capability MCP tool) may supply a list/tuple instead.
        active_modalities = set(scenario_context.get("active_payload_modalities") or [])

        required_all = trigger.get("required_payload_modalities_all")
        if required_all and not set(required_all).issubset(active_modalities):
            return False

        required_any = trigger.get("required_payload_modalities_any")
        if required_any and not (set(required_any) & active_modalities):
            return False

        return True
