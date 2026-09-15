"""EvidenceLedger — tracks, per (inspection_id, capability), which
`required_evidence`/`optional_evidence` specs have been satisfied so far.
Deliberately a plain bookkeeping structure: it does not itself decide
sufficiency policy (that's `recovery_policy`, applied by `pipeline.py`) or
resolve anything (that's `ObservationResolver`) — it just remembers what
has and hasn't resolved yet, including failed attempts, for diagnostics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .observation_resolver import ObservationResult
from .observation_store import ObservationRecord

__all__ = ["EvidenceEntry", "EvidenceLedger"]


@dataclass
class EvidenceEntry:
    evidence_spec: dict[str, Any]
    resolved_records: list[ObservationRecord] = field(default_factory=list)
    unavailable_results: list[ObservationResult] = field(default_factory=list)
    # observation_id -> whether that RESOLVED record came back forward-filled
    # (ObservationResult.is_forward_filled). Tracked separately from
    # resolved_records because ObservationRecord itself carries no
    # per-resolution flag — the same record could in principle be resolved
    # fresh in one capability run and forward-filled in another, depending
    # on that run's requested_at/max_age_s.
    forward_filled_ids: set[str] = field(default_factory=set)


class EvidenceLedger:
    def __init__(self, inspection_id: str, capability_id: str):
        self.inspection_id = inspection_id
        self.capability_id = capability_id
        self._entries: dict[str, EvidenceEntry] = {}

    def record(self, modality: str, evidence_spec: dict[str, Any], result: ObservationResult) -> None:
        entry = self._entries.setdefault(modality, EvidenceEntry(evidence_spec=evidence_spec))
        if result.status == "RESOLVED" and result.record is not None:
            entry.resolved_records.append(result.record)
            if result.is_forward_filled:
                entry.forward_filled_ids.add(result.record.observation_id)
        else:
            entry.unavailable_results.append(result)

    def is_satisfied(self, modality: str) -> bool:
        entry = self._entries.get(modality)
        if entry is None:
            return False
        return len(entry.resolved_records) >= entry.evidence_spec.get("min_count", 1)

    def resolved(self, modality: str) -> list[ObservationRecord]:
        entry = self._entries.get(modality)
        return list(entry.resolved_records) if entry else []

    def has_forward_filled(self, modality: str) -> bool:
        """True iff any RESOLVED record for this modality was forward-filled
        (stale, admitted only because recovery_policy.on_unavailable was
        "forward_fill" and nothing fresh qualified)."""
        entry = self._entries.get(modality)
        return bool(entry and entry.forward_filled_ids)

    def all_satisfied(self, evidence_specs: list[dict[str, Any]]) -> bool:
        return all(self.is_satisfied(spec["modality"]) for spec in evidence_specs)

    def unresolved(self, evidence_specs: list[dict[str, Any]]) -> list[str]:
        return [spec["modality"] for spec in evidence_specs if not self.is_satisfied(spec["modality"])]

    def summary(self) -> dict[str, dict[str, int]]:
        return {
            modality: {
                "resolved_count": len(entry.resolved_records),
                "unavailable_attempts": len(entry.unavailable_results),
                "forward_filled_count": len(entry.forward_filled_ids),
            }
            for modality, entry in self._entries.items()
        }
