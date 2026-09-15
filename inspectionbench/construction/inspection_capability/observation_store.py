"""In-memory index over `observation_records.json`, plus a JSONL append log
for observations created at runtime (currently only `rgb_gauge`, whose
`gauge_path` materializes dynamically in CouchDB rather than being
precomputable offline — see `build_observation_records.py` for the
precomputed modalities: `thermal`, `iot_timeseries`, `workorder_history`,
`physical_access_geometry`).

Indexed by `(asset_id, modality)` and kept sorted by `timestamp` so
`ObservationResolver.resolve()` never linear-scans the full record set —
relevant because a single asset's IoT time series alone can run to
thousands of records (e.g. `motor_01.json` has ~4096 readings).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

__all__ = ["ObservationRecord", "ObservationStore"]


#: Pass R2 (provenance-aware observation contract). Extends the L1/L2/L3
#: taxonomy (binding, plan section R0-PRE) for use at the individual-record
#: level: R0-PRE was defined over physical sensor evidence and explicitly
#: does not cover digital/enterprise records (IoT telemetry, work orders,
#: static geometry lookups) -- forcing those into L1/L2/L3 would misrepresent
#: telemetry as sensor-evidence provenance, so they get their own value
#: rather than being silently mapped onto the nearest physical-evidence class.
EVIDENCE_PROVENANCE_VALUES = (
    "L1_REAL_ASSET_EVIDENCE",
    "L2_ASSET_CLASS_EVIDENCE_REPLAY",
    "L3_EVIDENCE_DIAGNOSTIC",
    "SIMULATED_UNCLASSIFIED",
    "NOT_APPLICABLE_DIGITAL_ENTERPRISE",
)


@dataclass
class ObservationRecord:
    observation_id: str
    asset_id: str
    inspection_id: str
    modality: str
    timestamp: str
    quality: Optional[float]
    source_ref: dict[str, Any] = field(default_factory=dict)
    sensor_metadata: dict[str, Any] = field(default_factory=dict)
    #: Pass R2 fields. All default to None ("explicit unknown/null status" per
    #: the R2 stop condition) rather than a fabricated value -- most records
    #: predate any WORLD/site registry (that is R4's job), so world_id/
    #: site_id/inspection_point_id are genuinely unknown today, not merely
    #: unpopulated. See build_observation_records.py's
    #: `_inject_provenance_fields` for what IS knowable per modality today.
    world_id: Optional[str] = None
    site_id: Optional[str] = None
    inspection_point_id: Optional[str] = None
    episode_id: Optional[str] = None
    acquisition_method: Optional[str] = None
    evidence_class: Optional[str] = None
    evidence_provenance_class: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ObservationRecord":
        return cls(
            observation_id=d["observation_id"],
            asset_id=d["asset_id"],
            inspection_id=d["inspection_id"],
            modality=d["modality"],
            timestamp=d["timestamp"],
            quality=d.get("quality"),
            source_ref=d.get("source_ref", {}),
            sensor_metadata=d.get("sensor_metadata", {}),
            world_id=d.get("world_id"),
            site_id=d.get("site_id"),
            inspection_point_id=d.get("inspection_point_id"),
            episode_id=d.get("episode_id"),
            acquisition_method=d.get("acquisition_method"),
            evidence_class=d.get("evidence_class"),
            evidence_provenance_class=d.get("evidence_provenance_class"),
        )


class ObservationStore:
    """Loads `observation_records.json` once and indexes
    `dict[(asset_id, modality)] -> list[ObservationRecord]` sorted by
    timestamp (ascending, so `bisect`/tail-scan for "freshest within
    max_age" is cheap). `append()` supports runtime-created records
    (`rgb_gauge`) and optionally mirrors them to a JSONL log, following the
    same append-log convention `skill_library/skill_kg.py`'s `SkillRegistry`
    already uses for its own JSONL persistence."""

    def __init__(self, path: Optional[Path] = None, append_log_path: Optional[Path] = None):
        self._index: dict[tuple[str, str], list[ObservationRecord]] = {}
        self._append_log_path = append_log_path
        if path is not None and path.exists():
            payload = json.loads(path.read_text())
            for raw in payload.get("observations", []):
                self._insert(ObservationRecord.from_dict(raw))

    def _insert(self, record: ObservationRecord) -> None:
        key = (record.asset_id, record.modality)
        bucket = self._index.setdefault(key, [])
        # ISO-8601 timestamps sort lexicographically = chronologically, so a
        # simple insertion-position scan keeps the bucket sorted without
        # needing a custom comparator.
        idx = 0
        while idx < len(bucket) and bucket[idx].timestamp <= record.timestamp:
            idx += 1
        bucket.insert(idx, record)

    def query(self, asset_id: str, modality: str) -> list[ObservationRecord]:
        return list(self._index.get((asset_id, modality), []))

    def append(self, record: ObservationRecord, *, persist: bool = True) -> None:
        self._insert(record)
        if persist and self._append_log_path is not None:
            self._append_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._append_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def count(self) -> int:
        return sum(len(v) for v in self._index.values())
