"""ObservationResolver — the only path an agent (or capability) uses to get
evidence. Agents never open image/IoT/CSV files directly; they issue an
`ObservationRequest` and receive either a resolved `ObservationRecord` or a
typed `ObservationResult(status="UNAVAILABLE", ...)` explaining exactly which
check failed, so the calling capability can apply its own `recovery_policy`
(retry with relaxed constraints, request an alternate modality, escalate) —
the resolver itself never guesses, substitutes modalities, or invents data.

Evidence is treated as asynchronous: different modalities (RGB, thermal,
IoT, depth) arrive at different times and may not exist at all for a given
asset/inspection. `resolve()` reflects that directly — it degrades to
"UNAVAILABLE" rather than raising, and a missing modality is exactly as
valid an outcome as a stale or low-quality one, just with a different
`failed_checks` reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional

from .observation_store import ObservationRecord, ObservationStore

__all__ = [
    "ObservationRequest",
    "ObservationResult",
    "ObservationResolver",
]


@dataclass
class ObservationRequest:
    asset_id: str
    inspection_id: str
    modality: str
    max_age_s: int
    quality_threshold: float
    requested_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    # Threaded from the issuing capability's
    # observation_constraints.require_inspection_consistency. False (the
    # default) means the resolver falls back to the asset's cross-inspection
    # observation pool when nothing matches the current inspection_id,
    # rather than failing outright — see ObservationResolver.resolve().
    strict_inspection_consistency: bool = False
    # observation_ids already consumed by the caller this accumulation
    # round. resolve() is otherwise a pure, stateless "best match" query —
    # without this, a capability requiring min_count>1 of the same modality
    # (e.g. rgb_gauge x3) could never accumulate distinct observations,
    # since an identical request always returns the same best record.
    exclude_ids: frozenset[str] = field(default_factory=frozenset)
    # Forward-fill escape hatch (recovery_policy.on_unavailable="forward_fill"
    # — see InspectionPipeline._accumulate). When True, the freshness filter
    # is skipped entirely; asset/inspection-consistency and quality checks
    # still apply in full — this relaxes *only* recency, it never fabricates
    # a value, only reuses an old real one. Grounded in Ego-METAS
    # (arXiv:2606.02246v1)'s placeholder mechanism for an inactive sensor
    # (x̃_t = a_t·x_t + (1-a_t)·z_t, z_t = last realized feature) — the one
    # part of that paper that transfers to this resolver; see
    # docs/InspectionCapabilityFramework_Blueprint.md for what doesn't.
    ignore_freshness: bool = False


@dataclass
class ObservationResult:
    status: Literal["RESOLVED", "UNAVAILABLE"]
    record: Optional[ObservationRecord]
    reason: Optional[str]
    failed_checks: list[str] = field(default_factory=list)
    # True iff the resolved record would NOT have passed the normal
    # freshness check (only possible when ignore_freshness=True resolved
    # it). Computed by the resolver from the record's real age, not from
    # caller intent — mirrors Ego-METAS's requirement that a forward-filled
    # value stay visible to the consumer rather than look indistinguishable
    # from a fresh one. Always False on a normal (non-forward-filled) resolve.
    is_forward_filled: bool = False


def _parse_timestamp(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _age_seconds(record_timestamp: str, requested_at: str) -> float:
    return (_parse_timestamp(requested_at) - _parse_timestamp(record_timestamp)).total_seconds()


def _passes_quality(record: ObservationRecord, quality_threshold: float) -> bool:
    if record.quality is None:
        return quality_threshold <= 0.0
    return record.quality >= quality_threshold


class ObservationResolver:
    """Validates asset consistency, inspection consistency, freshness,
    modality, and quality (in that order) before returning the
    best-matching observation. "Best" = highest quality, ties broken by
    most recent timestamp."""

    def __init__(self, store: ObservationStore):
        self._store = store

    def known_observation_ids(self, asset_id: str, modality: str) -> frozenset[str]:
        """The set of observation_ids actually on record for (asset_id,
        modality) -- for a caller that needs to verify an observation_id it
        was handed genuinely came from a resolve() call, without exposing
        the store itself."""
        return frozenset(c.observation_id for c in self._store.query(asset_id=asset_id, modality=modality))

    def resolve(self, request: ObservationRequest) -> ObservationResult:
        # Asset + modality consistency: the store query is already scoped to
        # (asset_id, modality), so any candidate returned already satisfies
        # asset_consistency by construction — re-checked defensively below.
        candidates = self._store.query(asset_id=request.asset_id, modality=request.modality)
        candidates = [
            c for c in candidates
            if c.asset_id == request.asset_id and c.observation_id not in request.exclude_ids
        ]
        if not candidates:
            exhausted = bool(request.exclude_ids)
            return ObservationResult(
                status="UNAVAILABLE",
                record=None,
                reason=(
                    f"all known {request.modality!r} observations for asset {request.asset_id!r} "
                    "already consumed this round" if exhausted else
                    f"no {request.modality!r} observations exist for asset {request.asset_id!r}"
                ),
                failed_checks=["modality"],
            )

        same_inspection = [c for c in candidates if c.inspection_id == request.inspection_id]
        if same_inspection:
            pool = same_inspection
            inspection_consistent = True
        elif request.strict_inspection_consistency:
            return ObservationResult(
                status="UNAVAILABLE",
                record=None,
                reason=(
                    f"no {request.modality!r} observation for inspection {request.inspection_id!r} "
                    "and strict_inspection_consistency=True forbids cross-inspection reuse"
                ),
                failed_checks=["inspection_consistency"],
            )
        else:
            # Soft fallback (default): reuse the asset's cross-inspection
            # observation pool rather than failing outright.
            pool = candidates
            inspection_consistent = False

        if request.ignore_freshness:
            fresh = pool
        else:
            fresh = [c for c in pool if _age_seconds(c.timestamp, request.requested_at) <= request.max_age_s]
        if not fresh:
            failed = ["freshness"]
            if not inspection_consistent:
                failed.append("inspection_consistency")
            return ObservationResult(
                status="UNAVAILABLE",
                record=None,
                reason=f"no {request.modality!r} observation within {request.max_age_s}s of {request.requested_at}",
                failed_checks=failed,
            )

        qualified = [c for c in fresh if _passes_quality(c, request.quality_threshold)]
        if not qualified:
            return ObservationResult(
                status="UNAVAILABLE",
                record=None,
                reason=f"no {request.modality!r} observation meets quality >= {request.quality_threshold}",
                failed_checks=["quality"],
            )

        best = max(qualified, key=lambda c: (c.quality if c.quality is not None else 0.0, c.timestamp))
        forward_filled = _age_seconds(best.timestamp, request.requested_at) > request.max_age_s
        return ObservationResult(
            status="RESOLVED", record=best, reason=None, failed_checks=[], is_forward_filled=forward_filled
        )
