"""substrate.py — the ONLY module in this package that touches the real
observation substrate (ObservationStore / ObservationResolver / real
acoustic decode). Turns real, provenance-tagged records into the
booleans `semantics.py`'s rule R operates over, and into deterministic,
per-world ordered pools that both `frontier.py` (offline oracle) and
`env.py` (live execution) draw from identically.

Reused, never reimplemented: ObservationStore, ObservationResolver,
acoustic_perception.{resolve_audio_path,read_audio_bytes,decode_wav,
compute_acoustic_features}, EvidenceLedger's bookkeeping shape.

Nothing here reads `sensor_metadata` on the agent-visible path -- the
hidden label (`source_label`, `gold_action`, `machine_id`) is read ONLY
by `b2_families.py`'s world construction (to choose which real clip pool
a WORLD parameter selects) and by `pg_b2_substrate_gates.py`'s G1 check,
never by `env.py`'s delivery path (see `render_reading` below, whose
output is the whitelist itself).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

_ORCH_SRC = Path(__file__).resolve().parents[1]
if str(_ORCH_SRC) not in sys.path:
    sys.path.insert(0, str(_ORCH_SRC))

from inspection_capability.observation_store import ObservationRecord, ObservationStore  # noqa: E402
from inspection_capability.observation_resolver import ObservationRequest, ObservationResolver  # noqa: E402
from inspection_capability.acoustic_perception import (  # noqa: E402
    resolve_audio_path, read_audio_bytes, decode_wav, compute_acoustic_features, AudioResolutionError,
)

from .semantics import acoustic_indicator  # noqa: E402

__all__ = [
    "REPO_ROOT", "OBS_RECORDS_PATH", "get_store", "get_resolver",
    "AGREE_ACOUSTIC_ASSETS", "IOT_BAND_FIELD", "IOT_BAND_BOUNDS",
    "acoustic_pool_ids", "iot_pool_records", "thermal_pool_ids",
    "resolve_next", "reading_to_boolean", "render_reading",
    "WHITELISTED_FIELDS", "FORBIDDEN_FIELDS", "scan_for_leakage",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
OBS_RECORDS_PATH = (REPO_ROOT / "src" / "orchestrator" / "inspection_capability"
                     / "data" / "observation_records.json")

_STORE_CACHE: Optional[ObservationStore] = None


def get_store() -> ObservationStore:
    global _STORE_CACHE
    if _STORE_CACHE is None:
        _STORE_CACHE = ObservationStore(path=OBS_RECORDS_PATH, append_log_path=None)
    return _STORE_CACHE


def get_resolver() -> ObservationResolver:
    return ObservationResolver(get_store())


# ---------------------------------------------------------------------------
# Which assets have real acoustic coverage, and the MIMII asset<->machine
# mapping (checked directly against observation_records.json; see the
# repo-inspection notes in the approved plan). Acoustic is L2 asset-class
# replay evidence (real MIMII recordings, not asset-specific captures) --
# this is reported in every B2 result, matching plan section C's named
# substrate limit.
# ---------------------------------------------------------------------------
AGREE_ACOUSTIC_ASSETS: FrozenSet[str] = frozenset({"chiller_6", "hydraulic_pump_1"})
ACOUSTIC_MACHINE_IDS: Tuple[str, ...] = ("00", "02", "04", "06")

#: IoT band derivation (plan: "operating_band relative world parameter").
#: The real observation_records.json IoT records carry no anomaly label,
#: so "in band"/"out of band" is defined the same way scenario_gen.py
#: already defines it for its own synthetic gauge (a scalar compared
#: against a [lo, hi] band) -- applied here to one real telemetry field
#: per asset. Bounds are the empirical [P25, P75] of that field across the
#: asset's own real readings (grounded in real data, an explicit design
#: choice -- NOT claimed to be a physically validated operating envelope).
IOT_BAND_FIELD: Dict[str, str] = {
    "chiller_6": "Chiller 6 Chiller Efficiency",
    "hydraulic_pump_1": None,   # falls back to the first numeric field found
    "motor_01": None,
    "metro_pump_1": None,
}
#: Populated lazily by `_iot_band_bounds` and cached; NOT hand-picked.
IOT_BAND_BOUNDS: Dict[str, Tuple[float, float]] = {}


def _iot_field_for(asset: str, sample_readings: Dict[str, float]) -> str:
    field = IOT_BAND_FIELD.get(asset)
    numeric_fields = sorted(k for k, v in sample_readings.items() if isinstance(v, (int, float)))
    if field and field in numeric_fields:
        return field
    if not numeric_fields:
        raise ValueError(f"no numeric IoT field found for asset {asset!r} in {sample_readings!r}")
    # Deterministic fallback: first NUMERIC field name in sorted order.
    return numeric_fields[0]


def _iot_band_bounds(asset: str) -> Tuple[str, float, float]:
    """(field_name, lo, hi) — empirical P25/P75 of the chosen field over
    every real IoT record for this asset, computed once and cached."""
    key = asset
    cached = IOT_BAND_BOUNDS.get(key)
    field = IOT_BAND_FIELD.get(asset)
    records = get_store().query(asset_id=asset, modality="iot_timeseries")
    if not records:
        raise ValueError(f"no iot_timeseries records for asset {asset!r}")
    if field is None or field not in records[0].sensor_metadata.get("readings", {}):
        field = _iot_field_for(asset, records[0].sensor_metadata.get("readings", {}))
        IOT_BAND_FIELD[asset] = field
    if cached is not None:
        return field, cached[0], cached[1]
    values = sorted(
        v for r in records
        for v in [r.sensor_metadata.get("readings", {}).get(field)]
        if isinstance(v, (int, float))
    )
    n = len(values)
    lo = values[int(0.25 * (n - 1))]
    hi = values[int(0.75 * (n - 1))]
    IOT_BAND_BOUNDS[key] = (lo, hi)
    return field, lo, hi


# ---------------------------------------------------------------------------
# Pool construction (evaluator-only; picks WHICH real clips a WORLD's
# condition draws from -- this is where the hidden label is read).
# ---------------------------------------------------------------------------

def acoustic_pool_ids(asset: str, machine_id: str, label: str, limit: int) -> List[str]:
    """NOTE: callers building a B2 WORLD must never pass a
    (asset, machine_id) in `semantics.ACOUSTIC_EXCLUDED_KEYS` (G1-failed
    combos) -- `b2_families.py` enforces this; this function does not, so
    that G4/leakage and pool-listing utilities can still inspect it."""
    """Ordered list of up to `limit` real observation_ids for
    (asset, machine_id, label) -- 'abnormal' or 'normal' -- in the SAME
    order ObservationStore.query() returns them (insertion/file order,
    since all acoustic timestamps in this pool are identical, per the
    repo inspection). This determines the WORLD's draw order; `env.py`
    consumes it via real `resolve_next` calls with `exclude_ids`, never
    by indexing this list itself, so live execution and the offline
    oracle draw the identical sequence through the identical resolver path."""
    if asset not in AGREE_ACOUSTIC_ASSETS:
        return []
    records = get_store().query(asset_id=asset, modality="acoustic")
    ids = [r.observation_id for r in records
           if r.sensor_metadata.get("machine_id") == machine_id
           and r.sensor_metadata.get("source_label") == label]
    return ids[:limit]


def thermal_pool_ids(asset: str, fault_class: Optional[str], limit: int) -> List[str]:
    if asset != "motor_01":
        return []
    records = get_store().query(asset_id=asset, modality="thermal")
    if fault_class is None:
        ids = [r.observation_id for r in records if r.sensor_metadata.get("fault_class") == "Noload"]
    else:
        ids = [r.observation_id for r in records if r.sensor_metadata.get("fault_class") == fault_class]
    return ids[:limit]


def iot_pool_records(asset: str, band_state: str, limit: int) -> List[str]:
    """Ordered real observation_ids whose chosen field is in/out of the
    asset's empirical [P25, P75] band, per `band_state` ('in_band' |
    'out_of_band')."""
    field, lo, hi = _iot_band_bounds(asset)
    records = get_store().query(asset_id=asset, modality="iot_timeseries")
    want_in = band_state == "in_band"
    ids = []
    for r in records:
        v = r.sensor_metadata.get("readings", {}).get(field)
        if not isinstance(v, (int, float)):
            continue
        is_in = lo <= v <= hi
        if is_in == want_in:
            ids.append(r.observation_id)
    return ids[:limit]


# ---------------------------------------------------------------------------
# WorldPool -- the ordered, real observation_id lists behind one WorldSpec.
# Built ONCE per world (evaluator-side, reads the hidden label/fault_class
# to pick the right pool) and then consumed identically by `env.py` (live
# resolver draws, via `resolve_next`) and `frontier.py`'s oracle (via
# `world_pool_to_booleans`, which maps each id through `reading_to_boolean`
# -- the SAME function env.py's rendering path uses) -- so the offline
# oracle and live execution can never disagree about what a draw resolves to.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorldPool:
    asset: str
    acoustic_ids: Tuple[str, ...] = ()
    iot_ids: Tuple[str, ...] = ()
    thermal_ids: Tuple[str, ...] = ()


def build_world_pool(*, asset: str, condition: str, machine_id: Optional[str],
                      iot_band_state: str, thermal_fault_class: Optional[str],
                      max_draws: int = 8) -> WorldPool:
    """The ONE place a WorldSpec's condition selects which real clip pool
    to draw from (reads sensor_metadata.source_label/fault_class -- the
    hidden label -- exactly once, here, never at the agent-visible render
    path). `condition='fault'` selects MIMII 'abnormal' clips and the
    non-Noload thermal class; `condition='normal'` selects 'normal' clips
    and the Noload thermal class."""
    label = "abnormal" if condition == "fault" else "normal"
    acoustic_ids: Tuple[str, ...] = ()
    if machine_id is not None:
        acoustic_ids = tuple(acoustic_pool_ids(asset, machine_id, label, max_draws))
    thermal_ids: Tuple[str, ...] = ()
    if asset == "motor_01":
        fc = thermal_fault_class if condition == "fault" else None
        thermal_ids = tuple(thermal_pool_ids(asset, fc, max_draws))
    iot_ids = tuple(iot_pool_records(asset, iot_band_state, max_draws))
    return WorldPool(asset=asset, acoustic_ids=acoustic_ids, iot_ids=iot_ids, thermal_ids=thermal_ids)


def world_pool_to_booleans(pool: "WorldPool") -> Dict[str, Tuple[bool, ...]]:
    """Maps a WorldPool's real ids to the booleans `frontier.DrawPool`
    consumes, through the SAME `reading_to_boolean` env.py's live draws
    resolve to -- this is what keeps the offline oracle and live execution
    in agreement (plan section E's frontier is computed over exactly this)."""
    store = get_store()
    out: Dict[str, Tuple[bool, ...]] = {}
    for modality, ids in (("acoustic", pool.acoustic_ids), ("iot_timeseries", pool.iot_ids),
                          ("thermal", pool.thermal_ids)):
        recs_by_id = {r.observation_id: r for r in store.query(asset_id=pool.asset, modality=modality)}
        out[modality] = tuple(reading_to_boolean(recs_by_id[i]) for i in ids if i in recs_by_id)
    return out


# ---------------------------------------------------------------------------
# Live resolution -- real resolver, real exclude_ids bookkeeping. Consumed
# by BOTH env.py (live) and any offline pool-materialization helper (for
# the frontier oracle), so both paths draw identically.
# ---------------------------------------------------------------------------

def resolve_next(asset: str, modality: str, pool_ids: List[str], already_drawn: FrozenSet[str],
                  *, inspection_id: str = "PG-B2") -> Optional[ObservationRecord]:
    """Draw the next undrawn item of `pool_ids` through the REAL
    ObservationResolver: excludes every known id for (asset, modality)
    EXCEPT the pool, then excludes already-drawn pool members, so
    `resolve()` picks the pool's next entry deterministically (ties on
    quality/timestamp break to file order, matching `pool_ids`' own
    order by construction). Returns None if the pool is exhausted."""
    resolver = get_resolver()
    remaining = [i for i in pool_ids if i not in already_drawn]
    if not remaining:
        return None
    known = resolver.known_observation_ids(asset, modality)
    exclude = (known - {remaining[0]})
    req = ObservationRequest(asset_id=asset, inspection_id=inspection_id, modality=modality,
                              max_age_s=999_999_999, quality_threshold=0.0, exclude_ids=frozenset(exclude))
    result = resolver.resolve(req)
    if result.status != "RESOLVED":
        return None
    return result.record


# ---------------------------------------------------------------------------
# Reading -> boolean (feeds semantics.DeliveredEvidence). Acoustic uses the
# real decoded waveform + deterministic indicator; NEVER the hidden label.
# ---------------------------------------------------------------------------

_FEATURE_CACHE: Dict[str, Dict[str, Any]] = {}


def _acoustic_features(record: ObservationRecord) -> Dict[str, Any]:
    cached = _FEATURE_CACHE.get(record.observation_id)
    if cached is not None:
        return cached
    path = resolve_audio_path(record.source_ref)
    samples, sr = decode_wav(read_audio_bytes(path))
    features = compute_acoustic_features(samples, sr)
    _FEATURE_CACHE[record.observation_id] = features
    return features


def _acoustic_key(record: ObservationRecord) -> Tuple[str, str]:
    return (record.asset_id, record.sensor_metadata.get("machine_id", ""))


def reading_to_boolean(record: ObservationRecord) -> bool:
    """True/positive per modality's R-relevant sense: acoustic ->
    `semantics.acoustic_indicator` using the record's own (asset,
    machine_id) G1 calibration (real waveform, never the hidden label at
    this call site); iot -> in-band (per `_iot_band_bounds`); thermal ->
    hotspot present (`fault_class != 'Noload'`, the only real-data signal
    this pool has)."""
    if record.modality == "acoustic":
        positive, _ = acoustic_indicator(_acoustic_features(record), key=_acoustic_key(record))
        return positive
    if record.modality == "iot_timeseries":
        field, lo, hi = _iot_band_bounds(record.asset_id)
        v = record.sensor_metadata.get("readings", {}).get(field)
        return v is not None and lo <= v <= hi
    if record.modality == "thermal":
        return record.sensor_metadata.get("fault_class") != "Noload"
    raise ValueError(f"reading_to_boolean: unsupported modality {record.modality!r}")


# ---------------------------------------------------------------------------
# Whitelisted rendering (plan section C/G4) -- the delivery/leakage firewall
# for B2. `env.py` MUST render every delivered observation through this
# function; nothing else may reach the agent's context.
# ---------------------------------------------------------------------------

WHITELISTED_FIELDS: FrozenSet[str] = frozenset({
    "modality", "indicator", "indicator_value", "in_band", "value_relative",
    "hotspot", "status",
})
FORBIDDEN_FIELDS: FrozenSet[str] = frozenset({
    "source_ref", "source_label", "gold_action", "machine_id", "fault_class",
    "observation_id", "episode_id", "sensor_metadata",
})


def render_reading(record: ObservationRecord, minted_id: str) -> Dict[str, Any]:
    """The ONLY function permitted to turn a real ObservationRecord into
    agent-visible text/JSON. Returns exactly a WHITELISTED_FIELDS-shaped
    payload plus the minted (opaque) id -- never the record's own id,
    source_ref, or sensor_metadata."""
    if record.modality == "acoustic":
        positive, value = acoustic_indicator(_acoustic_features(record), key=_acoustic_key(record))
        payload = {"modality": "acoustic", "indicator": "positive" if positive else "negative",
                   "indicator_value": round(float(value), 3)}
    elif record.modality == "iot_timeseries":
        field, lo, hi = _iot_band_bounds(record.asset_id)
        v = record.sensor_metadata.get("readings", {}).get(field)
        in_band = v is not None and lo <= v <= hi
        payload = {"modality": "iot", "in_band": bool(in_band),
                   "value_relative": round(float((v - lo) / max(hi - lo, 1e-9)), 3) if v is not None else None}
    elif record.modality == "thermal":
        payload = {"modality": "thermal", "hotspot": record.sensor_metadata.get("fault_class") != "Noload"}
    else:
        raise ValueError(f"render_reading: unsupported modality {record.modality!r}")
    payload["observation_ref"] = minted_id
    assert not (set(payload) & FORBIDDEN_FIELDS), f"leaked forbidden field in {payload}"
    return payload


def scan_for_leakage(payload: Any) -> List[str]:
    """G4: recursively scans a rendered payload (or prompt string) for any
    FORBIDDEN_FIELDS key, or the literal substring '/abnormal/' or
    '/normal/' (MIMII source_ref paths embed the label in the path itself,
    e.g. '.../abnormal/00000000.wav' -- plan section C's named leakage
    hazard). Returns the list of violations found (empty = clean)."""
    hits: List[str] = []

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k in FORBIDDEN_FIELDS:
                    hits.append(f"{path}.{k}")
                _walk(v, f"{path}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                _walk(v, f"{path}[{i}]")
        elif isinstance(node, str):
            if "/abnormal/" in node or "/normal/" in node:
                hits.append(f"{path}=<path leaks label>")

    _walk(payload, "$")
    return hits
