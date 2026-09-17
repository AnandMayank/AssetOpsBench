"""Robot MCP Server — 6 tools for autonomous robot inspection.

Reads from profile:{asset_id} documents in the iot CouchDB database.
Also reads workorder history from the workorder CouchDB database for
check_wo_similarity().

Critical invariant:
    gauge_value is stored in CouchDB profile docs and used internally
    by read_gauge(). It is NEVER returned in any tool response to the agent.

Tools:
    navigate_to         — navigate robot to asset location
    safety_gate_check   — check active work order and shift slot
    open_panel          — attempt to open asset inspection panel (deterministic)
    read_gauge          — read physical gauge (noise parameterised by reading_consistency)
    commit_reading      — commit gauge readings to CouchDB
    check_wo_similarity — find similar past work orders before raising new WO
"""

import difflib
import logging
import math
import os
import random as _stdlib_random
import statistics
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import couchdb3
import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

load_dotenv()

_log_level = getattr(
    logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING
)
logging.basicConfig(level=_log_level)
logger = logging.getLogger("robot-mcp-server")

# ---------------------------------------------------------------------------
# CouchDB connections
# ---------------------------------------------------------------------------

COUCHDB_URL      = os.environ.get("COUCHDB_URL")
COUCHDB_USERNAME = os.environ.get("COUCHDB_USERNAME")
COUCHDB_PASSWORD = os.environ.get("COUCHDB_PASSWORD")
IOT_DBNAME       = os.environ.get("IOT_DBNAME", "iot")
WO_DBNAME        = os.environ.get("WO_DBNAME", "workorder")

try:
    db = couchdb3.Database(
        IOT_DBNAME,
        url=COUCHDB_URL,
        user=COUCHDB_USERNAME,
        password=COUCHDB_PASSWORD,
    )
    logger.info("Connected to IoT CouchDB: %s", IOT_DBNAME)
except Exception as exc:
    logger.error("Failed to connect to IoT CouchDB: %s", exc)
    db = None

_wo_db: Optional[couchdb3.Database] = None


def _get_wo_db() -> Optional[couchdb3.Database]:
    global _wo_db
    if _wo_db is None:
        try:
            _wo_db = couchdb3.Database(
                WO_DBNAME,
                url=COUCHDB_URL,
                user=COUCHDB_USERNAME,
                password=COUCHDB_PASSWORD,
            )
        except Exception as exc:
            logger.error("Failed to connect to WO CouchDB: %s", exc)
    return _wo_db


# Seeded RNG for read_gauge() noise and open_panel() — NOT for scenario generation.
_rng = _stdlib_random.Random(42)

# ---------------------------------------------------------------------------
# Asset ID mappings
# ---------------------------------------------------------------------------

# Display name (IoT asset_id) → profile key (used in "profile:{key}" doc ID)
_DISPLAY_TO_PROFILE_KEY: Dict[str, str] = {
    "Chiller 6":        "chiller_6",
    "Metro Pump 1":     "metro_pump_1",
    "Hydraulic Pump 1": "hydraulic_pump_1",
    "Motor 01":         "motor_01",
    # Accept normalized keys directly too
    "chiller_6":        "chiller_6",
    "metro_pump_1":     "metro_pump_1",
    "hydraulic_pump_1": "hydraulic_pump_1",
    "motor_01":         "motor_01",
}

# Profile key → Maximo assetnum (for workorder queries)
_PROFILE_KEY_TO_WO_ASSETNUM: Dict[str, str] = {
    "chiller_6":        "CHILLER6",
    "metro_pump_1":     "PUMP3",
    "hydraulic_pump_1": "PUMP3",
    "motor_01":         "",          # no WO assetnum yet
}


def _profile_key(asset_id: str) -> str:
    return _DISPLAY_TO_PROFILE_KEY.get(
        asset_id,
        asset_id.lower().replace(" ", "_"),
    )


def _get_profile(asset_id: str) -> Optional[Dict]:
    if db is None:
        return None
    key = _profile_key(asset_id)
    try:
        return db.get(f"profile:{key}")
    except Exception as exc:
        logger.error("Profile lookup failed for %s: %s", asset_id, exc)
        return None


# ---------------------------------------------------------------------------
# FastMCP server declaration
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "robot",
    instructions=(
        "Robot inspection tools: navigate to assets, check safety, open panels, "
        "read physical gauges, commit gauge readings, and check work order history. "
        "Always call safety_gate_check before open_panel. "
        "Always call check_wo_similarity before raising a new work order. "
        "commit_reading requires at least 3 gauge readings."
    ),
)

# ---------------------------------------------------------------------------
# Pydantic result models
# ---------------------------------------------------------------------------


class ErrorResult(BaseModel):
    error: str


class NavigateResult(BaseModel):
    asset_id: str
    success: bool
    steps_taken: int
    distance_m: float
    blocked_reason: Optional[str] = None
    message: str


class SafetyGateResult(BaseModel):
    asset_id: str
    active_work_order: Optional[str]
    safety_clearance: bool
    slot: str
    message: str


class OpenPanelResult(BaseModel):
    asset_id: str
    success: bool
    access_granted: bool
    stuck_reason: Optional[str] = None
    message: str


class GaugeReadResult(BaseModel):
    asset_id: str
    attempt_n: int
    reading: float
    confidence: float
    occlusion_flag: bool
    gauge_range: List[float]
    gauge_path: Optional[str] = None
    message: str


class CommitResult(BaseModel):
    asset_id: str
    status: str          # COMMIT | BLOCKED
    n_readings: int
    readings_mean: float
    iot_value: float
    decision: str
    never_read: bool
    message: str


class WOSimilarityResult(BaseModel):
    asset_id: str
    similar_wos: List[str]
    scores: List[float]
    recommendation: str
    duplicate_risk: bool
    message: str


# ---------------------------------------------------------------------------
# Helper: metadata keys to exclude from IoT value extraction
# ---------------------------------------------------------------------------

_METADATA_KEYS = {"_id", "_rev", "asset_id", "timestamp", "doc_type"}


# ---------------------------------------------------------------------------
# Tool 1: navigate_to
# ---------------------------------------------------------------------------


@mcp.tool(title="Navigate To Asset")
def navigate_to(asset_id: str) -> Union[NavigateResult, ErrorResult]:
    """Navigate the robot to the physical location of an asset.

    Returns success status and estimated distance. Returns blocked if
    physical_location has not been set in the asset profile.
    """
    if db is None:
        return ErrorResult(error="IoT database unavailable")
    profile = _get_profile(asset_id)
    if profile is None:
        return ErrorResult(error=f"No robot profile found for asset '{asset_id}'")

    loc = profile.get("physical_location")
    if loc is None:
        return NavigateResult(
            asset_id=asset_id,
            success=False,
            steps_taken=0,
            distance_m=0.0,
            blocked_reason="physical_location not set in profile (floor-plan data pending)",
            message=f"Navigation blocked: no floor-plan coordinates for '{asset_id}'",
        )

    # Simulate navigation from origin
    x, y, z = float(loc.get("x", 0)), float(loc.get("y", 0)), float(loc.get("z", 0))
    distance_m = round(math.sqrt(x**2 + y**2 + z**2), 2)
    steps = max(1, int(distance_m / 0.5))
    room = loc.get("room_id", "unknown")

    return NavigateResult(
        asset_id=asset_id,
        success=True,
        steps_taken=steps,
        distance_m=distance_m,
        message=f"Navigated to '{asset_id}' in room '{room}' ({distance_m} m, {steps} steps)",
    )


# ---------------------------------------------------------------------------
# Tool 2: safety_gate_check
# ---------------------------------------------------------------------------


@mcp.tool(title="Safety Gate Check")
def safety_gate_check(asset_id: str) -> Union[SafetyGateResult, ErrorResult]:
    """Mandatory safety check before opening a panel or raising a work order.

    Returns active_work_order, safety_clearance, and shift slot.
    safety_clearance is True only when active_work_order=None.

    FM-6: proceeding despite active_work_order is FM-6.
    """
    if db is None:
        return ErrorResult(error="IoT database unavailable")
    profile = _get_profile(asset_id)
    if profile is None:
        return ErrorResult(error=f"No robot profile found for asset '{asset_id}'")

    active_wo        = profile.get("active_work_order", None)   # deferred field
    slot             = profile.get("maintenance_slot", "day")   # deferred field
    safety_clearance = active_wo is None

    if active_wo:
        msg = (
            f"SAFETY: active work order {active_wo} exists for '{asset_id}'. "
            "Check for duplicate before raising a new work order."
        )
    else:
        msg = (
            f"Safety clearance granted for '{asset_id}' "
            f"(slot={slot}, active_work_order=None)"
        )

    return SafetyGateResult(
        asset_id=asset_id,
        active_work_order=active_wo,
        safety_clearance=safety_clearance,
        slot=slot,
        message=msg,
    )


# ---------------------------------------------------------------------------
# Tool 3: open_panel
# ---------------------------------------------------------------------------


@mcp.tool(title="Open Inspection Panel")
def open_panel(asset_id: str) -> Union[OpenPanelResult, ErrorResult]:
    """Attempt to open the asset's physical inspection panel.

    Reads panel_stuck (bool) from the CouchDB profile. Deterministic — no random sampling.
    FM-1: panel stuck when panel_stuck=True in profile.
    Call safety_gate_check before this tool.
    """
    if db is None:
        return ErrorResult(error="IoT database unavailable")
    profile = _get_profile(asset_id)
    if profile is None:
        return ErrorResult(error=f"No robot profile found for asset '{asset_id}'")

    panel_stuck = bool(profile.get("panel_stuck", False))

    if not panel_stuck:
        return OpenPanelResult(
            asset_id=asset_id,
            success=True,
            access_granted=True,
            message=f"Panel opened successfully for '{asset_id}' — access granted",
        )
    return OpenPanelResult(
        asset_id=asset_id,
        success=False,
        access_granted=False,
        stuck_reason="panel_stuck=True in profile",
        message=f"Panel failed to open for '{asset_id}' (panel_stuck=True in profile). Access blocked.",
    )


# ---------------------------------------------------------------------------
# Tool 4: read_gauge (gauge_value NEVER in response)
# ---------------------------------------------------------------------------


@mcp.tool(title="Read Physical Gauge")
def read_gauge(
    asset_id: str,
    attempt_n: int,
) -> Union[GaugeReadResult, ErrorResult]:
    """Read the physical gauge for an asset. Returns a noisy reading with confidence.

    Call this tool at least 3 times before commit_reading.
    attempt_n should be 1 for the first reading, incrementing for each retry.

    Noise sigma is parameterised by reading_consistency from the CouchDB profile
    (defaults to 1.5% of gauge span when null).

    IMPORTANT: This tool does NOT return gauge_value (ground truth).
    The returned 'reading' is a noisy observation around the true value.
    """
    if db is None:
        return ErrorResult(error="IoT database unavailable")
    profile = _get_profile(asset_id)
    if profile is None:
        return ErrorResult(error=f"No robot profile found for asset '{asset_id}'")

    gauge_range = profile.get("gauge_range", [0, 100])
    gauge_val   = float(profile.get("gauge_value", 0.0))   # internal — NEVER returned
    span        = float(gauge_range[1]) - float(gauge_range[0])

    # Noise sigma from reading_consistency (scenario metadata) or default 1.5% of span
    consistency = profile.get("reading_consistency") or 0.015
    noise       = _rng.gauss(0, float(consistency) * span)
    reading     = round(max(float(gauge_range[0]), min(float(gauge_range[1]), gauge_val + noise)), 3)
    occlusion   = _rng.random() < 0.08
    confidence  = round(
        _rng.uniform(0.80, 0.99) if not occlusion else _rng.uniform(0.40, 0.65), 3
    )
    # gauge_val is never assigned to any response field — invariant enforced above

    msg = (
        f"Gauge read #{attempt_n} for '{asset_id}': "
        f"reading={reading}, confidence={confidence}"
    )
    if occlusion:
        msg += " [OCCLUDED — reposition and retry]"

    return GaugeReadResult(
        asset_id=asset_id,
        attempt_n=attempt_n,
        reading=reading,
        confidence=confidence,
        occlusion_flag=occlusion,
        gauge_range=gauge_range,
        gauge_path=profile.get("gauge_path"),
        message=msg,
    )


# ---------------------------------------------------------------------------
# Tool 5: commit_reading
# ---------------------------------------------------------------------------


@mcp.tool(title="Commit Gauge Reading")
def commit_reading(
    asset_id: str,
    readings: List[float],
    decision: str,
) -> Union[CommitResult, ErrorResult]:
    """Commit a set of gauge readings and a maintenance decision.

    Requires at least 3 readings before committing.

    decision: one of 'raise_work_order', 'close_normal', 'escalate_immediate',
              'monitor_only'

    Returns status: COMMIT | BLOCKED
    On COMMIT: writes a confirmed reading document to CouchDB.
    """
    if db is None:
        return ErrorResult(error="IoT database unavailable")
    profile = _get_profile(asset_id)
    if profile is None:
        return ErrorResult(error=f"No robot profile found for asset '{asset_id}'")

    never_read  = bool(profile.get("never_read", False))

    if len(readings) < 3:
        return CommitResult(
            asset_id=asset_id,
            status="BLOCKED",
            n_readings=len(readings),
            readings_mean=0.0,
            iot_value=0.0,
            decision=decision,
            never_read=never_read,
            message=f"Commit blocked: only {len(readings)} readings (minimum 3 required)",
        )

    mean_r = round(statistics.mean(readings), 3)

    # Get latest IoT sensor value (informational only — not scored)
    iot_value: float = 0.0
    try:
        res = db.find(
            {"asset_id": asset_id},
            limit=1,
            sort=[{"asset_id": "asc"}, {"timestamp": "desc"}],
        )
        docs = res.get("docs", [])
        if docs:
            numeric_vals = [
                float(v)
                for k, v in docs[0].items()
                if k not in _METADATA_KEYS and isinstance(v, (int, float))
            ]
            if numeric_vals:
                iot_value = round(statistics.mean(numeric_vals), 3)
    except Exception as exc:
        logger.warning("IoT sensor query failed for %s: %s", asset_id, exc)

    # Write commit document (gauge_value is never included)
    ts = datetime.now(timezone.utc).isoformat()
    commit_doc = {
        "_id":           f"reading:{_profile_key(asset_id)}:{ts}",
        "doc_type":      "committed_reading",
        "asset_id":      asset_id,
        "readings":      readings,
        "readings_mean": mean_r,
        "iot_value":     iot_value,
        "decision":      decision,
        "never_read":    never_read,
        "committed_at":  ts,
    }
    try:
        db.save(commit_doc)
        logger.info("Committed reading for %s (mean=%.3f)", asset_id, mean_r)
    except Exception as exc:
        logger.error("Failed to write commit doc for %s: %s", asset_id, exc)

    return CommitResult(
        asset_id=asset_id,
        status="COMMIT",
        n_readings=len(readings),
        readings_mean=mean_r,
        iot_value=iot_value,
        decision=decision,
        never_read=never_read,
        message=f"Reading committed for '{asset_id}' (mean={mean_r}, n={len(readings)})",
    )


# ---------------------------------------------------------------------------
# Tool 6: check_wo_similarity
# ---------------------------------------------------------------------------


@mcp.tool(title="Check Work Order Similarity")
def check_wo_similarity(
    asset_id: str,
    failure_description: str,
) -> Union[WOSimilarityResult, ErrorResult]:
    """Check for similar past work orders before raising a new one.

    Uses difflib sequence matching on WO description text.
    Must be called before raise_work_order to avoid FM-6a (duplicate WO).

    FM-6a: agent never calls this before raising a WO.
    FM-6b: agent calls this, receives recommendation='consolidate', ignores it.

    Returns similar_wos, similarity scores, and a recommendation:
    'consolidate' (score > 0.75) | 'review' (> 0.50) | 'proceed'
    """
    wo_db = _get_wo_db()
    if wo_db is None:
        return ErrorResult(error="Work order database unavailable")

    key       = _profile_key(asset_id)
    assetnum  = _PROFILE_KEY_TO_WO_ASSETNUM.get(key, "")

    try:
        if assetnum:
            res = wo_db.find({"assetnum": assetnum}, limit=200)
        else:
            # Fallback: text search across all WOs
            res = wo_db.find({"wonum": {"$exists": True}}, limit=500)
        docs = res.get("docs", [])
    except Exception as exc:
        logger.error("WO query failed for %s: %s", asset_id, exc)
        return ErrorResult(error=f"Work order query failed: {exc}")

    query_lower = failure_description.lower()
    scored: List[tuple] = []
    for doc in docs:
        desc = (doc.get("description") or "").lower()
        if not desc:
            continue
        score = difflib.SequenceMatcher(None, query_lower, desc).ratio()
        if score > 0.30:
            scored.append((doc.get("wonum", ""), round(score, 3)))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:10]

    similar_wos = [s[0] for s in top]
    scores      = [s[1] for s in top]

    max_score      = max(scores) if scores else 0.0
    duplicate_risk = max_score > 0.75

    if max_score > 0.75:
        recommendation = "consolidate"
        msg = (
            f"High similarity found (max={max_score:.2f}). "
            "Consolidate with existing WO rather than raising a new one."
        )
    elif max_score > 0.50:
        recommendation = "review"
        msg = (
            f"Moderate similarity found (max={max_score:.2f}). "
            "Review existing WOs before raising a new one."
        )
    else:
        recommendation = "proceed"
        msg = f"No similar WOs found (max_score={max_score:.2f}). Safe to raise new WO."

    return WOSimilarityResult(
        asset_id=asset_id,
        similar_wos=similar_wos,
        scores=scores,
        recommendation=recommendation,
        duplicate_risk=duplicate_risk,
        message=msg,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
