"""tool_executor.py — Backend-agnostic executable tool interface (P0-1, P0-6, P0-7).

Defines the contract every execution backend implements, so MuJoCo (Stage 2) is
a backend swap rather than a rewrite. Two properties are structural rather than
conventional:

**The model never authors a tool result.** ``execute()`` returns a ``ToolResult``
built by the backend from hidden state. The model can request a call; it cannot
manufacture the payload.

**Modality masking lives here, not in the prompt.** ``available_tools(arm)``
removes the withheld channel's tools from the surface entirely, so a
PHYSICAL_ONLY arm has no IoT tool to call. Prompt redaction remains as defence
in depth, but it is no longer the mechanism.

Also provides ``render_gauge``, a deterministic dial renderer. ``capture_image``
in the MCP server returns a ``gauge_path``, and every pilot profile has
``gauge_path=None`` — so nothing could ever be delivered. P0-7 requires an image
the model actually consumes, so the backend renders one at the scenario's hidden
value and hands back base64 pixels. The needle position *is* the ground truth,
which is what makes reading it a real perceptual act rather than a lookup.
"""

from __future__ import annotations

import base64
import hashlib
import io
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

#: Which evidence channel each tool belongs to. Masking works on these.
TOOL_MODALITY: Dict[str, str] = {
    "capture_image": "physical",
    "read_gauge": "physical",
    "open_panel": "physical",
    "read_iot": "digital",
    "get_iot_reading": "digital",
    "get_sensor_history": "digital",
    "get_work_order": "enterprise",
    "check_wo_similarity": "enterprise",
    "get_asset_state": "enterprise",
    "navigate_to": "robot",
    "get_pose": "robot",
    "get_battery": "robot",
    "list_waypoints": "robot",
    "safety_gate_check": "robot",
    "sit": "robot",
    "stand": "robot",
    "dock": "robot",
    "power_on": "robot",
    "commit_reading": "enterprise",
    # FM-26 thermal (Pass 2) -- a distinct modality from "physical" because
    # thermal evidence is acquired and interpreted by a fundamentally
    # different sensing/perception path (real image decode + VLM) than the
    # gauge/panel physical channel, and A1/TEXT_CONTROL needs to withhold it
    # independently of PHYSICAL_ONLY. commit_thermal_decision mirrors
    # commit_reading's "enterprise" classification -- both are the terminal
    # write that records an operational decision, not evidence acquisition.
    "read_thermal_image": "thermal",
    "commit_thermal_decision": "enterprise",
    # THERMAL BRANCH Pass 3 action-space additions.
    "select_capability": "robot",     # deterministic trigger match, no evidence acquired
    "get_sensor_history": "digital",  # same channel as read_iot -- historical/context framing
    "read_vibration": "vibration",    # distinct modality: honestly UNAVAILABLE until Phase 12
    "read_acoustic": "acoustic",      # PASS 5/6: acoustic payload un-gated, MIMII-backed
    "commit_acoustic_decision": "enterprise",  # PASS 6: mirrors commit_thermal_decision's classification
    "escalate": "enterprise",         # generic procedural escalation, not a thermal decision
    # Family B (Evidence Acquisition), Phase 8H.2G: the sole acquisition tool.
    # Classified "robot" (a dispatch/meta-tool, like select_capability) rather
    # than a fixed evidence channel, because the modality it resolves is a
    # per-call argument, not a property of the tool itself -- a single B
    # episode's tool masking is expressed through which MODALITIES exist in
    # the observation store for that asset, not through withholding this tool.
    "request_observation": "robot",
}

STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_UNAVAILABLE = "unavailable"   # tool masked out of this arm


@dataclass(frozen=True)
class ToolCall:
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """What the executor did. Never authored by the model."""

    tool: str
    requested: bool = True
    executed: bool = False
    status: str = STATUS_FAILED
    error: Optional[str] = None
    observation_id: Optional[str] = None
    observation_hash: Optional[str] = None
    modality: Optional[str] = None
    #: PASS R2: the asset the delivered observation actually resolved
    #: against, when the backend genuinely knows it (thermal today). None
    #: for every other branch -- not populating this is honest "unknown",
    #: never a fabricated match.
    asset_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    #: base64 PNG, when this tool delivers an image the model can actually see.
    image_b64: Optional[str] = None

    @property
    def delivered(self) -> bool:
        """An observation reached the model only if the call ran, succeeded and
        produced content. Executed-but-empty is not delivery."""
        return (self.executed and self.status == STATUS_SUCCESS
                and self.observation_id is not None)

    def to_dict(self, include_image: bool = False) -> Dict[str, Any]:
        d = asdict(self)
        if not include_image:
            d.pop("image_b64", None)
        return d


class ToolExecutor(Protocol):
    """Implemented by CouchDBExecutor (Stage 1) and MuJoCoExecutor (Stage 2)."""

    backend_id: str
    backend_version: str

    def reset(self, scenario_id: str, arm_id: str, seed: int = 0) -> None: ...
    def available_tools(self) -> List[str]: ...
    def execute(self, call: ToolCall) -> ToolResult: ...
    def state_digest(self) -> str: ...


def mask_tools(all_tools: Sequence[str], withheld: Sequence[str]) -> List[str]:
    """Remove every tool belonging to a withheld modality (P0-6).

    Masking at the tool layer is what makes an arm real: a PHYSICAL_ONLY arm
    cannot call an IoT tool because the tool is not on the surface, so no
    redaction can be talked around.
    """
    blocked = set(withheld)
    return sorted(t for t in all_tools if TOOL_MODALITY.get(t, "robot") not in blocked)


# --------------------------------------------------------------------------
# Deterministic gauge rendering (P0-7)
# --------------------------------------------------------------------------

def render_gauge(value: float, vmin: float, vmax: float, unit: str = "",
                 label: str = "", size: int = 420,
                 occluded: bool = False) -> bytes:
    """Render an analog dial with the needle at ``value``. Deterministic.

    Returns PNG bytes. The needle angle encodes the hidden value, so a model
    must actually look at the image to read it — the point of P0-7. Rendering is
    intentionally plain; Stage 2 replaces this with a MuJoCo camera behind the
    same interface.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), (245, 245, 245))
    d = ImageDraw.Draw(img)
    cx = cy = size // 2
    r = int(size * 0.40)

    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255),
              outline=(30, 30, 30), width=4)

    # 270-degree sweep, 135 deg (lower-left) to 405 deg, clockwise.
    start, sweep = 135.0, 270.0
    span = (vmax - vmin) or 1.0

    for i in range(11):
        frac = i / 10.0
        ang = math.radians(start + sweep * frac)
        x0 = cx + int((r - 14) * math.cos(ang)); y0 = cy + int((r - 14) * math.sin(ang))
        x1 = cx + int(r * math.cos(ang));       y1 = cy + int(r * math.sin(ang))
        d.line([x0, y0, x1, y1], fill=(30, 30, 30), width=3 if i % 2 == 0 else 1)
        # Label every other major tick. These decisions turn on whether the
        # needle sits inside an operating band, so a dial with only three
        # labels would force the reading to rest on interpolation alone.
        if i % 2 == 0:
            tx = cx + int((r - 34) * math.cos(ang)); ty = cy + int((r - 34) * math.sin(ang))
            d.text((tx - 11, ty - 6), f"{vmin + span * frac:g}", fill=(20, 20, 20))

    clamped = max(vmin, min(vmax, value))
    ang = math.radians(start + sweep * ((clamped - vmin) / span))
    d.line([cx, cy, cx + int((r - 24) * math.cos(ang)), cy + int((r - 24) * math.sin(ang))],
           fill=(190, 20, 20), width=5)
    d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=(30, 30, 30))

    if unit:
        d.text((cx - 14, cy + int(r * 0.45)), unit, fill=(20, 20, 20))
    if label:
        d.text((10, 10), label[:52], fill=(60, 60, 60))

    if occluded:
        # Panel obstruction: the dial is physically blocked, not merely dim.
        d.rectangle([cx - r, cy - 18, cx + r, cy + int(r * 0.7)], fill=(70, 70, 70))
        d.text((cx - 54, cy + 6), "PANEL OBSTRUCTION", fill=(240, 240, 240))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def png_to_b64(png: bytes) -> str:
    return base64.b64encode(png).decode()


def observation_id(prefix: str, payload: Any) -> str:
    blob = repr(payload).encode()
    return f"{prefix}_{hashlib.sha256(blob).hexdigest()[:12]}"
