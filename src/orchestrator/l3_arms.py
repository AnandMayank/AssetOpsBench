"""l3_arms.py — Modality arms for the L3 evidence-dependency experiment.

The L1 arm failed because it was never proven able to emit the action it was
scored on. The corresponding hazard here is subtler and, on this corpus, real:
**the IoT value is written into the question prose.** R011's question states
"IoT sensors report pressure at 280 bar". An arm that merely stops offering an
IoT *tool* while leaving that sentence in place has withheld nothing — the agent
reads the number in its instructions. Three such arms would be three near-
identical prompts, and the resulting null would look like a clean result.

So an arm here is defined by what the rendered agent input actually contains,
and withholding means **redacting the value from every surface that carries it**:
question prose, tool availability, and evidence files. ``render_arm`` returns
the payload the agent would receive, and ``reconstructible`` checks whether a
withheld quantity survives anywhere in it.

Arms are declared per scenario according to that scenario's own evidence axis,
not applied uniformly. Three axes appear:

``physical_vs_digital``
    R009, R011, R015 — the gold decision depends on a physical gauge read and an
    IoT value. Supports FULL / PHYSICAL_ONLY / DIGITAL_ONLY.

``enterprise``
    R008, R010 — the gold decision depends on work-order state. Supports FULL /
    NO_ENTERPRISE.

``procedural``
    R006, R007 — the gold decision depends on *tool ordering* (``get_pose``
    before ``open_panel``), not on which evidence channel is available. **No
    modality ablation is semantically valid**, and inventing one would be the
    label-only ablation this module exists to prevent. These scenarios carry a
    FULL arm alone and are reported as not applicable for evidence dependency.

Arms that make the gold action unreachable by construction are marked
``insufficient_evidence_probe``. They are legitimate — FM-6a and FM-8 are
*about* acting without physical evidence — but they measure something different
from competence and are counted separately.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCEN_ROOT = (Path(__file__).resolve().parents[2].parent
             / "AssetOpsBenchScenarioGeneration" / "RobotInspection")

FULL = "FULL"
PHYSICAL_ONLY = "PHYSICAL_ONLY"
DIGITAL_ONLY = "DIGITAL_ONLY"
NO_ENTERPRISE = "NO_ENTERPRISE"

#: The fixed action interface, unchanged from the existing probe runners.
ALLOWED_ACTIONS = ("COMMIT", "ESCALATE", "ABORT")

#: Tool surfaces, so withholding a channel also withholds the means to fetch it.
PHYSICAL_TOOLS = ("navigate_to", "get_pose", "open_panel", "capture_image", "read_gauge")
DIGITAL_TOOLS = ("get_iot_reading", "get_sensor_history")
ENTERPRISE_TOOLS = ("get_work_orders", "get_similar_work_orders")


@dataclass(frozen=True)
class ArmSpec:
    scenario_id: str
    fm: str
    arm_id: str
    available_evidence: Tuple[str, ...]
    withheld_evidence: Tuple[str, ...]
    intended_evidence: Tuple[str, ...]
    necessary_evidence: Tuple[str, ...]
    expected_gold_action: str
    allowed_action_set: Tuple[str, ...] = ALLOWED_ACTIONS
    evidence_source_ids: Tuple[str, ...] = ()
    insufficient_evidence_probe: bool = False
    axis: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in asdict(self).items()}


#: Per-scenario evidence axis and gold, taken from manifest.json +
#: groundtruth.txt (which agree; the scenario CSV's fm_code disagrees for R006,
#: R008 and R011 and is not used).
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "R006": {"fm": "FM-5b", "gold": "COMMIT", "axis": "procedural",
             "asset": "motor_01", "iot_phrase": None},
    "R007": {"fm": "FM-5b", "gold": "COMMIT", "axis": "procedural",
             "asset": "hydraulic_pump_1", "iot_phrase": None},
    "R008": {"fm": "FM-5a", "gold": "ESCALATE", "axis": "enterprise",
             "asset": "metro_pump_1", "iot_phrase": None},
    "R009": {"fm": "FM-6a", "gold": "COMMIT", "axis": "physical_vs_digital",
             "asset": "chiller_6", "iot_phrase": r"avg\s*287\s*bar"},
    "R010": {"fm": "FM-6b", "gold": "ESCALATE", "axis": "enterprise",
             "asset": "motor_01", "iot_phrase": None},
    "R011": {"fm": "FM-7a", "gold": "ESCALATE", "axis": "physical_vs_digital",
             "asset": "hydraulic_pump_1", "iot_phrase": r"280\s*bar"},
    "R015": {"fm": "FM-8", "gold": "COMMIT", "axis": "physical_vs_digital",
             "asset": "chiller_6", "iot_phrase": None},
}

_REDACTED = "[WITHHELD]"


def scenario_dir(scenario_id: str) -> Optional[Path]:
    n = int(scenario_id.lstrip("Rr"))
    for cand in (SCEN_ROOT / f"scenario_R{n:02d}", SCEN_ROOT / f"scenario_R{n}"):
        if cand.is_dir():
            return cand
    return None


def question_text(scenario_id: str) -> str:
    d = scenario_dir(scenario_id)
    if d is None:
        raise FileNotFoundError(f"no scenario directory for {scenario_id}")
    return (d / "question.txt").read_text(errors="replace")


def arms_for(scenario_id: str) -> List[ArmSpec]:
    """Declare the arms this scenario's semantics genuinely support."""
    meta = SCENARIOS[scenario_id]
    fm, gold, axis = meta["fm"], meta["gold"], meta["axis"]
    src_iot = f"shared/iot/{meta['asset']}.json"
    src_wo = "shared/work_order/workorders.csv"
    src_img = f"capture_image({meta['asset']})"

    def spec(arm_id, avail, withheld, necessary, sources, probe=False) -> ArmSpec:
        return ArmSpec(scenario_id=scenario_id, fm=fm, arm_id=arm_id,
                       available_evidence=tuple(avail), withheld_evidence=tuple(withheld),
                       intended_evidence=("physical", "digital", "enterprise"),
                       necessary_evidence=tuple(necessary),
                       expected_gold_action=gold, evidence_source_ids=tuple(sources),
                       insufficient_evidence_probe=probe, axis=axis)

    if axis == "physical_vs_digital":
        return [
            spec(FULL, ("physical", "digital"), (), ("physical", "digital"),
                 (src_img, src_iot)),
            # Withholding IoT: the gold for FM-6a/FM-8 is a physical read, so
            # gold stays reachable; for FM-7a the contradiction cannot be seen,
            # so this is an insufficient-evidence probe.
            spec(PHYSICAL_ONLY, ("physical",), ("digital",), ("physical",), (src_img,),
                 probe=(fm == "FM-7a")),
            # Withholding the physical read is precisely the failure FM-6a and
            # FM-8 describe, so gold (COMMIT from an image) becomes unreachable
            # by construction. Legitimate, but a different measurement.
            spec(DIGITAL_ONLY, ("digital",), ("physical",), ("digital",), (src_iot,),
                 probe=True),
        ]
    if axis == "enterprise":
        return [
            spec(FULL, ("physical", "enterprise"), (), ("enterprise",),
                 (src_img, src_wo)),
            spec(NO_ENTERPRISE, ("physical",), ("enterprise",), ("enterprise",),
                 (src_img,), probe=True),
        ]
    # procedural: no evidence ablation is meaningful
    return [spec(FULL, ("physical",), (), ("tool_ordering",), (src_img,))]


def _redact_iot(text: str, pattern: Optional[str]) -> str:
    """Remove the IoT quantity from the question prose.

    Without this the arm is cosmetic: R011's question states the 280 bar reading
    outright, so an agent denied the IoT *tool* still has the number.
    """
    if not pattern:
        return text
    return re.sub(pattern, _REDACTED, text, flags=re.I)


def render_arm(spec: ArmSpec) -> Dict[str, Any]:
    """Build the payload the agent actually receives under this arm.

    Returns question text, offered tools and attached evidence source ids —
    everything that could carry the withheld quantity.
    """
    meta = SCENARIOS[spec.scenario_id]
    text = question_text(spec.scenario_id)
    tools = list(PHYSICAL_TOOLS + DIGITAL_TOOLS + ENTERPRISE_TOOLS)
    attached = list(spec.evidence_source_ids)

    if "digital" in spec.withheld_evidence:
        text = _redact_iot(text, meta.get("iot_phrase"))
        tools = [t for t in tools if t not in DIGITAL_TOOLS]
        attached = [a for a in attached if "/iot/" not in a]
    if "physical" in spec.withheld_evidence:
        tools = [t for t in tools if t not in PHYSICAL_TOOLS]
        attached = [a for a in attached if not a.startswith("capture_image")]
    if "enterprise" in spec.withheld_evidence:
        tools = [t for t in tools if t not in ENTERPRISE_TOOLS]
        attached = [a for a in attached if "work_order" not in a]

    return {
        "scenario_id": spec.scenario_id,
        "arm_id": spec.arm_id,
        "question": text,
        "tools": sorted(tools),
        "attached_evidence": sorted(attached),
        "allowed_actions": list(spec.allowed_action_set),
    }


def payload_signature(payload: Dict[str, Any]) -> str:
    """Canonical form of an agent-input payload, for arm-difference checks."""
    return json.dumps({k: payload[k] for k in
                       ("question", "tools", "attached_evidence")},
                      sort_keys=True)


def reconstructible(payload: Dict[str, Any], quantities: Sequence[str]) -> List[str]:
    """Withheld quantities that still appear somewhere in the rendered payload.

    Catches the case where a value is redacted from prose but survives in an
    attached evidence file, a tool name, or a restatement.
    """
    blob = payload_signature(payload).lower()
    return [q for q in quantities if q and q.lower() in blob]


def withheld_quantities(scenario_id: str, arm: ArmSpec) -> List[str]:
    """Literal strings that must not survive redaction under this arm."""
    meta = SCENARIOS[scenario_id]
    out: List[str] = []
    if "digital" in arm.withheld_evidence and meta.get("iot_phrase"):
        # The bare number, as it appears in prose.
        m = re.search(r"(\d+(?:\.\d+)?)", meta["iot_phrase"])
        if m:
            out.append(m.group(1))
    return out


def all_specs() -> List[ArmSpec]:
    return [s for sid in SCENARIOS for s in arms_for(sid)]


def manifest() -> Dict[str, Any]:
    """Machine-readable arm manifest for the whole L3 set."""
    return {
        "schema": "assetops.l3_arms/1",
        "allowed_action_set": list(ALLOWED_ACTIONS),
        "fm_code_source": "manifest.json + groundtruth.txt (scenario CSV disagrees "
                          "for R006, R008, R011 and is not used)",
        "arms": [s.to_dict() for s in all_specs()],
    }
