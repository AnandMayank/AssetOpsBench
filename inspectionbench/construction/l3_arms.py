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

``perceptual`` (Pass 2, FM-26 thermal)
    R049, R050, R088, R089 — the gold decision depends on the physical
    temperature distribution shown in a thermal image, not on a numeric
    IoT value. Supports three arms, run as a text-vs-image comparison
    rather than a physical/digital ablation:

    - ``TEXT_CONTROL`` (A1): the thermal *tool* is withheld, and the
      question prose is deliberately re-injected with the pre-digested
      finding Pass 1 stripped out (the opposite of every other arm's
      redaction). Gold is reachable here WITHOUT ever looking at pixels,
      by construction -- this is a leak control proving Pass 1's shortcut
      still exists if offered, never a competence arm. Always
      ``insufficient_evidence_probe=False``... except it is exactly the
      inverse case: gold *is* reachable, but not through genuine
      perception, which is why it is still reported separately from
      A2/A3 rather than blended with them.
    - ``IMAGE_GROUNDED`` (A2): the real thermal tool is available, no
      pre-digested finding anywhere; the agent must call
      ``read_thermal_image`` and reason over its structured observation.
    - ``IMAGE_REFERENCE`` (A3): as A2, plus a known-normal reference frame
      (R050 / 002.bmp) the agent may request alongside the query image.
      Explicitly underpowered at n<=4 -- reported as an anecdote, never a
      comparison arm (see scripts/run_thermal_pilot.py).

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
#: Pass 2 thermal (FM-26) arms -- a text-vs-image comparison, not a
#: physical/digital ablation. See the "perceptual" axis docstring above.
TEXT_CONTROL = "TEXT_CONTROL"
IMAGE_GROUNDED = "IMAGE_GROUNDED"  # = T1 in Part 16A's perception ablation
IMAGE_REFERENCE = "IMAGE_REFERENCE"
#: Part 16A thermal perception ablation -- same episodes, same downstream
#: agent/tools/scorer as IMAGE_GROUNDED; only the PERCEPTION PATHWAY differs.
#: T0_DETERMINISTIC: thermal_deterministic.py, no model call (T0a/T0b decided
#: automatically per-file by radiometric decodability, never chosen here).
#: T2_DIRECT_IMAGE: real pixels delivered to the DECISION agent directly
#: (perceptual fields suppressed at the executor); tests the structured-
#: observation interface itself, not perception quality.
#: T3_ALT_MODEL: identical VLM path as IMAGE_GROUNDED, different model_name.
T0_DETERMINISTIC = "T0_DETERMINISTIC"
T2_DIRECT_IMAGE = "T2_DIRECT_IMAGE"
T3_ALT_MODEL = "T3_ALT_MODEL"

#: PASS R3: FM-29 acoustic full-loop (R090/R091). One arm only -- this is
#: the mandatory-loop proof (plan A5), not a perception-pathway ablation
#: like thermal's Part 16A, so none of T0/T2/T3's variants apply here.
AUDIO_GROUNDED = "AUDIO_GROUNDED"

#: The fixed action interface, unchanged from the existing probe runners.
ALLOWED_ACTIONS = ("COMMIT", "ESCALATE", "ABORT")

#: FM-26's action space is a 4-way severity ladder, not the 3-way
#: COMMIT/ESCALATE/ABORT interface -- passed explicitly per-arm via
#: ArmSpec.allowed_action_set (already a per-arm override point), never by
#: changing ALLOWED_ACTIONS itself.
THERMAL_ACTIONS = ("COMMIT", "SCHEDULE_MAINTENANCE", "SHUTDOWN", "ESCALATE")

#: Tool surfaces, so withholding a channel also withholds the means to fetch it.
PHYSICAL_TOOLS = ("navigate_to", "get_pose", "open_panel", "capture_image", "read_gauge")
DIGITAL_TOOLS = ("get_iot_reading", "get_sensor_history")
ENTERPRISE_TOOLS = ("get_work_orders", "get_similar_work_orders")
THERMAL_TOOLS = ("read_thermal_image", "commit_thermal_decision")


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
    # --- L3 evidence-dependency pilot, authored 2026-08-12 (class A) ---------
    # Template: gold is determined by the physical read; the digital channel is
    # present as a shortcut. Withholding the shortcut must leave gold reachable.
    "R055": {"fm": "FM-6a", "gold": "COMMIT", "axis": "physical_vs_digital",
             "asset": "chiller_6", "iot_phrase": r"245\s*bar"},
    "R056": {"fm": "FM-6a", "gold": "ESCALATE", "axis": "physical_vs_digital",
             "asset": "metro_pump_1", "iot_phrase": r"1\.02\s*m³/s"},
    "R057": {"fm": "FM-8", "gold": "COMMIT", "axis": "physical_vs_digital",
             "asset": "motor_01", "iot_phrase": r"92\s*°C"},
    # R058's shortcut is a narrative clause, not a number: the history framing
    # that presents an exceedance as routine. Only that clause is redacted — the
    # 200 bar *limit* is legitimate context the gold decision needs.
    "R058": {"fm": "FM-7c", "gold": "ESCALATE", "axis": "physical_vs_digital",
             "asset": "hydraulic_pump_1",
             "iot_phrase": r"The maintenance history[^.]*\."},
    # --- FM-26 thermal, Pass 2 (2026-08-22) ----------------------------------
    # "gold" here is what score_l3_grounded actually scores against (a tuple
    # is set-valued gold, see l3_grounded_scoring._gold_action_set) -- taken
    # from each scenario's own manifest.json visual.fault_class /
    # groundtruth.txt, unchanged. reference_scenario_id names the
    # known-normal frame IMAGE_REFERENCE (A3) attaches; None for R050 itself
    # (it IS the reference frame -- not run against its own arm A3).
    "R049": {"fm": "FM-26", "gold": ("SHUTDOWN", "ESCALATE"), "axis": "perceptual",
             "asset": "motor_01", "reference_scenario_id": "R050"},
    "R050": {"fm": "FM-26", "gold": "COMMIT", "axis": "perceptual",
             "asset": "motor_01", "reference_scenario_id": None},
    "R088": {"fm": "FM-26", "gold": "COMMIT", "axis": "perceptual",
             "asset": "motor_01", "reference_scenario_id": "R050"},
    "R089": {"fm": "FM-26", "gold": "ESCALATE", "axis": "perceptual",
             "asset": "motor_01", "reference_scenario_id": "R050"},
    # --- FM-29 acoustic, Pass R3 (mandatory full-loop pair) -------------
    # gold here is a per-SCENARIO SME-adjudicated mapping of this one real
    # MIMII clip's source_label, not a bulk relabeling of MIMII's abnormal/
    # normal split (see build_observation_records._build_acoustic_scenario_
    # records's WORLD->GOLD firewall note). evidence_provenance_class is L2
    # (ASSET_CLASS_EVIDENCE_REPLAY): real external pump acoustic evidence
    # replayed onto hydraulic_pump_1, never claimed as evidence captured
    # from that AssetOpsBench asset.
    "R090": {"fm": "FM-29", "gold": "ESCALATE", "axis": "acoustic_perceptual",
             "asset": "hydraulic_pump_1"},
    "R091": {"fm": "FM-29", "gold": "COMMIT", "axis": "acoustic_perceptual",
             "asset": "hydraulic_pump_1"},
    # --- F4 evidence-sufficiency execution hardening, Pass R5.6 ----------
    # Secondary/controlled-construction family only -- never pooled with
    # real-evidence CC/PROC/ORDERING/CC_grounded results in any headline
    # table. Reuses existing, UNMODIFIED axis machinery: R092 is FM-26
    # ("perceptual" axis, thermal-required grounding, identical semantics
    # to R049/R050/R088/R089); R093 is FM-8 ("physical_vs_digital" axis,
    # physical/gauge-required grounding, identical semantics to R057). No
    # new FM scoring rule, no new REQUIRED_MODALITY/PROC_TOOLS entry.
    "R092": {"fm": "FM-26", "gold": ("SHUTDOWN", "ESCALATE"), "axis": "perceptual",
             "asset": "motor_01", "reference_scenario_id": None},
    "R093": {"fm": "FM-8", "gold": "COMMIT", "axis": "physical_vs_digital",
             "asset": "hydraulic_pump_1", "iot_phrase": None},
}

#: F4 evidence-sufficiency pair (Pass R5.6). R092 = insufficient first
#: modality (gauge ambiguous, thermal REQUIRED); R093 = sufficient first
#: modality (gauge clear, second modality available but NOT required).
F4_SCENARIOS = ("R092", "R093")

#: FM-26 thermal scored set (Pass 2). Kept separate from PILOT_SCENARIOS
#: (a different family/axis) rather than merged into it.
THERMAL_SCENARIOS = ("R049", "R050", "R088", "R089")

#: FM-29 acoustic scored set (Pass R3) -- the mandatory full-loop pair.
ACOUSTIC_SCENARIOS = ("R090", "R091")

#: The six-scenario evidence-dependency pilot. Apparatus validation, not a
#: powered benchmark: 6 scenarios x 2 competence arms cannot resolve a 10 pp
#: effect (see evaluation.paired_stats.min_pairs_for_effect).
PILOT_SCENARIOS = ("R009", "R015", "R055", "R056", "R057", "R058")

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


def _manifest(scenario_id: str) -> Dict[str, Any]:
    d = scenario_dir(scenario_id)
    if d is None:
        raise FileNotFoundError(f"no scenario directory for {scenario_id}")
    return json.loads((d / "manifest.json").read_text())


def required_minimal_procedure(scenario_id: str) -> List[str]:
    """P3 (Pass 6, SimuHome-informed design principle): the declared tool
    sequence this scenario's gold genuinely requires, read from the
    scenario's own manifest.json rather than a runner-level constant.

    Raises, never defaults silently: a scenario authored without this field
    is a real authoring gap (every scenario evaluated for ORDERING must
    declare its own procedure), not something to paper over with a shared
    fallback that could quietly apply the wrong sequence to a new family.
    """
    manifest = _manifest(scenario_id)
    procedure = manifest.get("required_minimal_procedure")
    if not procedure:
        raise KeyError(
            f"{scenario_id}: manifest.json has no required_minimal_procedure -- "
            "author one explicitly rather than relying on a shared default (P3)."
        )
    return list(procedure)


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
    if axis == "perceptual":
        gold_action = meta["gold"]
        gold_display = " or ".join(gold_action) if isinstance(gold_action, (list, tuple)) else gold_action
        src_thermal = f"read_thermal_image({meta['asset']})"

        def thermal_spec(arm_id: str, withheld: Sequence[str], sources, probe: bool) -> ArmSpec:
            return ArmSpec(
                scenario_id=scenario_id, fm=fm, arm_id=arm_id,
                available_evidence=() if withheld else ("thermal",),
                withheld_evidence=tuple(withheld),
                intended_evidence=("thermal",),
                necessary_evidence=() if withheld else ("thermal",),
                expected_gold_action=gold_display,
                allowed_action_set=THERMAL_ACTIONS,
                evidence_source_ids=tuple(sources),
                insufficient_evidence_probe=probe, axis=axis,
            )

        return [
            # A1: gold IS reachable here (the finding is handed to the agent
            # in prose) -- but not through genuine perception. probe=True
            # names it a leak control, never a competence arm; see the
            # class docstring's "perceptual" section.
            thermal_spec(TEXT_CONTROL, ("thermal",), (), probe=True),
            thermal_spec(IMAGE_GROUNDED, (), (src_thermal,), probe=False),
            thermal_spec(IMAGE_REFERENCE, (), (src_thermal,), probe=False),
            # Part 16A perception ablation: identical evidence declaration to
            # IMAGE_GROUNDED (the thermal tool IS available, nothing withheld)
            # -- only WHICH perception pathway runs behind that tool call
            # differs, which is a runner/executor-level dispatch decision on
            # arm_id, not an evidence-withholding decision belonging here.
            thermal_spec(T0_DETERMINISTIC, (), (src_thermal,), probe=False),
            thermal_spec(T2_DIRECT_IMAGE, (), (src_thermal,), probe=False),
            thermal_spec(T3_ALT_MODEL, (), (src_thermal,), probe=False),
        ]
    if axis == "acoustic_perceptual":
        # PASS R3: the mandatory full-loop proof (plan A5), not a
        # perception-pathway ablation -- one arm only. Mirrors the
        # "perceptual" axis's IMAGE_GROUNDED shape exactly, substituting
        # read_acoustic/commit_acoustic_decision for the thermal tools.
        gold_action = meta["gold"]
        src_acoustic = f"read_acoustic({meta['asset']})"
        return [
            ArmSpec(
                scenario_id=scenario_id, fm=fm, arm_id=AUDIO_GROUNDED,
                available_evidence=("acoustic",), withheld_evidence=(),
                intended_evidence=("acoustic",), necessary_evidence=("acoustic",),
                expected_gold_action=gold_action, allowed_action_set=THERMAL_ACTIONS,
                evidence_source_ids=(src_acoustic,),
                insufficient_evidence_probe=False, axis=axis,
            ),
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
    tools = list(PHYSICAL_TOOLS + DIGITAL_TOOLS + ENTERPRISE_TOOLS + THERMAL_TOOLS)
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
    if "thermal" in spec.withheld_evidence:
        tools = [t for t in tools if t not in THERMAL_TOOLS]
        attached = [a for a in attached if not a.startswith("read_thermal_image")]
        if spec.arm_id == TEXT_CONTROL:
            # The one arm that INJECTS rather than redacts: gold must be
            # reachable here without ever looking at pixels, by
            # construction, so this is the pre-Pass-1 shortcut restored
            # deliberately as a leak control (see class docstring). The
            # finding text lives in manifest.json's visual block, the same
            # authoritative source the real perception observation would
            # otherwise have to earn.
            findings = _manifest(spec.scenario_id).get("visual", {}).get("text_control_findings", "")
            if findings:
                text = text.rstrip() + "\n\nThermal image findings (already interpreted for you):\n" + findings
    elif spec.arm_id == IMAGE_REFERENCE:
        ref_sid = meta.get("reference_scenario_id")
        if ref_sid:
            text = (text.rstrip() + "\n\nA known-normal reference thermal frame of this asset "
                    f"is available. Call read_thermal_image with "
                    f"reference_inspection_id=\"{ref_sid}\" to include it alongside the query image.")

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
    """Literal strings that must not survive redaction under this arm.

    Derived by matching the redaction pattern against the *unredacted question*
    and taking what it actually matched, not by parsing the pattern itself. The
    earlier version scanned the pattern text for a number, so ``1\\.02`` yielded
    ``"1"`` — a string that appears all over any payload and produced a spurious
    leak report for R056.

    Numbers shorter than two characters are dropped: a bare digit cannot be
    distinguished from incidental text and would make the check fire constantly.
    """
    meta = SCENARIOS[scenario_id]
    out: List[str] = []
    pattern = meta.get("iot_phrase")
    if "digital" not in arm.withheld_evidence or not pattern:
        return out
    m = re.search(pattern, question_text(scenario_id), flags=re.I)
    if not m:
        return out
    matched = m.group(0)
    out.append(matched.strip())
    out += [n for n in re.findall(r"\d+(?:\.\d+)?", matched) if len(n) >= 2]
    return sorted(set(out))


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
