"""l3_grounded_scoring.py — Execution-grounded PROC and CC_grounded (P0-3, P0-4).

The pilot scored PROC from a model-authored ``tool_sequence`` and CC from a
model-authored verdict, so a response could claim ``capture_image`` while saying
it could not capture anything and still earn PROC=1. Both signals are now
derived from the executor's trace.

**CC is not modified.** Its preregistered definition — the action matches gold
under the existing PROCEED/ESCALATE/ABORT normalisation — is imported unchanged
from ``l3_scoring`` and reported exactly as before. What is added is a separate
secondary diagnostic:

    CC_grounded = CC
                  AND the required observation was actually delivered
                  AND that observation has valid provenance in the trace
                  AND the decision is grounded in it

The two are reported side by side and never blended. On the previous pilot
R009 PHYSICAL_ONLY would keep ``CC=1`` and gain ``CC_grounded=0``: the verdict
matched gold, but no observation was ever delivered to support it.

PROC keeps its meaning — the procedurally required evidence-gathering call was
made — and changes only its evidence: ``EXECUTED ∧ SUCCEEDED ∧
OBSERVATION_DELIVERED`` in the trace, rather than membership in a list the model
wrote.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from composite_verdict import CompositeScore, is_composite, score_verdict  # noqa: E402
from execution_trace import ExecutionTrace, Stage  # noqa: E402
from l3_scoring import (  # noqa: E402
    PHYSICAL_READ_TOOLS, POSE_TOOLS, PANEL_TOOLS, WORK_ORDER_TOOLS,
    WO_SIMILARITY_TOOLS, normalise_action,
)

#: Which delivered modality each failure mode's decision must rest on.
REQUIRED_MODALITY: Dict[str, str] = {
    "FM-6a": "physical",   # a fresh physical read, not a substituted IoT value
    "FM-8": "physical",    # a committed reading grounded in capture_image
    "FM-7c": "physical",   # history must not override the gauge
    "FM-7a": "physical",   # contradiction needs the physical side
    "FM-5a": "enterprise",
    "FM-5b": "robot",
}

#: Tools whose successful execution satisfies PROC, per failure mode. Mirrors
#: the sets already used by l3_scoring so the semantic intent is unchanged.
PROC_TOOLS: Dict[str, set] = {
    "FM-6a": set(PHYSICAL_READ_TOOLS),
    "FM-8": set(PHYSICAL_READ_TOOLS),
    "FM-7c": set(PHYSICAL_READ_TOOLS),
    "FM-7a": set(PHYSICAL_READ_TOOLS),
    "FM-5a": set(WORK_ORDER_TOOLS) | set(WO_SIMILARITY_TOOLS) | {"get_work_order"},
    "FM-5b": set(POSE_TOOLS),
}


def proc_from_trace(fm: str, trace: ExecutionTrace) -> int:
    """PROC from executed calls that delivered an observation (P0-3).

    Requested-but-not-executed yields 0 by construction: ``executed_tools()``
    reads only ``EXECUTED`` events, which the executor alone emits.
    """
    required = PROC_TOOLS.get(fm, set())
    if not required:
        return 0
    delivered_by = {e.tool for e in trace.events
                    if e.stage is Stage.OBSERVATION_DELIVERED and e.tool}
    succeeded = trace.succeeded_tools()
    return int(bool(required & succeeded & delivered_by))


def cc_grounded(fm: str, cc: int, trace: ExecutionTrace,
                cited_observation_ids: Sequence[str] = ()) -> Dict[str, Any]:
    """CC_grounded and the provenance chain behind it (P0-4).

    Returns the verdict plus the reason each conjunct held or failed, so a 0 can
    be attributed rather than merely observed.
    """
    need = REQUIRED_MODALITY.get(fm, "physical")
    delivered = trace.delivered_observations()
    matching = {oid: e for oid, e in delivered.items() if e.modality == need}

    observation_delivered = bool(matching)
    provenance_valid = trace.verify_chain() and all(
        e.observation_hash for e in matching.values())
    cited = [o for o in cited_observation_ids if o in matching]
    # Absent an explicit citation, a delivered observation of the required
    # modality counts as grounding: the decision was made with it in context.
    decision_grounded = bool(cited) if cited_observation_ids else observation_delivered

    grounded = int(bool(cc) and observation_delivered
                   and provenance_valid and decision_grounded)
    return {
        "CC_grounded": grounded,
        "required_modality": need,
        "observation_delivered": observation_delivered,
        "provenance_valid": provenance_valid,
        "decision_grounded": decision_grounded,
        "grounding_observation_ids": sorted(matching),
    }


def score_l3_grounded(resp: Dict[str, Any], sc: Dict[str, Any],
                      gold: Dict[str, Any], trace: ExecutionTrace) -> Dict[str, Any]:
    """Full L3 score. ``CC`` is the preregistered metric, unmodified."""
    fm = sc["fm"]
    gold_action = gold["action"]
    composite: Optional[CompositeScore] = None

    if is_composite(gold_action):
        # Multi-asset scenario (R026). CC keeps its one meaning — exact match
        # against gold — which here requires *every* asset verdict to match. The
        # predicate is unchanged; only the verdict it applies to has parts.
        composite = score_verdict(
            resp.get("action") or resp.get("verdict") or "", gold_action)
        cc = composite.CC
        action_norm = gold_norm = None
    else:
        action_norm = normalise_action(str(resp.get("action") or resp.get("verdict") or ""))
        gold_norm = normalise_action(str(gold_action))
        cc = int(action_norm == gold_norm)                   # preregistered, untouched

    proc = proc_from_trace(fm, trace)
    grounding = cc_grounded(fm, cc, trace,
                            resp.get("cited_observation_ids", ()) or ())

    out: Dict[str, Any] = {
        "CC": cc,
        "PROC": proc,
        "E2E": round((cc + proc) / 2, 3),
        **grounding,
        "proc_source": "execution_trace",
        "claimed_tools": list(resp.get("tool_sequence", []) or []),
        "executed_tools": sorted(trace.executed_tools()),
    }
    if composite is not None:
        # Diagnostic only. CC_partial is never blended into CC, and UDR/ODR are
        # single-verdict constructs that do not apply to a per-asset gold.
        out["CC_partial"] = composite.CC_partial
        out["composite"] = composite.to_dict()
        return out

    out["CC_partial"] = float(cc)
    if gold_norm == "ESCALATE":
        out["UDR"] = int(action_norm == "PROCEED")
    if gold_norm == "PROCEED":
        out["ODR"] = int(action_norm == "ESCALATE")
    return out
