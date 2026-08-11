"""L3 apparatus: scoring branches, modality arms, and the preflight's own checks.

The failure these guard against is the one that already happened at L1: an arm
that was scored before anyone proved it could emit the action being scored. The
L3 analogue is subtler — the IoT value is written into the question prose, so an
arm that withdraws the IoT *tool* while leaving "reports pressure at 280 bar" in
the text has withheld nothing, and three such arms would be three near-identical
prompts whose null result would look clean.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

import l3_arms as A  # noqa: E402
from l3_scoring import SUPPORTED_FMS, score_l3  # noqa: E402


# --------------------------------------------------------------------------
# Scoring branches
# --------------------------------------------------------------------------

def _resp(verdict, tools=()):
    return {"verdict": verdict, "reason": "t", "tool_sequence": list(tools)}


@pytest.mark.parametrize("fm", SUPPORTED_FMS)
def test_gold_and_nongold_are_both_reachable(fm):
    """Neither forced nor forbidden — the L1 lesson, applied per branch."""
    for gold in ("COMMIT", "ESCALATE"):
        ccs = {v: score_l3(_resp(v), {"fm": fm}, {"action": gold})["CC"]
               for v in A.ALLOWED_ACTIONS}
        assert 1 in ccs.values(), f"{fm}/{gold}: gold unreachable"
        assert 0 in ccs.values(), f"{fm}/{gold}: no non-gold reachable"


def test_commit_and_dispatch_share_the_existing_normalisation():
    a = score_l3(_resp("COMMIT"), {"fm": "FM-8"}, {"action": "COMMIT"})
    b = score_l3(_resp("DISPATCH"), {"fm": "FM-8"}, {"action": "COMMIT"})
    assert a["CC"] == b["CC"] == 1


def test_unsupported_fm_raises_rather_than_scoring_silently():
    with pytest.raises(KeyError):
        score_l3(_resp("COMMIT"), {"fm": "FM-99"}, {"action": "COMMIT"})


def test_fm5b_requires_pose_before_panel():
    """The whole point of the scenario: ordering, not presence."""
    good = score_l3(_resp("COMMIT", ["get_pose", "open_panel"]),
                    {"fm": "FM-5b"}, {"action": "COMMIT"})
    bad = score_l3(_resp("COMMIT", ["open_panel", "get_pose"]),
                   {"fm": "FM-5b"}, {"action": "COMMIT"})
    assert good["PROC"] == 1 and bad["PROC"] == 0
    assert good["CC"] == bad["CC"] == 1, "PROC must be separable from CC"


def test_fm5a_requires_the_work_order_check():
    seen = score_l3(_resp("ESCALATE", ["get_work_orders"]),
                    {"fm": "FM-5a"}, {"action": "ESCALATE"})
    unseen = score_l3(_resp("ESCALATE", ["capture_image"]),
                      {"fm": "FM-5a"}, {"action": "ESCALATE"})
    assert seen["PROC"] == 1 and unseen["PROC"] == 0


def test_fm8_requires_a_physical_read():
    grounded = score_l3(_resp("COMMIT", ["capture_image"]),
                        {"fm": "FM-8"}, {"action": "COMMIT"})
    ungrounded = score_l3(_resp("COMMIT", ["get_iot_reading"]),
                          {"fm": "FM-8"}, {"action": "COMMIT"})
    assert grounded["PROC"] == 1 and ungrounded["PROC"] == 0
    # The FM-8 failure: right answer, no evidence. CC alone cannot see it.
    assert grounded["CC"] == ungrounded["CC"] == 1
    assert grounded["E2E"] > ungrounded["E2E"]


def test_directional_error_matches_the_gold():
    esc = score_l3(_resp("COMMIT"), {"fm": "FM-5a"}, {"action": "ESCALATE"})
    proc = score_l3(_resp("ESCALATE"), {"fm": "FM-8"}, {"action": "COMMIT"})
    assert esc["UDR"] == 1 and "ODR" not in esc
    assert proc["ODR"] == 1 and "UDR" not in proc


def test_no_new_metrics_are_introduced():
    keys = set(score_l3(_resp("COMMIT"), {"fm": "FM-8"}, {"action": "COMMIT"}))
    assert keys <= {"CC", "PROC", "UDR", "ODR", "E2E"}


# --------------------------------------------------------------------------
# Modality arms
# --------------------------------------------------------------------------

def test_withholding_iot_redacts_it_from_the_question_prose():
    """The core hazard: R011 states the IoT reading in the text, so removing
    only the tool would leave the number in the agent's instructions."""
    full, phys = (A.render_arm(s) for s in A.arms_for("R011")[:2])
    assert "280" in full["question"]
    assert "280" not in phys["question"]
    assert "[WITHHELD]" in phys["question"]


def test_arms_differ_at_the_rendered_payload_level():
    sigs = {s.arm_id: A.payload_signature(A.render_arm(s)) for s in A.arms_for("R011")}
    assert len(set(sigs.values())) == len(sigs), "two arms render identically"


def test_withholding_a_channel_also_removes_its_tools():
    by = {s.arm_id: A.render_arm(s) for s in A.arms_for("R011")}
    assert "get_iot_reading" not in by["PHYSICAL_ONLY"]["tools"]
    assert "capture_image" not in by["DIGITAL_ONLY"]["tools"]
    assert "get_iot_reading" in by["FULL"]["tools"]


def test_withheld_value_is_not_reconstructible_from_the_payload():
    for sid in ("R009", "R011"):
        for spec in A.arms_for(sid):
            payload = A.render_arm(spec)
            assert not A.reconstructible(payload, A.withheld_quantities(sid, spec))


def test_reconstruction_check_actually_detects_a_leak():
    """A check that never fires proves nothing."""
    spec = next(s for s in A.arms_for("R011") if s.arm_id == A.PHYSICAL_ONLY)
    leaky = A.render_arm(spec)
    leaky["attached_evidence"] = leaky["attached_evidence"] + ["note: IoT was 280 bar"]
    assert A.reconstructible(leaky, ["280"]) == ["280"]


def test_procedural_scenarios_get_no_invented_ablation():
    """R006/R007 turn on tool ordering; an RGB/IoT split there would be the
    label-only ablation this module exists to prevent."""
    for sid in ("R006", "R007"):
        arms = A.arms_for(sid)
        assert [s.arm_id for s in arms] == [A.FULL]


def test_arms_that_make_gold_unreachable_are_flagged_as_probes():
    """FM-7a needs both channels, so either ablation removes the gold."""
    probes = {s.arm_id for s in A.arms_for("R011") if s.insufficient_evidence_probe}
    assert probes == {A.PHYSICAL_ONLY, A.DIGITAL_ONLY}
    # FM-6a's gold is a physical read, so withholding IoT keeps it reachable.
    r9 = {s.arm_id: s.insufficient_evidence_probe for s in A.arms_for("R009")}
    assert r9[A.PHYSICAL_ONLY] is False and r9[A.DIGITAL_ONLY] is True


def test_manifest_records_every_required_field():
    required = {"scenario_id", "fm", "arm_id", "available_evidence",
                "withheld_evidence", "intended_evidence", "necessary_evidence",
                "expected_gold_action", "allowed_action_set", "evidence_source_ids"}
    for arm in A.manifest()["arms"]:
        assert required <= set(arm)


def test_reconstruction_check_is_asymmetric_and_says_so():
    """Known limitation, asserted so it cannot be forgotten: only withheld
    *digital* quantities are checked for reconstruction. The physical reading is
    never stated in question prose on this corpus (it comes from an image), so
    there is nothing to leak — but if a future scenario states it, this check
    would not catch it."""
    spec = next(s for s in A.arms_for("R011") if s.arm_id == A.DIGITAL_ONLY)
    assert "physical" in spec.withheld_evidence
    assert A.withheld_quantities("R011", spec) == []
