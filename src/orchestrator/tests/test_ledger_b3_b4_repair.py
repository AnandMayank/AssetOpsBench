"""Verification for ledger B3 (leak detector fired 0/32, actual rate 15/32)
and B4 (R016->R066 confounded ordering and leak axes).

B3: replaced the literal "the answer is" / "you should X" detector, which had
never fired, with a conditional-rule detector (leak_detect.has_rule_leak).
Fifteen leaking scenarios get de-leaked twins (kind=de_leaked in the twin's
manifest), treated as a measured factor rather than an automatic defect.

B4: R016 previously had only one control, R066, which is ordering-free but
still states R016's decision rule -- confounding the ordering and leak axes
in one pair. R016 now has two independent single-axis controls: R066
(ordering removed, leak retained) and R078 (leak removed, ordering retained).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCEN = REPO_ROOT.parent / "AssetOpsBenchScenarioGeneration" / "RobotInspection"
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from classc_fixtures import DELEAK_FIXTURES, DELEAK_PAIRS, FIXTURES  # noqa: E402
from leak_detect import has_rule_leak  # noqa: E402

#: The 15 scenarios ledger B3 found leaking, by family.
LEAKING = {
    "B": ["R011", "R012", "R014"],
    "C_parents": ["R001", "R005", "R016", "R018"],
    "C_controls": ["R068", "R069", "R070"],
    "D_parents": ["R008", "R010", "R026"],
    "D_controls": ["R059", "R060"],
}
ALL_LEAKING = [s for group in LEAKING.values() for s in group]

#: Twin for every leaking scenario (parent -> twin ID), including B and R026
#: which have no executor fixtures yet.
TWIN_OF = {
    "R011": "R073", "R012": "R074", "R014": "R075",
    "R001": "R076", "R005": "R077", "R016": "R078", "R018": "R079",
    "R068": "R080", "R069": "R081", "R070": "R082",
    "R008": "R083", "R010": "R084", "R026": "R085",
    "R059": "R086", "R060": "R087",
}


def _dir(sid: str) -> Path:
    n = int(sid.lstrip("R"))
    for c in (SCEN / f"scenario_R{n:02d}", SCEN / f"scenario_R{n}"):
        if c.is_dir():
            return c
    raise FileNotFoundError(sid)


def _question_body(sid: str) -> str:
    q = (_dir(sid) / "question.txt").read_text(errors="replace")
    return re.sub(r"Return \{.*", "", q, flags=re.S)


def _gold(sid: str) -> str:
    gt = (_dir(sid) / "groundtruth.txt").read_text(errors="replace")
    m = re.search(r"Expected verdict:\s*([A-Z_]+)", gt)
    return m.group(1) if m else ""


# --- B3: the detector fires on exactly the 15, and on none of their twins ---

def test_all_15_known_leaking_scenarios_still_leak():
    assert set(ALL_LEAKING) == set(TWIN_OF)
    assert len(ALL_LEAKING) == 15
    for sid in ALL_LEAKING:
        assert has_rule_leak(_question_body(sid)), f"{sid} no longer detected as leaking"


def test_no_twin_leaks():
    for parent, twin in TWIN_OF.items():
        assert not has_rule_leak(_question_body(twin)), f"{twin} (twin of {parent}) still leaks"


def test_detector_does_not_fire_on_the_non_leaking_majority():
    """Negative control: scenarios never flagged by the manual scan must stay
    unflagged, so the regression test would catch the detector over-firing."""
    NOT_LEAKING = ["R006", "R007", "R017", "R023", "R024",
                   "R064", "R065", "R066", "R067", "R071", "R072",
                   "R021", "R022", "R025", "R061", "R062", "R063"]
    for sid in NOT_LEAKING:
        assert not has_rule_leak(_question_body(sid)), f"{sid} unexpectedly flagged"


# --- de-leaked twins preserve world and gold, vary only the leak axis -------

@pytest.mark.parametrize("parent,twin", sorted(DELEAK_PAIRS.items()))
def test_deleaked_twin_preserves_world_and_gold(parent, twin):
    p, t = FIXTURES[parent], FIXTURES[twin]
    assert t.asset == p.asset
    assert t.profile == p.profile
    assert t.robot_state == p.robot_state
    assert t.enterprise == p.enterprise
    assert t.waypoint_active == p.waypoint_active
    assert _gold(twin) == _gold(parent), "gold changed; leak axis is not isolated"


@pytest.mark.parametrize("parent,twin", sorted(DELEAK_PAIRS.items()))
def test_deleaked_twin_no_duplicate_and_no_answer_leak(parent, twin):
    q = " ".join(_question_body(twin).split())
    assert q != " ".join(_question_body(parent).split()), f"{twin} identical to {parent}"
    for phrase in ("the correct answer", "the verdict is", "you should commit",
                   "you should escalate", "you should abort", "gold:"):
        assert phrase not in q.lower(), f"{twin} leaks: {phrase!r}"


def test_deleak_fixtures_cover_every_c_d_leaking_scenario():
    c_d_leaking = (LEAKING["C_parents"] + LEAKING["C_controls"]
                  + LEAKING["D_parents"] + LEAKING["D_controls"])
    c_d_leaking = [s for s in c_d_leaking if s != "R026"]  # composite, excluded
    assert set(DELEAK_PAIRS) == set(c_d_leaking), (
        "DELEAK_FIXTURES/DELEAK_PAIRS must cover exactly the executable "
        "leaking C/D scenarios, no more, no fewer")


def test_all_c_d_twins_are_runnable_under_the_real_audit():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from classc_audit import audit as audit_c
    from classd_audit import audit as audit_d

    c_twins = ["R076", "R077", "R078", "R079", "R080", "R081", "R082"]
    d_twins = ["R083", "R084", "R086", "R087"]
    for sid in c_twins:
        a = audit_c(sid)
        assert a.verdict == "RUNNABLE", f"{sid}: {a.verdict}"
        assert not a.rule_leak
    for sid in d_twins:
        a = audit_d(sid)
        assert a.verdict == "RUNNABLE", f"{sid}: {a.verdict}"
        assert not a.rule_leak


# --- B4: R016 now has two independent single-axis controls -----------------

def test_r016_has_independent_ordering_and_leak_controls():
    """The original defect: R066 (R016's only control) removed BOTH the
    ordering constraint and the leak simultaneously, so no pair varied
    exactly one axis. Three scenarios now span the (ordering, leak) square
    on the two cells that matter:

        R016  ordering=constrained  leak=True    (baseline)
        R078  ordering=constrained  leak=False    <- isolates the leak axis
        R066  ordering=free         leak=False    <- isolates the ordering
                                                      axis, against R078,
                                                      not against R016
    """
    from classc_audit import audit as audit_c

    r016 = audit_c("R016")
    r078 = audit_c("R078")
    r066 = audit_c("R066")

    assert r016.rule_leak and not r016.ordering_free_by_design
    assert not r078.rule_leak and not r078.ordering_free_by_design, (
        "R078 must vary leak status only, relative to R016 -- ordering stays intact")
    assert not r066.rule_leak and r066.ordering_free_by_design

    # R016 -> R078: leak axis isolated (ordering held fixed at "constrained").
    # R078 -> R066: ordering axis isolated (leak held fixed at False) -- this,
    # not the original R016 -> R066 pair, is the clean ordering-axis control.
    assert _gold("R016") == _gold("R078") == _gold("R066")
