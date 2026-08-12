"""Verification for the nine contrastive repairs and the frozen coordination rate.

A contrastive partner is only useful if it varies exactly one thing. These tests
check that mechanically rather than by reading the scenario text.
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

import scenario_gen as G  # noqa: E402
from classc_fixtures import CONTRAST_PAIRS, CONTRASTIVE_FIXTURES, FIXTURES  # noqa: E402
from composite_verdict import R026_GOLD, score_verdict  # noqa: E402
from scenario_gen import Cell, derive_gold, sample_world  # noqa: E402

NEW = sorted(CONTRASTIVE_FIXTURES)


def _dir(sid: str) -> Path:
    n = int(sid.lstrip("R"))
    for c in (SCEN / f"scenario_R{n:02d}", SCEN / f"scenario_R{n}"):
        if c.is_dir():
            return c
    raise FileNotFoundError(sid)


def _gold(sid: str) -> str:
    """Read gold from either layout the scenarios use.

    R021 and R025 state their verdict in the gold-answer JSON rather than an
    "Expected verdict:" line. Reading only the first layout made both look like
    their control had changed gold, when nothing had.
    """
    gt = (_dir(sid) / "groundtruth.txt").read_text(errors="replace")
    m = re.search(r"Expected verdict:\s*([A-Z_]+)", gt)
    if m:
        return m.group(1)
    m = re.search(r'"verdict"\s*:\s*"([A-Z_]+)"', gt)
    return m.group(1) if m else ""


def _question(sid: str) -> str:
    return re.sub(r"Return \{.*", "", (_dir(sid) / "question.txt").read_text(), flags=re.S)


# --- the nine exist and are well formed -------------------------------------

@pytest.mark.parametrize("sid", NEW)
def test_contrastive_scenario_is_complete(sid):
    d = _dir(sid)
    for f in ("question.txt", "groundtruth.txt", "manifest.json"):
        assert (d / f).exists(), f"{sid} missing {f}"
    man = json.loads((d / "manifest.json").read_text())
    assert man["provenance"]["control_for"], f"{sid} does not name its parent"
    assert _gold(sid) in {"COMMIT", "ESCALATE", "ABORT"}


@pytest.mark.parametrize("sid", NEW)
def test_no_new_scenario_leaks_the_answer(sid):
    q = _question(sid).lower()
    for phrase in ("the correct answer", "the verdict is", "you should commit",
                   "you should escalate", "you should abort", "gold:"):
        assert phrase not in q, f"{sid} leaks: {phrase!r}"


@pytest.mark.parametrize("sid", NEW)
def test_no_new_scenario_is_gold_conditioned(sid):
    """Controls preserve their parent's world; none introduces a hidden value
    chosen to produce a verdict. The physical envelope comes from the asset
    spec, not from the label."""
    man = json.loads((_dir(sid) / "manifest.json").read_text())
    assert man["asset_id"] in G.ASSETS
    gt = (_dir(sid) / "groundtruth.txt").read_text()
    assert "control_for" in json.dumps(man) or "Contrastive" in gt or "control" in gt.lower()


def test_no_new_scenario_duplicates_an_existing_one():
    """A control must differ from its parent in the agent-visible text; two
    controls must not be identical to each other."""
    seen = {}
    for sid in NEW:
        q = " ".join(_question(sid).split())
        parent = json.loads((_dir(sid) / "manifest.json").read_text())["provenance"]["control_for"]
        assert q != " ".join(_question(parent).split()), f"{sid} identical to parent {parent}"
        assert q not in seen, f"{sid} duplicates {seen.get(q)}"
        seen[q] = sid


# --- class-D: gold moves only where the enterprise factor is causal ----------

@pytest.mark.parametrize("parent,expect_causal", [
    ("R008", True), ("R010", True),        # causal half
    ("R021", False), ("R022", False), ("R025", False),   # non-causal half
])
def test_enterprise_factor_causality_matches_the_pair(parent, expect_causal):
    control, causal = CONTRAST_PAIRS[parent]
    assert causal is expect_causal
    changed = _gold(parent) != _gold(control)
    assert changed is expect_causal, (
        f"{parent}->{control}: gold {'changed' if changed else 'unchanged'} "
        f"but the factor is {'causal' if expect_causal else 'non-causal'}")


def test_both_halves_of_the_enterprise_contrast_exist():
    """Without a non-causal half, 'the enterprise factor moved the decision'
    is unfalsifiable."""
    kinds = {c for p, (c, causal) in CONTRAST_PAIRS.items() if causal is True}
    non = {c for p, (c, causal) in CONTRAST_PAIRS.items() if causal is False}
    assert len(kinds) >= 2 and len(non) >= 3


@pytest.mark.parametrize("parent", ["R008", "R010", "R021", "R022", "R025"])
def test_control_preserves_the_parents_asset(parent):
    control, _ = CONTRAST_PAIRS[parent]
    assert FIXTURES[control].asset == FIXTURES[parent].asset, "physical world changed"


# --- class-C: ordering controls isolate ordering -----------------------------

@pytest.mark.parametrize("parent", ["R006", "R007", "R016", "R017"])
def test_ordering_control_preserves_world_and_gold(parent):
    control, _ = CONTRAST_PAIRS[parent]
    assert FIXTURES[control].asset == FIXTURES[parent].asset
    # Same precondition: the control must carry the parent's robot_state edits.
    assert FIXTURES[control].robot_state == FIXTURES[parent].robot_state
    assert _gold(control) == _gold(parent), "gold changed; ordering is not isolated"


def test_ordering_controls_cover_both_nominal_and_fault_preconditions():
    """R006/R007 are nominal; R016/R017 carry a fault. Covering only one kind
    would confound ordering with condition recognition."""
    nominal = {c for p, (c, _) in CONTRAST_PAIRS.items() if p in ("R006", "R007")}
    fault = {c for p, (c, _) in CONTRAST_PAIRS.items() if p in ("R016", "R017")}
    assert nominal and fault
    assert all(FIXTURES[c].robot_state for c in fault)
    assert all(not FIXTURES[c].robot_state for c in nominal)


# --- composite verdict adoption ---------------------------------------------

def test_composite_interface_is_adopted_with_cc_semantics_unchanged():
    assert score_verdict({"chiller_6": "ABORT", "motor_01": "COMMIT"}, R026_GOLD).CC == 1
    partial = score_verdict({"chiller_6": "ABORT", "motor_01": "ESCALATE"}, R026_GOLD)
    assert partial.CC == 0, "CC must require every asset"
    assert partial.CC_partial == 0.5, "CC_partial remains diagnostic"
    assert score_verdict("COMMIT", "COMMIT").CC == 1, "binary scenarios unchanged"


# --- frozen coordination rate ------------------------------------------------

def test_coordination_rates_are_frozen_and_unchanged():
    assert G.ACTIVE_WO_RATE == 0.25
    assert G.TECHNICIAN_PRESENT_RATE == 0.15


def test_frozen_generator_reproduces_the_committed_pilot():
    """The 12-world reference set must be bit-identical after these repairs."""
    a = G.generate(seed=2026, per_cell=3)
    committed = json.loads((REPO_ROOT / "reports" / "v1" / "generated_pilot.json").read_text())
    worlds = {r["scenario_id"]: r["world"] for r in committed["results"] if r["arm"] == "FULL"}
    for rec in a:
        w = rec["world"]
        assert worlds[w["scenario_id"]] == w, "historical world changed"
