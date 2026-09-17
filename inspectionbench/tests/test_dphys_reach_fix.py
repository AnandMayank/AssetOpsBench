"""Regression tests for the D-physical REACH observability fix
(scripts/phase8h2e_dphys_pilot.py's render_prompt / _episode_tests_reach /
_REACH_NOTE). See reports/benchmark/dphys_reach_fix_plan.md for the root
cause and reports/benchmark/dphys_reach_fix_report.md for the full writeup.

Zero model/API calls -- this tests prompt RENDERING only, never a real
generation call.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pytest  # noqa: E402

ASSET_PROFILES_PATH = (
    Path.home() / "AssetOpsBenchScenarioGeneration" / "RobotInspection" / "Scenarios"
    / "asset_profiles.json"
)
MANIFEST_PATH = REPO_ROOT / "reports" / "benchmark" / "d_physical_manifest.json"
pytestmark = pytest.mark.skipif(
    not ASSET_PROFILES_PATH.exists() or not MANIFEST_PATH.exists(),
    reason="AssetOpsBenchScenarioGeneration sibling repo or D-physical manifest not present",
)

import phase8h2e_dphys_pilot as P  # noqa: E402


def _episodes():
    return P.load_canonical_episodes()


def _find(stratum, **filters):
    eps = [e for e in _episodes() if e["constraint_stratum"] == stratum]
    for k, v in filters.items():
        eps = [e for e in eps if v in e["scenario_id"]]
    return eps


# --- A. Reach observation delivery ------------------------------------------

def test_reach_episode_prompt_includes_shoulder_height_and_formula():
    ep = _find("single:reach", asset="chiller_6")[0]
    prompt = P.render_prompt(ep)
    assert "arm shoulder height" in prompt
    assert "0.75" in prompt
    assert "required reach" in prompt
    assert "sqrt(standoff" in prompt


def test_reach_note_uses_the_same_fixed_constant_every_time():
    """SHOULDER_H is a manufacturer constant -- identical across every asset
    and margin condition, never derived from this episode's world or oracle
    output."""
    reach_eps = _find("single:reach")
    prompts = [P.render_prompt(e) for e in reach_eps]
    assert all("arm shoulder height: 0.75 m" in p for p in prompts)


def test_non_reach_single_binding_prompt_does_not_include_reach_note():
    for stratum in ("single:clearance", "single:joint_and_collision", "single:grasp_payload",
                    "single:stability", "single:energy"):
        eps = _find(stratum)
        if not eps:
            continue
        prompt = P.render_prompt(eps[0])
        assert "arm shoulder height" not in prompt, f"{stratum} unexpectedly got the reach note"


def test_coupled_stratum_involving_reach_gets_the_note():
    for stratum in ("dual:reach+clearance", "dual:reach+stability", "dual:energy+reach",
                    "triple:reach+clearance+stability"):
        eps = _find(stratum)
        if not eps:
            continue
        prompt = P.render_prompt(eps[0])
        assert "arm shoulder height" in prompt, f"{stratum} should include the reach note"


def test_coupled_stratum_not_involving_reach_omits_the_note():
    for stratum in ("dual:clearance+stability", "dual:grasp_payload+energy",
                    "dual:joint_and_collision+stability"):
        eps = _find(stratum)
        if not eps:
            continue
        prompt = P.render_prompt(eps[0])
        assert "arm shoulder height" not in prompt, f"{stratum} unexpectedly got the reach note"


def test_capx_prompts_are_unaffected_by_the_reach_fix():
    """CAP-X is explicitly out of scope for this fix (dphys_reach_fix_plan.md);
    its construct is candidate selection, not per-constraint attribution."""
    capx_eps = [e for e in _episodes() if e["template_id"] == "T-D-PHYS-CAP-X"]
    for ep in capx_eps:
        assert P._episode_tests_reach(ep) is False


# --- B. World/oracle consistency --------------------------------------------

def test_reach_note_values_are_consistent_with_the_worlds_own_geometry():
    """The prompt's standoff/panel_z numbers must match the same access dict
    the generator built (params-derived, not independently re-authored)."""
    ep = _find("single:reach", asset="chiller_6")[0]
    prompt = P.render_prompt(ep)
    assert str(ep["params"]["standoff"]) in prompt


# --- C. Matched reach cases --------------------------------------------------

def test_satisfied_and_violated_reach_prompts_differ_only_in_geometry_not_wording():
    inadm = _find("single:reach", asset="chiller_6", **{})
    reach_eps = [e for e in P.load_canonical_episodes() if e["constraint_stratum"] == "single:reach"
                and e["asset"] == "chiller_6"]
    assert len(reach_eps) == 2
    prompts = {e["params"]["margin"]: P.render_prompt(e) for e in reach_eps}
    assert set(prompts) == {"inadmissible", "comfortable"}
    # same formula/instructional text in both
    for p in prompts.values():
        assert "required reach = 3D straight-line distance" in p
    # differ only in the numeric panel height (the actual reach condition)
    assert prompts["inadmissible"] != prompts["comfortable"]


# --- D. No leakage ------------------------------------------------------------

#: Field names that would be gold/oracle leakage if they appeared as
#: DECLARATIVE statements about this episode. Excludes the "predicted_"
#: prefixed forms, which are the legitimate response-schema fields the
#: prompt instructs the agent to produce (not oracle output stated as fact).
LEAKED_TOKENS = ("VIOLATED", "SATISFIED", "admissibility_verdict", "AdmissibilityVerdict",
                 "replay_id", "energy_J")


def test_no_canonical_episode_prompt_leaks_evaluator_only_fields():
    for ep in _episodes():
        prompt = P.render_prompt(ep)
        for token in LEAKED_TOKENS:
            assert token not in prompt, f"{ep['episode_id']}: leaked {token!r} into the prompt"
        # "violated_constraints"/"limiting_constraint" ARE legitimate as the
        # "predicted_*" response-schema field names the agent must fill in --
        # only a BARE (non-"predicted_"-prefixed) occurrence would indicate
        # the oracle's own field leaking in as a stated fact.
        for bare in ("violated_constraints", "limiting_constraint"):
            for idx in range(len(prompt)):
                if prompt[idx:idx + len(bare)] == bare:
                    prefix = prompt[max(0, idx - 10):idx]
                    assert "predicted_" in prefix, (
                        f"{ep['episode_id']}: bare {bare!r} (not agent-response-schema-prefixed) "
                        f"found near: {prompt[max(0, idx-30):idx+30]!r}")
        # the oracle's own boolean verdict must never be STATED as fact
        assert '"admissible": true' not in prompt.lower()
        assert '"admissible": false' not in prompt.lower()
        # gold's own per-episode values must never appear verbatim as a
        # declarative statement (as opposed to the schema instructing the
        # agent to produce a value under that key)
        gold = ep["gold"]
        if gold["limiting_constraint"] not in (None, "MULTIPLE"):
            assert f'"{gold["limiting_constraint"]}"' not in prompt.split("Respond with")[0], (
                f"{ep['episode_id']}: gold limiting_constraint stated as fact before the response schema"
            )
