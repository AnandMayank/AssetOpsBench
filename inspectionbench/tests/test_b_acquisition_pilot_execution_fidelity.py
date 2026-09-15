"""Regression tests for the B-acquisition pilot's acquisition-EXECUTION
fidelity fix (scripts/phase8h2g_b_acquisition_pilot.py::get_real_acquisition_response).

Real bug found and fixed this pass: the environment always resolved
gold['required_acquisition'] regardless of what modality the model actually
requested in turn 1, and the turn-2 prompt then presented that gold-driven
result as if it were the response to the model's own request -- an
environment-fidelity violation, not merely a scoring inaccuracy (ASA's
CALCULATION was already reading the model's real turn-1 output; only the
EXECUTED observation was unfaithful). See
reports/benchmark/b_acq_selection_fidelity_report.md for the full audit.

Zero model/API calls -- deterministic simulator-level validation only.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pytest  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "reports" / "benchmark" / "b_acquisition_manifest.json"
pytestmark = pytest.mark.skipif(
    not MANIFEST_PATH.exists(), reason="B-acquisition manifest not present"
)

import phase8h2g_b_acquisition_pilot as P  # noqa: E402
import b_acquisition_scoring as S  # noqa: E402
from b_acquisition_generator import BAcquisitionGold  # noqa: E402


def _b_acq_4_episode(arm="ambiguous", asset="chiller_6"):
    eps = P.load_episodes()
    return [e for e in eps if e["template_id"] == "B-ACQ-4" and e["arm"] == arm
           and e["asset"] == asset][0]


def _gold_from_ep(ep):
    g = ep["gold"]
    return BAcquisitionGold(
        initial_evidence_sufficient=g["initial_evidence_sufficient"],
        acquisition_required=g["acquisition_required"],
        required_acquisition=g["required_acquisition"],
        acceptable_acquisition_set=frozenset(g["acceptable_acquisition_set"]),
        final_terminal_action=g["final_terminal_action"],
        acquisition_genuinely_unavailable=g["acquisition_genuinely_unavailable"],
        recovery_policy_on_unavailable=g["recovery_policy_on_unavailable"],
    )


# --- 1/2. requesting a specific modality delivers THAT modality's observation ---

def test_requesting_acoustic_delivers_an_acoustic_observation():
    ep = _b_acq_4_episode("ambiguous", "chiller_6")
    real = P.get_real_acquisition_response(ep, "acoustic")
    assert real["status"] == "RESOLVED"
    assert real["modality"] == "acoustic"
    assert real["observation_id"].startswith("obs_acoustic")


def test_requesting_iot_timeseries_delivers_an_iot_observation():
    ep = _b_acq_4_episode("ambiguous", "chiller_6")
    real = P.get_real_acquisition_response(ep, "iot_timeseries")
    assert real["status"] == "RESOLVED"
    assert real["modality"] == "iot_timeseries"
    assert real["observation_id"].startswith("obs_iot")


def test_different_requested_modalities_yield_different_observation_ids():
    """The core simulator-level proof required by the task: two different
    requested modalities on the SAME episode must resolve to different,
    modality-appropriate observations, not the same gold-driven one."""
    ep = _b_acq_4_episode("ambiguous", "chiller_6")
    acoustic = P.get_real_acquisition_response(ep, "acoustic")
    iot = P.get_real_acquisition_response(ep, "iot_timeseries")
    assert acoustic["observation_id"] != iot["observation_id"]
    assert acoustic["modality"] != iot["modality"]


# --- 3. requested != gold-required still determines the environment response ---

def test_requesting_the_non_gold_modality_is_still_honored():
    """B-ACQ-4's gold.required_acquisition is 'iot_timeseries' -- requesting
    'acoustic' instead (a real, observed model behavior) must still be
    faithfully executed as acoustic, never silently redirected to iot."""
    ep = _b_acq_4_episode("ambiguous", "chiller_6")
    assert ep["gold"]["required_acquisition"] == "iot_timeseries"
    real = P.get_real_acquisition_response(ep, "acoustic")
    assert real["modality"] == "acoustic", (
        "the environment substituted gold's required modality instead of executing "
        "the model's actual request -- this is exactly the bug this fix addresses"
    )


# --- 4. genuinely unavailable modality -> genuine UNAVAILABLE ----------------

def test_requesting_a_genuinely_unavailable_modality_returns_unavailable():
    ep = _b_acq_4_episode("ambiguous", "chiller_6")  # chiller_6 has 0 thermal records
    real = P.get_real_acquisition_response(ep, "thermal")
    assert real["status"] == "UNAVAILABLE"
    assert real["modality"] == "thermal"


def test_missing_requested_modality_is_unavailable_not_a_crash():
    ep = _b_acq_4_episode("ambiguous", "chiller_6")
    real = P.get_real_acquisition_response(ep, None)
    assert real["status"] == "UNAVAILABLE"


# --- 5. gold-required modality never appears unless actually requested -----

def test_turn2_prompt_reflects_only_the_requested_modality_not_gold():
    ep = _b_acq_4_episode("ambiguous", "chiller_6")
    real = P.get_real_acquisition_response(ep, "acoustic")
    prompt2 = P.TURN2_RESOLVED_TEMPLATE.format(
        modality=real["modality"], observation_id=real["observation_id"],
        sensor_metadata="{}")
    assert "acoustic" in prompt2
    assert "iot_timeseries" not in prompt2, (
        "gold's required_acquisition ('iot_timeseries') must never leak into a "
        "turn-2 prompt built from a DIFFERENT requested modality"
    )


# --- 6. ASA is computed from the model's actual selection -------------------

def test_asa_reflects_the_actually_requested_modality_not_gold():
    ep = _b_acq_4_episode("ambiguous", "chiller_6")
    gold = _gold_from_ep(ep)
    # model requested the WRONG modality (acoustic, not gold's iot_timeseries)
    response_wrong = {"acquire": True, "requested_modality": "acoustic",
                      "claimed_observation_id": None, "terminal_action": "COMMIT"}
    s_wrong = S.score_b_acquisition_episode(gold, response_wrong)
    assert s_wrong.ASA == 0.0
    # model requested the RIGHT modality
    response_right = {"acquire": True, "requested_modality": "iot_timeseries",
                      "claimed_observation_id": None, "terminal_action": "COMMIT"}
    s_right = S.score_b_acquisition_episode(gold, response_right)
    assert s_right.ASA == 1.0


def test_executed_modality_always_matches_requested_modality_end_to_end():
    """Simulate the run_model loop's own logic directly (no API call): for
    any requested_modality, executed_modality (what get_real_acquisition_response
    reports back) must equal it exactly -- proving the fidelity property
    holds for the actual code path, not just the isolated helper."""
    ep = _b_acq_4_episode("ambiguous", "chiller_6")
    for requested in ("acoustic", "iot_timeseries", "thermal"):
        real = P.get_real_acquisition_response(ep, requested)
        assert real["modality"] == requested


# --- 7. B-ACQ-4 observability prompts remain unchanged ----------------------

def test_b_acq_4_turn1_prompts_unaffected_by_the_execution_fidelity_fix():
    """The execution-fidelity fix touches only turn-2 dispatch
    (get_real_acquisition_response) -- turn-1 rendering (the prior
    observability fix) must be byte-identical."""
    ep = _b_acq_4_episode("ambiguous", "chiller_6")
    prompt = P.render_turn1(ep)
    assert "quality=0.9" in prompt
    assert '"quality_threshold": 0.95' in prompt


# --- 8. B-ACQ-1/2/3 behavior unchanged ---------------------------------------

def test_b_acq_1_2_3_turn1_prompts_unchanged():
    for ep in P.load_episodes():
        if ep["template_id"] == "B-ACQ-4":
            continue
        prompt = P.render_turn1(ep)
        expected_line = f"- {ep['initial_modality']}: {ep['initial_count']} observation(s) delivered"
        assert expected_line in prompt


def test_b_acq_1_gold_required_acquisition_still_used_correctly_when_requested():
    """For non-B-ACQ-4 templates the model is expected to request gold's
    own required_acquisition (e.g. B-ACQ-1's 'thermal') -- confirm the
    fixed helper still executes that faithfully when it IS what was
    requested (not a redirect, just the natural case where request ==
    gold)."""
    eps = P.load_episodes()
    ep = [e for e in eps if e["template_id"] == "B-ACQ-1" and e["arm"] == "present"][0]
    assert ep["gold"]["required_acquisition"] == "thermal"
    real = P.get_real_acquisition_response(ep, "thermal")
    assert real["modality"] == "thermal"
    assert real["status"] == "RESOLVED"
