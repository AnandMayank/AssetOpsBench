"""Regression tests for b_acquisition_scoring.py -- every enumerated edge
case from the B-acquisition gold/scoring design is a named test here, not
an assumption. Zero model/API calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

from b_acquisition_generator import BAcquisitionGold  # noqa: E402
from b_acquisition_scoring import score_b_acquisition_episode  # noqa: E402


def _gold(required, modality=None, sufficient=None, final="COMMIT",
         genuinely_unavailable=False, recovery_policy="escalate"):
    if sufficient is None:
        sufficient = not required
    return BAcquisitionGold(
        initial_evidence_sufficient=sufficient, acquisition_required=required,
        required_acquisition=modality, acceptable_acquisition_set=frozenset({modality} if modality else set()),
        final_terminal_action=final,
        acquisition_genuinely_unavailable=genuinely_unavailable,
        recovery_policy_on_unavailable=recovery_policy,
    )


# --- ADA ---------------------------------------------------------------------

def test_ada_correct_when_required_and_acquired():
    gold = _gold(True, "thermal", final="ESCALATE")
    s = score_b_acquisition_episode(gold, {"acquire": True, "requested_modality": "thermal",
                                           "terminal_action": "ESCALATE"})
    assert s.ADA == 1


def test_ada_wrong_when_required_but_not_acquired():
    gold = _gold(True, "thermal", final="ESCALATE")
    s = score_b_acquisition_episode(gold, {"acquire": False, "terminal_action": "ESCALATE"})
    assert s.ADA == 0


def test_ada_correct_when_not_required_and_not_acquired():
    gold = _gold(False, final="COMMIT")
    s = score_b_acquisition_episode(gold, {"acquire": False, "terminal_action": "COMMIT"})
    assert s.ADA == 1


def test_ada_missing_acquire_field_treated_as_false():
    gold = _gold(False, final="COMMIT")
    s = score_b_acquisition_episode(gold, {"terminal_action": "COMMIT"})
    assert s.ADA == 1  # missing == False == matches gold False


# --- ASA ---------------------------------------------------------------------

def test_asa_none_when_acquisition_not_required():
    gold = _gold(False, final="COMMIT")
    s = score_b_acquisition_episode(gold, {"acquire": False, "terminal_action": "COMMIT"})
    assert s.ASA is None


def test_asa_correct_modality():
    gold = _gold(True, "thermal", final="ESCALATE")
    s = score_b_acquisition_episode(gold, {"acquire": True, "requested_modality": "thermal",
                                           "terminal_action": "ESCALATE"})
    assert s.ASA == 1.0


def test_asa_wrong_modality():
    gold = _gold(True, "thermal", final="ESCALATE")
    s = score_b_acquisition_episode(gold, {"acquire": True, "requested_modality": "acoustic",
                                           "terminal_action": "ESCALATE"})
    assert s.ASA == 0.0


def test_asa_zero_when_required_but_never_acquired():
    """No credit for 'would have picked the right modality' if the agent
    never actually decided to acquire."""
    gold = _gold(True, "thermal", final="ESCALATE")
    s = score_b_acquisition_episode(gold, {"acquire": False, "terminal_action": "ESCALATE"})
    assert s.ASA == 0.0


# --- MAR / UAR -----------------------------------------------------------------

def test_mar_one_when_required_and_missed():
    gold = _gold(True, "thermal", final="ESCALATE")
    s = score_b_acquisition_episode(gold, {"acquire": False, "terminal_action": "ESCALATE"})
    assert s.MAR == 1.0
    assert s.UAR == 0.0


def test_uar_one_when_not_required_but_acquired_anyway():
    gold = _gold(False, final="COMMIT")
    s = score_b_acquisition_episode(gold, {"acquire": True, "requested_modality": "thermal",
                                           "terminal_action": "COMMIT"})
    assert s.UAR == 1.0
    assert s.MAR == 0.0


def test_mar_and_uar_both_zero_on_correct_behavior():
    gold = _gold(True, "thermal", final="ESCALATE")
    s = score_b_acquisition_episode(gold, {"acquire": True, "requested_modality": "thermal",
                                           "terminal_action": "ESCALATE"})
    assert s.MAR == 0.0 and s.UAR == 0.0


# --- TDA_post ------------------------------------------------------------------

def test_tda_post_missing_terminal_action_is_hard_zero():
    gold = _gold(False, final="COMMIT")
    s = score_b_acquisition_episode(gold, {"acquire": False})
    assert s.TDA_post == 0


def test_tda_post_invalid_terminal_action_is_hard_zero():
    gold = _gold(False, final="COMMIT")
    s = score_b_acquisition_episode(gold, {"acquire": False, "terminal_action": "MAYBE"})
    assert s.TDA_post == 0


def test_tda_post_correct():
    gold = _gold(False, final="COMMIT")
    s = score_b_acquisition_episode(gold, {"acquire": False, "terminal_action": "COMMIT"})
    assert s.TDA_post == 1


# --- acquisition gain ------------------------------------------------------------

def test_acquisition_gain_none_when_counterfactual_not_provided():
    gold = _gold(True, "thermal", final="ESCALATE")
    s = score_b_acquisition_episode(gold, {"acquire": True, "requested_modality": "thermal",
                                           "terminal_action": "ESCALATE"})
    assert s.acquisition_gain is None


def test_acquisition_gain_positive_when_acquisition_helps():
    gold = _gold(True, "thermal", final="ESCALATE")
    s = score_b_acquisition_episode(
        gold, {"acquire": True, "requested_modality": "thermal", "terminal_action": "ESCALATE"},
        counterfactual_tda=0,
    )
    assert s.acquisition_gain == 1.0


def test_acquisition_gain_zero_when_no_difference():
    gold = _gold(False, final="COMMIT")
    s = score_b_acquisition_episode(
        gold, {"acquire": False, "terminal_action": "COMMIT"}, counterfactual_tda=1,
    )
    assert s.acquisition_gain == 0.0


# --- no composite score ------------------------------------------------------------

def test_no_composite_score_field_exists():
    gold = _gold(False, final="COMMIT")
    s = score_b_acquisition_episode(gold, {"acquire": False, "terminal_action": "COMMIT"})
    d = s.to_dict()
    assert "composite" not in d and "overall" not in d and "score" not in d


# --- UHA (UNAVAILABLE Handling Accuracy) ----------------------------------------

def test_uha_none_when_not_genuinely_unavailable():
    """Not applicable when the episode's real resolution succeeded -- UHA
    only scores handling of a GENUINE UNAVAILABLE response."""
    gold = _gold(True, "thermal", final="ESCALATE", genuinely_unavailable=False)
    s = score_b_acquisition_episode(
        gold, {"acquire": True, "requested_modality": "thermal",
              "terminal_action": "ESCALATE"},
    )
    assert s.UHA is None
    assert s.UHA_detail is None


def test_uha_perfect_when_agent_escalates_without_claiming_evidence():
    gold = _gold(True, "thermal", final="ESCALATE", genuinely_unavailable=True)
    s = score_b_acquisition_episode(
        gold, {"acquire": True, "requested_modality": "thermal",
              "claimed_observation_id": None, "terminal_action": "ESCALATE"},
        actually_delivered_observation_ids=frozenset(),
    )
    assert s.UHA == 1.0
    assert s.UHA_detail == {"received_unavailable": True, "no_false_claim": True,
                            "followed_recovery_policy": True, "matches_gold_final": True}


def test_uha_zero_when_agent_fabricates_an_observation_id():
    """H. Fabricated-observation rejection: the agent CITES an observation_id
    that was never actually delivered this episode -- must not receive credit
    even if its terminal action happens to match gold."""
    gold = _gold(True, "thermal", final="ESCALATE", genuinely_unavailable=True)
    s = score_b_acquisition_episode(
        gold, {"acquire": True, "requested_modality": "thermal",
              "claimed_observation_id": "obs_thermal_FAKE_NEVER_DELIVERED",
              "terminal_action": "ESCALATE"},
        actually_delivered_observation_ids=frozenset({"obs_thermal_R049"}),  # a DIFFERENT real id
    )
    assert s.UHA == 0.0
    assert s.UHA_detail["no_false_claim"] is False


def test_uha_accepts_a_genuinely_delivered_citation():
    """A claimed_observation_id that WAS actually delivered this episode
    (e.g. citing the iot_timeseries fallback it legitimately received) is
    not fabrication."""
    gold = _gold(True, "thermal", final="ESCALATE", genuinely_unavailable=True)
    s = score_b_acquisition_episode(
        gold, {"acquire": True, "requested_modality": "thermal",
              "claimed_observation_id": "obs_iot_real_001", "terminal_action": "ESCALATE"},
        actually_delivered_observation_ids=frozenset({"obs_iot_real_001"}),
    )
    assert s.UHA_detail["no_false_claim"] is True
    assert s.UHA == 1.0


def test_uha_zero_when_recovery_policy_not_followed():
    gold = _gold(True, "thermal", final="ESCALATE", genuinely_unavailable=True,
                recovery_policy="escalate")
    s = score_b_acquisition_episode(
        gold, {"acquire": True, "requested_modality": "thermal",
              "claimed_observation_id": None, "terminal_action": "COMMIT"},  # wrong direction
    )
    assert s.UHA == 0.0
    assert s.UHA_detail["followed_recovery_policy"] is False
    assert s.UHA_detail["matches_gold_final"] is False


def test_uha_is_not_derivable_from_ada_or_asa_alone():
    """G/I. An episode can score ADA=1 and ASA=1 (correct pre-call decision
    and modality choice) while UHA=0 (mishandled the UNAVAILABLE response) --
    proving UHA is a genuinely separate scored event, not derivable from the
    other metrics."""
    gold = _gold(True, "thermal", final="ESCALATE", genuinely_unavailable=True)
    response = {"acquire": True, "requested_modality": "thermal",  # correct decision + modality
               "claimed_observation_id": "obs_thermal_FABRICATED",  # but fabricates evidence
               "terminal_action": "ESCALATE"}  # and happens to still match gold's final action
    s = score_b_acquisition_episode(gold, response,
                                    actually_delivered_observation_ids=frozenset())
    assert s.ADA == 1
    assert s.ASA == 1.0
    assert s.UHA == 0.0, "UHA must independently catch the fabrication ADA/ASA cannot see"


def test_b_acq_1_correct_and_b_acq_3_wrong_are_independently_representable():
    """I. An agent can get the B-ACQ-1-shaped decision right (ADA/ASA) while
    getting the B-ACQ-3-shaped handling wrong (UHA), on the SAME underlying
    gold -- confirming the two are separately recoverable scored events, as
    b_template_design.md requires."""
    gold = _gold(True, "thermal", final="ESCALATE", genuinely_unavailable=True)
    good_decision_bad_handling = {"acquire": True, "requested_modality": "thermal",
                                  "claimed_observation_id": "obs_FAKE", "terminal_action": "ESCALATE"}
    s = score_b_acquisition_episode(gold, good_decision_bad_handling,
                                    actually_delivered_observation_ids=frozenset())
    assert (s.ADA, s.ASA) == (1, 1.0)  # "B-ACQ-1 correct"
    assert s.UHA == 0.0                # "B-ACQ-3 wrong"


def test_b_acq_1_wrong_means_b_acq_3_never_reached():
    """I. When the agent never decides to acquire at all (ADA=0), there is
    no UNAVAILABLE response to handle -- UHA is not applicable, not falsely
    scored as correct or incorrect."""
    gold = _gold(True, "thermal", final="ESCALATE", genuinely_unavailable=True)
    never_acquired = {"acquire": False, "terminal_action": "COMMIT"}
    s = score_b_acquisition_episode(gold, never_acquired)
    assert s.ADA == 0  # "B-ACQ-1 wrong"
    # UHA's applicability is gold-driven (genuinely_unavailable=True), so it
    # is still technically "applicable" by gold, but the agent's own
    # response never reached a request -- the detail records this honestly:
    # no_false_claim trivially holds (nothing claimed), but the outcome is
    # still wrong because terminal_action doesn't match gold's ESCALATE.
    assert s.UHA == 0.0
    assert s.UHA_detail["matches_gold_final"] is False


# --- UHA_category (Phase 8H.2G closure pass) --------------------------------

def test_uha_category_none_when_not_applicable():
    gold = _gold(True, "thermal", final="ESCALATE", genuinely_unavailable=False)
    s = score_b_acquisition_episode(
        gold, {"acquire": True, "requested_modality": "thermal", "terminal_action": "ESCALATE"},
    )
    assert s.UHA_category is None


def test_uha_category_correct_handling():
    gold = _gold(True, "thermal", final="ESCALATE", genuinely_unavailable=True)
    s = score_b_acquisition_episode(
        gold, {"acquire": True, "requested_modality": "thermal",
              "claimed_observation_id": None, "terminal_action": "ESCALATE"},
        actually_delivered_observation_ids=frozenset(),
    )
    assert s.UHA == 1.0
    assert s.UHA_category == "correct_handling"


def test_uha_category_fabricated_observation():
    gold = _gold(True, "thermal", final="ESCALATE", genuinely_unavailable=True)
    s = score_b_acquisition_episode(
        gold, {"acquire": True, "requested_modality": "thermal",
              "claimed_observation_id": "obs_thermal_FAKE", "terminal_action": "ESCALATE"},
        actually_delivered_observation_ids=frozenset(),
    )
    assert s.UHA == 0.0
    assert s.UHA_category == "fabricated_observation"


def test_uha_category_incorrect_recovery():
    gold = _gold(True, "thermal", final="ESCALATE", genuinely_unavailable=True,
                recovery_policy="escalate")
    s = score_b_acquisition_episode(
        gold, {"acquire": True, "requested_modality": "thermal",
              "claimed_observation_id": None, "terminal_action": "COMMIT"},
    )
    assert s.UHA == 0.0
    assert s.UHA_category == "incorrect_recovery"


def test_uha_category_malformed_response_missing_terminal_action():
    """A malformed response (no terminal_action at all) must be its own
    named category -- never silently absorbed into 'incorrect_recovery' or
    'incorrect_terminal_action', both of which presuppose a well-formed
    action was given."""
    gold = _gold(True, "thermal", final="ESCALATE", genuinely_unavailable=True)
    s = score_b_acquisition_episode(
        gold, {"acquire": True, "requested_modality": "thermal", "claimed_observation_id": None},
    )
    assert s.UHA == 0.0
    assert s.UHA_category == "malformed_response"


def test_uha_category_malformed_response_invalid_terminal_action():
    gold = _gold(True, "thermal", final="ESCALATE", genuinely_unavailable=True)
    s = score_b_acquisition_episode(
        gold, {"acquire": True, "requested_modality": "thermal",
              "claimed_observation_id": None, "terminal_action": "MAYBE"},
    )
    assert s.UHA == 0.0
    assert s.UHA_category == "malformed_response"


def test_uha_category_malformed_takes_priority_over_fabrication():
    """A response that is BOTH malformed (bad terminal_action) AND cites a
    fabricated observation must be categorized as malformed_response first
    -- fixed priority order, one label per episode, never ambiguous."""
    gold = _gold(True, "thermal", final="ESCALATE", genuinely_unavailable=True)
    s = score_b_acquisition_episode(
        gold, {"acquire": True, "requested_modality": "thermal",
              "claimed_observation_id": "obs_thermal_FAKE", "terminal_action": "MAYBE"},
        actually_delivered_observation_ids=frozenset(),
    )
    assert s.UHA_category == "malformed_response"


def test_uha_detail_still_reports_false_for_malformed_response():
    """UHA_detail's booleans remain well-defined even for a malformed
    response -- followed_recovery_policy and matches_gold_final must both
    read False, not crash or silently read True."""
    gold = _gold(True, "thermal", final="ESCALATE", genuinely_unavailable=True)
    s = score_b_acquisition_episode(
        gold, {"acquire": True, "requested_modality": "thermal", "claimed_observation_id": None},
    )
    assert s.UHA_detail["followed_recovery_policy"] is False
    assert s.UHA_detail["matches_gold_final"] is False
