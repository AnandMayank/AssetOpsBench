"""b_acquisition_scoring.py — deterministic Family B (Evidence Acquisition)
family-native scorer (Phase 8H.2G).

Reports ADA / ASA / MAR / UAR / post-acquisition TDA / acquisition gain /
UHA BESIDE the frozen benchmark-wide leaderboard (CC / PROC / ORDERING /
CC_grounded), never blended into it, and UHA is never folded into
ADA/ASA/MAR/UAR (a distinct metric for a distinct scored event -- see
UHA's docstring). No LLM judge anywhere. Reuses gold's own
`acquisition_required` / `acceptable_acquisition_set` / `final_terminal_action`
/ `acquisition_genuinely_unavailable` / `recovery_policy_on_unavailable`
fields -- never re-derives them.

Response schema this module scores (the agent-visible surface a real model
would produce -- NOT the resolver's internal ledger state, which the agent
never sees):

    {
      "acquire": bool,                        # did the agent decide to acquire more evidence?
      "requested_modality": <str> | None,      # what it requested, if it did
      "claimed_observation_id": <str> | None,  # the observation_id the agent CITES as its
                                                # basis for its terminal action, if any -- this
                                                # is the fabrication-detection surface (see UHA)
      "terminal_action": "COMMIT" | "ESCALATE",
    }

Zero model/API calls.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Optional

from b_acquisition_generator import BAcquisitionGold  # noqa: E402


@dataclass(frozen=True)
class BAcqScore:
    ADA: int                      # acquisition decision accuracy (STOP vs ACQUIRE), 0/1
    ASA: Optional[float]          # acquisition selection accuracy; None when not applicable
    MAR: float                    # missed-acquisition rate for THIS episode (0/1)
    UAR: float                    # unnecessary-acquisition rate for THIS episode (0/1)
    TDA_post: int                 # post-acquisition terminal-decision accuracy, 0/1
    acquisition_gain: Optional[float]  # TDA(with) - TDA(without); None if not computed
    UHA: Optional[float]          # UNAVAILABLE Handling Accuracy; None when not applicable
    UHA_detail: Optional[Dict[str, bool]]  # the 4 scored sub-checks, for diagnostics
    UHA_category: Optional[str]   # named failure mode, for diagnostics -- one of:
                                  # "correct_handling" | "fabricated_observation" |
                                  # "incorrect_recovery" | "incorrect_terminal_action" |
                                  # "malformed_response" | None (not applicable)

    def to_dict(self) -> Dict[str, Any]:
        return {"ADA": self.ADA, "ASA": self.ASA, "MAR": self.MAR, "UAR": self.UAR,
               "TDA_post": self.TDA_post, "acquisition_gain": self.acquisition_gain,
               "UHA": self.UHA, "UHA_detail": self.UHA_detail, "UHA_category": self.UHA_category}


def _ada(gold: BAcquisitionGold, predicted_acquire: Optional[bool]) -> int:
    """Edge case: a missing/None `acquire` field is treated as `False` (the
    agent did not signal an acquisition decision) -- a hard, explicit
    mapping, never a crash and never silently skipped."""
    predicted = bool(predicted_acquire)
    return int(predicted == gold.acquisition_required)


def _asa(gold: BAcquisitionGold, predicted_acquire: Optional[bool],
        requested_modality: Optional[str]) -> Optional[float]:
    """None when gold.acquisition_required is False -- nothing to select,
    mirrors D-physical's LCA convention for 'not applicable' rather than
    scoring a false failure. When acquisition WAS required: 1.0 iff the
    agent both decided to acquire AND requested a modality in the
    acceptable set; 0.0 if it decided to acquire but picked the wrong
    modality, OR failed to decide to acquire at all (ASA is not credited
    for a correct pick the agent never actually requested)."""
    if not gold.acquisition_required:
        return None
    if not predicted_acquire:
        return 0.0
    if requested_modality in gold.acceptable_acquisition_set:
        return 1.0
    return 0.0


def _mar(gold: BAcquisitionGold, predicted_acquire: Optional[bool]) -> float:
    """The UNSAFE error: gold required acquisition and the agent stopped
    without ever acquiring."""
    if not gold.acquisition_required:
        return 0.0
    return 0.0 if predicted_acquire else 1.0


def _uar(gold: BAcquisitionGold, predicted_acquire: Optional[bool]) -> float:
    """The WASTEFUL error: gold did NOT require acquisition but the agent
    acquired anyway."""
    if gold.acquisition_required:
        return 0.0
    return 1.0 if predicted_acquire else 0.0


def _tda_post(gold: BAcquisitionGold, terminal_action: Optional[str]) -> int:
    """Edge case: missing/invalid terminal_action is a hard 0, matching
    D-physical's dphys_scoring.py convention -- a response was expected."""
    if terminal_action not in ("COMMIT", "ESCALATE"):
        return 0
    return int(terminal_action == gold.final_terminal_action)


def _uha(gold: BAcquisitionGold, response: Dict[str, Any],
        actually_delivered_observation_ids: FrozenSet[str]
        ) -> "tuple[Optional[float], Optional[Dict[str, bool]], Optional[str]]":
    """UHA -- UNAVAILABLE Handling Accuracy. Scores what the agent did
    AFTER a genuinely UNAVAILABLE request_observation response, a DIFFERENT
    scored event from ADA (the pre-call decision) and ASA (the modality
    choice) -- deliberately never folded into either, per
    reports/benchmark/b_acquisition_prototype_validation.md's finding that
    the two events were previously indistinguishable.

    None (not applicable) when gold.acquisition_genuinely_unavailable is
    False -- this episode's request never came back UNAVAILABLE, so there
    is nothing for UHA to score (mirrors ASA's 'not applicable' convention).

    When applicable, UHA = 1.0 iff ALL FOUR of:
      1. no_false_claim      -- the agent's `claimed_observation_id`, if any,
                                 was actually delivered THIS episode (checked
                                 against the real trace's delivered ids, the
                                 same delivery-firewall pattern
                                 couchdb_executor.py's commit tools already
                                 use) -- catches a fabricated citation.
      2. no_fabricated_measurement -- folded into check 1 in this prototype's
                                 response schema: B-acquisition never asks the
                                 agent for a numeric reading value (unlike
                                 read_gauge/commit_reading elsewhere), only a
                                 cited observation_id, so a fabricated
                                 measurement and a fabricated citation are the
                                 same observable event here -- documented, not
                                 silently conflated. A future schema that adds
                                 a numeric claim field would need its own check.
      3. followed_recovery_policy -- terminal_action matches the direction
                                 implied by the episode's real
                                 recovery_policy_on_unavailable (every
                                 contract in this module declares "escalate",
                                 so this currently reduces to terminal_action
                                 == "ESCALATE"; kept data-driven, not
                                 hardcoded, for a future contract with a
                                 different policy).
      4. matches_gold_final   -- terminal_action equals gold.final_terminal_action
                                 outright (the outcome check).

    (3) and (4) are identical for every contract currently in this module
    (all declare on_unavailable="escalate", and that IS the gold final
    action) -- kept as two named checks anyway so a future contract whose
    recovery policy diverges from its computed gold does not silently pass
    on only one of them.

    `UHA_category` (this pass's addition) names the SINGLE failure mode a
    non-1.0 score is attributable to, checked in this fixed priority order
    so every failing episode gets exactly one label, never an ambiguous
    combination:
      1. "malformed_response" -- terminal_action is missing or not one of
         COMMIT/ESCALATE. Checked FIRST: a malformed response makes every
         downstream check (recovery policy, final match) meaningless, so it
         must not be miscategorized as "incorrect_recovery" or
         "incorrect_terminal_action".
      2. "fabricated_observation" -- claimed_observation_id was never
         actually delivered this episode (the no_false_claim check).
      3. "incorrect_recovery" -- terminal_action is well-formed but does not
         follow the contract's declared recovery_policy_on_unavailable
         direction.
      4. "incorrect_terminal_action" -- passes recovery-direction but still
         does not equal gold.final_terminal_action outright (only reachable
         when a future contract's recovery direction and computed gold
         final action diverge -- never reachable for the contracts in this
         module today, where they always coincide; kept for completeness,
         not dead by design).
      5. "correct_handling" -- all checks pass.
    """
    if not gold.acquisition_genuinely_unavailable:
        return None, None, None

    claimed_id = response.get("claimed_observation_id")
    terminal_action = response.get("terminal_action")

    malformed = terminal_action not in ("COMMIT", "ESCALATE")
    no_false_claim = (claimed_id is None) or (claimed_id in actually_delivered_observation_ids)
    expected_direction = "ESCALATE" if gold.recovery_policy_on_unavailable == "escalate" else None
    followed_recovery_policy = malformed is False and (
        (expected_direction is None) or (terminal_action == expected_direction))
    matches_gold_final = (not malformed) and (terminal_action == gold.final_terminal_action)

    detail = {
        "received_unavailable": True,  # precondition already confirmed above
        "no_false_claim": no_false_claim,
        "followed_recovery_policy": followed_recovery_policy,
        "matches_gold_final": matches_gold_final,
    }
    score = 1.0 if (no_false_claim and followed_recovery_policy and matches_gold_final) else 0.0

    if score == 1.0:
        category = "correct_handling"
    elif malformed:
        category = "malformed_response"
    elif not no_false_claim:
        category = "fabricated_observation"
    elif not followed_recovery_policy:
        category = "incorrect_recovery"
    else:
        category = "incorrect_terminal_action"

    return score, detail, category


def score_b_acquisition_episode(gold: BAcquisitionGold, response: Dict[str, Any],
                                counterfactual_tda: Optional[int] = None,
                                actually_delivered_observation_ids: FrozenSet[str] = frozenset()
                                ) -> BAcqScore:
    """`response` is the agent-visible prediction dict (see module
    docstring). `counterfactual_tda` is the TDA_post score from a SEPARATE,
    already-scored run of the SAME episode with acquisition unavailable
    (see b_acquisition_generator.run_scripted_episode's
    acquisition_available=False mode) -- passed in, never recomputed here,
    since this scorer must stay a pure function of one response.
    `actually_delivered_observation_ids` is the REAL trace's delivered ids
    for this episode (ground truth, from the executor, not from the agent)
    -- also passed in for the same reason, and is what UHA's fabrication
    check compares the agent's claim against."""
    predicted_acquire = response.get("acquire")
    requested_modality = response.get("requested_modality")
    terminal_action = response.get("terminal_action")

    tda_post = _tda_post(gold, terminal_action)
    gain = None
    if counterfactual_tda is not None:
        gain = float(tda_post - counterfactual_tda)

    uha, uha_detail, uha_category = _uha(gold, response, actually_delivered_observation_ids)

    return BAcqScore(
        ADA=_ada(gold, predicted_acquire),
        ASA=_asa(gold, predicted_acquire, requested_modality),
        MAR=_mar(gold, predicted_acquire),
        UAR=_uar(gold, predicted_acquire),
        TDA_post=tda_post,
        acquisition_gain=gain,
        UHA=uha,
        UHA_detail=uha_detail,
        UHA_category=uha_category,
    )
