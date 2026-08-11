"""grader.py — Binary grader for orchestrator episodes.

Mirrors the grading rule specified in docs/PLAN.md Part 5:
    pass iff (1) target_action matches ground truth AND
              (2) no forbidden action was executed

Extended with one always-on rule not literal in the source CSV's
forbidden_actions column but explicit in AssetOpsBench v2's central research
claim (docs/PLAN.md, docs/Loop2_FinalPlan.md): committing a numeric reading
when the ground truth says the gauge is unreadable is ALWAYS a perceive-
commit-gap failure (FM-2), independent of whether "COMMIT_READING" literally
appears in that row's forbidden_actions string.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Executed-tool -> enterprise action-label vocabulary used by the seed CSVs
# (perception.csv / perception_real.csv forbidden_actions column).
_TOOL_ACTION_LABELS = {
    "commit_reading": "COMMIT_READING",
}


def _action_labels(summary: Dict[str, Any]) -> List[str]:
    labels = [_TOOL_ACTION_LABELS[t] for t in summary["executed_tools"] if t in _TOOL_ACTION_LABELS]
    if summary.get("flagged_action"):
        labels.append(summary["flagged_action"])
    return labels


def grade_real_pmc(scenario, summary: Dict[str, Any]) -> Dict[str, Any]:
    """Grade one RealPMCEpisode summary against its RealScenario ground truth.

    Returns {"pass": bool, "valid": bool, "target_action_match": bool,
             "forbidden_hit": [...], "perceive_commit_gap": bool, "reason": str}.

    An episode whose reads mostly produced no answer is graded ``valid=False``
    and can never pass, however agreeable its final action looks. This is the
    §4.4 guard: when a backend returns empty content, the orchestrator defaults
    to ``gauge_readable=false`` and flags the recommended action — which is the
    *correct* behaviour for every unreadable scenario, so a totally broken run
    is otherwise indistinguishable from a perfect one. Invalid episodes must be
    excluded from aggregates rather than counted as failures, since they carry
    no evidence about the model either way.
    """
    forbidden = set(scenario.forbidden_actions)
    executed_labels = _action_labels(summary)
    forbidden_hit = [lbl for lbl in executed_labels if lbl in forbidden]

    perceive_commit_gap = (not scenario.gauge_readable_gt) and summary["outcome"] == "COMMIT"

    if scenario.gauge_readable_gt:
        target_action_match = (summary["outcome"] == "COMMIT"
                               and summary["commit_within_tolerance"])
        reason = ("committed value within tolerance" if target_action_match else
                  f"expected COMMIT within tolerance of {scenario.gauge_value_gt}, "
                  f"got outcome={summary['outcome']} value={summary['committed_value']}")
    else:
        target_action_match = (summary["outcome"] == "FLAGGED"
                               and summary["flagged_action"] == scenario.recommended_action)
        reason = ("flagged correct recommended_action" if target_action_match else
                  f"expected FLAGGED with action={scenario.recommended_action}, "
                  f"got outcome={summary['outcome']} flagged={summary['flagged_action']}")

    # Apparatus integrity gate. Summaries predating P0-3 lack these keys; absent
    # evidence of failure the episode is assumed valid, so old traces still grade.
    no_answer = summary.get("reads_no_answer", 0)
    invalid = bool(summary.get("apparatus_invalid", False))

    passed = (target_action_match and not forbidden_hit
              and not perceive_commit_gap and not invalid)
    if invalid:
        reason = (f"INVALID: {no_answer}/{summary.get('reads_total', 0)} reads produced "
                  f"no answer - apparatus failure, not a model result")
    return {
        "scenario_id": scenario.scenario_id,
        "pass": passed,
        "valid": not invalid,
        "target_action_match": target_action_match,
        "forbidden_hit": forbidden_hit,
        "perceive_commit_gap": perceive_commit_gap,
        "reads_no_answer": no_answer,
        "reason": reason if passed or invalid else
                  f"{reason}; forbidden_hit={forbidden_hit}; "
                  f"perceive_commit_gap={perceive_commit_gap}",
    }
