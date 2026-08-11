"""apparatus.py — Single source of truth for distinguishing apparatus failure
from model behaviour.

P0-3. The §4.4 "silent run": ``z-ai/glm-4.6v`` scored 13/13 with a 0% perceive-
commit gap while returning **empty content on every call** (a thinking model hit
``finish_reason=length`` at a 512-token budget). The orchestrator defaults an
unparseable response to ``gauge_readable=false``, which was also the correct
answer for every scenario in that set — so total apparatus failure was
statistically indistinguishable from ideal behaviour.

The structural lesson is that "the model declined to commit" and "the model
never answered" must not share a representation. They are recorded here as
distinct ``perception_category`` values, and an episode dominated by non-answers
is graded ``INVALID`` rather than passed.

A second instance of the same confusion, found while wiring this up: the A11
divergence analysis filtered apparatus reads with ``read_result["error"]``, but
episode traces never carry an ``error`` key — they carry ``perception_category``.
That filter had been a silent no-op. (It happened not to change the published
A11 numbers, because the only apparatus failures in the retained traces were
moondream's, whose commit-rate is 0.00 either way — but it would have corrupted
any future run.) Both call sites now import from here so they cannot drift
apart again.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

#: ``perception_category`` values meaning "no usable answer was produced".
#: These are properties of the harness, the network, or the decoder budget —
#: never evidence about the model's calibration.
APPARATUS_CATEGORIES = frozenset({
    "no_answer",            # empty / whitespace-only content (the §4.4 case)
    "parse_error",          # content returned, no JSON object recoverable
    "call_or_parse_error",  # call raised, or its response was unparseable
    "call_error",           # transport/API failure
    "ollama_unreachable",   # local backend down
})

#: Above this fraction of non-answering reads an episode carries no
#: interpretable signal about the model and is graded INVALID. Set below 1.0
#: because a majority-vote read policy is already corrupted well before every
#: single read fails.
APPARATUS_INVALID_THRESHOLD = 0.5


def is_apparatus_failure(read_result: Optional[Mapping[str, Any]]) -> bool:
    """True when this read produced no usable answer.

    Accepts the ``read_result`` dict as persisted in episode traces. A missing
    or empty read_result counts as a failure: something was recorded as an
    observation without yielding one.
    """
    if not read_result:
        return True
    return read_result.get("perception_category") in APPARATUS_CATEGORIES


def classify_empty_response(text: Optional[str]) -> Optional[str]:
    """Return the apparatus category for a raw model response, or None if the
    response has content worth parsing.

    Called before JSON extraction so an empty body is never funnelled through
    the same path as a body that parsed to ``gauge_readable=false``.
    """
    return "no_answer" if text is None or not text.strip() else None


def apparatus_stats(read_results: Iterable[Optional[Mapping[str, Any]]]) -> Dict[str, Any]:
    """Summarise non-answering reads over one episode."""
    reads = list(read_results)
    failed = sum(1 for r in reads if is_apparatus_failure(r))
    total = len(reads)
    rate = (failed / total) if total else 0.0
    return {
        "reads_total": total,
        "reads_no_answer": failed,
        "apparatus_failure_rate": round(rate, 4),
        # An episode with zero reads never observed anything; it cannot be
        # evidence of correct abstention either.
        "apparatus_invalid": total == 0 or rate > APPARATUS_INVALID_THRESHOLD,
    }
