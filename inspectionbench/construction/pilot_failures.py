"""pilot_failures.py -- Phase 8H.1 multi-model pilot: failure classes + retry policy.

Three mutually-exclusive classes, using the EXISTING error vocabulary
(``scripts/run_generated_pilot._chat`` prefixes + ``apparatus.APPARATUS_CATEGORIES``)
-- no new taxonomy is invented here.

  INFRASTRUCTURE  transient: timeout, transport, connection, serialization.
                  -> retry the SAME episode_id, max 3, backoff 5/20/60s.
                     After 3 -> hard STOP, artifact preserved.

  SEMANTIC        benchmark/apparatus is wrong: missing required observation,
                  fixture not applied, KeyError on SCENARIO_PHYSICAL/FIXTURES,
                  assertion failure, world-hash drift, manifest SHA drift,
                  unexpected runner return structure.
                  -> STOP the entire pilot immediately. Do NOT repair mid-flight.

  MODEL_BEHAVIOR  wrong answer, grounding failure, refusal, odd tool choice,
                  parse_error / no_answer from a responsive endpoint.
                  -> NOT a failure. Recorded as data, scored normally.
                     Apparatus-failure rows are still excluded from rate
                     denominators (MetricSpec E) and counted separately, but
                     are never silently retried.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apparatus import APPARATUS_CATEGORIES  # noqa: E402

INFRASTRUCTURE = "INFRASTRUCTURE"
SEMANTIC = "SEMANTIC"
MODEL_BEHAVIOR = "MODEL_BEHAVIOR"

MAX_RETRIES = 3
BACKOFF_SECONDS = (5, 20, 60)

#: _chat error-string prefixes that mean the transport itself failed.
_INFRA_CHAT_PREFIXES = ("call_error:",)
#: _chat prefixes that mean the endpoint answered but the body was unusable --
#: model behaviour (possibly apparatus), never an infra retry.
_BEHAVIOR_CHAT_PREFIXES = ("parse_error:", "no_answer:")

#: exception types raised by the runners / dispatch that indicate the benchmark
#: definition is inconsistent -- always a hard STOP.
_SEMANTIC_EXC_NAMES = frozenset({
    "ManifestIntegrityError", "DispatchError", "KeyError", "AssertionError",
    "FileNotFoundError", "UnpairedDeltaError",
})


@dataclass
class FailureVerdict:
    cls: str                       # INFRASTRUCTURE | SEMANTIC | MODEL_BEHAVIOR
    retryable: bool
    stop_pilot: bool
    reason: str
    is_apparatus_failure: bool = False


def classify_exception(exc: BaseException) -> FailureVerdict:
    """An exception escaped a runner / dispatch call."""
    name = type(exc).__name__
    msg = f"{name}: {exc}"
    if name in _SEMANTIC_EXC_NAMES:
        return FailureVerdict(SEMANTIC, retryable=False, stop_pilot=True, reason=msg)
    lowered = str(exc).lower()
    if any(k in lowered for k in ("timeout", "timed out", "connection", "reset by peer",
                                  "temporarily unavailable", "json")):
        return FailureVerdict(INFRASTRUCTURE, retryable=True, stop_pilot=False, reason=msg)
    # unknown exception shape from inside a runner -> treat as semantic, STOP.
    return FailureVerdict(SEMANTIC, retryable=False, stop_pilot=True,
                          reason=f"unclassified runner exception -> STOP: {msg}")


def classify_call_errors(call_errors: List[str]) -> FailureVerdict:
    """A runner returned normally but carried ``errors`` / ``call_errors``.

    An empty list is a clean episode.
    """
    if not call_errors:
        return FailureVerdict(MODEL_BEHAVIOR, retryable=False, stop_pilot=False,
                              reason="no errors")
    joined = " | ".join(str(e) for e in call_errors)
    if any(str(e).startswith(p) for e in call_errors for p in _INFRA_CHAT_PREFIXES):
        return FailureVerdict(INFRASTRUCTURE, retryable=True, stop_pilot=False,
                              reason=f"transport failure in _chat: {joined}")
    if any(str(e).startswith(p) for e in call_errors for p in _BEHAVIOR_CHAT_PREFIXES):
        # endpoint answered, body unusable -> model behaviour, flag apparatus so
        # it drops out of rate denominators but is never auto-retried.
        return FailureVerdict(MODEL_BEHAVIOR, retryable=False, stop_pilot=False,
                              reason=f"unparseable/empty model output: {joined}",
                              is_apparatus_failure=True)
    return FailureVerdict(MODEL_BEHAVIOR, retryable=False, stop_pilot=False,
                          reason=f"other recorded error, treated as behaviour: {joined}")


def is_apparatus_category(cat: Optional[str]) -> bool:
    return cat in APPARATUS_CATEGORIES


def validate_runner_return(kind: str, ret: Dict[str, Any]) -> FailureVerdict:
    """Structural check on a runner's return dict. A missing REQUIRED field is a
    SEMANTIC stop -- the persisted artifact must be reconstructable."""
    required = {
        "a_episode": ("world_id", "arm", "gold", "verdict", "metric", "call_errors"),
        "b_incumbent": ("scenario_id", "fm", "gold", "arm", "verdict", "scores", "trace"),
        "classc": ("scenario_id", "fm", "gold", "verdict", "executed_order", "trace", "CC"),
        "classd": ("scenario_id", "fm", "gold", "verdict", "executed_tools", "trace", "CC"),
        "e_sequence": ("sequence_id", "episodes", "trace"),
    }[kind]
    missing = [k for k in required if k not in ret]
    if missing:
        return FailureVerdict(SEMANTIC, retryable=False, stop_pilot=True,
                              reason=f"{kind} runner return missing {missing} -- STOP")
    if kind == "e_sequence" and len(ret["episodes"]) != 3:
        return FailureVerdict(SEMANTIC, retryable=False, stop_pilot=True,
                              reason=f"e_sequence returned {len(ret['episodes'])} episodes, expected 3")
    return FailureVerdict(MODEL_BEHAVIOR, retryable=False, stop_pilot=False, reason="structure ok")
