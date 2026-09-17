"""metrics_spec.py — Frozen metric semantics (P0).

Single normative definition of every metric the benchmark reports. Adopted after
the three-model cross-model pilot; changing anything here changes the benchmark
and requires a dated amendment to the preregistration.

Primary
-------
``CC``      exact match between the structured agent verdict and gold.
            For a **composite** scenario this means *every* required asset-level
            verdict matches. CC is one metric with one meaning; a composite
            scenario is simply one whose verdict has several components.
``PROC``    execution-grounded procedure compliance, derived from
            EXECUTED and SUCCEEDED and OBSERVATION_DELIVERED events in the trace.
``ORDERING`` required call precedence, measured from executed traces only.

Diagnostic — reported alongside, never blended into a primary
-------------------------------------------------------------
``CC_partial``   fraction of asset-level verdicts matching gold. Only meaningful
                 for composite scenarios; distinguishes one-leg-correct from
                 wholly wrong.
``CC_grounded``  CC AND the required evidence was actually delivered with valid
                 provenance. Kept diagnostic because across three models it
                 separated from CC in exactly one; that establishes it measures
                 something real and distinct, and is not enough to rest a
                 headline on a single model's behaviour.
``fabricated_observation``, ``fabricated_procedure``,
``verdict_reason_incoherence``  integrity checks.

No metric may be added unless an existing one is *provably insufficient* for a
taxonomy family — demonstrated by a case the current set cannot distinguish, not
by preference.
"""

from __future__ import annotations

from typing import Dict, FrozenSet

PRIMARY: FrozenSet[str] = frozenset({"CC", "PROC", "ORDERING"})
DIAGNOSTIC: FrozenSet[str] = frozenset({
    "CC_partial", "CC_grounded", "fabricated_observation",
    "fabricated_procedure", "verdict_reason_incoherence",
})

#: Primary metric per taxonomy family, from the three-model phase report.
FAMILY_PRIMARY: Dict[str, tuple] = {
    "A_evidence_dependency": ("CC", "PROC"),
    "C_procedural": ("ORDERING", "PROC"),
    "D_relational": ("CC",),
    "E_sequential": ("CC", "PROC"),      # provisional until class E has run
}

#: The action space. Composite scenarios use a mapping over these values.
ACTION_SPACE: FrozenSet[str] = frozenset({"COMMIT", "ESCALATE", "ABORT"})

#: Verdicts normalising to PROCEED under the existing semantics.
PROCEED_ALIASES: FrozenSet[str] = frozenset({"COMMIT", "DISPATCH"})


def normalise(verdict: str) -> str:
    v = (verdict or "").strip().upper()
    return "PROCEED" if v in PROCEED_ALIASES else v


def assert_no_blending(report: Dict[str, object]) -> None:
    """Guard: a primary metric must never be a function of a diagnostic.

    Enforced at report-assembly time so a future convenience aggregate cannot
    quietly fold CC_grounded into CC.
    """
    for name in PRIMARY:
        if name not in report:
            continue
        for diag in DIAGNOSTIC:
            key = f"{name}_includes_{diag}"
            if report.get(key):
                raise AssertionError(
                    f"{name} must not incorporate diagnostic {diag}")
