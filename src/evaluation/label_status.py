"""label_status.py — Gate metrics on label trustworthiness (V7 enforcement).

V7 measured inter-labeling reliability on the perception catalog and found that
not every ground-truth field clears the κ ≥ 0.6 gate. ``gauge_value`` in
particular sits at κ ≈ 0.18 on the dual-labelled subset, which means **L1 MAE is
not a reportable number** over those scenarios: the target it is measured
against is not reliable enough to distinguish model error from label error.

The registry (``reports/v7/label_status.csv``, produced by
``scripts/v7_label_adjudication.py``) marks each (scenario_id, field) as
``trusted`` or ``contested``. This module is the enforcement point: metric code
asks for the eligible scenario set *before* aggregating, so an ungated metric is
a visible omission rather than a silently wrong number.

Typical use::

    from evaluation.label_status import LabelStatus

    ls = LabelStatus.load()
    eligible = ls.eligible("gauge_value", [s.scenario_id for s in scenarios])
    mae = mean(abs(pred[s] - gt[s]) for s in eligible)
    print(ls.coverage_note("gauge_value", [s.scenario_id for s in scenarios]))
    # -> "l1_mae computed over 0/20 scenarios (20 contested); NOT REPORTABLE"

Fail-open vs fail-closed: when the registry is absent, every scenario is treated
as trusted and ``registry_present`` is False, so a caller can distinguish "audit
says fine" from "audit never ran". Reporting code should refuse to publish a
gated metric when ``registry_present`` is False.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "reports" / "v7" / "label_status.csv"

TRUSTED = "trusted"      # two independent labelings agree
CONTESTED = "contested"  # two labelings disagree on this scenario
PENDING = "pending"      # only one labeling; corpus reliability not yet established

#: States whose labels may enter a metric. ``pending`` is included because
#: excluding it would drop the entire corpus on the strength of a stratum that
#: is biased by construction; it is counted and surfaced separately instead.
SCORABLE = (TRUSTED, PENDING)


@dataclass
class LabelStatus:
    """Per (scenario_id, field) trust verdicts from the V7 audit."""

    status: Dict[Tuple[str, str], str] = dc_field(default_factory=dict)
    reason: Dict[Tuple[str, str], str] = dc_field(default_factory=dict)
    gates_metric: Dict[str, str] = dc_field(default_factory=dict)
    registry_present: bool = False
    path: Optional[Path] = None

    @classmethod
    def load(cls, path: Path = DEFAULT_REGISTRY) -> "LabelStatus":
        if not path.exists():
            return cls(registry_present=False, path=path)
        status: Dict[Tuple[str, str], str] = {}
        reason: Dict[Tuple[str, str], str] = {}
        gates: Dict[str, str] = {}
        with path.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                key = (r["scenario_id"].strip(), r["field"].strip())
                status[key] = r["status"].strip()
                reason[key] = r.get("reason", "").strip()
                if r.get("gates_metric"):
                    gates[r["field"].strip()] = r["gates_metric"].strip()
        return cls(status=status, reason=reason, gates_metric=gates,
                   registry_present=True, path=path)

    def state(self, scenario_id: str, field: str) -> str:
        """Pairs absent from the registry are ``pending``: never audited, so no
        recorded objection and no established reliability either."""
        return self.status.get((scenario_id, field), PENDING)

    def is_scorable(self, scenario_id: str, field: str) -> bool:
        return self.state(scenario_id, field) in SCORABLE

    def eligible(self, field: str, scenario_ids: Iterable[str]) -> List[str]:
        """The subset of ``scenario_ids`` whose ``field`` label may be scored."""
        return [s for s in scenario_ids if self.is_scorable(s, field)]

    def contested(self, field: str, scenario_ids: Iterable[str]) -> List[str]:
        return [s for s in scenario_ids if self.state(s, field) == CONTESTED]

    def pending(self, field: str, scenario_ids: Iterable[str]) -> List[str]:
        return [s for s in scenario_ids if self.state(s, field) == PENDING]

    def contested_fields(self, scenario_ids: Iterable[str]) -> Set[str]:
        ids = set(scenario_ids)
        return {f for (s, f), v in self.status.items() if s in ids and v == CONTESTED}

    def reportable(self, field: str, scenario_ids: Sequence[str],
                   min_n: int = 1) -> bool:
        """A gated metric is reportable only if the audit ran and enough
        scenarios survive it."""
        return self.registry_present and len(self.eligible(field, scenario_ids)) >= min_n

    def coverage_note(self, field: str, scenario_ids: Sequence[str],
                      min_n: int = 1) -> str:
        """One-line provenance string to print or embed beside any gated metric.

        Reports ``contested`` and ``pending`` separately: the first is a known
        defect, the second is an unexamined region. Collapsing them would let an
        unaudited corpus read as a clean one.
        """
        metric = self.gates_metric.get(field, field)
        total = len(scenario_ids)
        if not self.registry_present:
            return (f"{metric}: V7 label audit has not run "
                    f"({self.path}); NOT REPORTABLE")
        keep = len(self.eligible(field, scenario_ids))
        n_cont = len(self.contested(field, scenario_ids))
        n_pend = len(self.pending(field, scenario_ids))
        note = f"{metric} computed over {keep}/{total} scenarios"
        detail = []
        if n_cont:
            detail.append(f"{n_cont} contested/excluded")
        if n_pend:
            detail.append(f"{n_pend} pending audit")
        if detail:
            note += " (" + ", ".join(detail) + ")"
        if keep < min_n:
            note += "; NOT REPORTABLE"
        return note


def eligible_scenarios(field: str, scenario_ids: Iterable[str],
                       registry: Path = DEFAULT_REGISTRY) -> List[str]:
    """Convenience wrapper for one-off filtering."""
    return LabelStatus.load(registry).eligible(field, scenario_ids)
