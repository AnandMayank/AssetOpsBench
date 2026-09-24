"""semantics.py — B2's five-way state separation (world truth / available
evidence / delivered evidence / sufficiency / terminal decision), the
sufficiency rule R, the WORLD-vs-EVIDENCE parameter classification, and the
cost model.

This is the ONLY module that defines what "sufficient" and "correct" mean
for B2. `env.py` executes against it; `frontier.py` searches over it;
`b2_scoring.py` scores against it. Nothing here calls a model or reads
observation_records.json directly (see `substrate.py` for that).

Plan sections: C (semantics), D (task families use these types), E
(frontier consumes `sufficiency`/`clause_relevant_modalities`), F (cost).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

_ORCH_SRC = Path(__file__).resolve().parents[1]
if str(_ORCH_SRC) not in sys.path:
    sys.path.insert(0, str(_ORCH_SRC))

# Reuse the existing content-hash convention (W-/S-/E- style prefixed
# sha256 of sorted-key JSON) rather than inventing a second hashing scheme.
# Read-only reuse: canonical_identity.py itself is never modified.
from canonical_identity import _hash as _ci_hash  # noqa: E402

__all__ = [
    "ParamClass", "param_class", "WorldSpec", "g_world",
    "ActionCost", "EvidenceRegime", "EpisodeSpec",
    "AcousticReading", "IoTReading", "ThermalReading", "DeliveredEvidence",
    "R_VERSION", "r_fault_clauses", "r_normal_clauses", "sufficiency",
    "clause_relevant_modalities", "TERMINAL_ACTIONS", "g_term_from_frontier",
    "commit_is_grounded", "check_gold_invariance",
    "ACTION_COST_FIELD", "action_cost",
    "ACOUSTIC_INDICATOR_BAND", "ACOUSTIC_THRESHOLD_DB", "acoustic_indicator",
]


# ---------------------------------------------------------------------------
# A/B. WORLD vs EVIDENCE parameter classification (plan section C.A/C.B)
# ---------------------------------------------------------------------------

class ParamClass(str, Enum):
    #: May change g_world. Anything not in this class must never change it.
    WORLD = "WORLD"
    #: May change what the agent can observe/acquire and how much it costs,
    #: never the underlying condition.
    EVIDENCE = "EVIDENCE"


WORLD_PARAM_NAMES: FrozenSet[str] = frozenset({
    "asset", "condition", "machine_id", "iot_band_state",
    "thermal_fault_class", "active_work_order", "technician_present",
})
EVIDENCE_PARAM_NAMES: FrozenSet[str] = frozenset({
    "initial_modalities", "available_modalities", "draw_variant",
    "costs", "budget", "acquisition_enabled",
})


def param_class(name: str) -> ParamClass:
    """Classify one B2 scenario-parameter field name. Raises KeyError for
    anything not explicitly classified -- an unclassified field must never
    silently pass through the gold-invariance check (fail loud, not open)."""
    if name in WORLD_PARAM_NAMES:
        return ParamClass.WORLD
    if name in EVIDENCE_PARAM_NAMES:
        return ParamClass.EVIDENCE
    raise KeyError(f"unclassified B2 parameter: {name!r} (add it to WORLD_PARAM_NAMES "
                    f"or EVIDENCE_PARAM_NAMES in semantics.py -- never assume)")


# ---------------------------------------------------------------------------
# WorldSpec / g_world -- evaluator-only world truth (plan section C, object W)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorldSpec:
    """WORLD-identity fields for one B2 world. Every field here is
    ParamClass.WORLD -- changing any of them is allowed to change
    `g_world`/`g_term`. Never rendered to the agent directly; only
    `DeliveredEvidence` built from resolved observations is agent-visible."""
    asset: str
    condition: str  # "normal" | "fault"
    machine_id: Optional[str] = None        # acoustic worlds: MIMII id "00".."06"
    iot_band_state: str = "in_band"         # "in_band" | "out_of_band"
    thermal_fault_class: Optional[str] = None
    active_work_order: bool = False
    technician_present: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        if self.condition not in ("normal", "fault"):
            raise ValueError(f"WorldSpec.condition must be 'normal' or 'fault', got {self.condition!r}")
        if self.iot_band_state not in ("in_band", "out_of_band"):
            raise ValueError(f"WorldSpec.iot_band_state invalid: {self.iot_band_state!r}")

    def canonical_fields(self) -> Dict[str, Any]:
        """WORLD_PARAM_NAMES only -- `seed` is a draw-order EVIDENCE
        concern (which item of the pool is drawn first), not world
        identity, so it is deliberately excluded here (mirrors
        canonical_identity's exclusion of scenario_id as PRESENTATION_ONLY)."""
        return {
            "asset": self.asset, "condition": self.condition,
            "machine_id": self.machine_id, "iot_band_state": self.iot_band_state,
            "thermal_fault_class": self.thermal_fault_class,
            "active_work_order": self.active_work_order,
            "technician_present": self.technician_present,
        }

    @property
    def world_id(self) -> str:
        return _ci_hash(self.canonical_fields(), "PGW")


def g_world(world: WorldSpec) -> str:
    """The world truth verdict. Trivial by construction (WorldSpec.condition
    IS the ground truth) -- kept as a named function so callers never read
    `.condition` directly and so the dependency is explicit/testable."""
    return world.condition


# ---------------------------------------------------------------------------
# EvidenceRegime / EpisodeSpec -- EVIDENCE/EXECUTION parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActionCost:
    """Ordinal design parameters (plan section F) -- NOT empirical prices.
    record < iot ~ capture << dispatch. `dispatch` is a one-time setup cost
    charged once per episode by the DISPATCH action; ACQUIRE_ACOUSTIC/
    ACQUIRE_THERMAL require a prior DISPATCH but do not re-charge it."""
    query_record: float = 0.5
    iot_read: float = 1.0
    acoustic_capture: float = 1.0
    thermal_capture: float = 1.0
    dispatch: float = 8.0

    @classmethod
    def with_dispatch_ratio(cls, ratio: float) -> "ActionCost":
        """Sensitivity setting (plan section F): ratio in {8, 20}. Every
        B2 result must be reported under at least two ratios."""
        return cls(dispatch=ratio)


@dataclass(frozen=True)
class EvidenceRegime:
    """EVIDENCE/EXECUTION parameters (plan section C.B). Changing any field
    here must never change `g_world`; it may change `g_term` only via the
    ESCALATE branch (frontier becomes empty), never by flipping which
    COMMIT verdict is correct."""
    initial_modalities: Tuple[str, ...] = ()
    available_modalities: FrozenSet[str] = field(
        default_factory=lambda: frozenset({"iot", "acoustic", "thermal", "record"}))
    draw_variant: str = "default"
    costs: ActionCost = field(default_factory=ActionCost)
    budget: Optional[float] = None
    acquisition_enabled: bool = True

    def regime_fields(self) -> Dict[str, Any]:
        return {
            "initial_modalities": list(self.initial_modalities),
            "available_modalities": sorted(self.available_modalities),
            "draw_variant": self.draw_variant,
            "budget": self.budget,
            "acquisition_enabled": self.acquisition_enabled,
            "dispatch_ratio": self.costs.dispatch / max(self.costs.iot_read, 1e-9),
        }


@dataclass(frozen=True)
class EpisodeSpec:
    world: WorldSpec
    regime: EvidenceRegime
    family: str                       # "A".."F"
    matched_group: Optional[str] = None   # evaluator-side only, never rendered
    variant: Optional[str] = None         # evaluator-side only, never rendered

    @property
    def episode_id(self) -> str:
        """Opaque id: hashes world identity + family + regime fields.
        Deliberately excludes `matched_group`/`variant` from the hashed
        (and therefore visible) payload -- those name the counterfactual
        condition and must stay evaluator-side (plan section H)."""
        payload = {"world_id": self.world.world_id, "family": self.family,
                   "regime": self.regime.regime_fields()}
        return f"PG-B2::{self.family}::{_ci_hash(payload, 'PGB2')}"


# ---------------------------------------------------------------------------
# DeliveredEvidence -- agent-visible object (plan section C, object E_t)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AcousticReading:
    positive: bool           # deterministic indicator verdict (agent-visible)
    indicator_value: float   # the raw statistic behind it (agent-visible)


@dataclass(frozen=True)
class IoTReading:
    in_band: bool
    value: float


@dataclass(frozen=True)
class ThermalReading:
    hotspot_present: bool


@dataclass
class DeliveredEvidence:
    """Everything the agent has actually received so far. R and g_term are
    defined ONLY over this object -- never over WorldSpec."""
    acoustic: List[AcousticReading] = field(default_factory=list)
    iot: List[IoTReading] = field(default_factory=list)
    thermal: List[ThermalReading] = field(default_factory=list)
    record_queried: bool = False

    def copy_with(self, **extra_items) -> "DeliveredEvidence":
        new = DeliveredEvidence(acoustic=list(self.acoustic), iot=list(self.iot),
                                 thermal=list(self.thermal), record_queried=self.record_queried)
        for k, v in extra_items.items():
            getattr(new, k).append(v) if isinstance(getattr(new, k), list) else setattr(new, k, v)
        return new


# ---------------------------------------------------------------------------
# Sufficiency rule R (plan section C.E) -- versioned, explicit, testable
# clause-by-clause. NEVER reads WorldSpec / hidden labels.
# ---------------------------------------------------------------------------

R_VERSION = "b2-R-v1"


def _acoustic_positive_count(e: DeliveredEvidence) -> int:
    return sum(1 for r in e.acoustic if r.positive)


def _acoustic_negative_count(e: DeliveredEvidence) -> int:
    return sum(1 for r in e.acoustic if not r.positive)


def _iot_out_of_band_streak(e: DeliveredEvidence) -> int:
    streak = 0
    for r in reversed(e.iot):
        if not r.in_band:
            streak += 1
        else:
            break
    return streak


def _iot_in_band_streak(e: DeliveredEvidence) -> int:
    streak = 0
    for r in reversed(e.iot):
        if r.in_band:
            streak += 1
        else:
            break
    return streak


def r_fault_clauses(e: DeliveredEvidence) -> Dict[str, bool]:
    """Each clause independently sufficient for R(e) == 'fault'."""
    return {
        "acoustic_2plus_positive": _acoustic_positive_count(e) >= 2,
        "acoustic1_iot_persist2": _acoustic_positive_count(e) >= 1 and _iot_out_of_band_streak(e) >= 2,
        "thermal_hotspot": any(r.hotspot_present for r in e.thermal),
    }


#: The two rule variants under audit (plan: G3 sensitivity analysis).
#: R_PRIMARY: as originally proposed, including the IoT-only NORMAL
#:            closure (3 consecutive in-band reads, no physical evidence
#:            at all, suffices to COMMIT_NORMAL).
#: R_STRICT:  the same rule with that closure REMOVED -- NORMAL always
#:            requires at least the 3-negative-acoustic clause (i.e. some
#:            physical evidence), matching the asymmetry FAULT already has
#:            (FAULT never has a telemetry-only closure path either).
#: Selection between them is NOT made here -- see
#: reports/playground/b2/rule_sensitivity_report.json and the audit's
#: explicit G3 decision; this module only makes both computable.
RULE_VARIANTS = ("primary", "strict")


def r_normal_clauses(e: DeliveredEvidence, rule: str = "primary") -> Dict[str, bool]:
    """Each clause independently sufficient for R(e) == 'normal'."""
    if rule not in RULE_VARIANTS:
        raise ValueError(f"unknown rule variant: {rule!r}")
    clauses = {
        "acoustic_3plus_negative_iot_inband": (
            _acoustic_negative_count(e) >= 3 and bool(e.iot) and e.iot[-1].in_band),
    }
    if rule == "primary":
        clauses["iot_inband_streak3"] = _iot_in_band_streak(e) >= 3
    return clauses


def sufficiency(e: DeliveredEvidence, rule: str = "primary") -> Optional[str]:
    """R(E_t) in {"normal", "fault", None==bottom}, under rule variant
    `rule` (default "primary", the original proposed rule -- backward
    compatible with every existing call site). Both a fault clause and a
    normal clause firing simultaneously (contradiction) is NOT resolved in
    favor of either -- it returns None (insufficient/contradictory),
    consistent with plan C.E: contradiction must be reconciled, not
    silently broken by clause priority."""
    fault_hit = any(r_fault_clauses(e).values())
    normal_hit = any(r_normal_clauses(e, rule=rule).values())
    if fault_hit and normal_hit:
        return None
    if fault_hit:
        return "fault"
    if normal_hit:
        return "normal"
    return None


def clause_relevant_modalities() -> Dict[str, FrozenSet[str]]:
    """Which modalities feed each clause -- used by the frontier search
    (reachability) and by the wasted-acquisition definition (plan E, F:
    "the modality of a_t appears in no clause of R still satisfiable")."""
    return {
        "acoustic_2plus_positive": frozenset({"acoustic"}),
        "acoustic1_iot_persist2": frozenset({"acoustic", "iot"}),
        "thermal_hotspot": frozenset({"thermal"}),
        "acoustic_3plus_negative_iot_inband": frozenset({"acoustic", "iot"}),
        "iot_inband_streak3": frozenset({"iot"}),
    }


# ---------------------------------------------------------------------------
# Terminal decision / gold invariant (plan section C.C/C.D/C.F/C.G)
# ---------------------------------------------------------------------------

TERMINAL_ACTIONS = ("COMMIT_NORMAL", "COMMIT_FAULT", "ESCALATE")


def g_term_from_frontier(world: WorldSpec, frontier_empty: bool) -> str:
    """The evidence-conditioned correct terminal action. `frontier_empty`
    is computed by `frontier.py` for a given (world, regime, state) --
    this function only encodes the invariant: COMMIT's verdict always
    equals g_world(world); ESCALATE fires iff no sufficient state remains
    reachable. This function NEVER returns the wrong-verdict COMMIT."""
    if frontier_empty:
        return "ESCALATE"
    return "COMMIT_FAULT" if g_world(world) == "fault" else "COMMIT_NORMAL"


def commit_is_grounded(verdict_claimed: str, delivered: DeliveredEvidence) -> bool:
    """True iff the claimed COMMIT verdict is actually supported by R over
    the evidence delivered so far (plan C.F) -- independent of whether it
    happens to match world truth. A lucky correct guess on R(e)=None
    evidence is NOT grounded."""
    r = sufficiency(delivered)
    if r is None:
        return False
    return (verdict_claimed == "COMMIT_FAULT" and r == "fault") or \
           (verdict_claimed == "COMMIT_NORMAL" and r == "normal")


def check_gold_invariance(world: WorldSpec, frontier_empty_by_regime: Dict[str, bool]) -> bool:
    """The gold-preservation check (mirrors canonical_identity's recompute-
    and-compare pattern, plan A.11/K): for a FIXED world, g_term across
    every regime must be in {ESCALATE, COMMIT_<g_world(world)>} only --
    never the other COMMIT value. `frontier_empty_by_regime` maps a regime
    label to whether ITS frontier is empty (computed by frontier.py)."""
    allowed_commit = "COMMIT_FAULT" if g_world(world) == "fault" else "COMMIT_NORMAL"
    for _, frontier_empty in frontier_empty_by_regime.items():
        term = g_term_from_frontier(world, frontier_empty)
        if term not in ("ESCALATE", allowed_commit):
            return False
    return True


# ---------------------------------------------------------------------------
# Cost accessor (plan section F)
# ---------------------------------------------------------------------------

ACTION_COST_FIELD: Dict[str, str] = {
    "QUERY_RECORD": "query_record",
    "ACQUIRE_IOT": "iot_read",
    "REINSPECT_IOT": "iot_read",
    "ACQUIRE_ACOUSTIC": "acoustic_capture",
    "ACQUIRE_THERMAL": "thermal_capture",
    "DISPATCH": "dispatch",
}


def action_cost(action: str, costs: ActionCost) -> float:
    if action not in ACTION_COST_FIELD:
        raise KeyError(f"no cost defined for action {action!r}")
    return getattr(costs, ACTION_COST_FIELD[action])


# ---------------------------------------------------------------------------
# Deterministic acoustic indicator (plan section G1 -- empirically checked,
# never assumed valid). Computed ONLY from real compute_acoustic_features()
# output; NEVER reads the hidden MIMII normal/abnormal label at run time --
# the per-(asset, machine_id) calibration table below is the ONE place the
# label is used, and only offline, once, to pick a (band, threshold,
# direction) per real-data run recorded in
# reports/playground/b2/substrate_gates_report.json (G1). "abnormal scores
# higher"/"lower" and the threshold come directly from that run's
# calibration field -- see that report for the exact sample and AUCs.
#
# G1's verdict on 2026-09-23: MARGINAL. 7 of 8 (asset, machine_id) pools
# clear AUC >= 0.65 in their best single-feature direction; chiller_6/02
# does NOT (0.612) and is therefore EXCLUDED from
# ACOUSTIC_CALIBRATED_KEYS -- families B/E/F must never construct a world
# on chiller_6/02's acoustic pool. No single (band, stat) pair generalizes
# across all 8 combinations (the winning band differs per machine_id, all
# via absolute band energy, never RMS-relative) -- consistent with real,
# noisy sensor data: this is why B2's sufficiency rule accumulates SEVERAL
# acoustic readings (>=2 positive / >=3 negative) rather than trusting one.
# ---------------------------------------------------------------------------

#: (asset, machine_id) -> (band, threshold_db, abnormal_scores_higher).
#: Populated from reports/playground/b2/substrate_gates_report.json's
#: g1.per_asset_machine[*].calibration field. chiller_6/02 intentionally
#: excluded (AUC 0.612 < the 0.65 gate) -- see module docstring above.
ACOUSTIC_CALIBRATION: Dict[Tuple[str, str], Tuple[str, float, bool]] = {
    ("chiller_6", "00"): ("0-500Hz", -45.318169713919914, True),
    ("chiller_6", "04"): ("0-500Hz", -46.475132577537465, True),
    ("chiller_6", "06"): ("500-2000Hz", -49.42609918105529, False),
    ("hydraulic_pump_1", "00"): ("0-500Hz", -46.89192534613899, True),
    ("hydraulic_pump_1", "02"): ("500-2000Hz", -50.57030015696242, True),
    ("hydraulic_pump_1", "04"): ("2000-8000Hz", -53.75995785760284, False),
    ("hydraulic_pump_1", "06"): ("500-2000Hz", -51.52744625850265, True),
}
#: Machine ids EXCLUDED from every acoustic-dependent B2 family (G1 gate).
ACOUSTIC_EXCLUDED_KEYS: FrozenSet[Tuple[str, str]] = frozenset({("chiller_6", "02")})

#: Global fallback ONLY for callers with no (asset, machine_id) context
#: (e.g. ad-hoc smoke tests) -- NOT used by b2_families.py, which always
#: passes a real key. Uninformative-by-construction as a safety default.
ACOUSTIC_INDICATOR_BAND = "500-2000Hz"
ACOUSTIC_THRESHOLD_DB = -50.0


def acoustic_indicator(features: Dict[str, Any],
                        key: Optional[Tuple[str, str]] = None) -> Tuple[bool, float]:
    """(positive, indicator_value). With `key=(asset, machine_id)` given
    and calibrated (in ACOUSTIC_CALIBRATION), uses that combo's real-data
    calibrated (band, threshold, direction) -- this is the path every B2
    world construction must use. Without a calibrated key, falls back to
    the uncalibrated global default (do not trust its AUC)."""
    if key is not None and key in ACOUSTIC_CALIBRATION:
        band, threshold, abnormal_higher = ACOUSTIC_CALIBRATION[key]
        value = float(features["band_energy_db"].get(band, -120.0))
        raw_positive = bool(value > threshold)
        return (raw_positive if abnormal_higher else not raw_positive), value
    band_db = float(features["band_energy_db"].get(ACOUSTIC_INDICATOR_BAND, -120.0))
    return bool(band_db > ACOUSTIC_THRESHOLD_DB), band_db
