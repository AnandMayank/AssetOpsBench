"""b2_families.py — builds the B2 episode pool: families A-F and matched
groups M0-M4 (plan section D/H), over worlds validated by G1/G2 (plan
section M). Computes each episode's gold (`B2Gold`: g_term, frontier
emptiness, C_min) offline via `frontier.py`, using the SAME real-data
boolean sequence `env.py` will draw live (via
`substrate.build_world_pool`/`world_pool_to_booleans`) -- so gold and live
execution can never disagree.

Reuses the existing B-ACQ coverage tables' SHAPE (asset lists, arm-registry
pattern) from `b_acquisition_generator.py` without importing/modifying it
-- B2 is a separate namespace (`PG-B2::...`, seeds not applicable, ids are
content hashes) and never touches the frozen B pool.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

_ORCH_SRC = Path(__file__).resolve().parents[1]
if str(_ORCH_SRC) not in sys.path:
    sys.path.insert(0, str(_ORCH_SRC))

from . import substrate  # noqa: E402
from .frontier import AbstractState, DrawPool, min_cost_to_sufficiency, reachable_sufficient_states  # noqa: E402
from .semantics import (  # noqa: E402
    ACOUSTIC_EXCLUDED_KEYS, ActionCost, EpisodeSpec, EvidenceRegime, WorldSpec,
    g_term_from_frontier, g_world,
)

__all__ = [
    "B2Gold", "AcousticWorldKey", "ACOUSTIC_WORLD_KEYS",
    "build_family_a", "build_family_b", "build_family_c", "build_family_d",
    "build_family_e", "build_family_f", "build_matched_groups",
    "build_pool", "compute_gold", "world_pool_for", "PRIMARY_HORIZON",
]

PRIMARY_HORIZON = 8

#: (asset, machine_id) pairs G1 validated (AUC >= 0.65) -- the only ones
#: any acoustic-dependent family may build a world on (plan section M).
AcousticWorldKey = Tuple[str, str]
ACOUSTIC_WORLD_KEYS: Tuple[AcousticWorldKey, ...] = tuple(sorted(
    (asset, mid)
    for asset in sorted(substrate.AGREE_ACOUSTIC_ASSETS)
    for mid in substrate.ACOUSTIC_MACHINE_IDS
    if (asset, mid) not in ACOUSTIC_EXCLUDED_KEYS
))

_TWO_DISPATCH_RATIOS: Tuple[float, ...] = (8.0, 20.0)


@dataclass(frozen=True)
class B2Gold:
    """Offline-computed gold for one EpisodeSpec (plan section E). `c_min`
    and `frontier_empty` are computed against `target_verdict=g_world` --
    i.e. the cheapest cost to reach a state whose OWN R() agrees with
    world truth specifically, NOT the cheapest sufficient state of either
    verdict (plan C.F/C.G: a cheaper-but-wrong-verdict state, as in the
    contradiction family E, does not make ESCALATE gold, and does not
    make COMMIT(wrong verdict) gold either -- it is simply not the
    frontier this gold cares about)."""
    world_truth: str          # "normal" | "fault"
    frontier_empty: bool      # True iff NO state agreeing with world_truth is reachable
    c_min: Optional[float]    # cost to the cheapest world_truth-agreeing state; None iff frontier_empty
    g_term: str               # "COMMIT_NORMAL" | "COMMIT_FAULT" | "ESCALATE"


def world_pool_for(world: WorldSpec, *, max_draws: int = PRIMARY_HORIZON + 2) -> DrawPool:
    """Real-data DrawPool for `world` -- the SAME booleans `env.py`
    resolves live (both go through `substrate.build_world_pool` ->
    `world_pool_to_booleans`/`reading_to_boolean`)."""
    wp = substrate.build_world_pool(
        asset=world.asset, condition=world.condition, machine_id=world.machine_id,
        iot_band_state=world.iot_band_state, thermal_fault_class=world.thermal_fault_class,
        max_draws=max_draws,
    )
    booleans = substrate.world_pool_to_booleans(wp)
    return DrawPool(acoustic=booleans.get("acoustic", ()), iot=booleans.get("iot_timeseries", ()),
                     thermal=booleans.get("thermal", ()))


def compute_gold(spec: EpisodeSpec, *, initial: Optional[AbstractState] = None,
                  rule: str = "primary") -> B2Gold:
    """`rule`: "primary" | "strict" (plan G3 sensitivity analysis, see
    semantics.RULE_VARIANTS). Default "primary" is backward compatible
    with every existing caller/test."""
    from .semantics import sufficiency as _sufficiency
    pool = world_pool_for(spec.world)
    state0 = initial if initial is not None else AbstractState()
    truth = g_world(spec.world)
    accept = lambda e, tv=truth, r=rule: _sufficiency(e, rule=r) == tv  # noqa: E731
    hits = reachable_sufficient_states(state0, pool, spec.regime, max_horizon=PRIMARY_HORIZON, accept=accept)
    c_min = hits[0][1] if hits else None
    empty = c_min is None
    term = g_term_from_frontier(spec.world, empty)
    return B2Gold(world_truth=truth, frontier_empty=empty, c_min=c_min, g_term=term)


def _default_iot_band(condition: str) -> str:
    """Ties the IoT WORLD parameter to the world's actual condition, so an
    IoT-only closure path is not accidentally reachable/unreachable for
    the wrong reason (bug caught by pool-level gold-distribution smoke
    testing: an unconditional 'in_band' let fault worlds falsely satisfy
    the normal-only 'iot_inband_streak3' clause). Family E deliberately
    OVERRIDES this to construct a genuine within-episode disagreement."""
    return "out_of_band" if condition == "fault" else "in_band"


def _acoustic_worlds(condition: str) -> List[WorldSpec]:
    return [WorldSpec(asset=asset, condition=condition, machine_id=mid,
                       iot_band_state=_default_iot_band(condition))
            for asset, mid in ACOUSTIC_WORLD_KEYS]


def _thermal_worlds() -> List[Tuple[WorldSpec, WorldSpec]]:
    """(normal, fault) motor_01 thermal world pairs -- one per real
    non-Noload fault_class available (Rotor-0, A&C&B10, Fan)."""
    fault_classes = ("Rotor-0", "A&C&B10", "Fan")
    return [(WorldSpec(asset="motor_01", condition="normal", iot_band_state=_default_iot_band("normal")),
              WorldSpec(asset="motor_01", condition="fault", thermal_fault_class=fc,
                        iot_band_state=_default_iot_band("fault")))
             for fc in fault_classes]


def _full_regime(*, dispatch_ratio: float = 8.0, draw_variant: str = "default",
                  budget: Optional[float] = None,
                  available: FrozenSet[str] = frozenset({"iot", "acoustic", "thermal", "record"}),
                  acquisition_enabled: bool = True) -> EvidenceRegime:
    return EvidenceRegime(available_modalities=available, draw_variant=draw_variant,
                           costs=ActionCost.with_dispatch_ratio(dispatch_ratio), budget=budget,
                           acquisition_enabled=acquisition_enabled)


# ---------------------------------------------------------------------------
# Family A: single-observation sufficiency (thermal, one-shot)
# ---------------------------------------------------------------------------

def build_family_a(*, dispatch_ratio: float = 8.0) -> List[EpisodeSpec]:
    specs: List[EpisodeSpec] = []
    for normal_w, fault_w in _thermal_worlds():
        for world in (normal_w, fault_w):
            regime = _full_regime(dispatch_ratio=dispatch_ratio)
            specs.append(EpisodeSpec(world=world, regime=regime, family="A"))
    return specs


# ---------------------------------------------------------------------------
# Family B: multi-observation sufficiency (acoustic, reuses G1-validated
# (asset, machine_id) pairs -- mirrors B-ACQ-2's asset/count-threshold
# shape from b_acquisition_generator.py without importing it)
# ---------------------------------------------------------------------------

def build_family_b(*, dispatch_ratio: float = 8.0) -> List[EpisodeSpec]:
    specs: List[EpisodeSpec] = []
    for condition in ("normal", "fault"):
        for world in _acoustic_worlds(condition):
            regime = _full_regime(dispatch_ratio=dispatch_ratio)
            specs.append(EpisodeSpec(world=world, regime=regime, family="B"))
    return specs


# ---------------------------------------------------------------------------
# Family C: redundant evidence (same worlds as A/B, evaluator marks which
# post-sufficiency acquisitions in a TRAJECTORY are wasted -- construction
# is identical to A/B; C is a SCORING lens (redundancy), not a distinct
# world set, so it reuses A's and a subset of B's worlds directly).
# ---------------------------------------------------------------------------

def build_family_c(*, dispatch_ratio: float = 8.0) -> List[EpisodeSpec]:
    specs: List[EpisodeSpec] = []
    for world in [w for pair in _thermal_worlds() for w in pair][:4]:
        specs.append(EpisodeSpec(world=world, regime=_full_regime(dispatch_ratio=dispatch_ratio), family="C"))
    for world in _acoustic_worlds("fault")[:4]:
        specs.append(EpisodeSpec(world=world, regime=_full_regime(dispatch_ratio=dispatch_ratio), family="C"))
    return specs


# ---------------------------------------------------------------------------
# Family D: unavailable critical modality
# ---------------------------------------------------------------------------

def build_family_d(*, dispatch_ratio: float = 8.0) -> List[EpisodeSpec]:
    specs: List[EpisodeSpec] = []
    for normal_w, fault_w in _thermal_worlds():
        for world in (normal_w, fault_w):
            regime = _full_regime(dispatch_ratio=dispatch_ratio,
                                   available=frozenset({"iot", "record"}))
            specs.append(EpisodeSpec(world=world, regime=regime, family="D"))
    for condition in ("normal", "fault"):
        for world in _acoustic_worlds(condition)[:2]:
            regime = _full_regime(dispatch_ratio=dispatch_ratio, available=frozenset({"iot", "record"}))
            specs.append(EpisodeSpec(world=world, regime=regime, family="D"))
    return specs


# ---------------------------------------------------------------------------
# Family E: contradiction (acoustic + IoT disagree on entry)
# ---------------------------------------------------------------------------

def build_family_e(*, dispatch_ratio: float = 8.0) -> List[EpisodeSpec]:
    specs: List[EpisodeSpec] = []
    for asset, mid in ACOUSTIC_WORLD_KEYS:
        # world truth fault, but IoT band-state set to "in_band" so the
        # FIRST iot read looks normal while acoustic (once acquired) will
        # tend positive -- a genuine within-episode disagreement the agent
        # must reconcile, not a fabricated contradiction (both readings are
        # real draws from real data; the "disagreement" is that this
        # asset's IoT band alone doesn't move with the acoustic condition).
        world = WorldSpec(asset=asset, condition="fault", machine_id=mid, iot_band_state="in_band")
        regime = _full_regime(dispatch_ratio=dispatch_ratio)
        specs.append(EpisodeSpec(world=world, regime=regime, family="E"))
    return specs


# ---------------------------------------------------------------------------
# Family F: cost-sensitive alternative paths (cheap IoT-only path vs
# DISPATCH; normal/fault twin per plan section D)
# ---------------------------------------------------------------------------

def build_family_f(*, dispatch_ratio: float = 8.0) -> List[EpisodeSpec]:
    specs: List[EpisodeSpec] = []
    normal_world = WorldSpec(asset="chiller_6", condition="normal", iot_band_state="in_band")
    specs.append(EpisodeSpec(world=normal_world, regime=_full_regime(dispatch_ratio=dispatch_ratio), family="F"))
    for asset, mid in ACOUSTIC_WORLD_KEYS[:2]:
        fault_world = WorldSpec(asset=asset, condition="fault", machine_id=mid, iot_band_state="in_band")
        specs.append(EpisodeSpec(world=fault_world, regime=_full_regime(dispatch_ratio=dispatch_ratio), family="F"))
    return specs


# ---------------------------------------------------------------------------
# Matched groups M0-M4 (plan section H) -- built over a fixed pivot world
# per group so the SAME world underlies every arm.
# ---------------------------------------------------------------------------

def build_matched_groups(*, dispatch_ratio: float = 8.0) -> Dict[str, List[EpisodeSpec]]:
    groups: Dict[str, List[EpisodeSpec]] = {}
    asset, mid = ACOUSTIC_WORLD_KEYS[0]
    pivot_fault = WorldSpec(asset=asset, condition="fault", machine_id=mid, iot_band_state="in_band")

    # M0: acquisition enabled vs disabled.
    groups["M0"] = [
        EpisodeSpec(world=pivot_fault, regime=_full_regime(dispatch_ratio=dispatch_ratio,
                                                            acquisition_enabled=True),
                    family="B", matched_group="M0", variant="enabled"),
        EpisodeSpec(world=pivot_fault, regime=_full_regime(dispatch_ratio=dispatch_ratio,
                                                            acquisition_enabled=False),
                    family="B", matched_group="M0", variant="disabled"),
    ]

    # M1: critical modality available vs unavailable (motor_01 thermal --
    # the family with an unambiguous single critical modality).
    normal_w, fault_w = _thermal_worlds()[0]
    groups["M1"] = [
        EpisodeSpec(world=fault_w, regime=_full_regime(dispatch_ratio=dispatch_ratio),
                    family="D", matched_group="M1", variant="available"),
        EpisodeSpec(world=fault_w, regime=_full_regime(dispatch_ratio=dispatch_ratio,
                                                        available=frozenset({"iot", "record"})),
                    family="D", matched_group="M1", variant="unavailable"),
    ]

    # M2: cheap path priced cheap vs expensive (dispatch ratio sensitivity
    # itself, applied to a family-F-shaped normal world).
    normal_world = WorldSpec(asset="chiller_6", condition="normal", iot_band_state="in_band")
    groups["M2"] = [
        EpisodeSpec(world=normal_world, regime=_full_regime(dispatch_ratio=8.0),
                    family="F", matched_group="M2", variant="cheap_ratio"),
        EpisodeSpec(world=normal_world, regime=_full_regime(dispatch_ratio=20.0),
                    family="F", matched_group="M2", variant="expensive_ratio"),
    ]

    # M3: draw order -- default vs an alternate draw variant label
    # (draw_variant is EVIDENCE-classified and never touches world_id;
    # the actual draw content differs only through which real clips are
    # in the pool at a given draw depth, i.e. genuinely different evidence
    # arriving in a different order for the SAME world).
    groups["M3"] = [
        EpisodeSpec(world=pivot_fault, regime=_full_regime(dispatch_ratio=dispatch_ratio, draw_variant="default"),
                    family="B", matched_group="M3", variant="draw_default"),
        EpisodeSpec(world=pivot_fault, regime=_full_regime(dispatch_ratio=dispatch_ratio, draw_variant="alt"),
                    family="B", matched_group="M3", variant="draw_alt"),
    ]

    # M4: no budget vs budget < C_min (computed after building the no-
    # budget arm's gold, so the hard cap provably empties the frontier).
    unbudgeted = EpisodeSpec(world=pivot_fault, regime=_full_regime(dispatch_ratio=dispatch_ratio),
                              family="B", matched_group="M4", variant="unbudgeted")
    gold_unbudgeted = compute_gold(unbudgeted)
    hard_budget = (gold_unbudgeted.c_min - 0.5) if gold_unbudgeted.c_min else 1.0
    budgeted = EpisodeSpec(world=pivot_fault,
                            regime=_full_regime(dispatch_ratio=dispatch_ratio, budget=hard_budget),
                            family="B", matched_group="M4", variant="budgeted")
    groups["M4"] = [unbudgeted, budgeted]
    return groups


# ---------------------------------------------------------------------------
# Pool assembly + admission (plan section H leakage controls / section K)
# ---------------------------------------------------------------------------

def build_pool(*, dispatch_ratio: float = 8.0, rule: str = "primary") -> List[Tuple[EpisodeSpec, B2Gold]]:
    """The PRIMARY family pool (A-F), deduplicated by episode_id. Gold is
    always world-truth-targeted (see B2Gold/compute_gold), so every
    episode here -- including family E's contradictions -- has a gold
    verdict that tracks the true condition, not whatever's cheapest to
    reach; no separate admission filter or diagnostic split is needed.
    `rule` selects the sufficiency rule variant (plan G3).

    Matched-pair (M0-M4) analysis is NOT read off this pool -- call
    `build_matched_groups()` directly for that (plan section H)."""
    families = (build_family_a(dispatch_ratio=dispatch_ratio) + build_family_b(dispatch_ratio=dispatch_ratio) +
                build_family_c(dispatch_ratio=dispatch_ratio) + build_family_d(dispatch_ratio=dispatch_ratio) +
                build_family_e(dispatch_ratio=dispatch_ratio) + build_family_f(dispatch_ratio=dispatch_ratio))
    seen_ids: Dict[str, EpisodeSpec] = {}
    for spec in families:
        seen_ids.setdefault(spec.episode_id, spec)
    return [(spec, compute_gold(spec, rule=rule)) for spec in seen_ids.values()]
