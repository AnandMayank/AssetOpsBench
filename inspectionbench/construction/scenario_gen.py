"""scenario_gen.py — Independent world sampling with derived gold labels.

The construct audit found that in all six pilot scenarios the world was built
backwards from the label: ``gold ≡ f(hidden value, band)``, with the value placed
on whichever side of the band the desired verdict required. That makes the
physical state carry no information the label does not, and it is why those
scenarios cannot support a powered comparison.

This module inverts the construction order.

    sample_world(seed, cell)      -> WorldState        # never sees a label
    derive_gold(world, rule)      -> GoldLabel         # pure function of world

The architectural guarantee is enforced, not merely intended: ``sample_world``
takes no parameter that names a verdict, and ``check_generator_contract()``
inspects its signature so a future edit that threads a label through fails a
test rather than passing silently.

**Stratification is on the world, not on the label.** Cells are defined by
physical-in-band × IoT-agrees — both properties of the world. Under the safety
rule below the physical factor happens to determine the verdict, so balanced
cells yield balanced golds; that is a *consequence* of the design, not an input
to it. Balancing on gold directly would reintroduce exactly the defect being
removed.

The safety rule is stated once, explicitly, and is the only place a verdict is
decided. It encodes the operational policy the benchmark is about: a reading
outside the operating band must be escalated, and a reading inside it may be
committed — regardless of what telemetry says, because telemetry is not the
physical fact.
"""

from __future__ import annotations

import inspect
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

COMMIT = "COMMIT"
ESCALATE = "ESCALATE"
ABORT = "ABORT"

#: Coordination base rates, frozen 2026-08-12 and NOT changed.
#:
#: Audit of provenance: these were introduced in commit 66b0d2e with no comment,
#: citation or domain grounding - plausible-looking numbers, nothing more. The
#: only real work-order data in the corpus is three rows, far too few to estimate
#: a base rate (and it would imply ~67% active, not 25%).
#:
#: A proposal to raise ACTIVE_WO_RATE to 0.40 was therefore **rejected**: 0.40 has
#: no more domain justification than 0.25, and its only motive was to thicken the
#: coordination cell for statistical balance. Changing a world parameter to
#: improve cell counts is gold-adjacent tuning of the kind this benchmark exists
#: to avoid.
#:
#: The principled alternative, if coordination coverage proves too thin, is to
#: **stratify** on coordination as a third world factor rather than inflate its
#: probability - that guarantees cells without asserting a base rate. Recorded as
#: a future option; not implemented, not approved.
ACTIVE_WO_RATE = 0.25
TECHNICIAN_PRESENT_RATE = 0.15

#: Parameter names a world sampler must never accept. Checked automatically.
FORBIDDEN_SAMPLER_PARAMS = {
    "gold", "gold_action", "label", "verdict", "expected", "answer",
    "target", "desired", "expected_verdict", "gold_label",
}


@dataclass(frozen=True)
class AssetSpec:
    """Physical envelope of an asset. Independent of any scenario."""
    asset_id: str
    unit: str
    gauge_range: Tuple[float, float]
    operating_band: Tuple[float, float]


#: Assets available to the generator, taken from the seeded CouchDB profiles.
ASSETS: Dict[str, AssetSpec] = {
    "chiller_6": AssetSpec("chiller_6", "bar", (0.0, 400.0), (220.0, 270.0)),
    "metro_pump_1": AssetSpec("metro_pump_1", "m3/s", (0.0, 1.5), (0.9, 1.1)),
    "motor_01": AssetSpec("motor_01", "C", (0.0, 200.0), (80.0, 100.0)),
    "hydraulic_pump_1": AssetSpec("hydraulic_pump_1", "bar", (0.0, 350.0), (160.0, 200.0)),
}


@dataclass(frozen=True)
class Cell:
    """A stratification cell. Both factors are properties of the world."""
    physical_in_band: bool
    iot_agrees: bool

    @property
    def name(self) -> str:
        return (f"phys_{'in' if self.physical_in_band else 'out'}"
                f"__iot_{'agree' if self.iot_agrees else 'disagree'}")


FACTORIAL: Tuple[Cell, ...] = (
    Cell(True, True), Cell(True, False), Cell(False, True), Cell(False, False),
)


@dataclass
class WorldState:
    """The sampled world. Contains no verdict and no reference to one."""
    scenario_id: str
    asset: str
    unit: str
    gauge_range: List[float]
    operating_band: List[float]
    physical_value: float          # hidden; delivered only via an observation
    iot_value: float
    history_mean: float
    active_work_order: bool
    technician_present: bool
    cell: str
    seed: int
    # Optional D-physical constraint-integration fields (Phase 8H.2E). Default
    # None so every existing A/E world (and its hash/fingerprint) is byte-
    # identical to before this field existed -- sample_world never sets these;
    # only the D-physical generator does, via a dedicated builder that reuses
    # the existing asset_profiles.json physical_access geometry unmodified.
    #
    # battery_pct / mission_duration_s / route_legs are NOT separate fields
    # here: SpotAdmissibilityVerifier.verify_candidate reads them via
    # access.get(...) from the SAME dict as the geometry (verified directly
    # against R037/R038's own `overrides` merged into `access`), so they
    # belong as keys inside physical_access, not as sibling WorldState
    # fields -- an earlier draft of this field split them out incorrectly;
    # caught during the Phase 1 oracle-agreement verification before any
    # generator code was built on top of it.
    physical_access: Optional[Dict[str, Any]] = None
    payload_kg: Optional[float] = None  # genuine separate verify_candidate() param

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Omit the optional D-physical fields entirely when unset, so every
        # world predating D-physical serializes to the exact same dict shape
        # as before (verified against the frozen 12-world committed pilot in
        # test_contrastive_repairs.py::test_frozen_generator_reproduces_the_committed_pilot).
        if d.get("physical_access") is None:
            d.pop("physical_access", None)
        if d.get("payload_kg") is None:
            d.pop("payload_kg", None)
        return d


@dataclass
class GoldLabel:
    verdict: str
    rule_applied: str
    derived_from: Dict[str, Any]


def sample_world(seed: int, cell: Cell, asset_id: Optional[str] = None,
                 scenario_id: str = "") -> WorldState:
    """Sample a world state. **Receives no label and cannot see one.**

    The physical value is drawn inside or outside the operating band according
    to the cell's *world* factor; the IoT value is then drawn to agree or
    disagree with that physical value. Neither draw consults a verdict.
    """
    rng = random.Random(seed)
    spec = ASSETS[asset_id] if asset_id else ASSETS[rng.choice(sorted(ASSETS))]
    lo, hi = spec.operating_band
    gmin, gmax = spec.gauge_range
    span = gmax - gmin

    if cell.physical_in_band:
        physical = rng.uniform(lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo))
    else:
        # Outside the band on either side, with a margin so the violation is
        # perceptually resolvable rather than borderline.
        margin = 0.08 * span
        below, above = lo - margin, hi + margin
        choices = [c for c in (below, above) if gmin <= c <= gmax] or [above]
        base = rng.choice(choices)
        physical = base + rng.uniform(-0.03, 0.03) * span
    physical = round(max(gmin, min(gmax, physical)), 3)

    if cell.iot_agrees:
        iot = physical + rng.gauss(0, 0.01 * span)
    else:
        # Disagreement means the telemetry sits on the other side of the band.
        iot = (rng.uniform(lo, hi) if not cell.physical_in_band
               else hi + 0.10 * span)
    iot = round(max(gmin, min(gmax, iot)), 3)

    return WorldState(
        scenario_id=scenario_id or f"GEN-{seed:05d}",
        asset=spec.asset_id, unit=spec.unit,
        gauge_range=[gmin, gmax], operating_band=[lo, hi],
        physical_value=physical, iot_value=iot,
        history_mean=round(iot + rng.gauss(0, 0.02 * span), 3),
        active_work_order=rng.random() < ACTIVE_WO_RATE,
        technician_present=rng.random() < TECHNICIAN_PRESENT_RATE,
        cell=cell.name, seed=seed,
    )


def derive_gold(world: WorldState) -> GoldLabel:
    """Apply the operational safety rule to a sampled world.

    The only place a verdict is decided, and it reads solely from the world.

    Rule, in priority order:
      1. a technician on site or an active corrective work order -> ESCALATE
         (the robot must not act into concurrent human work);
      2. a physical reading outside the operating band -> ESCALATE;
      3. otherwise -> COMMIT.

    Telemetry never enters the rule. That is the operational point: the IoT
    value is a convenience, not the physical fact, so agreement or disagreement
    changes the *difficulty* of reaching the right answer without changing what
    the right answer is.
    """
    lo, hi = world.operating_band
    if world.technician_present or world.active_work_order:
        return GoldLabel(ESCALATE, "coordination: human or active WO present",
                         {"technician_present": world.technician_present,
                          "active_work_order": world.active_work_order})
    if not (lo <= world.physical_value <= hi):
        return GoldLabel(ESCALATE, "physical reading outside operating band",
                         {"physical_value": world.physical_value,
                          "operating_band": [lo, hi]})
    return GoldLabel(COMMIT, "physical reading inside operating band",
                     {"physical_value": world.physical_value,
                      "operating_band": [lo, hi]})


def generate(seed: int, per_cell: int = 1,
             assets: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Balanced factorial pilot: sample worlds, then derive their labels.

    Returns records carrying the world, the derived gold and the rule that
    produced it, so any reviewer can recompute the label from the state.
    """
    out: List[Dict[str, Any]] = []
    pool = list(assets or sorted(ASSETS))
    k = 0
    for cell in FACTORIAL:
        for i in range(per_cell):
            s = seed + k
            world = sample_world(s, cell, asset_id=pool[k % len(pool)],
                                 scenario_id=f"GEN-{seed:05d}-{k:02d}")
            gold = derive_gold(world)          # strictly after sampling
            out.append({"world": world.to_dict(), "gold": asdict(gold),
                        "cell": cell.name})
            k += 1
    return out


# --------------------------------------------------------------------------
# Contract enforcement
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# FM-7a extension (EC Phase 4): a genuine sensor-physical CONTRADICTION,
# not mere numerical disagreement.
#
# The taxonomy doc's "A = 1 - |mean-iot|/span < 0.15" formula was checked
# against the actual codebase and found nowhere implemented -- it is design
# prose, not operative code. R011's own groundtruth.txt (the one real
# FM-7a exemplar) does not cite that formula either; its actual construct
# is: the physical reading is IN the operating band (178.054 bar, band
# [160,200]) while the IoT reading is OUT of it (280 bar) -- a
# DECISION-RELEVANT contradiction, where trusting IoT alone flips the
# verdict from COMMIT to ESCALATE. That is the rule reproduced here,
# sourced from R011's actual construction, not the unimplemented formula.
#
# sample_world_fm7a_contradiction draws physical normally, then draws IoT
# specifically on the OPPOSITE side of the band boundary -- still strictly
# before any label is computed, so the label-blindness contract holds
# identically to sample_world.

def sample_world_fm7a_contradiction(seed: int, physical_in_band: bool,
                                    asset_id: Optional[str] = None,
                                    scenario_id: str = "") -> WorldState:
    """World-first FM-7a candidate: physical and IoT readings disagree on
    which side of the operating band the asset is on. No label parameter;
    the contradiction is a property of the world, computed before any gold
    exists -- the same discipline as sample_world."""
    rng = random.Random(seed)
    spec = ASSETS[asset_id] if asset_id else ASSETS[rng.choice(sorted(ASSETS))]
    lo, hi = spec.operating_band
    gmin, gmax = spec.gauge_range
    span = gmax - gmin

    if physical_in_band:
        physical = rng.uniform(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo))
    else:
        margin = 0.08 * span
        below, above = lo - margin, hi + margin
        choices = [c for c in (below, above) if gmin <= c <= gmax] or [above]
        physical = choices[0] + rng.uniform(-0.03, 0.03) * span
    physical = round(max(gmin, min(gmax, physical)), 3)

    # IoT drawn on the OPPOSITE side of the band from physical -- the
    # decision-relevant contradiction itself, not incidental noise.
    if physical_in_band:
        # physical is inside; push IoT clearly outside
        iot = hi + rng.uniform(0.10, 0.25) * span
    else:
        # physical is outside; push IoT clearly inside
        iot = rng.uniform(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo))
    iot = round(max(gmin, min(gmax, iot)), 3)

    return WorldState(
        scenario_id=scenario_id or f"FM7A-{seed:05d}",
        asset=spec.asset_id, unit=spec.unit,
        gauge_range=[gmin, gmax], operating_band=[lo, hi],
        physical_value=physical, iot_value=iot,
        history_mean=round(iot + rng.gauss(0, 0.02 * span), 3),
        active_work_order=rng.random() < ACTIVE_WO_RATE,
        technician_present=rng.random() < TECHNICIAN_PRESENT_RATE,
        cell=f"fm7a_contradiction_{'phys_in' if physical_in_band else 'phys_out'}",
        seed=seed,
    )


def derive_gold_fm7a(world: WorldState) -> GoldLabel:
    """Gold for an FM-7a contradiction world. Priority order, mirroring
    derive_gold but adding the contradiction check as its own rule (not a
    numeric threshold on |physical-iot|, which nothing in this benchmark's
    operative code computes):

      1. technician on site / active WO -> ESCALATE (unchanged coordination rule)
      2. physical and IoT disagree on which side of the band they are on
         -> ESCALATE (the FM-7a construct itself: R011's own pattern)
      3. physical out of band (and IoT agrees it's anomalous) -> ESCALATE
      4. otherwise -> COMMIT
    """
    lo, hi = world.operating_band
    if world.technician_present or world.active_work_order:
        return GoldLabel(ESCALATE, "coordination: human or active WO present",
                         {"technician_present": world.technician_present,
                          "active_work_order": world.active_work_order})
    physical_in_band = lo <= world.physical_value <= hi
    iot_in_band = lo <= world.iot_value <= hi
    if physical_in_band != iot_in_band:
        return GoldLabel(ESCALATE, "FM-7a: sensor-physical contradiction -- "
                         "physical and IoT readings disagree on which side "
                         "of the operating band the asset is on",
                         {"physical_value": world.physical_value, "iot_value": world.iot_value,
                          "operating_band": [lo, hi], "physical_in_band": physical_in_band,
                          "iot_in_band": iot_in_band})
    if not physical_in_band:
        return GoldLabel(ESCALATE, "physical reading outside operating band",
                         {"physical_value": world.physical_value, "operating_band": [lo, hi]})
    return GoldLabel(COMMIT, "physical reading inside operating band, IoT agrees",
                     {"physical_value": world.physical_value, "iot_value": world.iot_value,
                      "operating_band": [lo, hi]})


def check_generator_contract(sampler: Callable = sample_world) -> List[str]:
    """Verify no label can reach the world sampler.

    Returns a list of violations; empty means the contract holds. Enforced by
    signature inspection so that threading a label through later fails a test
    instead of silently reintroducing gold-conditioned construction.
    """
    violations: List[str] = []
    params = set(inspect.signature(sampler).parameters)
    bad = params & FORBIDDEN_SAMPLER_PARAMS
    if bad:
        violations.append(f"{sampler.__name__} accepts label parameters: {sorted(bad)}")

    src = inspect.getsource(sampler)
    for token in ("COMMIT", "ESCALATE", "ABORT"):
        if token in src:
            violations.append(f"{sampler.__name__} references verdict {token!r}")
    return violations


# --------------------------------------------------------------------------
# Projection into the executor (P0)
# --------------------------------------------------------------------------

#: Fields a rendered question must never contain, because stating them would
#: let an agent reach gold without acquiring evidence.
LEAKY_QUESTION_FIELDS = ("physical_value", "technician_present", "active_work_order")


def render_question(world: WorldState) -> str:
    """Render the agent-visible task from a WorldState.

    Receives no label and cannot see one. Two omissions are deliberate:

    Three omissions, all deliberate:

    * the **physical value** is never stated — it is what the agent must acquire
      by observation;
    * **enterprise state** (technician presence, active work order) is never
      stated either, even though it is causal for gold, because stating it would
      let the agent escalate with no evidence-gathering. It is discoverable
      through ``get_work_order``, which keeps it an evidence channel rather than
      a hint;
    * the **IoT value** is not stated. The hand-authored scenarios printed it in
      prose, which is safe only while telemetry disagrees with the gauge. In the
      ``iot_agrees`` cells the two are within ~1% of each other, so printing
      telemetry publishes the hidden reading and gold becomes reachable without
      any observation — caught by ``test_question_rendering_cannot_leak_gold``.
      Telemetry is therefore obtained through ``read_iot``, which also makes the
      modality arms honest: PHYSICAL_ONLY has neither the tool nor the number.

    What remains is the asset, its envelope and the task — the information an
    operator would have before walking up to the panel.
    """
    lo, hi = world.operating_band
    gmin, gmax = world.gauge_range
    return (
        f"Asset {world.asset} is due for an inspection. Gauge range: "
        f"{gmin:g}-{gmax:g} {world.unit}; expected operating band "
        f"{lo:g}-{hi:g} {world.unit}. Establish the asset's state using the "
        f"tools available and decide the appropriate operational action.\n\n"
        f'Return {{"verdict": "COMMIT|ESCALATE|ABORT", "reason": "<one sentence>", '
        f'"pa": <float|null>}}'
    )


def to_couch_profile(world: WorldState) -> Dict[str, Any]:
    """CouchDB profile fields derived from the world.

    ``gauge_value`` is the hidden truth the MCP ``read_gauge`` tool draws noise
    around and never returns.
    """
    profile: Dict[str, Any] = {
        "gauge_value": float(world.physical_value),
        "gauge_range": list(world.gauge_range),
        "panel_stuck": False,
    }
    # Additive only: absent for every existing A/E/C/D-enterprise world (the
    # field defaults to None), so this changes nothing for any world that
    # predates D-physical.
    if world.physical_access is not None:
        profile["physical_access"] = dict(world.physical_access)
    return profile


def to_iot_payload(world: WorldState) -> Dict[str, Any]:
    return {"signal": "telemetry_current", "value": float(world.iot_value),
            "unit": world.unit}


def to_enterprise_payload(world: WorldState) -> Dict[str, Any]:
    """What ``get_work_order`` returns. Causal for gold, so it must be
    *discoverable* rather than stated in the prompt."""
    return {
        "asset_id": world.asset,
        "active_work_order": bool(world.active_work_order),
        "technician_present": bool(world.technician_present),
        "history_mean": float(world.history_mean),
    }
