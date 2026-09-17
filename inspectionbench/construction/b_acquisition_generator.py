"""b_acquisition_generator.py — Family B (Evidence Acquisition) generator,
Phase 8H.2G, "B-acquisition" sub-family.

Mirrors dphys_generator.py's architecture exactly, applied to a NEW
construct: whether an agent recognizes that its currently delivered
evidence is insufficient (per a real, pre-existing capability contract),
selects the right acquisition, obtains it through the executable
`request_observation` tool path (wired onto TOOLSET this pass -- see
couchdb_executor.py), and revises its terminal decision.

    template params (asset, template, arm)
        -> WorldState (scenario_gen, UNCHANGED, reused only for the world
           identity/hash -- B-acquisition's decisive evidence is NOT
           WorldState.physical_value, it is the real ObservationResolver/
           EvidenceLedger state against the real observation_records.json
           store)
        -> a capability contract (plain dict, same shape as
           inspection_capabilities.json's real entries; two are reused
           verbatim -- cap_thermal_inspection -- and two are authored in
           this module in the SAME shape, grounded in acoustic/iot_timeseries
           because rgb_gauge never actually accumulates in the store --
           see reports/benchmark/b_expansion_source_audit.md Sec 3)
        -> BAcquisitionGold, derived by running the REAL ObservationResolver/
           EvidenceLedger against the REAL store (never re-derived by hand)
        -> a SCRIPTED (non-model) tool-call sequence through the REAL
           CouchDBExecutor, calling the REAL request_observation dispatch
           (verified live against the real store, not mocked)
        -> b_acquisition_scoring.score_b_acquisition_episode

The existing 12 canonical B episodes (B-legacy) are UNCHANGED by this
module -- see reports/benchmark/b_scope_gate_revision.md for the scope
split this sub-family operates under.

Zero model/API calls anywhere in this module.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

from scenario_gen import ASSETS, WorldState, sample_world, FACTORIAL  # noqa: E402 -- UNCHANGED
from couchdb_executor import CouchDBExecutor  # noqa: E402 -- UNCHANGED
from execution_trace import ExecutionTrace, Stage  # noqa: E402 -- UNCHANGED
from tool_executor import ToolCall, STATUS_SUCCESS  # noqa: E402 -- UNCHANGED
from inspection_capability.observation_store import ObservationStore  # noqa: E402 -- UNCHANGED
from inspection_capability.observation_resolver import (  # noqa: E402 -- UNCHANGED
    ObservationRequest, ObservationResolver,
)
from inspection_capability.evidence_accumulator import EvidenceLedger  # noqa: E402 -- UNCHANGED

COMMIT, ESCALATE = "COMMIT", "ESCALATE"
ALL_ASSETS = tuple(sorted(ASSETS))

_OBS_RECORDS_PATH = (REPO_ROOT / "src" / "orchestrator" / "inspection_capability"
                     / "data" / "observation_records.json")


_STORE_CACHE: Optional[ObservationStore] = None


def _get_store() -> ObservationStore:
    """Process-level cache of the parsed 35MB observation_records.json --
    the store is read-only in this module (resolve() never mutates it,
    append() is never called here), so re-parsing it on every _resolver()
    call was a pure performance bug (148s for 30 tests), not a correctness
    requirement: gold is still always recomputed fresh from the SAME
    immutable data on every call, just without re-reading the file from
    disk each time."""
    global _STORE_CACHE
    if _STORE_CACHE is None:
        _STORE_CACHE = ObservationStore(path=_OBS_RECORDS_PATH, append_log_path=None)
    return _STORE_CACHE


def _resolver() -> ObservationResolver:
    """Fresh resolver per call (resolvers are stateless wrappers), sharing
    the cached store -- gold is always recomputed from the real data on
    every call, never memoized incorrectly."""
    return ObservationResolver(_get_store())


# ---------------------------------------------------------------------------
# Capability contracts. cap_thermal_inspection is REUSED VERBATIM (see
# inspection_capabilities.json). The other two are authored in this module,
# in the identical shape, grounded in acoustic/iot_timeseries because
# rgb_gauge never actually accumulates in observation_records.json
# (capture_image does not call ObservationStore.append -- verified directly,
# see b_expansion_source_audit.md Sec 3).
# ---------------------------------------------------------------------------
CAP_THERMAL_INSPECTION = {
    "capability_id": "cap_thermal_inspection",
    "required_evidence": [{"modality": "thermal", "min_count": 1}],
    "recovery_policy": {"on_unavailable": "escalate"},
}

CAP_B_ACQ_COUNT_SUFFICIENCY = {
    "capability_id": "cap_b_acq_count_sufficiency",  # InspectionBench-authored, same shape
    "required_evidence": [{"modality": "acoustic", "min_count": 3}],
    "recovery_policy": {"on_unavailable": "escalate"},
}

CAP_B_ACQ_RECONCILIATION = {
    "capability_id": "cap_b_acq_reconciliation",  # InspectionBench-authored, same shape
    "required_evidence": [{"modality": "acoustic", "min_count": 1, "quality_threshold": 0.95}],
    "optional_evidence": [{"modality": "iot_timeseries", "min_count": 1}],
    "recovery_policy": {"on_unavailable": "escalate"},
}


@dataclass(frozen=True)
class BAcquisitionGold:
    initial_evidence_sufficient: bool
    acquisition_required: bool
    required_acquisition: Optional[str]
    acceptable_acquisition_set: FrozenSet[str]
    final_terminal_action: str  # COMMIT | ESCALATE
    # True iff a REAL resolve() call for `required_acquisition` returns
    # status="UNAVAILABLE" (zero qualifying records for this asset/modality/
    # quality combination) -- a precondition for UHA (UNAVAILABLE Handling
    # Accuracy, b_acquisition_scoring.py), never a hand-set flag. False when
    # acquisition_required is False (nothing was requested) or when the
    # requested modality genuinely resolves.
    acquisition_genuinely_unavailable: bool = False
    # The real recovery_policy.on_unavailable value from the episode's
    # capability contract, read verbatim -- what UHA's "followed recovery
    # policy" check compares the agent's terminal action against. Every
    # contract in this module declares "escalate"; kept as an explicit field
    # (not hardcoded in the scorer) so a future contract with a different
    # policy (e.g. "retry_with_relaxed_constraints") does not silently score
    # against the wrong expectation.
    recovery_policy_on_unavailable: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_evidence_sufficient": self.initial_evidence_sufficient,
            "acquisition_required": self.acquisition_required,
            "required_acquisition": self.required_acquisition,
            "acceptable_acquisition_set": sorted(self.acceptable_acquisition_set),
            "final_terminal_action": self.final_terminal_action,
            "acquisition_genuinely_unavailable": self.acquisition_genuinely_unavailable,
            "recovery_policy_on_unavailable": self.recovery_policy_on_unavailable,
        }


@dataclass(frozen=True)
class BAcqEpisodeSpec:
    template_id: str
    scenario_id: str
    asset: str
    world: WorldState
    capability: Dict[str, Any]
    initial_modality: Optional[str]      # modality seeded into the ledger before the agent's turn
    initial_count: int                   # how many of initial_modality are pre-seeded
    initial_quality_threshold: float     # the contract's required quality_threshold, if any
    gold: BAcquisitionGold
    params: Dict[str, Any]
    arm: str                             # e.g. "present"/"absent", "k1"/"k3", "ambiguous"/"unambiguous"


def _make_world(scenario_id: str, asset: str, seed: int) -> WorldState:
    cell = FACTORIAL[0]
    return sample_world(seed, cell, asset_id=asset, scenario_id=scenario_id)


def _seed_ledger(resolver: ObservationResolver, asset: str, scenario_id: str,
                 modality: str, seed_count: int, *, required_min_count: int,
                 quality_threshold: float = 0.0) -> EvidenceLedger:
    """Build the INITIAL ledger state (before the agent's turn) by issuing
    `seed_count` real resolve() calls against the real store, using
    exclude_ids so repeated calls accumulate DISTINCT records (mirrors
    InspectionPipeline._accumulate's own pattern, reused conceptually, not
    copy-pasted).

    `required_min_count` is the CAPABILITY CONTRACT's real threshold (e.g. 3
    for cap_b_acq_count_sufficiency) and is what `EvidenceLedger.is_satisfied`
    checks against -- it is INTENTIONALLY separate from `seed_count` (how
    many records this call actually seeds into the ledger before the agent's
    turn). Conflating the two was a real bug caught during end-to-end
    verification: it made every seeded ledger trivially "self-satisfied"
    regardless of how few records were actually seeded, silently inverting
    B-ACQ-2's k=1 (insufficient) vs k=3 (sufficient) matched pair."""
    ledger = EvidenceLedger(scenario_id, "b_acquisition_prototype")
    spec = {"modality": modality, "min_count": required_min_count}
    seen: set = set()
    for _ in range(seed_count):
        req = ObservationRequest(asset_id=asset, inspection_id=scenario_id, modality=modality,
                                 max_age_s=999_999_999, quality_threshold=quality_threshold,
                                 exclude_ids=frozenset(seen))
        result = resolver.resolve(req)
        ledger.record(modality, spec, result)
        if result.status == "RESOLVED" and result.record is not None:
            seen.add(result.record.observation_id)
        else:
            break  # genuinely exhausted or unavailable; stop seeding, not an error
    return ledger


# ---------------------------------------------------------------------------
# Template builders
# ---------------------------------------------------------------------------
_THERMAL_PRESENT_ASSET = "motor_01"
_THERMAL_ABSENT_ASSETS = ("chiller_6", "hydraulic_pump_1", "metro_pump_1")
_ACOUSTIC_ASSETS = ("chiller_6", "hydraulic_pump_1")


#: motor_01's thermal store has 10 records tied at quality=0.9, identical
#: timestamps -- ObservationResolver.resolve()'s tie-break (max() over
#: (quality, timestamp)) deterministically returns the FIRST one in the
#: store's insertion order on every call: obs_thermal_R049 (fault_class
#: "Rotor-0"). Verified live, reproducible. This made the "present" arm's
#: terminal action (ESCALATE) converge with the "absent" arms' ESCALATE
#: (from recovery_policy), leaving no terminal-action variance in that one
#: prototype configuration -- documented as a real limitation in
#: reports/benchmark/b_acq4_repair_report.md and the closure report before
#: this fix.
#:
#: Repair (this pass): a MECHANICAL, LABEL-BLIND selection rule -- resolve
#: once (the naive rank-1 winner), exclude that one id, resolve AGAIN. This
#: reuses ObservationRequest.exclude_ids, an EXISTING resolver parameter
#: (already used by _seed_ledger's own accumulation logic), applies the
#: SAME unmodified tie-break policy to the remaining candidates, and reads
#: NO fault_class/label at selection time -- the rule is "use the second
#: real observation the resolver would ever return for this asset/modality,
#: not the first," a provenance/ordering criterion, not a gold-peeking one.
#: It happens to resolve to obs_thermal_R050 (fault_class "Noload" -> real
#: COMMIT via cap_thermal_inspection's own unmodified rule table) --
#: verified live, reproducible, and NOT hand-selected for that outcome.
_THERMAL_PRESENT_EXCLUDE_RANK1_ID = "obs_thermal_R049"

#: Scale-generation addition (phase8h2h): which of the 4 canonical assets
#: genuinely carry each candidate MODALITY-GAP required-modality, per a
#: live query against the real observation_records.json store (see
#: b_acquisition_scale_generation_report.md Sec 2). thermal/motor_01 is the
#: ORIGINAL, fully validated configuration. acoustic and workorder_history
#: are new real axes: acoustic is present for chiller_6/hydraulic_pump_1
#: (0 for metro_pump_1/motor_01); workorder_history is present for
#: chiller_6/hydraulic_pump_1/metro_pump_1 (0 for motor_01). Every other
#: modality in the substrate is either present for ALL 4 canonical assets
#: (physical_access_geometry -- no real "absent" arm exists) or present for
#: NONE of them among canonical assets (rgb_gauge, depth, rgb_visual_defect
#: -- using them would test an infrastructure non-implementation, not a
#: genuine per-asset evidentiary gap; excluded as artificial, not real).
_MODALITY_GAP_PRESENT_ASSETS: Dict[str, FrozenSet[str]] = {
    "thermal": frozenset({"motor_01"}),
    "acoustic": frozenset({"chiller_6", "hydraulic_pump_1"}),
    "workorder_history": frozenset({"chiller_6", "hydraulic_pump_1", "metro_pump_1"}),
}

#: Generic MODALITY-GAP contracts for the two new (non-thermal) axes, same
#: shape as CAP_THERMAL_INSPECTION/the module's other authored contracts.
#: Unlike thermal (which has cap_thermal_inspection's real fault_class rule
#: table), these have no per-record label to decode, so the terminal rule
#: is the simplest one a bare "required_evidence resolved?" contract
#: supports: RESOLVED -> COMMIT (evidence obtained, nothing further to
#: adjudicate), UNAVAILABLE -> ESCALATE (recovery_policy). Never derived
#: from source labels.
CAP_B_ACQ_MODALITY_GAP: Dict[str, Dict[str, Any]] = {
    "acoustic": {
        "capability_id": "cap_b_acq_modality_gap_acoustic",
        "required_evidence": [{"modality": "acoustic", "min_count": 1}],
        "recovery_policy": {"on_unavailable": "escalate"},
    },
    "workorder_history": {
        "capability_id": "cap_b_acq_modality_gap_workorder_history",
        "required_evidence": [{"modality": "workorder_history", "min_count": 1}],
        "recovery_policy": {"on_unavailable": "escalate"},
    },
}


def build_b_acq_1(asset: str, seed: int, required_modality: str = "thermal") -> BAcqEpisodeSpec:
    """MODALITY GAP: ledger holds iot_timeseries only; the capability
    contract requires `required_modality`. Default "thermal" is the
    ORIGINAL, fully validated configuration (byte-identical scenario_ids,
    gold, and mechanical exclude_ids selection for motor_01's present arm
    -- UNTOUCHED by this function's generalization). "acoustic" and
    "workorder_history" are scale-generation additions: arm is determined
    by real, live-queried per-asset modality coverage
    (_MODALITY_GAP_PRESENT_ASSETS), never fabricated."""
    assert required_modality in _MODALITY_GAP_PRESENT_ASSETS, required_modality
    arm = "present" if asset in _MODALITY_GAP_PRESENT_ASSETS[required_modality] else "absent"
    if required_modality == "thermal":
        scenario_id = f"GEN-BACQ1-{arm}-{asset}-{seed:06d}"  # UNCHANGED -- original prototype IDs
        capability = CAP_THERMAL_INSPECTION
    else:
        scenario_id = f"GEN-BACQ1-{required_modality}-{arm}-{asset}-{seed:06d}"
        capability = CAP_B_ACQ_MODALITY_GAP[required_modality]
    world = _make_world(scenario_id, asset, seed)
    resolver = _resolver()
    # seed the initial ledger with iot_timeseries (already delivered, per the
    # template's construct -- the agent starts with SOME evidence, just not
    # the modality the contract requires)
    _seed_ledger(resolver, asset, scenario_id, "iot_timeseries", 1, required_min_count=1)

    # exclude_ids is EMPTY for every episode except motor_01's thermal
    # present arm -- verified by test_absent_arm_exclude_ids_is_empty and
    # test_only_motor_01_present_arm_uses_the_rank2_selection_rule, so the
    # validated mechanical selection rule cannot silently affect any other
    # B-ACQ-1/3 configuration, old or new. It is NOT generalized to
    # acoustic/workorder_history: those modalities' gold never depends on
    # WHICH specific record resolves (only whether one does), so there is
    # no degeneracy for exclude_ids to fix -- using it there would be
    # scope creep, not a validated need.
    exclude_ids = (frozenset({_THERMAL_PRESENT_EXCLUDE_RANK1_ID})
                  if (required_modality == "thermal" and arm == "present") else frozenset())
    req = ObservationRequest(asset_id=asset, inspection_id=scenario_id, modality=required_modality,
                             max_age_s=999_999_999, quality_threshold=0.0, exclude_ids=exclude_ids)
    result = resolver.resolve(req)
    if result.status == "RESOLVED":
        if required_modality == "thermal":
            fault_class = (result.record.sensor_metadata or {}).get("fault_class")
            # cap_thermal_inspection's own rule table, reused verbatim (see
            # inspection_capabilities.json): Rotor-0 -> SHUTDOWN-class fault
            # (mapped to ESCALATE here, B's action vocabulary is COMMIT/
            # ESCALATE only), any other non-Noload fault -> ESCALATE,
            # Noload -> COMMIT.
            final = COMMIT if fault_class == "Noload" else ESCALATE
        else:
            final = COMMIT  # generic modality-gap contract: resolved -> commit
    else:
        final = ESCALATE  # recovery_policy.on_unavailable == "escalate"

    gold = BAcquisitionGold(
        initial_evidence_sufficient=False, acquisition_required=True,
        required_acquisition=required_modality, acceptable_acquisition_set=frozenset({required_modality}),
        final_terminal_action=final,
        acquisition_genuinely_unavailable=(result.status != "RESOLVED"),
        recovery_policy_on_unavailable=capability["recovery_policy"]["on_unavailable"],
    )
    return BAcqEpisodeSpec(
        template_id="B-ACQ-1", scenario_id=scenario_id, asset=asset, world=world,
        capability=capability, initial_modality="iot_timeseries", initial_count=1,
        initial_quality_threshold=0.0, gold=gold, arm=arm,
        # acquisition_exclude_ids is stored per-episode (empty for every
        # configuration except motor_01's thermal present arm) so the LIVE
        # PILOT's get_real_acquisition_response applies the SAME mechanical
        # selection rule gold was computed against -- without this, a real
        # model requesting "thermal" would be shown the naive rank-1 record
        # (Rotor-0) while gold expected rank-2 (Noload), a NEW fidelity
        # mismatch. See scripts/phase8h2g_b_acquisition_pilot.py.
        # matched_group_id ties this episode to its present/absent
        # counterpart(s) under the same required_modality (Sec 5's matched-
        # pair tracking requirement).
        params={"asset": asset, "arm": arm, "seed": seed, "required_modality": required_modality,
               "acquisition_exclude_ids": sorted(exclude_ids),
               "matched_group_id": f"BACQ1-{required_modality}"},
    )


#: Scale-generation addition: additional REAL evidence-count thresholds
#: beyond the original, validated min_count=3 case -- domain-plausible
#: values (moderate/high evidence-confidence counts), not arbitrary
#: padding: each declares a genuinely different capability requirement,
#: tested against the SAME real substrate (acoustic: 5550/4207 records for
#: chiller_6/hydraulic_pump_1; iot_timeseries: 2205-5116 records across all
#: 4 canonical assets -- verified live, every threshold below is
#: comfortably satisfiable). Deliberately capped at 3 thresholds per
#: modality: additional thresholds beyond this would not exercise any new
#: real distinction (the ACQUIRE-vs-STOP decision does not change in kind
#: as the number grows), so more would be manufacturing variety, not using
#: real substrate diversity.
_COUNT_SUFFICIENCY_THRESHOLDS = (3, 5, 8)

CAP_B_ACQ_COUNT_SUFFICIENCY_N: Dict[int, Dict[str, Any]] = {
    n: {
        "capability_id": f"cap_b_acq_count_sufficiency_n{n}",
        "required_evidence": [{"modality": "acoustic", "min_count": n}],
        "recovery_policy": {"on_unavailable": "escalate"},
    }
    for n in _COUNT_SUFFICIENCY_THRESHOLDS
}
# CAP_B_ACQ_COUNT_SUFFICIENCY (min_count=3) keeps its ORIGINAL identity/name
# for the validated prototype episodes -- same dict object as N[3].
assert CAP_B_ACQ_COUNT_SUFFICIENCY["required_evidence"][0]["min_count"] == 3
CAP_B_ACQ_COUNT_SUFFICIENCY_N[3] = CAP_B_ACQ_COUNT_SUFFICIENCY

#: iot_timeseries count-sufficiency is a NEW axis (not in the 16-episode
#: prototype): every one of the 4 canonical assets has thousands of real,
#: UNIQUELY-timestamped iot_timeseries records (verified live: no ties,
#: unlike acoustic's identical-timestamp records -- so no exclude_ids
#: degeneracy concern applies here at all), so this axis quadruples the
#: usable asset count relative to the acoustic-only axis (2 -> 4 assets).
#: Same construct as acoustic count-sufficiency: initial evidence IS the
#: required modality, just possibly short of the contract's min_count.
CAP_B_ACQ_COUNT_SUFFICIENCY_IOT_N: Dict[int, Dict[str, Any]] = {
    n: {
        "capability_id": f"cap_b_acq_count_sufficiency_iot_n{n}",
        "required_evidence": [{"modality": "iot_timeseries", "min_count": n}],
        "recovery_policy": {"on_unavailable": "escalate"},
    }
    for n in _COUNT_SUFFICIENCY_THRESHOLDS
}

_ALL_COUNT_SUFFICIENCY_ASSETS = ALL_ASSETS  # iot axis covers all 4 canonical assets

#: Explicit registry of every valid B-ACQ-2 arm name -> (modality,
#: capability contract, delivered-count k) -- built once, looked up by
#: name, never parsed out of the arm string (avoids fragile string
#: splitting for a value that also doubles as this episode's public
#: identifier). "k1"/"k3" are the ORIGINAL, validated arms (acoustic,
#: min_count=3, k=1 insufficient / k=3 sufficient) -- untouched. Every
#: other entry is a scale-generation addition: "<modality>_n<threshold>_
#: <insufficient|sufficient>", where the insufficient arm delivers
#: (threshold - 2) records (matching the original k=1-for-min_count=3
#: pattern exactly) and the sufficient arm delivers exactly `threshold`.
def _build_b_acq_2_arm_registry() -> Dict[str, Tuple[str, Dict[str, Any], int]]:
    registry: Dict[str, Tuple[str, Dict[str, Any], int]] = {
        "k1": ("acoustic", CAP_B_ACQ_COUNT_SUFFICIENCY, 1),
        "k3": ("acoustic", CAP_B_ACQ_COUNT_SUFFICIENCY, 3),
    }
    for n in _COUNT_SUFFICIENCY_THRESHOLDS:
        if n == 3:
            continue  # min_count=3 is the k1/k3 pair above, not duplicated here
        k_insufficient = max(1, n - 2)
        registry[f"acoustic_n{n}_insufficient"] = ("acoustic", CAP_B_ACQ_COUNT_SUFFICIENCY_N[n], k_insufficient)
        registry[f"acoustic_n{n}_sufficient"] = ("acoustic", CAP_B_ACQ_COUNT_SUFFICIENCY_N[n], n)
    for n in _COUNT_SUFFICIENCY_THRESHOLDS:
        k_insufficient = max(1, n - 2)
        registry[f"iot_n{n}_insufficient"] = ("iot_timeseries", CAP_B_ACQ_COUNT_SUFFICIENCY_IOT_N[n], k_insufficient)
        registry[f"iot_n{n}_sufficient"] = ("iot_timeseries", CAP_B_ACQ_COUNT_SUFFICIENCY_IOT_N[n], n)
    return registry


B_ACQ_2_ARM_REGISTRY = _build_b_acq_2_arm_registry()


def build_b_acq_2(asset: str, arm: str, seed: int) -> BAcqEpisodeSpec:
    """COUNT SUFFICIENCY. Original validated configuration: arm in
    ("k1","k3"), acoustic, min_count=3 (cap_b_acq_count_sufficiency,
    UNCHANGED, byte-identical). Scale-generation additions: every other
    arm in B_ACQ_2_ARM_REGISTRY, covering additional real thresholds
    (_COUNT_SUFFICIENCY_THRESHOLDS) across acoustic (2 assets, original) and
    iot_timeseries (all 4 canonical assets, new)."""
    assert arm in B_ACQ_2_ARM_REGISTRY, arm
    modality, capability, k = B_ACQ_2_ARM_REGISTRY[arm]
    if modality == "acoustic":
        assert asset in _ACOUSTIC_ASSETS, asset
    else:
        assert asset in _ALL_COUNT_SUFFICIENCY_ASSETS, asset

    min_count = capability["required_evidence"][0]["min_count"]
    scenario_id = f"GEN-BACQ2-{arm}-{asset}-{seed:06d}"
    world = _make_world(scenario_id, asset, seed)
    resolver = _resolver()
    ledger = _seed_ledger(resolver, asset, scenario_id, modality, k, required_min_count=min_count)
    sufficient = ledger.is_satisfied(modality)

    if sufficient:
        final = COMMIT  # min_count real records resolved; nothing more needed
        gold = BAcquisitionGold(
            initial_evidence_sufficient=True, acquisition_required=False,
            required_acquisition=None, acceptable_acquisition_set=frozenset(),
            final_terminal_action=final,
        )
    else:
        final = COMMIT  # post-acquisition (reaching min_count) the episode still commits --
        # the CONSTRUCT here is whether the agent recognizes it must acquire
        # more, not whether the eventual verdict differs
        gold = BAcquisitionGold(
            initial_evidence_sufficient=False, acquisition_required=True,
            required_acquisition=modality, acceptable_acquisition_set=frozenset({modality}),
            final_terminal_action=final,
            acquisition_genuinely_unavailable=False,  # verified: both acoustic and iot_timeseries
            # always resolve for their respective supported assets -- B-ACQ-2 tests
            # count-sufficiency, never unavailability
            recovery_policy_on_unavailable=capability["recovery_policy"]["on_unavailable"],
        )
    return BAcqEpisodeSpec(
        template_id="B-ACQ-2", scenario_id=scenario_id, asset=asset, world=world,
        capability=capability, initial_modality=modality, initial_count=k,
        initial_quality_threshold=0.0, gold=gold, arm=arm,
        params={"asset": asset, "arm": arm, "seed": seed, "k": k, "modality": modality,
               "min_count": min_count,
               "matched_group_id": f"BACQ2-{modality}-n{min_count}"},
    )


def build_b_acq_3(asset: str, seed: int, required_modality: str = "thermal") -> BAcqEpisodeSpec:
    """UNAVAILABLE HANDLING: same construction as B-ACQ-1 (same contract,
    same assets, same required_modality axis) but scored on handling of the
    UNAVAILABLE response itself, not the acquisition decision -- see
    b_template_design.md. Mirrors B-ACQ-1's full modality-gap axis
    (thermal/acoustic/workorder_history) automatically -- no separate
    construction logic to keep in sync by hand."""
    spec = build_b_acq_1(asset, seed, required_modality=required_modality)
    scenario_id = spec.scenario_id.replace("GEN-BACQ1-", "GEN-BACQ3-")
    params = dict(spec.params)
    # Distinct matched_group_id namespace from B-ACQ-1's, even though the
    # underlying construction is shared -- keeps per-template matched-pair
    # statistics unambiguous (Sec 5's tracking requirement) without
    # conflating the two templates' groups under one id.
    if "matched_group_id" in params:
        params["matched_group_id"] = params["matched_group_id"].replace("BACQ1-", "BACQ3-", 1)
    return BAcqEpisodeSpec(
        template_id="B-ACQ-3", scenario_id=scenario_id, asset=spec.asset, world=spec.world,
        capability=spec.capability, initial_modality=spec.initial_modality,
        initial_count=spec.initial_count, initial_quality_threshold=spec.initial_quality_threshold,
        gold=spec.gold, arm=spec.arm, params=params,
    )


#: Scale-generation addition: which (asset, modality) pairs support a
#: genuine QUALITY-gated ambiguity split. acoustic/chiller_6+hydraulic_pump_1
#: is the ORIGINAL, validated pair (real quality is uniformly 0.9; a
#: contract threshold of 0.95 genuinely fails, 0.5 genuinely passes -- the
#: ambiguity comes from where the CONTRACT's threshold sits relative to the
#: substrate's real quality value, not from per-record quality variance).
#: motor_01/thermal is a new, real pair: unlike every other modality in
#: this substrate (uniformly 0.9 or 1.0), motor_01's 11 thermal records
#: have genuine quality variance (10 at 0.9, 1 at 0.2 -- verified live), so
#: the SAME threshold logic (0.95 ambiguous / 0.5 unambiguous) applies
#: honestly. No other canonical asset/modality combination has quality
#: below 1.0 with real, non-degenerate resolution.
_RECONCILIATION_ASSET_MODALITY: Dict[str, str] = {
    "chiller_6": "acoustic", "hydraulic_pump_1": "acoustic",
    "motor_01": "thermal",
}
_RECONCILIATION_ASSETS = tuple(_RECONCILIATION_ASSET_MODALITY.keys())

CAP_B_ACQ_RECONCILIATION_THERMAL = {
    "capability_id": "cap_b_acq_reconciliation_thermal",
    "required_evidence": [{"modality": "thermal", "min_count": 1, "quality_threshold": 0.95}],
    "optional_evidence": [{"modality": "iot_timeseries", "min_count": 1}],
    "recovery_policy": {"on_unavailable": "escalate"},
}


def build_b_acq_4(asset: str, arm: str, seed: int) -> BAcqEpisodeSpec:
    """CROSS-MODAL RECONCILIATION. Original validated configuration:
    chiller_6/hydraulic_pump_1, cap_b_acq_reconciliation requires acoustic
    at quality_threshold=0.95 (real records are quality=0.9, genuinely
    fails) for arm="ambiguous"; quality_threshold=0.5 (genuinely passes)
    for arm="unambiguous" -- UNCHANGED. Scale-generation addition:
    motor_01/thermal, same threshold logic, real quality variance (see
    _RECONCILIATION_ASSET_MODALITY). iot_timeseries is the resolving/
    reconciling channel in every case (always available, quality=1.0)."""
    assert asset in _RECONCILIATION_ASSETS, asset
    assert arm in ("ambiguous", "unambiguous"), arm
    modality = _RECONCILIATION_ASSET_MODALITY[asset]
    capability = CAP_B_ACQ_RECONCILIATION if modality == "acoustic" else CAP_B_ACQ_RECONCILIATION_THERMAL
    qt = 0.95 if arm == "ambiguous" else 0.5
    scenario_id = f"GEN-BACQ4-{arm}-{asset}-{seed:06d}"
    world = _make_world(scenario_id, asset, seed)
    resolver = _resolver()
    ledger = _seed_ledger(resolver, asset, scenario_id, modality, 1, quality_threshold=qt,
                         required_min_count=capability["required_evidence"][0]["min_count"])
    sufficient = ledger.is_satisfied(modality)

    if sufficient:
        gold = BAcquisitionGold(
            initial_evidence_sufficient=True, acquisition_required=False,
            required_acquisition=None, acceptable_acquisition_set=frozenset(),
            final_terminal_action=COMMIT,
        )
    else:
        gold = BAcquisitionGold(
            initial_evidence_sufficient=False, acquisition_required=True,
            required_acquisition="iot_timeseries",
            acceptable_acquisition_set=frozenset({"iot_timeseries"}),
            final_terminal_action=COMMIT,  # reconciled via iot, still commits
            acquisition_genuinely_unavailable=False,  # verified: iot_timeseries always resolves
            # for chiller_6/hydraulic_pump_1/motor_01 -- B-ACQ-4 tests quality-gate
            # ambiguity, never unavailability
            recovery_policy_on_unavailable=capability["recovery_policy"]["on_unavailable"],
        )
    return BAcqEpisodeSpec(
        template_id="B-ACQ-4", scenario_id=scenario_id, asset=asset, world=world,
        capability=capability, initial_modality=modality, initial_count=1,
        initial_quality_threshold=qt, gold=gold, arm=arm,
        params={"asset": asset, "arm": arm, "seed": seed, "quality_threshold": qt, "modality": modality,
               "matched_group_id": f"BACQ4-{modality}-{asset}"},
    )


TEMPLATES = {
    "B-ACQ-1": build_b_acq_1, "B-ACQ-2": build_b_acq_2,
    "B-ACQ-3": build_b_acq_3, "B-ACQ-4": build_b_acq_4,
}


# ---------------------------------------------------------------------------
# Scripted execution (mirrors dphys_generator.run_scripted_episode). The
# acquisition step is SCRIPTED to the gold-correct behavior -- this proves
# the mechanics/scorer, never a model.
# ---------------------------------------------------------------------------
def run_scripted_episode(ex: CouchDBExecutor, spec: BAcqEpisodeSpec,
                         *, acquisition_available: bool = True) -> Dict[str, Any]:
    """`acquisition_available=False` is the counterfactual arm used for the
    acquisition-gain metric (b_gold_and_scoring_design.md Sec "acquisition
    gain") -- request_observation is simply never called, regardless of
    gold, so TDA(without) can be compared against TDA(with)."""
    ex.reset_from_world(spec.world, "spot_1", seed=spec.world.seed)
    trace = ExecutionTrace(spec.scenario_id, "FULL")

    def _call(tool: str, args: Dict[str, Any]):
        trace.append(Stage.REQUESTED, tool=tool, args=args)
        res = ex.execute(ToolCall(tool, args))
        if res.executed:
            trace.append(Stage.EXECUTED, tool=tool, status=res.status, error=res.error)
        if res.status == STATUS_SUCCESS:
            trace.append(Stage.SUCCEEDED, tool=tool)
        if res.delivered:
            trace.append(Stage.OBSERVATION_DELIVERED, tool=tool, observation_id=res.observation_id,
                         observation_hash=res.observation_hash, modality=res.modality)
        return res

    requested_modalities: List[str] = []
    primary_result = None  # the ToolResult of the FIRST (primary) acquisition
    # request -- the one UHA/claimed_observation_id reasons about; B-ACQ-2's
    # follow-up accumulation calls are recorded in the trace but are not
    # what UHA scores (UHA is about handling the primary request's outcome)
    # scripted-to-gold: request the required acquisition iff gold requires it
    # AND the counterfactual arm allows it
    if spec.gold.acquisition_required and acquisition_available:
        target = spec.gold.required_acquisition
        qt = spec.initial_quality_threshold if target == spec.initial_modality else 0.0
        # acquisition_exclude_ids: propagates the SAME mechanical, label-blind
        # rank-2 selection rule used by gold computation (see
        # _THERMAL_PRESENT_EXCLUDE_RANK1_ID) into the REAL tool call, so gold
        # and actual execution never diverge. Empty for every configuration
        # except motor_01's B-ACQ-1/B-ACQ-3 present arm.
        exclude_ids = spec.params.get("acquisition_exclude_ids", [])
        primary_result = _call("request_observation", {"modality": target, "max_age_s": 999_999_999,
                                                        "quality_threshold": qt,
                                                        "exclude_ids": exclude_ids})
        requested_modalities.append(target)
        if spec.template_id == "B-ACQ-2":
            # accumulate up to min_count=3 (already have initial_count)
            need = spec.capability["required_evidence"][0]["min_count"] - spec.initial_count
            for _ in range(max(0, need)):
                _call("request_observation", {"modality": target, "max_age_s": 999_999_999,
                                              "quality_threshold": 0.0})

    # HONEST claimed_observation_id: the real delivered id when the primary
    # request resolved, or None -- the scripted runner NEVER fabricates one
    # when the request came back UNAVAILABLE (this is what proves the
    # scorer/UHA mechanics, not a model, so it must model correct behavior).
    claimed_observation_id = None
    primary_response_status = None
    if primary_result is not None:
        primary_response_status = primary_result.payload.get("status")
        if primary_result.delivered:
            claimed_observation_id = primary_result.observation_id

    verdict = spec.gold.final_terminal_action  # scripted to equal gold
    trace.append(Stage.DECISION, detail={"verdict": verdict})
    return {
        "trace_events": [e.to_dict() for e in trace.events],
        "trace_chain_valid": trace.verify_chain(),
        "verdict": verdict,
        "requested_modalities": requested_modalities,
        "acquired": bool(requested_modalities),
        "claimed_observation_id": claimed_observation_id,
        "primary_response_status": primary_response_status,
    }
