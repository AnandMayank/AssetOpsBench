"""Regression tests for b_acquisition_generator.py -- Family B (Evidence
Acquisition), Phase 8H.2G. Zero model/API calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

import b_acquisition_generator as B  # noqa: E402
from couchdb_executor import CouchDBExecutor  # noqa: E402


# --- construct distinction from A: no regime withholding masks the tool ----

def test_request_observation_never_masked_for_b_acquisition():
    """B-acquisition never withholds a TOOL -- insufficiency comes from the
    ledger's real state, not from tool masking (A's construct). Verified
    against every arm/regime this executor supports."""
    ex = CouchDBExecutor()
    for withheld in ([], ["physical"], ["digital"], ["thermal"], ["acoustic"]):
        ex._withheld = withheld
        assert "request_observation" in ex.available_tools(), (
            f"request_observation must remain available regardless of withheld={withheld}"
        )


# --- gold validity, per template -------------------------------------------

def test_b_acq_1_absent_assets_are_genuinely_unavailable():
    for asset in B._THERMAL_ABSENT_ASSETS:
        spec = B.build_b_acq_1(asset, seed=1)
        assert spec.arm == "absent"
        assert spec.gold.acquisition_required is True
        assert spec.gold.required_acquisition == "thermal"
        assert spec.gold.final_terminal_action == "ESCALATE"


def test_b_acq_1_present_asset_resolves_a_real_thermal_record():
    spec = B.build_b_acq_1(B._THERMAL_PRESENT_ASSET, seed=1)
    assert spec.arm == "present"
    assert spec.gold.acquisition_required is True
    assert spec.gold.required_acquisition == "thermal"
    assert spec.gold.final_terminal_action in ("COMMIT", "ESCALATE")


def test_b_acq_2_k1_is_insufficient_k3_is_sufficient():
    for asset in B._ACOUSTIC_ASSETS:
        k1 = B.build_b_acq_2(asset, "k1", seed=2)
        k3 = B.build_b_acq_2(asset, "k3", seed=2)
        assert k1.gold.initial_evidence_sufficient is False
        assert k1.gold.acquisition_required is True
        assert k1.gold.required_acquisition == "acoustic"
        assert k3.gold.initial_evidence_sufficient is True
        assert k3.gold.acquisition_required is False
        assert k3.gold.required_acquisition is None


def test_b_acq_2_matched_pair_shares_asset_and_world_identity():
    """The two arms differ ONLY in the evidence condition -- same asset,
    same task, per the matched-pair design."""
    k1 = B.build_b_acq_2("chiller_6", "k1", seed=5)
    k3 = B.build_b_acq_2("chiller_6", "k3", seed=5)
    assert k1.asset == k3.asset == "chiller_6"
    assert k1.capability["capability_id"] == k3.capability["capability_id"]


def test_b_acq_3_shares_construction_with_b_acq_1_but_different_template_id():
    for asset in B.ALL_ASSETS:
        acq1 = B.build_b_acq_1(asset, seed=3)
        acq3 = B.build_b_acq_3(asset, seed=3)
        assert acq1.template_id == "B-ACQ-1"
        assert acq3.template_id == "B-ACQ-3"
        assert acq1.gold.to_dict() == acq3.gold.to_dict()
        assert acq1.arm == acq3.arm


def test_b_acq_4_ambiguous_fails_quality_gate_unambiguous_passes():
    for asset in B._ACOUSTIC_ASSETS:
        amb = B.build_b_acq_4(asset, "ambiguous", seed=4)
        unamb = B.build_b_acq_4(asset, "unambiguous", seed=4)
        assert amb.gold.acquisition_required is True
        assert amb.gold.required_acquisition == "iot_timeseries"
        assert unamb.gold.acquisition_required is False


def test_different_seeds_same_params_produce_identical_gold():
    """Gold is a function of the real, static observation store, not the
    seed -- proves seed jitter alone does not create distinct episodes."""
    s1 = B.build_b_acq_1("chiller_6", seed=1)
    s2 = B.build_b_acq_1("chiller_6", seed=999)
    assert s1.gold.to_dict() == s2.gold.to_dict()


# --- executable acquisition (real trace, real OBSERVATION_DELIVERED) -------

def test_scripted_episode_delivers_a_real_observation_when_gold_requires_one():
    ex = CouchDBExecutor()
    spec = B.build_b_acq_1(B._THERMAL_PRESENT_ASSET, seed=1)
    result = B.run_scripted_episode(ex, spec)
    delivered = [e for e in result["trace_events"] if e["stage"] == "OBSERVATION_DELIVERED"]
    assert any(e["modality"] == "thermal" for e in delivered)
    assert result["trace_chain_valid"] is True


def test_scripted_episode_produces_no_delivery_when_gold_says_sufficient():
    ex = CouchDBExecutor()
    spec = B.build_b_acq_2("chiller_6", "k3", seed=2)
    result = B.run_scripted_episode(ex, spec)
    acq_events = [e for e in result["trace_events"]
                 if e["stage"] == "REQUESTED" and e["tool"] == "request_observation"]
    assert acq_events == []
    assert result["acquired"] is False


def test_counterfactual_acquisition_available_false_never_requests():
    ex = CouchDBExecutor()
    spec = B.build_b_acq_1(B._THERMAL_PRESENT_ASSET, seed=1)
    result = B.run_scripted_episode(ex, spec, acquisition_available=False)
    acq_events = [e for e in result["trace_events"]
                 if e["stage"] == "REQUESTED" and e["tool"] == "request_observation"]
    assert acq_events == []


# --- no evaluator-only leakage ----------------------------------------------

def test_trace_events_never_carry_gold_fields():
    ex = CouchDBExecutor()
    forbidden = {"final_terminal_action", "acquisition_required", "acceptable_acquisition_set",
                "initial_evidence_sufficient"}
    for builder, args in [
        (B.build_b_acq_1, ("chiller_6", 1)),
        (B.build_b_acq_2, ("chiller_6", "k1", 2)),
        (B.build_b_acq_3, ("motor_01", 3)),
        (B.build_b_acq_4, ("hydraulic_pump_1", "ambiguous", 4)),
    ]:
        spec = builder(*args)
        result = B.run_scripted_episode(ex, spec)
        for event in result["trace_events"]:
            assert forbidden.isdisjoint(event.keys()), f"{spec.scenario_id}: leaked field in {event}"


# --- F. genuine UNAVAILABLE response ----------------------------------------

def test_absent_arm_scripted_episode_receives_genuine_unavailable_status():
    ex = CouchDBExecutor()
    for asset in B._THERMAL_ABSENT_ASSETS:
        spec = B.build_b_acq_1(asset, seed=1)
        assert spec.gold.acquisition_genuinely_unavailable is True
        result = B.run_scripted_episode(ex, spec)
        assert result["primary_response_status"] == "UNAVAILABLE"
        assert result["claimed_observation_id"] is None, (
            "the scripted runner must not claim an observation id when none was delivered"
        )


def test_present_arm_scripted_episode_receives_genuine_resolved_status():
    ex = CouchDBExecutor()
    spec = B.build_b_acq_1(B._THERMAL_PRESENT_ASSET, seed=1)
    assert spec.gold.acquisition_genuinely_unavailable is False
    result = B.run_scripted_episode(ex, spec)
    assert result["primary_response_status"] == "RESOLVED"
    assert result["claimed_observation_id"] is not None


# --- K. no freshness-based B sufficiency ------------------------------------

def test_ledger_seeding_never_restricts_by_freshness():
    """B's sufficiency axis is scoped to evidence COUNT and MODALITY
    COVERAGE (per the approved plan's binding E-overlap guardrail) --
    freshness (max_age_s) must never be the thing that makes an episode's
    initial evidence insufficient. Verified structurally: every internal
    resolve() call the generator issues uses an effectively-infinite
    max_age_s, so a real record's age is never what determines gold."""
    import inspect
    src = inspect.getsource(B._seed_ledger) + inspect.getsource(B.build_b_acq_1) + \
        inspect.getsource(B.build_b_acq_2) + inspect.getsource(B.build_b_acq_4)
    assert "max_age_s=999_999_999" in src or "max_age_s = 999_999_999" in src


# --- L. no A-style tool masking (construct distinctness) -------------------

def test_gold_derivation_never_calls_executor_regime_masking():
    """Gold for every B-acquisition template is derived directly from
    ObservationResolver/EvidenceLedger against the real store -- never from
    a CouchDBExecutor.execute() call with a withheld-tools regime, which is
    A's construct, not B's."""
    import inspect
    for builder in (B.build_b_acq_1, B.build_b_acq_2, B.build_b_acq_4):
        src = inspect.getsource(builder)
        assert "withheld" not in src
        assert "REGIME_WITHHELD" not in src


# --- M. B-ACQ-1 motor_01 present-arm de-degeneracy fix (final prototype gate) ---

def test_absent_arm_exclude_ids_is_empty():
    """The rank-2 selection rule is scoped to ONLY motor_01's present arm --
    every absent-arm configuration (and every other template) must carry an
    empty exclude_ids, verified directly on the built spec's params, not
    inferred."""
    for asset in B._THERMAL_ABSENT_ASSETS:
        spec = B.build_b_acq_1(asset, seed=1)
        assert spec.params.get("acquisition_exclude_ids") == []


def test_only_motor_01_present_arm_uses_the_rank2_selection_rule():
    spec = B.build_b_acq_1(B._THERMAL_PRESENT_ASSET, seed=1)
    assert spec.arm == "present"
    assert spec.params.get("acquisition_exclude_ids") == [B._THERMAL_PRESENT_EXCLUDE_RANK1_ID]


def test_b_acq_1_present_arm_now_resolves_noload_commit_deterministically():
    """The de-degeneracy fix's actual payoff: motor_01's present arm now
    deterministically yields a genuinely different terminal action (COMMIT)
    than the absent arms (ESCALATE) -- real terminal-action variance, not
    forced. Re-run across several seeds to confirm determinism (seed must
    not affect which real record the resolver returns)."""
    for seed in (1, 2, 999):
        spec = B.build_b_acq_1(B._THERMAL_PRESENT_ASSET, seed=seed)
        assert spec.gold.final_terminal_action == "COMMIT"
        assert spec.gold.acquisition_genuinely_unavailable is False


def test_b_acq_3_present_arm_inherits_the_same_fix():
    """B-ACQ-3 internally calls build_b_acq_1 and shares its construction --
    the fix must propagate automatically, not be reimplemented."""
    spec = B.build_b_acq_3(B._THERMAL_PRESENT_ASSET, seed=3)
    assert spec.template_id == "B-ACQ-3"
    assert spec.params.get("acquisition_exclude_ids") == [B._THERMAL_PRESENT_EXCLUDE_RANK1_ID]
    assert spec.gold.final_terminal_action == "COMMIT"


def test_b_acq_2_and_b_acq_4_are_unaffected_by_the_exclude_ids_fix():
    """Regression guard: the fix must not leak into templates that never
    touch thermal/motor_01's tie-break at all."""
    for asset in B._ACOUSTIC_ASSETS:
        for arm in ("k1", "k3"):
            spec = B.build_b_acq_2(asset, arm, seed=1)
            assert spec.params.get("acquisition_exclude_ids", []) == []
        for arm in ("ambiguous", "unambiguous"):
            spec = B.build_b_acq_4(asset, arm, seed=1)
            assert spec.params.get("acquisition_exclude_ids", []) == []


def test_scripted_episode_present_arm_propagates_exclude_ids_to_the_real_call():
    """The environment-fidelity requirement: run_scripted_episode's REAL
    request_observation call must carry the same exclude_ids gold was
    computed against, so the live-resolved record matches gold's
    expectation (Noload/COMMIT), not the naive rank-1 record (Rotor-0)."""
    ex = CouchDBExecutor()
    spec = B.build_b_acq_1(B._THERMAL_PRESENT_ASSET, seed=1)
    result = B.run_scripted_episode(ex, spec)
    assert result["primary_response_status"] == "RESOLVED"
    assert result["verdict"] == "COMMIT"
    assert result["claimed_observation_id"] != B._THERMAL_PRESENT_EXCLUDE_RANK1_ID


# --- N. Scale-generation additions (phase8h2h) ------------------------------

def test_b_acq_1_original_thermal_configuration_unaffected_by_generalization():
    """build_b_acq_1's default (required_modality="thermal") must remain
    byte-identical to the original, validated 16-episode prototype -- same
    scenario_ids, same gold, for every one of the 4 canonical assets."""
    for asset in B.ALL_ASSETS:
        default_spec = B.build_b_acq_1(asset, seed=1)
        explicit_spec = B.build_b_acq_1(asset, seed=1, required_modality="thermal")
        assert default_spec.scenario_id == explicit_spec.scenario_id
        assert default_spec.gold.to_dict() == explicit_spec.gold.to_dict()
        assert "GEN-BACQ1-" in default_spec.scenario_id
        assert "-thermal-" not in default_spec.scenario_id  # no modality tag for the original axis


def test_b_acq_1_acoustic_and_workorder_axes_use_real_per_asset_coverage():
    """present/absent for the new axes must match the real, live-queried
    per-asset modality coverage in observation_records.json, never a
    fabricated split."""
    for asset in ("chiller_6", "hydraulic_pump_1"):
        spec = B.build_b_acq_1(asset, seed=1, required_modality="acoustic")
        assert spec.arm == "present"
        assert spec.gold.acquisition_genuinely_unavailable is False
    for asset in ("metro_pump_1", "motor_01"):
        spec = B.build_b_acq_1(asset, seed=1, required_modality="acoustic")
        assert spec.arm == "absent"
        assert spec.gold.acquisition_genuinely_unavailable is True
    for asset in ("chiller_6", "hydraulic_pump_1", "metro_pump_1"):
        spec = B.build_b_acq_1(asset, seed=1, required_modality="workorder_history")
        assert spec.arm == "present"
    spec = B.build_b_acq_1("motor_01", seed=1, required_modality="workorder_history")
    assert spec.arm == "absent"


def test_b_acq_1_new_axes_never_use_exclude_ids():
    """The mechanical exclude_ids selection rule stays scoped to thermal/
    motor_01 only -- it must never be applied to the new axes, which have
    no fault_class-decoding degeneracy to fix."""
    for modality in ("acoustic", "workorder_history"):
        for asset in B.ALL_ASSETS:
            spec = B.build_b_acq_1(asset, seed=1, required_modality=modality)
            assert spec.params["acquisition_exclude_ids"] == []


def test_b_acq_3_mirrors_b_acq_1_across_all_modality_axes():
    for modality in ("thermal", "acoustic", "workorder_history"):
        for asset in B.ALL_ASSETS:
            s1 = B.build_b_acq_1(asset, seed=5, required_modality=modality)
            s3 = B.build_b_acq_3(asset, seed=5, required_modality=modality)
            assert s3.template_id == "B-ACQ-3"
            assert s3.gold.to_dict() == s1.gold.to_dict()
            assert s3.arm == s1.arm


def test_b_acq_1_and_b_acq_3_matched_group_ids_are_distinct_namespaces():
    s1 = B.build_b_acq_1("chiller_6", seed=1, required_modality="acoustic")
    s3 = B.build_b_acq_3("chiller_6", seed=1, required_modality="acoustic")
    assert s1.params["matched_group_id"] == "BACQ1-acoustic"
    assert s3.params["matched_group_id"] == "BACQ3-acoustic"


def test_b_acq_2_original_k1_k3_arms_unaffected_by_registry_generalization():
    for asset in B._ACOUSTIC_ASSETS:
        for arm in ("k1", "k3"):
            spec = B.build_b_acq_2(asset, arm, seed=1)
            assert spec.capability is B.CAP_B_ACQ_COUNT_SUFFICIENCY
            assert spec.params["k"] == (1 if arm == "k1" else 3)


def test_b_acq_2_new_thresholds_are_real_matched_insufficient_sufficient_pairs():
    for arm_prefix, assets in (("acoustic_n5", B._ACOUSTIC_ASSETS),
                               ("acoustic_n8", B._ACOUSTIC_ASSETS),
                               ("iot_n3", B.ALL_ASSETS), ("iot_n5", B.ALL_ASSETS),
                               ("iot_n8", B.ALL_ASSETS)):
        for asset in assets:
            insufficient = B.build_b_acq_2(asset, f"{arm_prefix}_insufficient", seed=1)
            sufficient = B.build_b_acq_2(asset, f"{arm_prefix}_sufficient", seed=1)
            assert insufficient.gold.initial_evidence_sufficient is False
            assert insufficient.gold.acquisition_required is True
            assert sufficient.gold.initial_evidence_sufficient is True
            assert sufficient.gold.acquisition_required is False
            # same asset/world/task, only the delivered count differs
            assert insufficient.asset == sufficient.asset == asset
            assert insufficient.capability["capability_id"] == sufficient.capability["capability_id"]


def test_b_acq_2_iot_axis_has_no_tie_break_degeneracy():
    """iot_timeseries records have unique real timestamps for every
    canonical asset (verified live against the store) -- no exclude_ids
    mechanism is needed or used for this axis."""
    for asset in B.ALL_ASSETS:
        spec = B.build_b_acq_2(asset, "iot_n3_sufficient", seed=1)
        assert spec.gold.initial_evidence_sufficient is True


def test_b_acq_4_motor_01_thermal_axis_is_a_real_quality_ambiguity_pair():
    ambiguous = B.build_b_acq_4("motor_01", "ambiguous", seed=1)
    unambiguous = B.build_b_acq_4("motor_01", "unambiguous", seed=1)
    assert ambiguous.gold.initial_evidence_sufficient is False
    assert ambiguous.gold.acquisition_required is True
    assert unambiguous.gold.initial_evidence_sufficient is True
    assert unambiguous.gold.acquisition_required is False
    assert ambiguous.initial_modality == unambiguous.initial_modality == "thermal"
    assert ambiguous.capability is B.CAP_B_ACQ_RECONCILIATION_THERMAL


def test_b_acq_4_original_acoustic_pair_unaffected_by_generalization():
    for asset in B._ACOUSTIC_ASSETS:
        spec = B.build_b_acq_4(asset, "ambiguous", seed=1)
        assert spec.capability is B.CAP_B_ACQ_RECONCILIATION
        assert spec.initial_modality == "acoustic"


def test_scale_generation_determinism_across_new_axes():
    """Regenerating the same (asset, arm/modality, seed) config twice must
    produce byte-identical scenario_id and gold, across every new axis."""
    configs = [
        lambda: B.build_b_acq_1("chiller_6", 1, required_modality="acoustic"),
        lambda: B.build_b_acq_1("motor_01", 1, required_modality="workorder_history"),
        lambda: B.build_b_acq_2("metro_pump_1", "iot_n5_insufficient", 1),
        lambda: B.build_b_acq_3("hydraulic_pump_1", 1, required_modality="acoustic"),
        lambda: B.build_b_acq_4("motor_01", "ambiguous", 1),
    ]
    for factory in configs:
        a, b = factory(), factory()
        assert a.scenario_id == b.scenario_id
        assert a.gold.to_dict() == b.gold.to_dict()


def test_scale_generation_no_leakage_across_new_axes():
    ex = CouchDBExecutor()
    forbidden = {"final_terminal_action", "acquisition_required",
                "acceptable_acquisition_set", "initial_evidence_sufficient"}
    specs = [
        B.build_b_acq_1("chiller_6", 1, required_modality="acoustic"),
        B.build_b_acq_1("motor_01", 1, required_modality="workorder_history"),
        B.build_b_acq_2("metro_pump_1", "iot_n5_insufficient", 1),
        B.build_b_acq_2("chiller_6", "acoustic_n8_sufficient", 1),
        B.build_b_acq_3("hydraulic_pump_1", 1, required_modality="acoustic"),
        B.build_b_acq_4("motor_01", "ambiguous", 1),
    ]
    for spec in specs:
        result = B.run_scripted_episode(ex, spec)
        for ev in result["trace_events"]:
            assert forbidden.isdisjoint(ev.keys()), (spec.scenario_id, ev)


def test_scale_generation_execution_fidelity_across_new_axes():
    """requested_modality == executed_modality for every new axis's
    acquire=True path, through the REAL executor/tool dispatch."""
    ex = CouchDBExecutor()
    specs = [
        B.build_b_acq_1("chiller_6", 1, required_modality="acoustic"),
        B.build_b_acq_1("motor_01", 1, required_modality="workorder_history"),
        B.build_b_acq_2("metro_pump_1", "iot_n5_insufficient", 1),
        B.build_b_acq_4("motor_01", "ambiguous", 1),
    ]
    for spec in specs:
        result = B.run_scripted_episode(ex, spec)
        if result["acquired"]:
            assert result["requested_modalities"][0] == spec.gold.required_acquisition


def test_b_acq_2_arm_registry_covers_exactly_the_documented_thresholds():
    thresholds = {int(a.split("_n")[1].split("_")[0]) for a in B.B_ACQ_2_ARM_REGISTRY if "_n" in a}
    assert thresholds == set(B._COUNT_SUFFICIENCY_THRESHOLDS)
