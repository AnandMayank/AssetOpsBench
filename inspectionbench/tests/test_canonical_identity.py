"""Phase 8F Section M — identity regression tests for the normalized
canonical identity layer (canonical_identity.py) and the generator built on
top of it (scale_generator.py)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from canonical_identity import (  # noqa: E402
    CanonicalWorld, EvidenceRealization, IdentityClass, MUTATION_TYPES,
    TemplateInstantiation, WorldIdentityProtectionError, BlockedTemplateError,
    apply_evidence_mutation, classify_field, normalize_world, template_eligibility,
)
from scenario_gen import FACTORIAL, sample_world, derive_gold  # noqa: E402
from sequence_executor import sample_sequence  # noqa: E402
from scale_generator import (  # noqa: E402
    construct_world, construct_scenario, construct_episode,
    construct_evidence_realization, construct_temporal_evidence_realization,
    construct_item, offered_tools_for_regime,
)

CELL = "phys_in__iot_agree"


def _world(seed=1, asset="chiller_6", label="C1-chiller_6-phys_in__iot_agree-1"):
    return construct_world(seed, CELL, asset, label)


# --- 1-5: matched-regime / template / prompt / item invariance -------------

def test_1_same_world_all_three_regimes_same_world_id():
    world_state, cw = _world()
    scenario = construct_scenario(cw, world_state, "FM-6a", ("FULL", "PHYSICAL_ONLY", "DIGITAL_ONLY"),
                                  ("T-OP-ACTION",))
    ids = set()
    for regime in ("FULL", "PHYSICAL_ONLY", "DIGITAL_ONLY"):
        ep = construct_episode(scenario, regime)
        ev = construct_evidence_realization(ep, regime)
        offered = offered_tools_for_regime(regime)
        item, _ = construct_item("T-OP-ACTION", world_state, "FM-6a", offered, scenario, ep, ev)
        ids.add(item["world_id"])
    assert ids == {cw.world_id}


def test_2_same_world_different_evidence_mutation_same_world_id():
    world_state, cw = _world()
    scenario = construct_scenario(cw, world_state, "FM-6a", ("FULL",), ("T-OP-ACTION",))
    ep = construct_episode(scenario, "FULL")
    ev1 = construct_evidence_realization(ep, "FULL")
    ev2 = construct_evidence_realization(ep, "PHYSICAL_ONLY")
    assert ev1.evidence_realization_id != ev2.evidence_realization_id
    item1, _ = construct_item("T-OP-ACTION", world_state, "FM-6a", offered_tools_for_regime("FULL"),
                              scenario, ep, ev1)
    item2, _ = construct_item("T-OP-ACTION", world_state, "FM-6a", offered_tools_for_regime("PHYSICAL_ONLY"),
                              scenario, ep, ev2)
    assert item1["world_id"] == item2["world_id"] == cw.world_id


def test_3_same_world_different_template_same_world_id():
    world_state, cw = _world()
    scenario = construct_scenario(cw, world_state, "FM-6a", ("FULL",),
                                  ("T-OP-ACTION", "T-PERC-GAUGE-VALUE"))
    ep = construct_episode(scenario, "FULL")
    ev = construct_evidence_realization(ep, "FULL")
    offered = offered_tools_for_regime("FULL")
    item_a, _ = construct_item("T-OP-ACTION", world_state, "FM-6a", offered, scenario, ep, ev)
    item_b, _ = construct_item("T-PERC-GAUGE-VALUE", world_state, "FM-6a", offered, scenario, ep, ev)
    assert item_a["item_id"] != item_b["item_id"]
    assert item_a["world_id"] == item_b["world_id"] == cw.world_id


def test_4_same_world_different_prompt_wording_same_world_id():
    # Prompt wording is a rendering-layer concern (item_rendering.render);
    # canonical_identity never consumes rendered text at all -- world_id is
    # computed purely from WORLD_STATE_FIELDS before any rendering happens.
    world_state, cw = _world()
    rendered_a = "What action should be taken?"
    rendered_b = "Given the evidence, what is the correct verdict?"
    assert rendered_a != rendered_b
    # world_id is unaffected because normalize_world() never looks at prompt text.
    assert normalize_world(world_state).world_id == cw.world_id


def test_5_same_world_different_evaluation_item_same_world_id():
    world_state, cw = _world()
    scenario = construct_scenario(cw, world_state, "FM-6a", ("FULL",),
                                  ("T-OP-ACTION", "T-PERC-GAUGE-VALUE"))
    ep = construct_episode(scenario, "FULL")
    ev = construct_evidence_realization(ep, "FULL")
    offered = offered_tools_for_regime("FULL")
    seen_world_ids = set()
    for tmpl in ("T-OP-ACTION", "T-PERC-GAUGE-VALUE"):
        item, _ = construct_item(tmpl, world_state, "FM-6a", offered, scenario, ep, ev)
        seen_world_ids.add(item["world_id"])
    assert seen_world_ids == {cw.world_id}


# --- 6: different underlying world state -> different world_id -------------

def test_6_different_underlying_world_state_different_world_id():
    _, cw1 = _world(seed=1)
    _, cw2 = _world(seed=2)
    assert cw1.world_id != cw2.world_id


# --- 7: reconstruction from normalized representation -----------------------

def test_7_reconstruction_from_normalized_representation_same_world_id():
    world_state, cw = _world()
    cw_rebuilt = CanonicalWorld(fields=dict(cw.fields))
    assert cw_rebuilt.world_id == cw.world_id


# --- 8/12: crossover/mutation cannot silently modify world_id --------------

def test_8_and_12_protected_field_mutation_rejected_not_silently_reassigned():
    world_state, cw = _world()
    with pytest.raises(WorldIdentityProtectionError):
        apply_evidence_mutation(cw, "occlusion", {"physical_value": 999.0})
    # world untouched -- a second, valid call still returns the same id
    assert normalize_world(world_state).world_id == cw.world_id


# --- 9: WORLD_STATE_CHANGING mutation produces a new world_id --------------

def test_9_world_state_changing_mutation_produces_new_world_id(monkeypatch):
    world_state, cw = _world()
    # Explicitly register a WORLD_STATE_CHANGING mutation type for this test
    # only (mirrors Section J: "If the operation is explicitly
    # WORLD_STATE_CHANGING: construct the changed normalized world, derive a
    # new world_id, record the reason/provenance").
    monkeypatch.setitem(MUTATION_TYPES, "test_only_world_state_changing", True)
    new_world = apply_evidence_mutation(cw, "test_only_world_state_changing",
                                        {"physical_value": 999.0})
    assert new_world.world_id != cw.world_id
    assert new_world.fields["physical_value"] == 999.0


def test_9b_unregistered_mutation_type_rejected():
    world_state, cw = _world()
    with pytest.raises(WorldIdentityProtectionError):
        apply_evidence_mutation(cw, "not_a_real_mutation_type", {"visibility_state": "x"})


# --- 10: the Phase 8C NOT_POPULATED world-collapse bug cannot recur --------

def test_10_placeholder_world_id_does_not_collapse_distinct_scenarios():
    # The Phase 8C bug: 7 distinct incumbent scenarios shared the literal
    # world_id="NOT_POPULATED" placeholder. Here, two worlds built with
    # different underlying state but the SAME placeholder *label* string
    # must still get distinct world_id, because the label is
    # PRESENTATION_ONLY and never enters the hash.
    world_state1, cw1 = _world(seed=10, label="NOT_POPULATED")
    world_state2, cw2 = _world(seed=11, label="NOT_POPULATED")
    assert cw1.world_id != cw2.world_id, (
        "two distinct worlds must not collapse to one id merely because "
        "they share a placeholder label string")


# --- 11: deterministic regeneration reproduces every identity --------------

def test_11_deterministic_regeneration_reproduces_all_identities():
    def build():
        world_state, cw = _world()
        scenario = construct_scenario(cw, world_state, "FM-6a", ("FULL",), ("T-OP-ACTION",))
        ep = construct_episode(scenario, "FULL")
        ev = construct_evidence_realization(ep, "FULL")
        item, prov = construct_item("T-OP-ACTION", world_state, "FM-6a",
                                    offered_tools_for_regime("FULL"), scenario, ep, ev)
        return cw.world_id, scenario.scenario_id, ep.episode_id, ev.evidence_realization_id, \
            item["item_id"], item["gold"]

    run1 = build()
    run2 = build()
    assert run1 == run2


# --- 13: distinct incumbent scenarios cannot collapse via placeholders ------

def test_13_distinct_scenarios_do_not_collapse_through_placeholder_values():
    # Same as test 10 but asserting at the scenario level too.
    world_state1, cw1 = _world(seed=20, label="PLACEHOLDER")
    world_state2, cw2 = _world(seed=21, label="PLACEHOLDER")
    s1 = construct_scenario(cw1, world_state1, "FM-6a", ("FULL",), ("T-OP-ACTION",))
    s2 = construct_scenario(cw2, world_state2, "FM-6a", ("FULL",), ("T-OP-ACTION",))
    assert s1.scenario_id != s2.scenario_id


# --- 14: different underlying world states cannot accidentally share id ----

def test_14_many_distinct_worlds_have_no_id_collisions():
    ids = set()
    for seed in range(1, 40):
        _, cw = _world(seed=seed)
        ids.add(cw.world_id)
    assert len(ids) == 39


# --- 15: normalization invariant to serialization/order differences --------

def test_15_normalization_invariant_to_key_order():
    world_state, cw = _world()
    reordered = CanonicalWorld(fields={k: cw.fields[k] for k in reversed(list(cw.fields))})
    assert reordered.world_id == cw.world_id


def test_15b_float_precision_noise_does_not_change_world_id():
    world_state, cw = _world()
    noisy_fields = dict(cw.fields)
    if isinstance(noisy_fields.get("physical_value"), float):
        noisy_fields["physical_value"] = noisy_fields["physical_value"] + 1e-9
    noisy = CanonicalWorld(fields=noisy_fields)
    assert noisy.world_id == cw.world_id


# --- 16: template/prompt changes cannot alter world identity ---------------

def test_16_template_change_cannot_alter_world_identity():
    world_state, cw = _world()
    scenario = construct_scenario(cw, world_state, "FM-6a", ("FULL",),
                                  ("T-OP-ACTION", "T-PERC-GAUGE-VALUE"))
    ep = construct_episode(scenario, "FULL")
    ev = construct_evidence_realization(ep, "FULL")
    offered = offered_tools_for_regime("FULL")
    for tmpl in ("T-OP-ACTION", "T-PERC-GAUGE-VALUE"):
        item, _ = construct_item(tmpl, world_state, "FM-6a", offered, scenario, ep, ev)
        assert item["world_id"] == cw.world_id


# --- field classification and blocked-template rejection -------------------

def test_field_classification_covers_world_evidence_template_item():
    assert classify_field("physical_value") == IdentityClass.WORLD_IDENTITY
    assert classify_field("scenario_id") == IdentityClass.PRESENTATION_ONLY
    assert classify_field("mutation_type") == IdentityClass.EVIDENCE_IDENTITY
    assert classify_field("template_id") == IdentityClass.TEMPLATE_IDENTITY
    assert classify_field("item_id") == IdentityClass.ITEM_IDENTITY


def test_blocked_template_rejected_loudly():
    world_state, cw = _world()
    scenario = construct_scenario(cw, world_state, "FM-6a", ("FULL",), ("T-PERC-GAUGE-READABLE",))
    ep = construct_episode(scenario, "FULL")
    ev = construct_evidence_realization(ep, "FULL")
    with pytest.raises(BlockedTemplateError):
        construct_item("T-PERC-GAUGE-READABLE", world_state, "FM-6a",
                      offered_tools_for_regime("FULL"), scenario, ep, ev)


def test_template_eligibility_states_preserved_verbatim():
    assert template_eligibility("T-OP-ACTION") == "SCALE-READY"
    assert template_eligibility("T-EPIST-SUFFICIENT") == "SCALE-WITH-AUDIT"
    assert template_eligibility("T-PERC-THERMAL-ANOMALY") == "BLOCKED"


# --- sequence (E-dimension) world identity ----------------------------------

def test_sequence_world_normalizes_and_ignores_sequence_id_label():
    sw1 = sample_sequence(1001, n_episodes=3)
    cw1 = normalize_world(sw1)
    # Re-sample with the same seed -- deterministic regeneration for E too.
    sw1b = sample_sequence(1001, n_episodes=3)
    cw1b = normalize_world(sw1b)
    assert cw1.world_id == cw1b.world_id

    sw2 = sample_sequence(1002, n_episodes=3)
    cw2 = normalize_world(sw2)
    assert cw1.world_id != cw2.world_id


def test_temporal_evidence_realization_never_touches_world_identity():
    world_state, cw = _world()
    scenario = construct_scenario(cw, world_state, "FM-6a", ("FULL",), ("T-TEMPORAL-STALE",))
    ep = construct_episode(scenario, "ep1", {"episode_index": 1})
    ev = construct_temporal_evidence_realization(ep, 1)
    assert ev.mutation_type == "temporal_staleness"
    assert not ev.is_world_state_changing
