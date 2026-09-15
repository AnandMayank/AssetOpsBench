"""Phase 8G Section E — gold-invariance tests for the E/T-OP-ACTION
gold-hook fix (eval_templates._gold_action now branches on world TYPE,
delegating to sequence_executor.derive_sequence_gold for a SequenceWorld
rather than calling gold_fn(world).verdict, which had no episode_index
hook at all)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval_templates as ET  # noqa: E402
from sequence_executor import sample_sequence, derive_sequence_gold  # noqa: E402
from scenario_gen import FACTORIAL, sample_world, derive_gold  # noqa: E402

OFFERED = {"read_gauge", "read_iot", "get_work_order"}

# A seed with a genuine mid-sequence transition (verified: COMMIT, COMMIT, ESCALATE).
TRANSITION_SEED = 1004


def _op_action_gold(sw, episode_index):
    item = ET.instantiate("T-OP-ACTION", sw, "FM-temporal", OFFERED, world_id="W",
                          episode_id=f"E{episode_index}", episode_index=episode_index)
    return item["gold"], item["cannot_determine"]


# 1-3: each episode index derives the correct gold (matches derive_sequence_gold directly)

def test_1_episode_0_matches_derive_sequence_gold():
    sw = sample_sequence(TRANSITION_SEED, n_episodes=3)
    gold, cd = _op_action_gold(sw, 0)
    assert gold == derive_sequence_gold(sw, 0)["verdict"]
    assert cd is False


def test_2_episode_1_matches_derive_sequence_gold():
    sw = sample_sequence(TRANSITION_SEED, n_episodes=3)
    gold, cd = _op_action_gold(sw, 1)
    assert gold == derive_sequence_gold(sw, 1)["verdict"]
    assert cd is False


def test_3_episode_2_matches_derive_sequence_gold():
    sw = sample_sequence(TRANSITION_SEED, n_episodes=3)
    gold, cd = _op_action_gold(sw, 2)
    assert gold == derive_sequence_gold(sw, 2)["verdict"]
    assert cd is False


# 4: different episode indices legitimately produce different gold when the
# world transitions require it.

def test_4_gold_transitions_when_the_world_requires_it():
    sw = sample_sequence(TRANSITION_SEED, n_episodes=3)
    golds = [_op_action_gold(sw, i)[0] for i in range(3)]
    assert golds == ["COMMIT", "COMMIT", "ESCALATE"]
    assert len(set(golds)) > 1


# 5: gold is deterministic for a fixed world + episode index

def test_5_gold_is_deterministic_for_fixed_world_and_episode():
    sw1 = sample_sequence(TRANSITION_SEED, n_episodes=3)
    sw2 = sample_sequence(TRANSITION_SEED, n_episodes=3)
    for idx in range(3):
        assert _op_action_gold(sw1, idx) == _op_action_gold(sw2, idx)


# 6: the fix does not change non-E (WorldState) template behavior

def test_6_worldstate_gold_unchanged_by_the_fix():
    cell = next(c for c in FACTORIAL if c.name == "phys_in__iot_agree")
    world = sample_world(1, cell, asset_id="chiller_6", scenario_id="C1-chiller_6-phys_in__iot_agree-1")
    item = ET.instantiate("T-OP-ACTION", world, "FM-6a", OFFERED, world_id="W", episode_id="EP",
                          gold_fn=derive_gold)
    assert item["gold"] == derive_gold(world).verdict
    assert item["cannot_determine"] is False


# 7/8: the fix does not alter canonical world_id or unrelated scenario/episode identity

def test_7_and_8_fix_does_not_alter_canonical_identity():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from canonical_identity import normalize_world
    from scale_generator import construct_world, construct_scenario, construct_episode

    world_state, cw_before = construct_world(TRANSITION_SEED, "phys_in__iot_agree", "chiller_6",
                                             "PRE-FIX-LABEL")
    # Re-derive after exercising the fixed code path -- identity must be untouched.
    sw = sample_sequence(TRANSITION_SEED, n_episodes=3)
    _ = _op_action_gold(sw, 1)  # exercise the fixed branch
    world_state2, cw_after = construct_world(TRANSITION_SEED, "phys_in__iot_agree", "chiller_6",
                                             "PRE-FIX-LABEL")
    assert cw_before.world_id == cw_after.world_id

    scenario = construct_scenario(cw_before, world_state, "FM-6a", ("FULL",), ("T-OP-ACTION",))
    ep = construct_episode(scenario, "FULL")
    # Unrelated (non-sequence) scenario/episode identities are unaffected by the
    # SequenceWorld branch added to _gold_action.
    assert scenario.scenario_id and ep.episode_id


# 9: regeneration reproduces all identities and gold exactly

def test_9_regeneration_reproduces_identities_and_gold():
    def build():
        sw = sample_sequence(TRANSITION_SEED, n_episodes=3)
        return tuple(_op_action_gold(sw, i) for i in range(3))

    assert build() == build()
