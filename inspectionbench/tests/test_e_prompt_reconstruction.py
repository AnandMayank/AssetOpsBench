"""test_e_prompt_reconstruction.py — regression test for E's G6 leakage
repair: proves prompt reconstruction is deterministic and leak-free,
without relying on a never-persisted original."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from e_prompt_reconstruction import render_visit_prompt  # noqa: E402
from sequence_executor import sample_sequence  # noqa: E402
from couchdb_executor import TOOLSET, mask_tools  # noqa: E402


def test_reconstruction_is_deterministic_across_calls():
    tools = mask_tools(TOOLSET, [])
    for seed in (1, 2, 3):
        w1 = sample_sequence(seed, n_episodes=3)
        w2 = sample_sequence(seed, n_episodes=3)
        for k in range(3):
            assert render_visit_prompt(w1, k, tools) == render_visit_prompt(w2, k, tools)


def test_reconstructed_prompt_never_states_gold_or_verdict():
    tools = mask_tools(TOOLSET, [])
    w = sample_sequence(1, n_episodes=3)
    p = render_visit_prompt(w, 0, tools)
    for banned in ("commit", "escalate", "abort", "gold", "expected verdict"):
        assert banned not in p.lower() or banned in ("commit", "escalate", "abort")
    # the allowed-verdicts line legitimately lists the action space in caps;
    # it must not additionally STATE which one is correct
    assert "the correct verdict is" not in p.lower()
    assert "expected verdict" not in p.lower()


def test_reconstructed_prompt_contains_the_asset_and_band_only():
    tools = mask_tools(TOOLSET, [])
    w = sample_sequence(1, n_episodes=3)
    p = render_visit_prompt(w, 0, tools)
    assert w.asset in p
    assert str(w.operating_band[0]) in p and str(w.operating_band[1]) in p
    # the hidden per-episode VALUE (what the agent must discover) is never
    # in the prompt -- only the band/range are stated
    assert str(w.value_at(0)) not in p
