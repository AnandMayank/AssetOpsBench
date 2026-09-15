"""Fidelity tests for the InspectionBench V3 primary evaluation panel
(src/orchestrator/eval_model_panel.py) -- Phase 8H.2J.

Zero API calls: these test configuration correctness and the served-model
comparison LOGIC only (fed real, previously-captured response.model strings
from the feasibility gate, not live calls).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

import eval_model_panel as EMP  # noqa: E402


# --- panel configuration -----------------------------------------------

def test_primary_panel_has_exactly_five_models():
    assert len(EMP.PRIMARY_PANEL) == 5


def test_primary_panel_slots_are_unique_and_ordered_1_to_5():
    slots = [m.slot for m in EMP.PRIMARY_PANEL]
    assert slots == [1, 2, 3, 4, 5]


def test_primary_panel_tokenrouter_ids_are_unique():
    ids = [m.tokenrouter_id for m in EMP.PRIMARY_PANEL]
    assert len(ids) == len(set(ids))


def test_only_claude_is_an_exact_factorybench_match():
    exact = {m.display_name for m in EMP.PRIMARY_PANEL if m.is_exact_factorybench_match}
    assert exact == {"Claude Sonnet 4.6"}


def test_every_non_exact_model_discloses_a_successor_reason():
    for m in EMP.PRIMARY_PANEL:
        if not m.is_exact_factorybench_match:
            assert m.successor_reason, f"{m.display_name}: successor claimed with no disclosed reason"
            assert len(m.successor_reason) > 20  # not a placeholder


def test_qwen_9b_is_excluded_not_in_primary_panel():
    ids = [m.tokenrouter_id for m in EMP.PRIMARY_PANEL]
    assert "tokenrouter/qwen/qwen3.5-9b" not in ids
    excluded_ids = [e["tokenrouter_id"] for e in EMP.EXCLUDED_MODELS]
    assert "tokenrouter/qwen/qwen3.5-9b" in excluded_ids


def test_exclusion_reason_is_a_protocol_incompatibility_not_a_capability_claim():
    reason = EMP.EXCLUDED_MODELS[0]["exclusion_reason"]
    assert "max_tokens" in reason
    assert "not a capability gap" in reason


def test_get_model_returns_the_matching_config():
    m = EMP.get_model("tokenrouter/anthropic/claude-sonnet-4.6")
    assert m.display_name == "Claude Sonnet 4.6"


def test_get_model_raises_for_an_unconfigured_model():
    """Never silently run an unconfigured model against V3."""
    import pytest
    with pytest.raises(KeyError):
        EMP.get_model("tokenrouter/some/random-model")


def test_all_primary_models_use_temperature_zero():
    for m in EMP.PRIMARY_PANEL:
        assert m.temperature == 0.0


# --- served-model verification (fed real captured values, no API call) -----

REAL_CAPTURED_RESPONSES = {
    # requested_id -> actual response.model observed live at the feasibility
    # gate. Deliberately inconsistent formatting per provider -- this is
    # exactly why normalized substring matching is required (see
    # eval_model_panel._normalize_model_token's docstring).
    "tokenrouter/anthropic/claude-sonnet-4.6": "claude-sonnet-4-6",
    "tokenrouter/openai/gpt-5.2": "gpt-5.2-2025-12-11",
    "tokenrouter/deepseek/deepseek-v4-pro": "deepseek-v4-pro",
    "tokenrouter/mistralai/mistral-medium-3-5": "mistralai/mistral-medium-3-5",
    "tokenrouter/qwen/qwen3.5-397b-a17b": "qwen/qwen3.5-397b-a17b",
}


def test_served_model_verification_passes_for_all_five_real_captured_responses():
    for req, served in REAL_CAPTURED_RESPONSES.items():
        check = EMP.verify_served_model(req, served)
        assert check.ok, f"{req}: {check.reason}"


def test_served_model_verification_rejects_a_genuine_mismatch():
    check = EMP.verify_served_model("tokenrouter/openai/gpt-5.2", "anthropic/claude-haiku-4.5")
    assert check.ok is False
    assert "MISMATCH" in check.reason


def test_served_model_verification_rejects_cross_family_mismatch_within_same_provider():
    """A provider serving a DIFFERENT model of its own (not just a dated
    snapshot suffix) must still be caught."""
    check = EMP.verify_served_model("tokenrouter/qwen/qwen3.5-397b-a17b", "qwen/qwen3.5-9b")
    assert check.ok is False


def test_served_model_none_is_unverifiable_not_a_silent_pass_disguised_as_match():
    check = EMP.verify_served_model("tokenrouter/anthropic/claude-sonnet-4.6", None)
    assert check.ok is True
    assert "unverifiable" in check.reason


# --- frozen benchmark identity recorded in this config ----------------------

def test_manifest_version_constants_match_the_actual_v3_manifest():
    import json
    v3 = json.loads((REPO_ROOT / EMP.MANIFEST_PATH).read_text())
    assert v3["final_canonical_total"] == EMP.MANIFEST_FINAL_CANONICAL_TOTAL == 4075


def test_frozen_93_sha_constant_matches_the_actual_file_on_disk():
    import hashlib
    actual = hashlib.sha256((REPO_ROOT / "reports/ec/phase8h1_pilot_manifest.json").read_bytes()).hexdigest()
    assert actual == EMP.FROZEN_93_SHA256
