"""Regression tests for the B-ACQ-4 turn-1 prompt observability fix
(scripts/phase8h2g_b_acquisition_pilot.py::render_turn1). Two real findings
were fixed together: (1) the delivered observation's real `quality` value
was never shown, and (2) the rendered `quality_threshold` requirement was
the same generic constant for both arms even though gold for "unambiguous"
was computed against a different, per-episode threshold. See
reports/benchmark/b_acquisition_prototype_implementation.md for the
original finding and reports/benchmark/b_acq4_repair_report.md for this fix.

Zero model/API calls -- tests prompt RENDERING only.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pytest  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "reports" / "benchmark" / "b_acquisition_manifest.json"
pytestmark = pytest.mark.skipif(
    not MANIFEST_PATH.exists(), reason="B-acquisition manifest not present"
)

import phase8h2g_b_acquisition_pilot as P  # noqa: E402


def _episodes():
    return P.load_episodes()


def _b_acq_4():
    return [e for e in _episodes() if e["template_id"] == "B-ACQ-4"]


# --- the real observation quality is exposed -------------------------------

def test_b_acq_4_prompt_exposes_real_quality_value():
    for ep in _b_acq_4():
        prompt = P.render_turn1(ep)
        assert "quality=0.9" in prompt, f"{ep['episode_id']}: real quality not shown"


def test_quality_value_sourced_from_the_actual_delivered_record():
    """Not a fabricated/hardcoded number -- re-derive it independently via
    the same resolver path and confirm it matches what the prompt shows."""
    import b_acquisition_generator as B
    from b_acquisition_generator import ObservationRequest
    for ep in _b_acq_4():
        resolver = B._resolver()
        req = ObservationRequest(asset_id=ep["asset"], inspection_id=ep["scenario_id"],
                                 modality="acoustic", max_age_s=999_999_999, quality_threshold=0.0)
        result = resolver.resolve(req)
        assert result.status == "RESOLVED"
        prompt = P.render_turn1(ep)
        assert f"quality={result.record.quality}" in prompt


# --- the ambiguous/unambiguous requirement is now distinguishable ----------

def test_ambiguous_and_unambiguous_show_different_quality_thresholds():
    for asset in ("chiller_6", "hydraulic_pump_1"):
        amb = [e for e in _b_acq_4() if e["asset"] == asset and e["arm"] == "ambiguous"][0]
        unamb = [e for e in _b_acq_4() if e["asset"] == asset and e["arm"] == "unambiguous"][0]
        p_amb, p_unamb = P.render_turn1(amb), P.render_turn1(unamb)
        assert "0.95" in p_amb
        assert "0.5" in p_unamb
        assert p_amb != p_unamb, "ambiguous and unambiguous prompts must now differ"


def test_rendered_threshold_matches_the_real_per_episode_gold_basis():
    """The shown quality_threshold must equal ep['params']['quality_threshold']
    -- the SAME value gold was actually computed against -- not the generic
    shared capability constant."""
    for ep in _b_acq_4():
        prompt = P.render_turn1(ep)
        expected = ep["params"]["quality_threshold"]
        assert f'"quality_threshold": {expected}' in prompt


# --- no gold leakage ---------------------------------------------------------

LEAKED_TOKENS = ("acquisition_required", "final_terminal_action",
                 "acceptable_acquisition_set", "acquisition_genuinely_unavailable",
                 "ambiguous", "unambiguous", "COMMIT", "ESCALATE")


def test_no_gold_field_or_arm_label_or_answer_leaks_into_the_prompt():
    for ep in _b_acq_4():
        prompt = P.render_turn1(ep)
        for token in LEAKED_TOKENS:
            assert token not in prompt, f"{ep['episode_id']}: leaked {token!r}"


def test_no_gold_field_leaks_for_the_zero_record_case():
    """Defensive: even if a future asset genuinely has zero records for a
    B-ACQ-4-shaped modality, the zero-count branch must not leak gold
    either."""
    fake_ep = {"template_id": "B-ACQ-4", "asset": "metro_pump_1",
              "scenario_id": "TEST-NONEXISTENT", "initial_modality": "thermal",
              "initial_count": 1, "capability": {"required_evidence": [
                  {"modality": "thermal", "min_count": 1, "quality_threshold": 0.95}]},
              "params": {"quality_threshold": 0.95}}
    prompt = P.render_turn1(fake_ep)
    assert "0 observation(s) delivered" in prompt
    for token in LEAKED_TOKENS:
        assert token not in prompt


# --- B-ACQ-1/2/3 behavior unchanged -----------------------------------------

def test_b_acq_1_2_3_prompts_unchanged_by_the_quality_fix():
    """The fix is scoped to B-ACQ-4 only -- every other template's
    ledger_state rendering must remain the plain count-only form."""
    for ep in _episodes():
        if ep["template_id"] == "B-ACQ-4":
            continue
        prompt = P.render_turn1(ep)
        expected_line = f"- {ep['initial_modality']}: {ep['initial_count']} observation(s) delivered"
        assert expected_line in prompt
        assert "quality=" not in prompt


def test_b_acq_1_2_3_do_not_use_the_real_ledger_helper():
    """Structural: _real_initial_ledger_observation must only be reachable
    from the B-ACQ-4 branch of render_turn1."""
    import inspect
    src = inspect.getsource(P.render_turn1)
    # the helper call must be inside the `if ep["template_id"] == "B-ACQ-4":` block
    lines = src.splitlines()
    branch_idx = next(i for i, l in enumerate(lines) if 'template_id"] == "B-ACQ-4"' in l)
    call_idx = next(i for i, l in enumerate(lines) if "_real_initial_ledger_observation(ep)" in l)
    assert call_idx > branch_idx
