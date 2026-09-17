"""Regression tests for the B-ACQ-4 turn-1 DISCLOSURE fix
(scripts/phase8h2g_b_acquisition_pilot.py::render_turn1).

Real defect found by adversarial audit: gold for B-ACQ-4's "ambiguous" arm
computes `required_acquisition="iot_timeseries"` from the capability
contract's `optional_evidence` field (CAP_B_ACQ_RECONCILIATION /
CAP_B_ACQ_RECONCILIATION_THERMAL), but `render_turn1` only ever rendered
`required_evidence` -- `iot_timeseries` was never named anywhere in the
agent-facing prompt. All 9 live pilot rows (3 models x 3 assets) requested
exactly the modality that WAS printed (acoustic/thermal), never the
correct one, making ASA=0.0 a prompt-disclosure artifact, not a capability
finding. See reports/benchmark/b_v3_repair_report.md.

Fix: render `capability["optional_evidence"]` (a real, generator-computed
capability-contract field -- never gold, never a source label, never
fault_class) alongside `required_evidence`, generically over
`cap.get("optional_evidence", [])`. B-ACQ-1/2/3's contracts never declare
this field, so their prompts are unaffected.

Zero model/API calls -- tests prompt RENDERING and the real resolver path
only.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pytest  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "reports" / "benchmark" / "b_acquisition_final_manifest.json"
pytestmark = pytest.mark.skipif(
    not MANIFEST_PATH.exists(), reason="B-acquisition final manifest not present"
)

import json  # noqa: E402
import phase8h2g_b_acquisition_pilot as P  # noqa: E402
import b_acquisition_generator as B  # noqa: E402
from b_acquisition_generator import ObservationRequest  # noqa: E402


def _episodes():
    m = json.loads(MANIFEST_PATH.read_text())
    return m["episodes"]


def _b_acq_4():
    return [e for e in _episodes() if e["template_id"] == "B-ACQ-4"]


def _b_acq_4_ambiguous():
    return [e for e in _b_acq_4() if e["arm"] == "ambiguous"]


# --- 1. the complementary acquisition modality is now disclosed ------------

def test_optional_evidence_modality_is_named_in_the_prompt_when_declared():
    for ep in _b_acq_4():
        cap = ep["capability"]
        optional = cap.get("optional_evidence", [])
        assert optional, f"{ep['episode_id']}: expected a declared optional_evidence"
        prompt = P.render_turn1(ep)
        for entry in optional:
            assert entry["modality"] in prompt, (
                f"{ep['episode_id']}: complementary modality {entry['modality']!r} "
                f"not disclosed in the turn-1 prompt")


def test_ambiguous_arm_gold_required_acquisition_is_now_a_named_option():
    """The exact defect: gold expects the agent to request
    required_acquisition, which for the ambiguous arm comes from
    optional_evidence -- confirm that value is literally present in the
    rendered prompt text (not merely a coincidental substring)."""
    for ep in _b_acq_4_ambiguous():
        assert ep["gold"]["required_acquisition"] == "iot_timeseries"
        prompt = P.render_turn1(ep)
        assert '"modality": "iot_timeseries"' in prompt


# --- 2/3/4. no gold / fault-label / evaluator-only field leaks -------------

LEAKED_TOKENS = ("acquisition_required", "final_terminal_action",
                 "acceptable_acquisition_set", "acquisition_genuinely_unavailable",
                 "ambiguous", "unambiguous", "COMMIT", "ESCALATE",
                 "fault_class", "recovery_policy_on_unavailable")


def test_no_gold_fault_label_or_evaluator_only_field_leaks_after_the_fix():
    for ep in _b_acq_4():
        prompt = P.render_turn1(ep)
        for token in LEAKED_TOKENS:
            assert token not in prompt, f"{ep['episode_id']}: leaked {token!r}"


def test_optional_evidence_rendering_handles_a_contract_with_none_declared():
    """Defensive: a capability dict with no optional_evidence key (every
    B-ACQ-1/2/3 contract) must render with no extra line and no crash."""
    fake_ep = {"template_id": "B-ACQ-1", "asset": "chiller_6",
              "scenario_id": "TEST-NO-OPTIONAL", "initial_modality": "iot_timeseries",
              "initial_count": 1,
              "capability": {"required_evidence": [{"modality": "thermal", "min_count": 1}]},
              "params": {}}
    prompt = P.render_turn1(fake_ep)
    assert "OPTIONAL/COMPLEMENTARY" not in prompt
    for token in LEAKED_TOKENS:
        assert token not in prompt


# --- 5/6/7. requested modality remains agent-selected; executed/delivered --
# modality trace through the REAL resolver path -----------------------------

def test_requested_modality_is_never_prefilled_by_render_turn1():
    """render_turn1 must return a PROMPT (a string) that ASKS the agent to
    choose -- it must never itself decide or embed a specific
    requested_modality answer field."""
    for ep in _b_acq_4():
        prompt = P.render_turn1(ep)
        assert '"requested_modality": "<modality name>" or null' in prompt
        assert '"acquire": true or false' in prompt


def test_executed_modality_matches_whatever_is_actually_requested():
    """Once the agent (now correctly informed) requests iot_timeseries,
    the REAL environment must resolve exactly that modality -- never
    gold's required_acquisition by coincidence, never a substitution."""
    for ep in _b_acq_4_ambiguous():
        real = P.get_real_acquisition_response(ep, "iot_timeseries")
        assert real["modality"] == "iot_timeseries"
        assert real["status"] == "RESOLVED"
        # cross-check independently against the real resolver, not the
        # pilot script's own return value alone
        resolver = B._resolver()
        req = ObservationRequest(asset_id=ep["asset"], inspection_id=ep["scenario_id"],
                                 modality="iot_timeseries", max_age_s=999_999_999,
                                 quality_threshold=0.0)
        result = resolver.resolve(req)
        assert result.status == "RESOLVED"
        assert real["observation_id"] == result.record.observation_id


def test_delivered_observation_id_corresponds_to_the_requested_modality_not_acoustic():
    """The observation actually delivered for an iot_timeseries request
    must not be the acoustic record the (broken) prompt used to steer
    every model toward."""
    for ep in _b_acq_4_ambiguous():
        real = P.get_real_acquisition_response(ep, "iot_timeseries")
        assert real["observation_id"].startswith("obs_iot") or "iot" in real["observation_id"].lower() \
            or real["modality"] == "iot_timeseries"


# --- B-ACQ-1/2/3 unaffected --------------------------------------------------

def test_b_acq_1_2_3_prompts_unchanged_by_the_disclosure_fix():
    """B-ACQ-1/2/3 contracts never declare optional_evidence -- their
    rendered prompts must be byte-identical to before this fix: no new
    'OPTIONAL/COMPLEMENTARY' line, ledger_state line unchanged."""
    for ep in _episodes():
        if ep["template_id"] == "B-ACQ-4":
            continue
        prompt = P.render_turn1(ep)
        assert "OPTIONAL/COMPLEMENTARY" not in prompt
        expected_line = f"- {ep['initial_modality']}: {ep['initial_count']} observation(s) delivered"
        assert expected_line in prompt
