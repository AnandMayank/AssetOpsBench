"""P0-5: the run contract and E3's framing conditions.

Two classes of silent failure are covered:

* runs that look comparable but were not (a decoding budget differing 4x
  between runners, with nothing recording it);
* framing conditions that differ in more than the framing, which would make any
  measured E3 effect uninterpretable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

from framing import (  # noqa: E402
    BENCHMARK, DEPLOYMENT, FRAMINGS, NEUTRAL, SAFETY_AUDIT,
    apply_framing, assert_framings_valid,
    assert_prompts_differ_only_by_framing, check_framings, framing_text, pairwise,
)
from run_contract import (  # noqa: E402
    ContractViolation, RunContract, assert_all_comparable, group_by_track,
)


def _contract(**over) -> RunContract:
    base = dict(task="pmc_l1", model="m", api="tokenrouter", temperature=0.0,
                max_tokens=4096, system_prompt_id="gauge_read/v1")
    base.update(over)
    return RunContract(**base)


# --------------------------------------------------------------------------
# Run contract
# --------------------------------------------------------------------------

def test_model_identity_does_not_break_comparability():
    """Comparing models is the point; only the harness must match."""
    _contract(model="a").assert_comparable(_contract(model="b"))


def test_decoding_drift_is_caught():
    """The live drift: probe runners use max_tokens=1024, providers 4096."""
    with pytest.raises(ContractViolation) as e:
        _contract(max_tokens=4096).assert_comparable(_contract(max_tokens=1024))
    assert "DRIFT" in str(e.value) and "max_tokens" in str(e.value)


def test_temperature_drift_is_caught():
    with pytest.raises(ContractViolation) as e:
        _contract(temperature=0.0).assert_comparable(_contract(temperature=1.0))
    assert "temperature" in str(e.value)


def test_track_difference_is_reported_as_a_track_not_as_drift():
    """A text-only model is a separate track, not a misconfiguration."""
    vision = _contract(model="v", channels=("rgb", "iot"))
    text = _contract(model="t", channels=("iot",))
    with pytest.raises(ContractViolation) as e:
        vision.assert_comparable(text)
    msg = str(e.value)
    assert "TRACK" in msg and "report separately" in msg
    assert "DRIFT" not in msg


def test_fingerprint_ignores_model_and_seed_but_not_budget():
    a = _contract(model="a", seed="s1")
    assert a.fingerprint() == _contract(model="b", seed="s2").fingerprint()
    assert a.fingerprint() != _contract(model="a", max_tokens=1024).fingerprint()


def test_framing_variant_changes_the_fingerprint():
    """E3 conditions are different contracts; they must not silently pool."""
    a = _contract()
    assert a.fingerprint() != a.with_framing(DEPLOYMENT).fingerprint()


def test_channel_ablation_is_a_track_change():
    full = _contract()
    assert not full.same_track(full.with_channels("iot"))


def test_unknown_channel_is_rejected():
    with pytest.raises(ValueError):
        _contract().with_channels("lidar")


def test_unknown_prompt_variant_is_rejected():
    with pytest.raises(ValueError):
        _contract().with_prompt_variant("hinted")


def test_stamp_is_serialisable_and_carries_the_fingerprint():
    import json
    s = _contract().stamp()
    json.dumps(s)
    assert s["contract_fingerprint"] == _contract().fingerprint()


def test_group_by_track_partitions_mixed_model_sets():
    cs = [_contract(model="a"), _contract(model="b"),
          _contract(model="c").with_channels("iot")]
    tracks = group_by_track(cs)
    assert len(tracks) == 2
    assert sorted(len(v) for v in tracks.values()) == [1, 2]


def test_assert_all_comparable_accepts_a_single_contract():
    assert_all_comparable([_contract()])


# --------------------------------------------------------------------------
# Framing
# --------------------------------------------------------------------------

def test_shipped_framings_pass_every_integrity_check():
    assert_framings_valid()


def test_all_three_conditions_plus_a_neutral_control_exist():
    assert set(FRAMINGS) == {NEUTRAL, DEPLOYMENT, BENCHMARK, SAFETY_AUDIT}
    assert FRAMINGS[NEUTRAL] == ""


def test_a_framing_that_leaks_the_rule_under_test_is_rejected():
    """If the framing states the rule, the effect is E2's, not E3's."""
    bad = dict(FRAMINGS)
    bad[DEPLOYMENT] = ("You are operating a real facility. If the gauge is "
                       "unreadable, abstain and do not guess a value at all.")
    with pytest.raises(AssertionError) as e:
        assert_framings_valid(bad)
    assert "no_operational_knowledge" in str(e.value)


def test_a_framing_that_discloses_the_study_is_rejected():
    bad = dict(FRAMINGS)
    bad[BENCHMARK] = ("You are being evaluated. We are studying whether "
                      "evaluation awareness changes how agents behave here.")
    with pytest.raises(AssertionError):
        assert_framings_valid(bad)


def test_wildly_unequal_framing_lengths_are_rejected():
    bad = dict(FRAMINGS)
    bad[BENCHMARK] = "Do the task."
    with pytest.raises(AssertionError) as e:
        assert_framings_valid(bad)
    assert "length" in str(e.value)


def test_apply_framing_only_prepends_and_neutral_is_a_no_op():
    body = "SYSTEM: read the gauge."
    assert apply_framing(body, NEUTRAL) == body
    out = apply_framing(body, BENCHMARK)
    assert out.endswith(body) and out.startswith(framing_text(BENCHMARK))


def test_prompts_identical_apart_from_framing_pass():
    body = "SYSTEM: read the gauge."
    assert_prompts_differ_only_by_framing(
        {n: apply_framing(body, n) for n in (DEPLOYMENT, BENCHMARK, SAFETY_AUDIT)}
    )


def test_a_swapped_task_prompt_between_conditions_is_caught():
    """The real hazard: a runner that also changes the task or tools."""
    rendered = {
        DEPLOYMENT: apply_framing("SYSTEM: read the gauge.", DEPLOYMENT),
        BENCHMARK: apply_framing("SYSTEM: read the gauge carefully.", BENCHMARK),
    }
    with pytest.raises(AssertionError) as e:
        assert_prompts_differ_only_by_framing(rendered)
    assert "outside the framing paragraph" in str(e.value)


def test_pairwise_yields_the_three_contrasts_holm_corrects():
    assert len(pairwise()) == 3


def test_unknown_framing_name_is_rejected():
    with pytest.raises(ValueError):
        framing_text("production")


def test_check_framings_reports_every_condition():
    names = {c.name for c in check_framings()}
    for f in (DEPLOYMENT, BENCHMARK, SAFETY_AUDIT):
        assert f"no_operational_knowledge[{f}]" in names


# --------------------------------------------------------------------------
# The guard must fire before the scenario loop, not inside it.
# --------------------------------------------------------------------------

def test_runner_refuses_framing_on_a_backend_that_ignores_it():
    """Regression: the first version of this guard lived in
    _build_vision_provider, whose ValueError is caught by the per-scenario
    error handler and recorded as a result row. The run then completed and was
    labelled with a framing the backend never applied — the same shape as the
    Sec 4.4 silent run. It must fail before any scenario executes."""
    import subprocess
    r = subprocess.run(
        [sys.executable, str(REPO_ROOT / "src" / "orchestrator" / "run_pmc_benchmark.py"),
         "--n", "2", "--backend", "moondream", "--framing", "benchmark"],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 2, f"expected refusal, got {r.returncode}\n{r.stdout}{r.stderr}"
    assert "does not apply the E3 framing" in r.stderr
    assert "[1/2]" not in r.stdout, "a scenario ran before the guard fired"
