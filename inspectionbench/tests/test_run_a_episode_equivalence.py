"""Behavioral-equivalence regression for the Phase 8H.1 change to
scripts/phase8h_live_pilot.py:run_a_episode.

The signature gained keyword-only model/api_key/base_url that default to the
module globals. This test proves that, given identical _chat responses:

  run_a_episode(world, arm)                       # existing call sites
  run_a_episode(world, arm, model=MODEL,          # new multi-model call site,
                api_key=API_KEY, base_url=BASE_URL)#   passing the current defaults

produce the SAME execution semantics and the SAME return structure -- the only
difference being which (model, api_key, base_url) triple reaches _chat.

Deterministic: _chat is replaced with a recording stub. No network, no API,
no model call. The CouchDB executor runs in-process (same as the rest of the
orchestrator suite).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _load_live_pilot():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "phase8h_live_pilot", REPO_ROOT / "scripts" / "phase8h_live_pilot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load_live_pilot()


class _FakeExecutor:
    """Minimal in-memory stand-in for CouchDBExecutor so run_a_episode's
    execution path runs with no CouchDB and no network. Deterministic."""

    def __init__(self):
        self._reset_args = None

    def reset_from_world(self, world, arm, seed=1, withheld=None):
        self._reset_args = (world.scenario_id, arm, seed, tuple(withheld or []))

    def available_tools(self):
        return ["read_gauge", "capture_image", "get_asset_state"]

    def execute(self, call):
        from tool_executor import ToolResult, STATUS_SUCCESS
        return ToolResult(tool=call.tool, requested=True, executed=True,
                          status=STATUS_SUCCESS, modality="physical",
                          observation_id=f"obs::{call.tool}",
                          observation_hash="deadbeef0000", asset_id=None,
                          payload={"value": 245.0})


class _RecordingChat:
    """Stands in for run_l3_pilot_executed._chat. Records every call's
    (model, api_key, base_url) and returns fixed two-step responses."""

    def __init__(self):
        self.calls = []

    def __call__(self, model, messages, api_key, base_url):
        self.calls.append((model, api_key, base_url))
        if len(self.calls) % 2 == 1:  # step 1: request tools
            return {"tool_calls": [{"tool": "read_gauge", "args": {}}]}, None
        return {"verdict": "ESCALATE", "reason": "stub", "tool_sequence": ["read_gauge"]}, None


def _make_world():
    from scenario_gen import FACTORIAL, sample_world
    cell = next(c for c in FACTORIAL if c.name == "phys_in__iot_agree")
    return sample_world(3000, cell, asset_id="chiller_6",
                        scenario_id="A-chiller_6-phys_in__iot_agree-3000")


def _structure(ret):
    """The comparable shape: keys, types, and the config-independent values."""
    return {
        "keys": sorted(ret.keys()),
        "world_id": ret["world_id"],
        "arm": ret["arm"],
        "gold": ret["gold"],
        "verdict": ret["verdict"],
        "metric_type": type(ret["metric"]).__name__,
        "metric_keys": sorted(ret["metric"].to_dict().keys()),
        "integrity_flags_type": type(ret["integrity_flags"]).__name__,
        "call_errors": ret["call_errors"],
    }


@pytest.fixture
def stub_chat(monkeypatch):
    stub = _RecordingChat()
    monkeypatch.setattr(MOD.R, "_chat", stub)
    monkeypatch.setattr(MOD, "CouchDBExecutor", _FakeExecutor)
    return stub


def test_default_call_uses_module_globals(stub_chat):
    MOD.run_a_episode(_make_world(), "FULL")
    assert stub_chat.calls, "chat was never invoked"
    for (model, api_key, base_url) in stub_chat.calls:
        assert model == MOD.MODEL
        assert api_key == MOD.API_KEY
        assert base_url == MOD.BASE_URL


def test_explicit_defaults_are_identical_to_the_bare_call(stub_chat):
    bare = MOD.run_a_episode(_make_world(), "FULL")
    calls_after_bare = list(stub_chat.calls)
    stub_chat.calls.clear()

    explicit = MOD.run_a_episode(_make_world(), "FULL",
                                 model=MOD.MODEL, api_key=MOD.API_KEY, base_url=MOD.BASE_URL)

    assert _structure(bare) == _structure(explicit)
    assert bare["metric"].to_dict() == explicit["metric"].to_dict()
    # same number of _chat round-trips, same triples
    assert calls_after_bare == stub_chat.calls


def test_explicit_config_is_the_only_thing_that_changes(stub_chat):
    MOD.run_a_episode(_make_world(), "FULL",
                      model="probe/model-x", api_key="KEY-X", base_url="https://x.invalid/v1")
    assert stub_chat.calls
    for (model, api_key, base_url) in stub_chat.calls:
        assert (model, api_key, base_url) == ("probe/model-x", "KEY-X", "https://x.invalid/v1")


def test_return_keys_are_the_frozen_set(stub_chat):
    ret = MOD.run_a_episode(_make_world(), "FULL")
    assert sorted(ret.keys()) == sorted(
        ["world_id", "arm", "gold", "verdict", "metric", "integrity_flags", "call_errors"])
