"""P0 generator-integrity CI: world -> gold causality and executor projection.

Each test corresponds to a way the previous scenario set was built backwards
from its label. Live-CouchDB tests skip when it is down; the preflight does not.
"""

from __future__ import annotations

import base64
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

import scenario_gen as G  # noqa: E402
from scenario_gen import (  # noqa: E402
    COMMIT, ESCALATE, FACTORIAL, Cell, WorldState, check_generator_contract,
    derive_gold, generate, render_question, sample_world, to_couch_profile,
)
from tool_executor import ToolCall  # noqa: E402


def _executor():
    try:
        from couchdb_executor import CouchDBExecutor
        ex = CouchDBExecutor()
        if getattr(ex._robot, "db", None) is None:
            pytest.skip("CouchDB unavailable")
        return ex
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"executor unavailable: {exc}")


def _world(seed=31, cell=Cell(True, True), asset="chiller_6") -> WorldState:
    return sample_world(seed, cell, asset_id=asset)


# 1. no gold/verdict dependency ---------------------------------------------
def test_no_gold_or_verdict_dependency_in_sampling():
    assert check_generator_contract() == []


def test_rendering_and_projection_never_receive_gold():
    """Every world->artifact function takes a WorldState and nothing else."""
    import inspect
    for fn in (render_question, to_couch_profile, G.to_iot_payload,
               G.to_enterprise_payload):
        params = set(inspect.signature(fn).parameters)
        assert not (params & G.FORBIDDEN_SAMPLER_PARAMS), fn.__name__


# 2. deterministic generation by seed ---------------------------------------
def test_deterministic_generation_by_seed():
    a, b = generate(seed=909, per_cell=2), generate(seed=909, per_cell=2)
    assert a == b
    assert generate(seed=910, per_cell=2) != a


# 3. balanced world-cell coverage -------------------------------------------
def test_balanced_world_cell_coverage():
    recs = generate(seed=5, per_cell=4)
    cells = Counter(r["cell"] for r in recs)
    assert len(cells) == len(FACTORIAL) and set(cells.values()) == {4}


# 4. IoT mutation does not change gold (IoT is non-causal) -------------------
@pytest.mark.parametrize("cell", FACTORIAL)
def test_iot_mutation_does_not_change_gold(cell):
    w = _world(cell=cell)
    before = derive_gold(w).verdict
    for v in (w.gauge_range[0], w.gauge_range[1], w.operating_band[0]):
        w.iot_value = float(v)
        w.history_mean = float(v)
        assert derive_gold(w).verdict == before


# 5. physical mutation across the boundary changes gold ---------------------
def test_physical_mutation_across_safety_boundary_changes_gold():
    w = _world(cell=Cell(True, True))
    w.technician_present = False
    w.active_work_order = False
    assert derive_gold(w).verdict == COMMIT
    lo, hi = w.operating_band
    w.physical_value = hi + 0.1 * (hi - lo)     # cross the boundary
    assert derive_gold(w).verdict == ESCALATE


# 6. enterprise mutation changes gold only where causal ---------------------
def test_enterprise_mutation_changes_gold_only_where_causal():
    """Causal when the physical reading is in band (it can flip COMMIT to
    ESCALATE); non-causal when already out of band, since gold is ESCALATE
    either way."""
    w = _world(cell=Cell(True, True))
    w.technician_present = False
    w.active_work_order = False
    assert derive_gold(w).verdict == COMMIT
    w.technician_present = True
    assert derive_gold(w).verdict == ESCALATE          # causal here

    out = _world(seed=44, cell=Cell(False, True))
    out.technician_present = False
    out.active_work_order = False
    assert derive_gold(out).verdict == ESCALATE
    out.active_work_order = True
    assert derive_gold(out).verdict == ESCALATE        # non-causal here


# 7. question rendering cannot leak gold ------------------------------------
def test_question_rendering_cannot_leak_gold():
    """The physical value and the enterprise state are omitted: stating either
    would let an agent reach gold with no evidence-gathering."""
    for rec in generate(seed=808, per_cell=3):
        w = WorldState(**rec["world"])
        q = render_question(w)
        assert f"{w.physical_value:g}" not in q, "physical value leaked"
        # When telemetry agrees with the gauge, printing it publishes the
        # hidden reading; it must come from read_iot instead.
        assert f"{w.iot_value:g}" not in q, "IoT value leaked"
        for token in ("technician", "work order", "work_order"):
            assert token not in q.lower(), f"enterprise state leaked: {token}"
        body = q.split("Return {")[0]
        for verdict in ("COMMIT", "ESCALATE", "ABORT"):
            assert verdict not in body, "verdict named outside the format line"


# 8. CouchDB state matches the generated WorldState -------------------------
def test_couchdb_state_matches_the_generated_world():
    ex = _executor()
    for rec in generate(seed=606, per_cell=1):
        w = WorldState(**rec["world"])
        ex.reset_from_world(w, "FULL", seed=1)
        doc = ex._robot.db.get(f"profile:{w.asset}")
        assert float(doc["gauge_value"]) == pytest.approx(w.physical_value)
        assert list(doc["gauge_range"]) == list(w.gauge_range)


# 9. rendered observation matches the generated WorldState ------------------
def test_rendered_observation_matches_the_generated_world():
    """Two worlds differing only in physical value must render different images;
    the same world must render identically."""
    ex = _executor()
    w1 = _world(seed=71, cell=Cell(True, True))
    ex.reset_from_world(w1, "FULL", seed=1)
    a = ex.execute(ToolCall("capture_image")).image_b64
    ex.reset_from_world(w1, "FULL", seed=1)
    b = ex.execute(ToolCall("capture_image")).image_b64
    assert a == b and base64.b64decode(a)[:8] == b"\x89PNG\r\n\x1a\n"

    w2 = WorldState(**w1.to_dict())
    w2.physical_value = w1.operating_band[1] + 20
    ex.reset_from_world(w2, "FULL", seed=1)
    assert ex.execute(ToolCall("capture_image")).image_b64 != a


def test_read_gauge_tracks_the_generated_world_without_revealing_it():
    ex = _executor()
    w = _world(seed=91, cell=Cell(False, True), asset="hydraulic_pump_1")
    ex.reset_from_world(w, "FULL", seed=1)
    r = ex.execute(ToolCall("read_gauge", {"attempt_n": 1}))
    span = w.gauge_range[1] - w.gauge_range[0]
    assert abs(r.payload["reading"] - w.physical_value) < 0.05 * span
    assert "gauge_value" not in r.payload


def test_iot_and_enterprise_tools_project_the_world():
    ex = _executor()
    w = _world(seed=101, cell=Cell(True, False))
    w.technician_present = True
    ex.reset_from_world(w, "FULL", seed=1)
    assert ex.execute(ToolCall("read_iot")).payload["value"] == pytest.approx(w.iot_value)
    ent = ex.execute(ToolCall("get_work_order")).payload
    assert ent["technician_present"] is True
