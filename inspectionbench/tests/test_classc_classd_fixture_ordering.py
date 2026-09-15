"""Regression tests for the C/D fixture-application-ordering fix
(Phase 8H.2 blocker repair, run_classc_pilot.py / run_classd_pilot.py /
phase8h1_run_pilot.py).

Bug: ``ex.reset()`` unconditionally writes ``panel_stuck=False`` (and
``gauge_value``/``gauge_range``) to the profile document. The frozen pilot
path never called ``FixtureSession`` at all, and even ``run_classc_pilot.py``'s
own ``main()`` -- previously treated as the reference-correct implementation
-- wrapped a ``FixtureSession`` *around* a call to ``run()``, which calls
``ex.reset()`` as its own first line, clobbering any profile field the
fixture had just applied. Forensically proven for R001's ``panel_stuck=True``
(observation-hash reconstruction, Phase 8H.2 C-integrity triage).

Fix: reset FIRST, fixture SECOND, then ``run(..., skip_reset=True)`` so
``run()`` never re-clears what the fixture just wrote. ``skip_reset``
defaults to ``False`` so every existing call site that omits it is
unchanged. No model/API call in this test file -- ``run()`` itself is never
invoked (that would require a real ``_chat``); only the reset/fixture
ordering that precedes it is tested directly against the real executor and
real CouchDB-backed profile documents.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from couchdb_executor import CouchDBExecutor  # noqa: E402
from classc_fixtures import FIXTURES, FixtureSession  # noqa: E402


def test_reset_alone_clears_panel_stuck():
    """Establishes the bug mechanism directly: reset() always writes False,
    regardless of any fixture -- this is the fact the fix works around, not
    a fact the fix changes."""
    ex = CouchDBExecutor()
    ex.reset("R001", "FULL", seed=1)
    doc = ex._robot.db.get("profile:chiller_6")
    assert doc.get("panel_stuck") is False


def test_reset_then_fixture_survives_into_the_episode():
    """The fixed order: reset() first, FixtureSession second. panel_stuck
    must read True inside the session -- this is what run(skip_reset=True)
    now relies on."""
    ex = CouchDBExecutor()
    fx = FIXTURES["R001"]
    ex.reset("R001", "FULL", seed=1)
    with FixtureSession(ex._robot.db, fx) as sess:
        assert sess.verify_applied() == []
        doc = ex._robot.db.get("profile:chiller_6")
        assert doc.get("panel_stuck") is True


def test_fixture_then_reset_reproduces_the_original_bug():
    """The OLD (buggy) order, kept as a named regression test: fixture
    first, reset second -- exactly what run()'s internal unconditional
    ex.reset() call did to a FixtureSession applied around it. If this
    stops reproducing the clobber, the bug's own mechanism has changed and
    the fix above should be re-examined."""
    ex = CouchDBExecutor()
    fx = FIXTURES["R001"]
    with FixtureSession(ex._robot.db, fx):
        ex.reset("R001", "FULL", seed=1)
        doc = ex._robot.db.get("profile:chiller_6")
        assert doc.get("panel_stuck") is False, (
            "if this fails, reset() no longer clobbers panel_stuck and the "
            "skip_reset fix may be unnecessary -- do not assume, re-verify")


def test_fixture_session_restores_state_on_exit():
    """No leakage between episodes: after the session closes, panel_stuck
    reverts to whatever reset() had last established."""
    ex = CouchDBExecutor()
    fx = FIXTURES["R001"]
    ex.reset("R001", "FULL", seed=1)
    with FixtureSession(ex._robot.db, fx):
        pass
    doc = ex._robot.db.get("profile:chiller_6")
    assert doc.get("panel_stuck") is False


def test_run_classc_pilot_run_accepts_skip_reset_and_fixture_verified():
    """Signature-level check that the new keyword-only parameters exist
    with the documented defaults, without calling run() (which would make
    a real _chat call)."""
    import inspect
    import run_classc_pilot as C

    sig = inspect.signature(C.run)
    assert "skip_reset" in sig.parameters
    assert sig.parameters["skip_reset"].default is False
    assert sig.parameters["skip_reset"].kind == inspect.Parameter.KEYWORD_ONLY
    assert "fixture_verified" in sig.parameters
    assert sig.parameters["fixture_verified"].default is True


def test_run_classd_pilot_run_accepts_skip_reset():
    import inspect
    import run_classd_pilot as D

    sig = inspect.signature(D.run)
    assert "skip_reset" in sig.parameters
    assert sig.parameters["skip_reset"].default is False
    assert sig.parameters["skip_reset"].kind == inspect.Parameter.KEYWORD_ONLY


def test_phase8h1_run_pilot_dispatch_uses_reset_then_fixture_order():
    """Source-fidelity check (not a duplicated hand-written copy): the real
    dispatch function must call ex.reset(...) before FixtureSession(...) for
    both classc and classd, and must pass skip_reset=True into run(). Greps
    the actual file so this breaks if the fix is ever reverted."""
    src = (REPO_ROOT / "scripts" / "phase8h1_run_pilot.py").read_text()
    classc_block = src[src.index('if unit.kind == "classc"'):src.index('if unit.kind == "classd"')]
    classd_block = src[src.index('if unit.kind == "classd"'):src.index('if unit.kind == "e_sequence"')]
    for block, label in ((classc_block, "classc"), (classd_block, "classd")):
        reset_idx = block.index("ex.reset(")
        fixture_idx = block.index("FixtureSession(")
        assert reset_idx < fixture_idx, f"{label}: reset() must precede FixtureSession()"
        assert "skip_reset=True" in block, f"{label}: run() must be called with skip_reset=True"
