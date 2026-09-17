"""Shared fixtures for orchestrator/kinematics unit tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _mujoco_available() -> bool:
    try:
        import mujoco  # noqa: F401
        return True
    except ImportError:
        return False


def _rosclaw_available() -> bool:
    # Uses the same Python-3.10-safe loader digital_twin_firewall.py/
    # spot_admissibility_verifier.py use at runtime (kinematics/_rosclaw_compat.py)
    # rather than a bare import, so tests aren't skipped on envs (e.g.
    # gauge_train310, vidbot) where rosclaw's own __init__.py chain needs
    # Python 3.11+ features unrelated to firewall.decorator.
    try:
        from orchestrator.kinematics._rosclaw_compat import load_digital_twin_firewall_cls
        load_digital_twin_firewall_cls()
        return True
    except ImportError:
        return False


requires_mujoco = pytest.mark.skipif(not _mujoco_available(), reason="mujoco not installed")
requires_rosclaw = pytest.mark.skipif(not _rosclaw_available(), reason="rosclaw not importable (set ROSCLAW_SRC)")

REGISTRY_PATH = (
    Path.home() / "AssetOpsBenchScenarioGeneration" / "RobotInspection" / "shared" / "robot_assets_registry.json"
)


@pytest.fixture
def registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text())
