"""Guards that this package never touched the frozen InspectionBench
substrate (plan section K: 'frozen unchanged')."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

import pilot_dispatch  # noqa: E402


def test_frozen_93_manifest_sha_and_counts():
    # load_manifest() itself raises ManifestIntegrityError on any SHA or
    # count mismatch (pilot_dispatch.py:58-75) -- a successful call IS the
    # check; we additionally assert the episode count explicitly.
    manifest = pilot_dispatch.load_manifest()
    assert len(manifest["episodes"]) == 93


def test_v3_final_manifest_total_is_4075():
    path = REPO_ROOT / "reports" / "benchmark" / "final_benchmark_manifest_v3.json"
    data = json.loads(path.read_text())
    assert data["final_canonical_total"] == 4075


def test_b_acquisition_final_manifest_still_66():
    path = REPO_ROOT / "reports" / "benchmark" / "b_acquisition_final_manifest.json"
    data = json.loads(path.read_text())
    assert data["total"] == 66
    assert len(data["episodes"]) == 66


def test_d_physical_manifest_still_70():
    path = REPO_ROOT / "reports" / "benchmark" / "d_physical_manifest.json"
    data = json.loads(path.read_text())
    assert data["canonical_final"] == 70
