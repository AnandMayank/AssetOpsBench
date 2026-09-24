"""No B2 episode/world id collides with any frozen-manifest id (plan K)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

from playground.b2_families import build_pool  # noqa: E402

FROZEN_MANIFESTS = [
    "reports/benchmark/final_benchmark_manifest_v3.json",
    "reports/benchmark/final_benchmark_manifest.json",
    "reports/ec/phase8h1_pilot_manifest.json",
    "reports/benchmark/cd_4000_final_manifest.json",
    "reports/benchmark/b_acquisition_final_manifest.json",
    "reports/benchmark/d_physical_manifest.json",
]


def test_pg_b2_ids_use_a_disjoint_namespace():
    pool = build_pool()
    for spec, _ in pool:
        assert spec.episode_id.startswith("PG-B2::")
        assert spec.world.world_id.startswith("PGW-")


def test_no_pg_b2_id_appears_in_any_frozen_manifest_text():
    pool = build_pool()
    b2_ids = {spec.episode_id for spec, _ in pool} | {spec.world.world_id for spec, _ in pool}
    for rel in FROZEN_MANIFESTS:
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        text = path.read_text()
        for eid in b2_ids:
            assert eid not in text, f"{eid} collides with content of {rel}"
