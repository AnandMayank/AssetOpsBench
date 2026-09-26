#!/usr/bin/env python3
"""phase8j0_build_e_expanded_v2_pool.py -- builds the E-expanded-v2 pool
(30 MORE sequences x 3 episodes = 90 canonical E episodes, on top of the
existing E-expanded-v1's 30 sequences/90 episodes) ENTIRELY OFFLINE. Zero
API calls, zero model execution.

Same generator, same pattern as phase8h2y_build_e_expanded_pool.py
(sequence_executor.sample_sequence / derive_sequence_gold, unmodified).
Fresh seed range that cannot collide with E-v1 (5000-5005) or
E-expanded-v1 (6000-6029). Combined with v1 (90) + the historical 18,
this gives 90 (v1) + 90 (v2) = 180 canonical E episodes per model for
the swapped panel's scale-up, at the user's explicit request to expand
E's defensibility within a fixed compute-time budget -- purpose
unchanged from v1: get the reobservation-necessity confusion matrix
(Precision/Recall/F1 for "reacquires exactly when the tracked state
actually changed") to a larger, more defensible per-model N.

Outputs (all new, additive; does not touch E-v1 or E-expanded-v1):
  inspectionbench/manifests/e_temporal_expanded_v2.json
  reports/benchmark/e_expanded_v2/matched_sequence_audit.csv
  reports/benchmark/e_expanded_v2/matched_sequence_audit.json
  reports/benchmark/e_expanded_v2/necessity_diversity_audit.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

from sequence_executor import sample_sequence, derive_sequence_gold  # noqa: E402
from canonical_identity import normalize_world  # noqa: E402

MANIFEST_OUT = REPO_ROOT / "inspectionbench" / "manifests" / "e_temporal_expanded_v2.json"
AUDIT_DIR = REPO_ROOT / "reports" / "benchmark" / "e_expanded_v2"

N_SEQUENCES = 30                  # v1 used 30 (6000-6029); this doubles v1+v2 to 60 seq / 180 ep
N_EPISODES_PER_SEQ = 3
SEED_BASE = 7000                  # E-v1: 5000-5005, E-expanded-v1: 6000-6029; 7000+ cannot collide
TOOL_CONTRACT = "SequenceExecutor (shared battery_budget)"
REQUIRED_MODALITY_E = "physical"
GENERATION_PROVENANCE = (
    "sequence_executor.sample_sequence (unmodified, identical to E-v1/E-expanded-v1's "
    "generator) -- world sampled strictly before any gold label is computed "
    "(derive_sequence_gold is a pure function of the world and episode prefix, "
    "label-blindness contract preserved); 30 independent seeds produce genuinely "
    "distinct SequenceWorld instances (different asset, base_value, drift process "
    "draw), not repeats."
)


def build_sequences() -> List[Dict[str, Any]]:
    sequences = []
    for i in range(N_SEQUENCES):
        seed = SEED_BASE + i
        world = sample_sequence(seed, n_episodes=N_EPISODES_PER_SEQ)
        canon = normalize_world(world)
        golds = [derive_sequence_gold(world, ep) for ep in range(N_EPISODES_PER_SEQ)]
        sequences.append({
            "world": world, "seed": seed, "sequence_id": world.sequence_id,
            "asset_id": world.asset, "process_kind": world.process.kind,
            "world_hash": canon.world_id, "golds": golds,
        })
    return sequences


def build_manifest(sequences: List[Dict[str, Any]]):
    episodes = []
    seq_summaries = []
    for s in sequences:
        seq_id = s["sequence_id"]
        for ep in range(N_EPISODES_PER_SEQ):
            gold = s["golds"][ep]
            eid = f"E3::{seq_id}::ep{ep}"
            episodes.append({
                "episode_id": eid, "dimension": "E", "world_id": seq_id,
                "world_seed": s["seed"], "normalized_world_hash": s["world_hash"],
                "scenario_id": seq_id, "scenario_template": "sample_sequence",
                "scenario_seed": s["seed"],
                "scenario_parameters": {"episode_index": ep, "n_episodes": N_EPISODES_PER_SEQ,
                                        "process_kind": s["process_kind"]},
                "regime": "FULL",
                "gold_action": gold["verdict"],
                "band_crossed_by_now": gold["band_crossed_by_now"],
                "required_modality": REQUIRED_MODALITY_E,
                "tool_contract": TOOL_CONTRACT,
                "asset_id": s["asset_id"],
                "generation_provenance": GENERATION_PROVENANCE,
                "pool_version": "e_temporal_expanded_v2",
            })
        seq_summaries.append({
            "sequence_id": seq_id, "asset_id": s["asset_id"], "seed": s["seed"],
            "process_kind": s["process_kind"], "world_hash": s["world_hash"],
            "gold_verdicts_by_episode": [g["verdict"] for g in s["golds"]],
            "band_crossed_by_episode": [g["band_crossed_by_now"] for g in s["golds"]],
        })

    manifest = {
        "pool_version": "e_temporal_expanded_v2",
        "dimension": "E",
        "description": (
            "E-expanded-v2: 30 MORE sequences x 3 episodes = 90 canonical E episodes, "
            "additive to E-expanded-v1's 30 sequences/90 episodes. Combined per-model N "
            "for E across v1+v2 is 180. Built at explicit user request to scale up E's "
            "defensibility for the swapped primary panel within a fixed compute-time "
            "budget, after C was confirmed to already be at its maximum defensible N (17) "
            "using only already-authored, non-control, non-twin fixtures. Does NOT "
            "replace, modify, or get merged with E-v1 or E-expanded-v1."
        ),
        "target_sequences": N_SEQUENCES, "target_episodes": N_SEQUENCES * N_EPISODES_PER_SEQ,
        "actual_sequences": len(sequences), "actual_episodes": len(episodes),
        "protocol_reference": (
            "IDENTICAL to E-v1/E-expanded-v1: scripts/run_class_e_pilot.py::run_sequence, "
            "sequence_executor.SequenceExecutor, shared battery_budget, "
            "metric_contract.score_episode + sequence_executor.stale_state_reuse scoring. "
            "No protocol change of any kind."
        ),
        "generation_provenance": GENERATION_PROVENANCE,
        "seed_range": [SEED_BASE, SEED_BASE + N_SEQUENCES - 1],
        "prior_seed_ranges_for_collision_check": {"e_v1": [5000, 5005], "e_expanded_v1": [6000, 6029]},
        "episodes": episodes,
    }
    return manifest, seq_summaries


def audit_diversity(seq_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    from collections import Counter
    process_dist = Counter(s["process_kind"] for s in seq_summaries)
    asset_dist = Counter(s["asset_id"] for s in seq_summaries)
    n_with_transition = 0
    for s in seq_summaries:
        crossed = s["band_crossed_by_episode"]
        if any(crossed[i] != crossed[i - 1] for i in range(1, len(crossed))):
            n_with_transition += 1
    return {
        "n_sequences": len(seq_summaries),
        "process_kind_distribution": dict(process_dist),
        "asset_distribution": dict(asset_dist),
        "n_sequences_with_a_band_transition": n_with_transition,
        "n_sequences_stationary_throughout": sum(
            1 for s in seq_summaries if len(set(s["band_crossed_by_episode"])) == 1),
    }


def main() -> int:
    print(f"Building E-expanded-v2 pool: {N_SEQUENCES} sequences x {N_EPISODES_PER_SEQ} episodes "
         f"= {N_SEQUENCES * N_EPISODES_PER_SEQ} episodes")
    sequences = build_sequences()
    assert len(sequences) == N_SEQUENCES

    manifest, seq_summaries = build_manifest(sequences)
    assert manifest["actual_episodes"] == N_SEQUENCES * N_EPISODES_PER_SEQ

    eids = [e["episode_id"] for e in manifest["episodes"]]
    assert len(eids) == len(set(eids)), "duplicate episode_id found"
    seq_ids = [s["sequence_id"] for s in seq_summaries]
    assert len(seq_ids) == len(set(seq_ids)), "duplicate sequence_id found"
    for e in manifest["episodes"]:
        assert e["gold_action"] in ("COMMIT", "ESCALATE"), f"invalid gold_action: {e}"
        assert e["normalized_world_hash"], f"missing world_hash: {e['episode_id']}"
    e_v1_seeds = set(range(5000, 5006))
    e_exp_v1_seeds = set(range(6000, 6030))
    e2_seeds = set(s["seed"] for s in seq_summaries)
    assert not (e_v1_seeds & e2_seeds), "seed collision with E-v1"
    assert not (e_exp_v1_seeds & e2_seeds), "seed collision with E-expanded-v1"
    print(f"Integrity: 0 duplicate episode/sequence IDs, 0 missing gold actions, "
         f"0 missing world hashes, 0 seed collision with E-v1 or E-expanded-v1 -- PASS")

    diversity = audit_diversity(seq_summaries)
    print(json.dumps(diversity, indent=2))

    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {MANIFEST_OUT} ({manifest['actual_episodes']} episodes)")

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIT_DIR / "matched_sequence_audit.json").write_text(json.dumps(seq_summaries, indent=2))
    with open(AUDIT_DIR / "matched_sequence_audit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sequence_id", "asset_id", "seed", "process_kind",
                                           "world_hash", "gold_verdicts_by_episode",
                                           "band_crossed_by_episode"])
        w.writeheader()
        for s in seq_summaries:
            row = dict(s)
            row["gold_verdicts_by_episode"] = "|".join(row["gold_verdicts_by_episode"])
            row["band_crossed_by_episode"] = "|".join(str(x) for x in row["band_crossed_by_episode"])
            w.writerow(row)
    (AUDIT_DIR / "necessity_diversity_audit.json").write_text(json.dumps(diversity, indent=2))
    print(f"Wrote {AUDIT_DIR}/matched_sequence_audit.{{json,csv}}, necessity_diversity_audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
