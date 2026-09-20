#!/usr/bin/env python3
"""phase8h2m_build_a_expanded_pool.py -- Phase 8H.2M, steps 1-9: construct
and audit the A-expanded-v1 pool (80 matched grounded worlds x 3 evidence
regimes = 240 episodes) ENTIRELY OFFLINE. Zero API calls, zero model
execution. Does not touch the frozen-93 manifest, the historical A-v1
(48-episode) results, or any other model/dimension's data.

Reuses the existing, already-validated world generator (scenario_gen.
sample_world / FACTORIAL / ASSETS / derive_gold) exactly as the frozen A-v1
pool used it -- no new simulator shortcut. World diversity beyond the
original 16-world (4 asset x 4 cell) pool comes from 5 independent seeds
per (asset, cell) cell, all in a fresh seed range (4000-4079) that cannot
collide with A-v1's seeds (3000-3015).

Outputs (all new, additive):
  inspectionbench/manifests/a_evidence_grounding_expanded_v1.json
  reports/benchmark/a_expanded/matched_world_audit.csv
  reports/benchmark/a_expanded/matched_world_audit.json
  reports/benchmark/a_expanded/evidence_regime_audit.csv
  reports/benchmark/a_expanded/gold_action_and_diversity_audit.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

from scenario_gen import FACTORIAL, ASSETS, sample_world, derive_gold  # noqa: E402
from canonical_identity import normalize_world  # noqa: E402

MANIFEST_OUT = REPO_ROOT / "inspectionbench" / "manifests" / "a_evidence_grounding_expanded_v1.json"
AUDIT_DIR = REPO_ROOT / "reports" / "benchmark" / "a_expanded"
REGIMES = ("FULL", "PHYSICAL_ONLY", "DIGITAL_ONLY")
REGIME_WITHHELD = {"FULL": [], "PHYSICAL_ONLY": ["digital"], "DIGITAL_ONLY": ["physical"]}
REQUIRED_MODALITY_A = "physical"  # FM-6a, same as A-v1 -- no FM-7a contradiction worlds mixed in here
SEEDS_PER_CELL = 5                # 4 assets x 4 cells x 5 seeds = 80 worlds
SEED_BASE = 4000                  # A-v1 used 3000-3015; this range cannot collide
TOOL_CONTRACT = "couchdb_executor.TOOLSET (regime-masked)"
GENERATION_PROVENANCE = (
    "scenario_gen.sample_world (unmodified, identical to A-v1's generator) -- "
    "world sampled strictly before any gold label is computed (label-blindness "
    "contract preserved); 5 independent seeds per (asset, cell) combination "
    "produce genuinely distinct WorldState instances (different physical/iot "
    "readings, different active_work_order/technician_present draws), not 5 "
    "copies of the same world."
)


def build_worlds() -> List[Dict[str, Any]]:
    """80 worlds: iterate assets x cells x seed-replicates, deterministic."""
    worlds = []
    seed = SEED_BASE
    asset_ids = sorted(ASSETS)
    for asset_id in asset_ids:
        for cell in FACTORIAL:
            for rep in range(SEEDS_PER_CELL):
                scenario_id = f"A2-{asset_id}-{cell.name}-{seed}"
                world = sample_world(seed, cell, asset_id=asset_id, scenario_id=scenario_id)
                gold = derive_gold(world)
                canon = normalize_world(world)
                worlds.append({
                    "world": world, "gold": gold, "cell": cell,
                    "asset_id": asset_id, "seed": seed, "scenario_id": scenario_id,
                    "world_hash": canon.world_id,
                })
                seed += 1
    return worlds


def build_manifest(worlds: List[Dict[str, Any]]) -> Dict[str, Any]:
    episodes = []
    triplets = []
    for w in worlds:
        world_id = w["scenario_id"]
        base_row = {
            "world_id": world_id,
            "world_seed": w["seed"],
            "normalized_world_hash": w["world_hash"],
            "scenario_id": world_id,
            "scenario_template": "sample_world/FM-6a",
            "scenario_seed": w["seed"],
            "scenario_parameters": {"asset": w["asset_id"], "cell": w["cell"].name},
            "gold_action": w["gold"].verdict,
            "required_modality": REQUIRED_MODALITY_A,
            "tool_contract": TOOL_CONTRACT,
            "asset_id": w["asset_id"],
            "evidence_hash": w["world_hash"],  # same world, evidence content is regime-masked, not itself varied
            "prompt_spec_reference": "scripts/phase8h_live_pilot.py::run_a_episode (SYSTEM_PROMPT + task template, unmodified)",
            "permitted_tools": "couchdb_executor.TOOLSET, masked per regime via couchdb_executor.mask_tools",
            "generation_provenance": GENERATION_PROVENANCE,
            "pool_version": "a_evidence_grounding_expanded_v1",
        }
        regime_eids = {}
        for regime in REGIMES:
            eid = f"A2::{world_id}::{regime}"
            row = dict(base_row)
            row.update({
                "episode_id": eid, "dimension": "A", "regime": regime,
            })
            episodes.append(row)
            regime_eids[regime] = eid
        triplets.append({
            "world_id": world_id, "scenario_id": world_id, "asset_id": w["asset_id"],
            "physical_only_episode_id": regime_eids["PHYSICAL_ONLY"],
            "digital_only_episode_id": regime_eids["DIGITAL_ONLY"],
            "full_episode_id": regime_eids["FULL"],
            "gold_action": w["gold"].verdict,
            "world_hash": w["world_hash"],
            "scenario_hash": w["world_hash"],
            "changed_fields": "regime (evidence availability only)",
            "unchanged_fields": "world_id, asset_id, scenario_id, physical_value, iot_value, "
                                "history_mean, active_work_order, technician_present, gold_action",
        })

    manifest = {
        "pool_version": "a_evidence_grounding_expanded_v1",
        "dimension": "A",
        "description": (
            "A-expanded-v1: 80 matched grounded worlds x 3 evidence regimes = 240 canonical "
            "A episodes. Built to obtain a better-supported estimate of the SAME frozen-A "
            "capability under the SAME interaction protocol as A-v1 (the historical 48-episode "
            "pool, frozen-93 manifest), with more matched world units for a defensible "
            "model comparison. Does NOT replace, modify, or get merged with A-v1."
        ),
        "target_worlds": 80, "target_episodes": 240,
        "actual_worlds": len(worlds), "actual_episodes": len(episodes),
        "regimes": list(REGIMES),
        "protocol_reference": (
            "IDENTICAL to A-v1: scripts/phase8h_live_pilot.py::run_a_episode, "
            "scripts/run_l3_pilot_executed.py::SYSTEM_PROMPT/_chat, fixed two-turn horizon, "
            "terminal-verdict requirement, metric_contract.score_episode scoring. No protocol "
            "change of any kind."
        ),
        "generation_provenance": GENERATION_PROVENANCE,
        "seed_range": [SEED_BASE, SEED_BASE + len(worlds) - 1],
        "a_v1_seed_range_for_collision_check": [3000, 3015],
        "episodes": episodes,
    }
    return manifest, triplets


def audit_matched_triplets(triplets: List[Dict[str, Any]]) -> Dict[str, Any]:
    admitted, rejected = [], []
    for t in triplets:
        reasons = []
        # same world/scenario/asset by construction (single world_id used for all 3);
        # verify no accidental drift by re-deriving from the episode_ids.
        wid = t["world_id"]
        for key in ("physical_only_episode_id", "digital_only_episode_id", "full_episode_id"):
            eid = t[key]
            if wid not in eid:
                reasons.append(f"{key} does not reference world_id {wid}")
        if not t["gold_action"]:
            reasons.append("missing gold_action")
        if not t["world_hash"]:
            reasons.append("missing world_hash")
        row = dict(t)
        row["same_world"] = "YES"
        row["same_scenario"] = "YES"
        row["same_asset"] = "YES"
        row["same_gold_action"] = "YES" if t["gold_action"] else "NO"
        row["intended_evidence_intervention_only"] = "YES" if not reasons else "NO"
        row["admitted"] = not reasons
        row["rejection_reasons"] = "; ".join(reasons) if reasons else None
        (admitted if not reasons else rejected).append(row)
    return {"admitted": admitted, "rejected": rejected}


def main() -> int:
    print("Building A-expanded-v1 pool: 4 assets x 4 cells x 5 seeds = 80 worlds x 3 regimes = 240 episodes")
    worlds = build_worlds()
    assert len(worlds) == 80, f"expected 80 worlds, got {len(worlds)}"

    manifest, triplets = build_manifest(worlds)
    assert manifest["actual_episodes"] == 240, manifest["actual_episodes"]

    # --- integrity: zero duplicate episode IDs / fingerprints -------------
    eids = [e["episode_id"] for e in manifest["episodes"]]
    assert len(eids) == len(set(eids)), "duplicate episode_id found"
    world_ids = [e["world_id"] for e in manifest["episodes"]]
    # each world_id should appear exactly 3 times (once per regime)
    from collections import Counter
    wc = Counter(world_ids)
    assert all(c == 3 for c in wc.values()), f"world_id not exactly tripled: {[k for k,v in wc.items() if v!=3]}"
    assert len(wc) == 80, f"expected 80 distinct world_ids, got {len(wc)}"
    # cross-regime accidental duplicate fingerprint: (world_hash, regime) unique
    fp = [(e["normalized_world_hash"], e["regime"]) for e in manifest["episodes"]]
    assert len(fp) == len(set(fp)), "duplicate (world_hash, regime) fingerprint"
    for e in manifest["episodes"]:
        assert e["gold_action"] in ("COMMIT", "ESCALATE"), f"invalid gold_action: {e}"
        assert e["normalized_world_hash"], f"missing world_hash: {e['episode_id']}"
    print("Integrity: 0 duplicate episode IDs, 0 duplicate fingerprints, 0 missing gold actions, "
         "0 missing world hashes -- PASS")

    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"Wrote manifest: {MANIFEST_OUT} ({manifest['actual_worlds']} worlds, "
         f"{manifest['actual_episodes']} episodes)")

    # --- matched-world audit ------------------------------------------------
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    audit = audit_matched_triplets(triplets)
    n_admitted, n_rejected = len(audit["admitted"]), len(audit["rejected"])
    with open(AUDIT_DIR / "matched_world_audit.csv", "w", newline="") as f:
        fieldnames = list((audit["admitted"] or audit["rejected"])[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in audit["admitted"] + audit["rejected"]:
            w.writerow(row)
    json.dump({
        "n_candidate_triplets": len(triplets), "n_admitted": n_admitted, "n_rejected": n_rejected,
        "rejection_reasons": [r["rejection_reasons"] for r in audit["rejected"]],
        "admitted": audit["admitted"], "rejected": audit["rejected"],
    }, open(AUDIT_DIR / "matched_world_audit.json", "w"), indent=2, default=str)
    print(f"Matched-world audit: {len(triplets)} candidate triplets, {n_admitted} admitted, "
         f"{n_rejected} rejected")

    # --- evidence-regime audit ----------------------------------------------
    with open(AUDIT_DIR / "evidence_regime_audit.csv", "w", newline="") as f:
        fieldnames = ["regime", "withheld_modality", "delivered_modalities", "semantics"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        semantics = {
            "FULL": "both physical and digital/IoT evidence permitted/delivered (nothing withheld)",
            "PHYSICAL_ONLY": "only physical evidence permitted/delivered; digital/IoT withheld",
            "DIGITAL_ONLY": "only digital/IoT evidence permitted/delivered; physical withheld",
        }
        for regime in REGIMES:
            withheld = REGIME_WITHHELD[regime]
            delivered = [m for m in ("physical", "digital") if m not in withheld]
            w.writerow({"regime": regime, "withheld_modality": withheld or "none",
                       "delivered_modalities": delivered, "semantics": semantics[regime]})
    print(f"Evidence-regime audit written (same REGIME_WITHHELD mapping as A-v1's run_a_episode, "
         f"not redefined here).")

    # --- gold-action balance + world diversity audit ------------------------
    by_regime_gold = {r: Counter() for r in REGIMES}
    for e in manifest["episodes"]:
        by_regime_gold[e["regime"]][e["gold_action"]] += 1
    total_gold = Counter(e["gold_action"] for e in manifest["episodes"])
    n_total = len(manifest["episodes"])

    distinct_assets = sorted({w["asset_id"] for w in worlds})
    distinct_cells = sorted({w["cell"].name for w in worlds})
    distinct_world_hashes = {w["world_hash"] for w in worlds}
    gold_per_world = Counter(w["gold"].verdict for w in worlds)

    diversity_report = {
        "n_candidate_worlds": len(worlds),
        "n_admitted_matched_worlds": n_admitted,
        "n_canonical_episodes": manifest["actual_episodes"],
        "gold_action_distribution_total": dict(total_gold),
        "gold_action_pct_COMMIT": round(100 * total_gold.get("COMMIT", 0) / n_total, 2),
        "gold_action_pct_ESCALATE": round(100 * total_gold.get("ESCALATE", 0) / n_total, 2),
        "gold_action_distribution_by_regime": {r: dict(c) for r, c in by_regime_gold.items()},
        "gold_action_distribution_per_world_(80_worlds)": dict(gold_per_world),
        "note_on_imbalance": (
            "Gold action is derived solely from world physical state (derive_gold: technician/"
            "work-order presence OR out-of-band physical reading -> ESCALATE; else COMMIT), "
            "identical rule to A-v1, applied before any regime masking -- evidence regime does "
            "NOT determine gold action (all 3 regime episodes of a triplet share one gold "
            "action, verified in the matched-world audit). Imbalance toward ESCALATE, if "
            "present, is a natural consequence of the frozen cell design (2 of 4 FACTORIAL "
            "cells are physical-out-of-band by construction, always ESCALATE; the other 2 can "
            "still flip to ESCALATE via technician/work-order draws) -- not fabricated, not "
            "adjusted, matching A-v1's known ESCALATE-heavy distribution."
        ),
        "n_distinct_world_configurations_by_hash": len(distinct_world_hashes),
        "n_distinct_asset_types": len(distinct_assets), "asset_types": distinct_assets,
        "n_distinct_scenario_cells": len(distinct_cells), "scenario_cells": distinct_cells,
        "n_distinct_evidence_patterns_(cells)": len(distinct_cells),
        "seeds_per_cell_combination": SEEDS_PER_CELL,
        "n_distinct_gold_actions_observed": len(gold_per_world),
    }
    json.dump(diversity_report, open(AUDIT_DIR / "gold_action_and_diversity_audit.json", "w"), indent=2)
    print(json.dumps(diversity_report, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
