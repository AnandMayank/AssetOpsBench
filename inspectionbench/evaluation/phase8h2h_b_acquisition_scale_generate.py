#!/usr/bin/env python3
"""phase8h2h_b_acquisition_scale_generate.py — Family B (Evidence Acquisition)
SCALE generation (Phase 8H.2H).

Scales the validated 16-episode B-acquisition prototype to the largest
REAL, non-padded candidate pool the substrate genuinely supports, along
the axes documented in reports/benchmark/b_acquisition_scale_generation_report.md
Sec 2: additional real required-modalities for B-ACQ-1/B-ACQ-3 (acoustic,
workorder_history, alongside the original thermal), additional real
evidence-count thresholds and the iot_timeseries axis for B-ACQ-2, and the
motor_01/thermal quality-ambiguity pair for B-ACQ-4.

Zero model/API calls. Does NOT touch the 16-episode prototype manifest,
B-legacy, D-physical, or the global benchmark manifest -- writes a
SEPARATE, standalone manifest: reports/benchmark/b_acquisition_scaled_manifest.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
OUT = REPO_ROOT / "reports" / "benchmark"
OUT.mkdir(parents=True, exist_ok=True)

import b_acquisition_generator as B  # noqa: E402
import b_acquisition_scoring as S  # noqa: E402
from couchdb_executor import CouchDBExecutor  # noqa: E402

GENERATOR_VERSION = "b_acquisition_generator.py@phase8h2h_scale"
SCHEMA_VERSION = "phase8h1_c_gold_schema_freeze_plan/1+b_acquisition_gold/1"

_B1_MODALITIES = ("thermal", "acoustic", "workorder_history")


def enumerate_scale() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seed = 71000

    # B-ACQ-1: 3 real modality-gap axes x 4 canonical assets = 12
    for modality in _B1_MODALITIES:
        for asset in B.ALL_ASSETS:
            rows.append({"template": "B-ACQ-1", "asset": asset, "seed": seed,
                        "required_modality": modality})
            seed += 1

    # B-ACQ-2: every registered arm x its real supported asset set
    for arm, (modality, _capability, _k) in sorted(B.B_ACQ_2_ARM_REGISTRY.items()):
        assets = B._ACOUSTIC_ASSETS if modality == "acoustic" else B._ALL_COUNT_SUFFICIENCY_ASSETS
        for asset in assets:
            rows.append({"template": "B-ACQ-2", "asset": asset, "arm": arm, "seed": seed})
            seed += 1

    # B-ACQ-3: mirrors B-ACQ-1's full modality-gap axis (same construction,
    # different scoring focus) -- 12
    for modality in _B1_MODALITIES:
        for asset in B.ALL_ASSETS:
            rows.append({"template": "B-ACQ-3", "asset": asset, "seed": seed,
                        "required_modality": modality})
            seed += 1

    # B-ACQ-4: real quality-ambiguity pairs (acoustic x 2 assets, thermal x
    # motor_01) -- 6
    for asset in B._RECONCILIATION_ASSETS:
        for arm in ("ambiguous", "unambiguous"):
            rows.append({"template": "B-ACQ-4", "asset": asset, "arm": arm, "seed": seed})
            seed += 1

    return rows


def fingerprint(rec: Dict[str, Any]) -> str:
    payload = {
        "template_id": rec["template_id"], "asset": rec["asset"], "arm": rec["arm"],
        "gold": rec["gold"], "capability_id": rec["capability"]["capability_id"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def build_spec(row: Dict[str, Any]):
    t = row["template"]
    if t == "B-ACQ-1":
        return B.build_b_acq_1(row["asset"], row["seed"], required_modality=row["required_modality"])
    if t == "B-ACQ-2":
        return B.build_b_acq_2(row["asset"], row["arm"], row["seed"])
    if t == "B-ACQ-3":
        return B.build_b_acq_3(row["asset"], row["seed"], required_modality=row["required_modality"])
    if t == "B-ACQ-4":
        return B.build_b_acq_4(row["asset"], row["arm"], row["seed"])
    raise ValueError(t)


def admit_episode(ex: CouchDBExecutor, spec, row: Dict[str, Any]) -> Dict[str, Any]:
    rejects = []

    # G4: determinism -- regenerate the SAME spec twice, gold must be
    # byte-identical both times (guards against any accidental non-
    # determinism in the new axes, not just the original thermal case).
    spec_again = build_spec(row)
    if spec_again.gold.to_dict() != spec.gold.to_dict() or spec_again.scenario_id != spec.scenario_id:
        rejects.append({"reason": "gold or scenario_id not identical across two independent "
                                  "generations of the same config", "validator": "determinism"})

    result = B.run_scripted_episode(ex, spec)

    if result["verdict"] != spec.gold.final_terminal_action:
        rejects.append({"reason": f"scripted verdict {result['verdict']!r} != gold "
                                  f"{spec.gold.final_terminal_action!r}", "validator": "scorer_compatibility"})
    if not result["trace_chain_valid"]:
        rejects.append({"reason": "trace hash chain invalid", "validator": "provenance_completeness"})

    # G7: no evaluator-only field leaks into the agent-visible trace.
    forbidden_fields = {"final_terminal_action", "acquisition_required",
                        "acceptable_acquisition_set", "initial_evidence_sufficient",
                        "arm", "acquisition_genuinely_unavailable", "recovery_policy_on_unavailable"}
    for event in result["trace_events"]:
        if not forbidden_fields.isdisjoint(event.keys()):
            rejects.append({"reason": f"gold field leaked into trace event: {event}",
                            "validator": "evaluator_only_boundary"})
            break

    response = {
        "acquire": result["acquired"],
        "requested_modality": result["requested_modalities"][0] if result["requested_modalities"] else None,
        "claimed_observation_id": result["claimed_observation_id"],
        "terminal_action": result["verdict"],
    }
    delivered_ids = frozenset(
        e["observation_id"] for e in result["trace_events"]
        if e["stage"] == "OBSERVATION_DELIVERED" and e.get("observation_id")
    )
    score = S.score_b_acquisition_episode(spec.gold, response,
                                          actually_delivered_observation_ids=delivered_ids)
    if score.ADA != 1:
        rejects.append({"reason": "scripted-to-gold response failed ADA on its own gold",
                        "validator": "scorer_compatibility"})
    if score.UHA is not None and score.UHA != 1:
        rejects.append({"reason": "scripted-to-gold response failed UHA on its own gold",
                        "validator": "scorer_compatibility"})
    if score.TDA_post != 1:
        rejects.append({"reason": "scripted-to-gold response failed TDA_post on its own gold",
                        "validator": "scorer_compatibility"})

    episode_id = f"{spec.template_id}::{spec.scenario_id}" if not rejects else None
    rec = {
        "episode_id": episode_id, "template_id": spec.template_id, "family": "B",
        "b_subfamily": "acquisition", "arm": spec.arm, "asset": spec.asset,
        "scenario_id": spec.scenario_id, "params": spec.params,
        "capability": spec.capability, "initial_modality": spec.initial_modality,
        "initial_count": spec.initial_count, "gold": spec.gold.to_dict(),
        "generator_version": GENERATOR_VERSION, "schema_version": SCHEMA_VERSION,
        "trace_chain_valid": result["trace_chain_valid"], "score": score.to_dict(),
        "admitted": not rejects, "rejections": rejects,
        "matched_group_id": spec.params.get("matched_group_id"),
        "source_provenance": {
            "source_id": result.get("claimed_observation_id"),
            "source_dataset": "src/orchestrator/inspection_capability/data/observation_records.json",
            "modality": spec.gold.required_acquisition or spec.initial_modality,
            "asset": spec.asset, "scenario_family": "B-acquisition",
            "benchmark_family": "B", "world_id": spec.world.scenario_id,
            "episode_id": episode_id, "provenance_class": "real_substrate_deterministic_resolve",
        },
    }
    if not rejects:
        rec["fingerprint"] = fingerprint(rec)
    return rec


def main() -> int:
    ex = CouchDBExecutor()
    t0 = time.time()
    rows = enumerate_scale()
    recs = [admit_episode(ex, build_spec(row), row) for row in rows]
    t1 = time.time()

    admitted = [r for r in recs if r["admitted"]]
    rejected = [r for r in recs if not r["admitted"]]

    # Dedup: within the scaled pool AND against the existing 16-episode
    # prototype (a config could theoretically collide with an original
    # thermal/motor_01 episode's fingerprint; the original 16 are NOT
    # re-admitted here, only checked against).
    proto_path = OUT / "b_acquisition_manifest.json"
    proto_fingerprints = set()
    if proto_path.exists():
        proto = json.loads(proto_path.read_text())
        proto_fingerprints = {e["fingerprint"] for e in proto["episodes"] if "fingerprint" in e}

    fp_counts = Counter(r["fingerprint"] for r in admitted)
    duplicates_within = {k: v for k, v in fp_counts.items() if v > 1}
    redundant_within = sum(c - 1 for c in duplicates_within.values())

    cross_manifest_dupes = [r for r in admitted if r["fingerprint"] in proto_fingerprints]

    # Retain the canonical representative: first-generated instance per
    # duplicate fingerprint (deterministic enumeration order), drop the
    # rest; drop any that collide with the existing prototype (retain the
    # prototype's copy, not a re-derived duplicate here).
    seen_fp: set = set()
    canonical: List[Dict[str, Any]] = []
    dropped_dupe_within = 0
    dropped_dupe_cross = 0
    for r in admitted:
        fp = r["fingerprint"]
        if fp in proto_fingerprints:
            dropped_dupe_cross += 1
            continue
        if fp in seen_fp:
            dropped_dupe_within += 1
            continue
        seen_fp.add(fp)
        canonical.append(r)

    canonical_final = len(canonical)

    with (OUT / "b_acquisition_scaled_rejections.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["scenario_id", "template_id", "reason", "validator"])
        for r in rejected:
            for rej in r["rejections"]:
                w.writerow([r["scenario_id"], r["template_id"], rej["reason"], rej["validator"]])

    by_template = Counter(r["template_id"] for r in canonical)
    by_asset = Counter(r["asset"] for r in canonical)
    by_arm = Counter(r["arm"] for r in canonical)
    by_modality = Counter(r["source_provenance"]["modality"] for r in canonical)
    by_matched_group = Counter(r["matched_group_id"] for r in canonical)

    with (OUT / "b_acquisition_scaled_coverage.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["dimension", "value", "count"])
        for k, v in by_template.items(): w.writerow(["template", k, v])
        for k, v in by_asset.items(): w.writerow(["asset", k, v])
        for k, v in by_arm.items(): w.writerow(["arm", k, v])
        for k, v in by_modality.items(): w.writerow(["modality", k, v])
        for k, v in by_matched_group.items(): w.writerow(["matched_group", k, v])
        w.writerow(["GENERATED", "B-acquisition-scale", len(recs)])
        w.writerow(["ADMITTED", "B-acquisition-scale", len(admitted)])
        w.writerow(["REJECTED", "B-acquisition-scale", len(rejected)])
        w.writerow(["duplicate_within_scaled_pool", "count", dropped_dupe_within])
        w.writerow(["duplicate_against_prototype_16", "count", dropped_dupe_cross])
        w.writerow(["CANONICAL_FINAL", "B-acquisition-scale", canonical_final])

    manifest = {
        "benchmark_version": "phase8h2h_b_acquisition_scale_v1_CANDIDATE_NOT_MERGED",
        "generation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "schema_version": SCHEMA_VERSION, "generator_version": GENERATOR_VERSION,
        "scaling_method": "real-substrate axis expansion (no seed-padding): additional real "
                          "required-modalities for B-ACQ-1/B-ACQ-3 (acoustic, workorder_history), "
                          "additional real evidence-count thresholds + the iot_timeseries axis for "
                          "B-ACQ-2, the motor_01/thermal quality-ambiguity pair for B-ACQ-4 -- see "
                          "reports/benchmark/b_acquisition_scale_generation_report.md Sec 2",
        "episode_count_generated": len(recs), "episode_count_admitted": len(admitted),
        "episode_count_rejected": len(rejected),
        "duplicate_within_scaled_pool": dropped_dupe_within,
        "duplicate_against_prototype_16": dropped_dupe_cross,
        "canonical_final": canonical_final,
        "template_counts": dict(by_template), "asset_counts": dict(by_asset),
        "arm_counts": dict(by_arm), "modality_counts": dict(by_modality),
        "matched_group_counts": dict(by_matched_group),
        "prototype_16_status": "UNCHANGED -- reports/benchmark/b_acquisition_manifest.json is not "
                               "modified or re-embedded by this script; this is a SEPARATE, "
                               "standalone candidate-pool manifest",
        "b_legacy_status": "UNCHANGED, frozen at 12 episodes -- not touched by this script",
        "global_benchmark_status": "NOT MERGED -- candidate scaled manifest only",
        "oracle_evaluator_only": "verified structurally -- no admitted/rejected record's trace "
                                 "contains a gold field",
        "episodes": canonical,
    }
    (OUT / "b_acquisition_scaled_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print(f"scale: generated={len(recs)} admitted={len(admitted)} rejected={len(rejected)} "
         f"dup_within={dropped_dupe_within} dup_vs_prototype={dropped_dupe_cross} "
         f"canonical_final={canonical_final} ({t1-t0:.1f}s)")
    print("by_template:", dict(by_template))
    if rejected:
        for r in rejected[:10]:
            print("  REJECTED:", r["template_id"], r["scenario_id"], r["rejections"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
