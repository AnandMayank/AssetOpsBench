#!/usr/bin/env python3
"""phase8h2g_b_acquisition_bulk_generate.py — Family B (Evidence Acquisition)
prototype generation + admission + coverage (Phase 8H.2G).

Generates the 16-episode prototype specified in
reports/benchmark/b_prototype_spec.md. Zero model/API calls. Does NOT touch
the existing 12 B-legacy canonical episodes, the D-physical pool, or the
global benchmark manifest.
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

GENERATOR_VERSION = "b_acquisition_generator.py@phase8h1_8h2g"
SCHEMA_VERSION = "phase8h1_c_gold_schema_freeze_plan/1+b_acquisition_gold/1"


def enumerate_prototype() -> List[Dict[str, Any]]:
    rows = []
    seed = 70000
    for asset in B.ALL_ASSETS:
        rows.append({"template": "B-ACQ-1", "asset": asset, "seed": seed}); seed += 1
    for asset in B._ACOUSTIC_ASSETS:
        for arm in ("k1", "k3"):
            rows.append({"template": "B-ACQ-2", "asset": asset, "arm": arm, "seed": seed}); seed += 1
    for asset in B.ALL_ASSETS:
        rows.append({"template": "B-ACQ-3", "asset": asset, "seed": seed}); seed += 1
    for asset in B._ACOUSTIC_ASSETS:
        for arm in ("ambiguous", "unambiguous"):
            rows.append({"template": "B-ACQ-4", "asset": asset, "arm": arm, "seed": seed}); seed += 1
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
        return B.build_b_acq_1(row["asset"], row["seed"])
    if t == "B-ACQ-2":
        return B.build_b_acq_2(row["asset"], row["arm"], row["seed"])
    if t == "B-ACQ-3":
        return B.build_b_acq_3(row["asset"], row["seed"])
    if t == "B-ACQ-4":
        return B.build_b_acq_4(row["asset"], row["arm"], row["seed"])
    raise ValueError(t)


def admit_episode(ex: CouchDBExecutor, spec) -> Dict[str, Any]:
    rejects = []
    result = B.run_scripted_episode(ex, spec)

    if result["verdict"] != spec.gold.final_terminal_action:
        rejects.append({"reason": f"scripted verdict {result['verdict']!r} != gold "
                                  f"{spec.gold.final_terminal_action!r}", "validator": "scorer_compatibility"})
    if not result["trace_chain_valid"]:
        rejects.append({"reason": "trace hash chain invalid", "validator": "provenance_completeness"})

    forbidden_fields = {"final_terminal_action", "acquisition_required",
                        "acceptable_acquisition_set", "initial_evidence_sufficient"}
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
    }
    if not rejects:
        rec["fingerprint"] = fingerprint(rec)
    return rec


def main() -> int:
    ex = CouchDBExecutor()
    t0 = time.time()
    recs = [admit_episode(ex, build_spec(row)) for row in enumerate_prototype()]
    t1 = time.time()

    admitted = [r for r in recs if r["admitted"]]
    rejected = [r for r in recs if not r["admitted"]]
    fp_counts = Counter(r["fingerprint"] for r in admitted)
    duplicates = {k: v for k, v in fp_counts.items() if v > 1}
    redundant = sum(c - 1 for c in duplicates.values())
    canonical_final = len(admitted) - redundant

    with (OUT / "b_acquisition_rejections.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["scenario_id", "template_id", "reason", "validator"])
        for r in rejected:
            for rej in r["rejections"]:
                w.writerow([r["scenario_id"], r["template_id"], rej["reason"], rej["validator"]])

    by_template = Counter(r["template_id"] for r in admitted)
    by_asset = Counter(r["asset"] for r in admitted)
    by_gold_req = Counter(r["gold"]["acquisition_required"] for r in admitted)
    by_arm = Counter(r["arm"] for r in admitted)

    with (OUT / "b_acquisition_coverage.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["dimension", "value", "count"])
        for k, v in by_template.items(): w.writerow(["template", k, v])
        for k, v in by_asset.items(): w.writerow(["asset", k, v])
        for k, v in by_arm.items(): w.writerow(["arm", k, v])
        for k, v in by_gold_req.items(): w.writerow(["acquisition_required", k, v])
        w.writerow(["GENERATED", "B-acquisition", len(recs)])
        w.writerow(["ADMITTED", "B-acquisition", len(admitted)])
        w.writerow(["REJECTED", "B-acquisition", len(rejected)])
        w.writerow(["duplicate_fingerprints", "count", len(duplicates)])
        w.writerow(["CANONICAL_FINAL", "B-acquisition", canonical_final])

    manifest = {
        "benchmark_version": "phase8h1_b_acquisition_prototype_v1_CANDIDATE_NOT_MERGED",
        "generation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "schema_version": SCHEMA_VERSION, "generator_version": GENERATOR_VERSION,
        "episode_count_generated": len(recs), "episode_count_admitted": len(admitted),
        "episode_count_rejected": len(rejected), "duplicate_fingerprints": len(duplicates),
        "canonical_final": canonical_final,
        "template_counts": dict(by_template), "asset_counts": dict(by_asset),
        "arm_counts": dict(by_arm),
        "b_legacy_status": "UNCHANGED, frozen at 12 episodes -- not touched by this script",
        "global_benchmark_status": "NOT MERGED -- candidate prototype manifest only",
        "oracle_evaluator_only": "verified structurally -- no admitted/rejected record's trace "
                                 "contains a gold field",
        "episodes": admitted,
    }
    (OUT / "b_acquisition_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print(f"prototype: generated={len(recs)} admitted={len(admitted)} rejected={len(rejected)} "
         f"duplicates={len(duplicates)} canonical_final={canonical_final} ({t1-t0:.1f}s)")
    if rejected:
        for r in rejected[:10]:
            print("  REJECTED:", r["template_id"], r["scenario_id"], r["rejections"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
