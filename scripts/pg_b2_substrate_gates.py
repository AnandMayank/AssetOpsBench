"""pg_b2_substrate_gates.py — Phase 1a, G1-G4: read-only substrate checks
that MUST pass before the B2 pool is built or any model is run (plan
section M). Writes reports/playground/b2/substrate_gates_report.json.

G1: does the deterministic acoustic indicator (semantics.acoustic_indicator,
    computed ONLY from real compute_acoustic_features() output) carry
    information about the hidden MIMII normal/abnormal label? Sweeps
    candidate (band, threshold) pairs and reports AUC (Mann-Whitney U /
    n_pos*n_neg, no external ML dependency) per asset/machine_id.

G2: do the acoustic asset pools (chiller_6 <- MIMII 'fan', hydraulic_pump_1
    <- MIMII 'pump') share any underlying clips? Sets the true independent-
    world count.

G3: the sufficiency rule R is isolated in semantics.py, versioned
    (R_VERSION), and every clause is independently testable -- this gate
    reports the rule for SME sign-off rather than validating it itself
    (SME endorsement is an external dependency, never claimed here).

G4: scans representative + exhaustive `render_reading` payloads for any
    FORBIDDEN_FIELDS leak or MIMII path-label leak, fails closed.

Zero model/API calls. Does not modify any frozen artifact.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

from playground import substrate  # noqa: E402
from playground.semantics import acoustic_indicator  # noqa: E402

OUT_PATH = REPO_ROOT / "reports" / "playground" / "b2" / "substrate_gates_report.json"

BANDS = ("0-500Hz", "500-2000Hz", "2000-8000Hz", f"8000-{None}")  # last resolved per-record
CANDIDATE_BANDS = ("0-500Hz", "500-2000Hz", "2000-8000Hz")
CANDIDATE_STATS = ("band_rel_rms", "band_abs_db", "crest_factor", "dominant_freq_mag_db")


def _auc(pos_scores, neg_scores) -> float:
    """Mann-Whitney U statistic / (n_pos*n_neg) == AUC, no sklearn needed.
    Ties count as 0.5. O(n log n) via rank sums on the merged, sorted list."""
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    combined = sorted([(s, 1) for s in pos_scores] + [(s, 0) for s in neg_scores])
    ranks = {}
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    rank_sum_pos = sum(ranks[idx] for idx, (s, label) in enumerate(combined) if label == 1)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def _score_for(features, band, stat) -> float:
    if stat == "band_rel_rms":
        return features["band_energy_db"].get(band, -120.0) - features["rms_level_db"]
    if stat == "band_abs_db":
        return features["band_energy_db"].get(band, -120.0)
    if stat == "crest_factor":
        return features["crest_factor"]
    if stat == "dominant_freq_mag_db":
        return features["dominant_freq_mag_db"]
    raise ValueError(stat)


def gate_g1(*, sample_per_class: int = 60, seed: int = 42) -> dict:
    rng = random.Random(seed)
    store = substrate.get_store()
    results = {}
    best_overall = None
    for asset in sorted(substrate.AGREE_ACOUSTIC_ASSETS):
        for machine_id in substrate.ACOUSTIC_MACHINE_IDS:
            all_recs = store.query(asset_id=asset, modality="acoustic")
            pos = [r for r in all_recs if r.sensor_metadata.get("machine_id") == machine_id
                   and r.sensor_metadata.get("source_label") == "abnormal"]
            neg = [r for r in all_recs if r.sensor_metadata.get("machine_id") == machine_id
                   and r.sensor_metadata.get("source_label") == "normal"]
            if not pos or not neg:
                results[f"{asset}/{machine_id}"] = {"status": "NO_DATA", "n_pos": len(pos), "n_neg": len(neg)}
                continue
            pos_s = rng.sample(pos, min(sample_per_class, len(pos)))
            neg_s = rng.sample(neg, min(sample_per_class, len(neg)))
            per_candidate = {}
            for band in CANDIDATE_BANDS:
                for stat in CANDIDATE_STATS:
                    key = f"{band}|{stat}" if stat != "crest_factor" and stat != "dominant_freq_mag_db" \
                        else stat
                    if key in per_candidate:
                        continue
                    try:
                        pos_scores = [_score_for(substrate._acoustic_features(r), band, stat) for r in pos_s]
                        neg_scores = [_score_for(substrate._acoustic_features(r), band, stat) for r in neg_s]
                    except Exception as exc:  # AudioResolutionError or similar -- report, don't crash the gate
                        per_candidate[key] = {"status": "DECODE_ERROR", "error": str(exc)}
                        continue
                    auc = _auc(pos_scores, neg_scores)
                    directed_positive_is_abnormal = auc >= 0.5
                    auc_directed = max(auc, 1.0 - auc)
                    pos_mean = sum(pos_scores) / len(pos_scores)
                    neg_mean = sum(neg_scores) / len(neg_scores)
                    threshold = (pos_mean + neg_mean) / 2.0
                    per_candidate[key] = {
                        "auc": auc, "auc_best_direction": auc_directed,
                        "abnormal_scores_higher": directed_positive_is_abnormal,
                        "pos_mean": pos_mean, "neg_mean": neg_mean, "calibrated_threshold": threshold,
                    }
            best_key = max((k for k, v in per_candidate.items() if "auc" in v),
                           key=lambda k: per_candidate[k]["auc_best_direction"], default=None)
            best = per_candidate[best_key] if best_key else None
            results[f"{asset}/{machine_id}"] = {
                "status": "OK", "n_pos_sampled": len(pos_s), "n_neg_sampled": len(neg_s),
                "n_pos_total": len(pos), "n_neg_total": len(neg),
                "candidates": per_candidate, "best_candidate": best_key,
                "best_auc_best_direction": best["auc_best_direction"] if best else None,
                "calibration": ({"band_stat_key": best_key, "threshold": best["calibrated_threshold"],
                                  "abnormal_scores_higher": best["abnormal_scores_higher"]}
                                 if best else None),
            }
            if best_key and (best_overall is None or
                              per_candidate[best_key]["auc_best_direction"] > best_overall[1]):
                best_overall = (f"{asset}/{machine_id}:{best_key}", per_candidate[best_key]["auc_best_direction"])

    aucs = [v["best_auc_best_direction"] for v in results.values() if v.get("status") == "OK"]
    verdict = "PASS" if aucs and min(aucs) >= 0.65 else ("MARGINAL" if aucs and max(aucs) >= 0.65 else "FAIL")
    return {
        "gate": "G1_acoustic_indicator_informativeness", "verdict": verdict,
        "per_asset_machine": results, "best_overall": best_overall,
        "note": ("AUC computed against the HIDDEN MIMII label, using only real "
                 "compute_acoustic_features() output -- never used to build the shipped "
                 "indicator's threshold from labels beyond this validation check itself. "
                 "verdict=PASS requires EVERY asset/machine_id combination to clear 0.65 "
                 "AUC (best direction); MARGINAL means only some do; FAIL means none do."),
    }


def gate_g2() -> dict:
    store = substrate.get_store()
    pools = {}
    for asset in sorted(substrate.AGREE_ACOUSTIC_ASSETS):
        recs = store.query(asset_id=asset, modality="acoustic")
        pools[asset] = {r.sensor_metadata.get("sha256") for r in recs if r.sensor_metadata.get("sha256")}
    overlap = pools.get("chiller_6", set()) & pools.get("hydraulic_pump_1", set())
    machine_types = {
        asset: {r.sensor_metadata.get("machine_type") for r in store.query(asset_id=asset, modality="acoustic")}
        for asset in sorted(substrate.AGREE_ACOUSTIC_ASSETS)
    }
    independent_worlds = len(substrate.AGREE_ACOUSTIC_ASSETS) * len(substrate.ACOUSTIC_MACHINE_IDS)
    return {
        "gate": "G2_acoustic_pool_independence", "verdict": "PASS" if not overlap else "FAIL",
        "sha256_overlap_count": len(overlap), "machine_types_per_asset": {k: sorted(v) for k, v in machine_types.items()},
        "independent_acoustic_worlds_per_condition": independent_worlds,
        "note": ("chiller_6 draws from MIMII 'fan' clips, hydraulic_pump_1 from MIMII "
                 "'pump' clips -- disjoint machine types by construction; this check "
                 "confirms no literal clip (by content sha256) is shared. "
                 f"{independent_worlds} (asset, machine_id) acoustic pools exist per "
                 "condition (normal/fault), i.e. up to 2x that many worlds total -- "
                 "this is the number used for world-clustered statistics (plan I)."),
    }


def gate_g3() -> dict:
    from playground import semantics
    return {
        "gate": "G3_sufficiency_rule_explicit", "verdict": "REPORTED_PENDING_SME_SIGNOFF",
        "rule_version": semantics.R_VERSION,
        "fault_clause_names": sorted(semantics.clause_relevant_modalities().keys() & {
            "acoustic_2plus_positive", "acoustic1_iot_persist2", "thermal_hotspot"}),
        "normal_clause_names": ["acoustic_3plus_negative_iot_inband", "iot_inband_streak3"],
        "open_question_for_sme": (
            "Is IoT-only closure (3 consecutive in-band reads, with NO physical/acoustic/"
            "thermal evidence at all) an acceptable basis for COMMIT_NORMAL, given the same "
            "closure path is NOT offered for COMMIT_FAULT (fault always requires acoustic or "
            "thermal)? This asymmetry is intentional in the draft rule (telemetry alone can "
            "rule an asset IN as healthy but not confirm a fault) but needs domain sign-off."
        ),
        "note": "Code-level testability (one test per clause) is verified by "
                "playground/tests/test_semantics_r_clauses.py, run separately. SME "
                "endorsement of the rule's real-world validity is NOT established by this gate.",
    }


def gate_g4() -> dict:
    store = substrate.get_store()
    violations = []
    checked = 0
    for asset in sorted(substrate.AGREE_ACOUSTIC_ASSETS):
        for machine_id in substrate.ACOUSTIC_MACHINE_IDS:
            for label in ("abnormal", "normal"):
                ids = substrate.acoustic_pool_ids(asset, machine_id, label, limit=2)
                recs = store.query(asset_id=asset, modality="acoustic")
                for rid in ids:
                    rec = next(r for r in recs if r.observation_id == rid)
                    rendered = substrate.render_reading(rec, minted_id=f"PGOBS-{checked:06d}")
                    hits = substrate.scan_for_leakage(rendered)
                    checked += 1
                    if hits:
                        violations.append({"asset": asset, "machine_id": machine_id, "label": label,
                                            "record_id": rid, "hits": hits})
    for asset in ("motor_01",):
        for fc in (None, "Rotor-0", "A&C&B10", "Fan"):
            ids = substrate.thermal_pool_ids(asset, fc, limit=2)
            recs = store.query(asset_id=asset, modality="thermal")
            for rid in ids:
                rec = next(r for r in recs if r.observation_id == rid)
                rendered = substrate.render_reading(rec, minted_id=f"PGOBS-{checked:06d}")
                hits = substrate.scan_for_leakage(rendered)
                checked += 1
                if hits:
                    violations.append({"asset": asset, "fault_class": fc, "record_id": rid, "hits": hits})
    for asset in sorted({"chiller_6", "hydraulic_pump_1", "motor_01", "metro_pump_1"}):
        for band_state in ("in_band", "out_of_band"):
            ids = substrate.iot_pool_records(asset, band_state, limit=2)
            recs = store.query(asset_id=asset, modality="iot_timeseries")
            for rid in ids:
                rec = next(r for r in recs if r.observation_id == rid)
                rendered = substrate.render_reading(rec, minted_id=f"PGOBS-{checked:06d}")
                hits = substrate.scan_for_leakage(rendered)
                checked += 1
                if hits:
                    violations.append({"asset": asset, "band_state": band_state, "record_id": rid, "hits": hits})
    return {
        "gate": "G4_leakage_scan", "verdict": "PASS" if not violations else "FAIL",
        "n_payloads_checked": checked, "violations": violations,
    }


def main() -> int:
    report = {
        "schema": "pg_b2.substrate_gates/1",
        "g1": gate_g1(),
        "g2": gate_g2(),
        "g3": gate_g3(),
        "g4": gate_g4(),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, default=str))
    for gate in ("g1", "g2", "g3", "g4"):
        g = report[gate]
        print(f"[{g['verdict']:>24}] {g['gate']}")
    print(f"\nFull report: {OUT_PATH}")
    hard_fail = report["g1"]["verdict"] == "FAIL" or report["g2"]["verdict"] == "FAIL" \
        or report["g4"]["verdict"] == "FAIL"
    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
