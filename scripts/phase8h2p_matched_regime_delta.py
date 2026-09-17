#!/usr/bin/env python3
"""phase8h2p_matched_regime_delta.py -- matched within-world FULL vs
PHYSICAL_ONLY vs DIGITAL_ONLY paired delta for the A family.

Each of the 16 A worlds is realized under all 3 evidence-access regimes
(same world_id, same gold, same scenario -- see canonical_identity's
normalize_world / world_id invariant, Section 3.2 of the paper). This
computes, per model, the PAIRED delta in TDA and GSR between FULL and
each withheld-modality regime on the SAME world -- a cleaner causal
comparison than a physically-changed-fault-state counterfactual, since
only evidence access changes, never the world.

A pair is included only when BOTH sides have a valid (non-empty) verdict
(matches the reviewed evaluable-N convention for A) -- excluded pairs are
counted and reported, not silently dropped from view.

Read-only; writes only to reports/benchmark/matched_regime_delta/.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FROZEN93 = REPO_ROOT / "reports" / "benchmark" / "v3_full_results" / "frozen93"
OUT_DIR = REPO_ROOT / "reports" / "benchmark" / "matched_regime_delta"

MODELS = [
    ("Claude Sonnet 4.6", "Claude_Sonnet_4.6"),
    ("GPT-5.2", "GPT-5.2"),
    ("DeepSeek V4 Pro", "DeepSeek_V4_Pro"),
    ("Mistral Medium 3.5", "Mistral_Medium_3.5"),
    ("Qwen3.5-397B-A17B", "Qwen3.5-397B-A17B"),
]


def load_by_world(path: Path):
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    a_rows = [r for r in rows if r["dim"] == "A"]
    by_world = defaultdict(dict)
    for r in a_rows:
        by_world[r["world_id"]][r["arm"]] = r
    return by_world


def paired_delta(by_world, withheld_arm: str):
    """Returns (n_pairs_included, n_pairs_excluded, mean_TDA_delta,
    mean_GSR_delta, n_full_wins_tda, n_withheld_wins_tda,
    n_full_wins_gsr, n_withheld_wins_gsr) for FULL vs withheld_arm."""
    included = excluded = 0
    tda_deltas, gsr_deltas = [], []
    full_wins_tda = withheld_wins_tda = full_wins_gsr = withheld_wins_gsr = 0
    for world_id, arms in by_world.items():
        full = arms.get("FULL")
        other = arms.get(withheld_arm)
        if full is None or other is None:
            excluded += 1
            continue
        full_verdict = full["runner_return"].get("verdict", "")
        other_verdict = other["runner_return"].get("verdict", "")
        if full_verdict == "" or other_verdict == "":
            excluded += 1
            continue
        included += 1
        full_tda = full["runner_return"]["metric"]["TDA"]
        other_tda = other["runner_return"]["metric"]["TDA"]
        full_gsr = full["runner_return"]["metric"]["GSR"]
        other_gsr = other["runner_return"]["metric"]["GSR"]
        tda_deltas.append(full_tda - other_tda)
        gsr_deltas.append(full_gsr - other_gsr)
        if full_tda > other_tda:
            full_wins_tda += 1
        elif other_tda > full_tda:
            withheld_wins_tda += 1
        if full_gsr > other_gsr:
            full_wins_gsr += 1
        elif other_gsr > full_gsr:
            withheld_wins_gsr += 1
    mean_tda = sum(tda_deltas) / len(tda_deltas) if tda_deltas else None
    mean_gsr = sum(gsr_deltas) / len(gsr_deltas) if gsr_deltas else None
    return {
        "n_pairs_included": included, "n_pairs_excluded": excluded,
        "mean_TDA_delta_FULL_minus_withheld": round(mean_tda, 4) if mean_tda is not None else None,
        "mean_GSR_delta_FULL_minus_withheld": round(mean_gsr, 4) if mean_gsr is not None else None,
        "FULL_wins_TDA": full_wins_tda, "withheld_wins_TDA": withheld_wins_tda,
        "FULL_wins_GSR": full_wins_gsr, "withheld_wins_GSR": withheld_wins_gsr,
    }


def main():
    results = {}
    for display, suffix in MODELS:
        by_world = load_by_world(FROZEN93 / f"raw_{suffix}.jsonl")
        results[display] = {
            "FULL_vs_PHYSICAL_ONLY": paired_delta(by_world, "PHYSICAL_ONLY"),
            "FULL_vs_DIGITAL_ONLY": paired_delta(by_world, "DIGITAL_ONLY"),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "matched_regime_delta_v1.json").write_text(json.dumps(results, indent=2))

    lines = ["# Matched within-world FULL vs withheld-modality paired delta (A family)",
             "",
             "16 worlds x 3 regimes (FULL/PHYSICAL_ONLY/DIGITAL_ONLY), same world_id, same gold.",
             "Pairs excluded when either side has an empty verdict (matches the evaluable-N",
             "convention used for the main table's A column).",
             "",
             "| Model | vs PHYSICAL_ONLY: n incl/excl | mean TDA delta | mean GSR delta | | vs DIGITAL_ONLY: n incl/excl | mean TDA delta | mean GSR delta |",
             "|---|---|---|---|---|---|---|---|"]
    for display, _ in MODELS:
        p = results[display]["FULL_vs_PHYSICAL_ONLY"]
        d = results[display]["FULL_vs_DIGITAL_ONLY"]
        def fmt(x):
            return f"{x['n_pairs_included']}/{x['n_pairs_excluded']}", (f"{x['mean_TDA_delta_FULL_minus_withheld']:+.3f}" if x['mean_TDA_delta_FULL_minus_withheld'] is not None else "n/a"), (f"{x['mean_GSR_delta_FULL_minus_withheld']:+.3f}" if x['mean_GSR_delta_FULL_minus_withheld'] is not None else "n/a")
        pn, pt, pg = fmt(p)
        dn, dt, dg = fmt(d)
        lines.append(f"| {display} | {pn} | {pt} | {pg} | | {dn} | {dt} | {dg} |")

    lines.append("")
    lines.append("Positive delta = FULL scores higher than the withheld-modality regime on the same world.")
    md = "\n".join(lines)
    (OUT_DIR / "matched_regime_delta_v1.md").write_text(md + "\n")
    print(md)
    print(f"\nWrote {OUT_DIR}/matched_regime_delta_v1.json and .md")


if __name__ == "__main__":
    main()
