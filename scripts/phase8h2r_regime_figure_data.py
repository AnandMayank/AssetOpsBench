#!/usr/bin/env python3
"""Computes per-model GSR (and TDA) under FULL/PHYSICAL_ONLY/DIGITAL_ONLY,
restricted to worlds where ALL THREE regimes have a valid verdict (a
stricter, fully-matched triplet -- not just pairwise), for the figure.
Read-only.
"""
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FROZEN93 = REPO_ROOT / "reports" / "benchmark" / "v3_full_results" / "frozen93"

MODELS = [
    ("GPT-5.2", "GPT-5.2"),
    ("DeepSeek V4 Pro", "DeepSeek_V4_Pro"),
    ("Mistral Medium 3.5", "Mistral_Medium_3.5"),
    ("Qwen3.5-397B-A17B", "Qwen3.5-397B-A17B"),
]

out = {}
for display, suffix in MODELS:
    rows = [json.loads(l) for l in (FROZEN93 / f"raw_{suffix}.jsonl").read_text().splitlines()]
    a_rows = [r for r in rows if r["dim"] == "A"]
    by_world = defaultdict(dict)
    for r in a_rows:
        by_world[r["world_id"]][r["arm"]] = r
    triplets = []
    for world_id, arms in by_world.items():
        if all(a in arms for a in ("FULL", "PHYSICAL_ONLY", "DIGITAL_ONLY")):
            if all(arms[a]["runner_return"].get("verdict", "") != "" for a in ("FULL", "PHYSICAL_ONLY", "DIGITAL_ONLY")):
                triplets.append(world_id)
    n = len(triplets)
    result = {"n_full_triplets": n, "n_worlds_total": len(by_world)}
    for arm in ("FULL", "PHYSICAL_ONLY", "DIGITAL_ONLY"):
        gsr = [by_world[w][arm]["runner_return"]["metric"]["GSR"] for w in triplets]
        tda = [by_world[w][arm]["runner_return"]["metric"]["TDA"] for w in triplets]
        result[f"{arm}_GSR"] = round(sum(gsr) / n, 4) if n else None
        result[f"{arm}_TDA"] = round(sum(tda) / n, 4) if n else None
    out[display] = result

out_path = REPO_ROOT / "reports" / "benchmark" / "matched_regime_delta" / "regime_triplet_figure_data.json"
out_path.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
print(f"\nWrote {out_path}")
