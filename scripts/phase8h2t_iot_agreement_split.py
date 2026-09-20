#!/usr/bin/env python3
"""phase8h2t_iot_agreement_split.py -- FULL-regime TDA/GSR split by the
world's iot_agree vs iot_disagree tag. Unlike the FULL-vs-DIGITAL_ONLY
comparison, this stays within FULL (physical evidence always delivered,
so GSR has no structural floor here) and asks whether the presence of
CONTRADICTING telemetry corrupts grounding even when the required
evidence is available. NOT a matched within-world pair (agree/disagree
are different worlds, not the same world under two access conditions) --
reported as a between-world comparison, weaker than Section 3.2's
within-world invariant, and labeled as such. Read-only.
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FROZEN93 = REPO_ROOT / "reports" / "benchmark" / "v3_full_results" / "frozen93"

MODELS = [
    ("Claude Sonnet 4.6", "Claude_Sonnet_4.6"),
    ("GPT-5.2", "GPT-5.2"),
    ("DeepSeek V4 Pro", "DeepSeek_V4_Pro"),
    ("Mistral Medium 3.5", "Mistral_Medium_3.5"),
    ("Qwen3.5-397B-A17B", "Qwen3.5-397B-A17B"),
]

out = {}
for display, suffix in MODELS:
    rows = [json.loads(l) for l in (FROZEN93 / f"raw_{suffix}.jsonl").read_text().splitlines()]
    full = [r for r in rows if r["dim"] == "A" and r["arm"] == "FULL"]
    result = {}
    for tag in ("iot_agree", "iot_disagree"):
        subset = [r for r in full if tag in r["world_id"]]
        evaluable = [r for r in subset if r["runner_return"].get("verdict", "") != ""]
        n = len(subset)
        n_eval = len(evaluable)
        tda = sum(r["runner_return"]["metric"]["TDA"] for r in evaluable)
        gsr = sum(r["runner_return"]["metric"]["GSR"] for r in evaluable)
        result[tag] = {
            "n": n, "n_evaluable": n_eval,
            "TDA": round(tda / n_eval, 4) if n_eval else None,
            "GSR": round(gsr / n_eval, 4) if n_eval else None,
        }
    out[display] = result

out_path = REPO_ROOT / "reports" / "benchmark" / "matched_regime_delta" / "iot_agreement_split_v1.json"
out_path.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
print(f"\nWrote {out_path}")
