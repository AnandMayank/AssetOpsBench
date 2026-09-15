#!/usr/bin/env python3
"""phase8h2j_v3_full_aggregate.py — aggregates the full V3 primary-panel run
raw results into per-family, per-model tables. Reads only; writes only to
reports/benchmark/v3_full_results/. Never touches benchmark/manifest files.
"""
from __future__ import annotations
import glob
import json
from pathlib import Path
from collections import Counter, defaultdict

REPO = Path(__file__).resolve().parent.parent
FROZEN93_DIR = REPO / "reports/benchmark/v3_full_results/frozen93"
B_DIR = REPO / "reports/benchmark/b_acquisition_pilot"
D_DIR = REPO / "reports/benchmark/dphys_pilot"
OUT = REPO / "reports/benchmark/v3_full_results"

MODELS = [
    ("Claude Sonnet 4.6", "anthropic/claude-sonnet-4.6", "tokenrouter/anthropic/claude-sonnet-4.6"),
    ("GPT-5.2", "openai/gpt-5.2", "tokenrouter/openai/gpt-5.2"),
    ("DeepSeek V4 Pro", "deepseek/deepseek-v4-pro", "tokenrouter/deepseek/deepseek-v4-pro"),
    ("Mistral Medium 3.5", "mistralai/mistral-medium-3-5", "tokenrouter/mistralai/mistral-medium-3-5"),
    ("Qwen3.5-397B-A17B", "qwen/qwen3.5-397b-a17b", "tokenrouter/qwen/qwen3.5-397b-a17b"),
]


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def load_frozen93_rows(label):
    p = FROZEN93_DIR / f"raw_{label.replace(' ', '_').replace('.', '.')}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def aggregate_frozen93(rows):
    by_dim = defaultdict(list)
    for r in rows:
        by_dim[r["dim"]].append(r)
    out = {}

    # A: TDA, GSR
    a = by_dim.get("A", [])
    tda = [rr["runner_return"].get("metric", {}).get("TDA") for rr in a]
    gsr = [rr["runner_return"].get("metric", {}).get("GSR") for rr in a]
    out["A"] = {"n": len(a), "TDA_mean": _mean(tda), "GSR_mean": _mean(gsr)}

    # B-legacy (12): CC, PROC, CC_grounded from the `scores` sub-dict
    b = by_dim.get("B", [])
    cc = [(rr["runner_return"].get("scores") or {}).get("CC") for rr in b]
    proc = [(rr["runner_return"].get("scores") or {}).get("PROC") for rr in b]
    ccg = [(rr["runner_return"].get("scores") or {}).get("CC_grounded") for rr in b]
    out["B_legacy"] = {"n": len(b), "CC_mean": _mean(cc), "PROC_mean": _mean(proc),
                       "CC_grounded_mean": _mean(ccg)}

    # C: CC, ordering_satisfied
    c = by_dim.get("C", [])
    cc_c = [rr["runner_return"].get("CC") for rr in c]
    ordsat = [1 if rr["runner_return"].get("ordering_satisfied") else 0 for rr in c]
    out["C"] = {"n": len(c), "CC_mean": _mean(cc_c), "ordering_satisfied_rate": _mean(ordsat)}

    # D-enterprise (6): CC
    d = by_dim.get("D", [])
    cc_d = [rr["runner_return"].get("CC") for rr in d]
    out["D_enterprise"] = {"n": len(d), "CC_mean": _mean(cc_d)}

    # E: CC, CC_grounded, PROC, stale_state_reuse, unnecessary_reobservation
    e = by_dim.get("E", [])
    cc_e = [rr["runner_return"].get("CC") for rr in e]
    ccg_e = [rr["runner_return"].get("CC_grounded") for rr in e]
    proc_e = [rr["runner_return"].get("PROC") for rr in e]
    stale = [1 if rr["runner_return"].get("stale_state_reuse") else 0 for rr in e]
    unnec = [1 if rr["runner_return"].get("unnecessary_reobservation") else 0 for rr in e]
    out["E"] = {"n": len(e), "CC_mean": _mean(cc_e), "CC_grounded_mean": _mean(ccg_e),
               "PROC_mean": _mean(proc_e), "stale_state_reuse_rate": _mean(stale),
               "unnecessary_reobservation_rate": _mean(unnec)}

    return out


def load_b_acq_summary(label_tr):
    safe = label_tr.replace("/", "_")
    p = B_DIR / f"b_acq_pilot_summary_{safe}_v3_full.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_d_phys_summary(label_tr):
    safe = label_tr.replace("/", "_")
    p = D_DIR / f"dphys_pilot_summary_{safe}_v3_full.json"
    return json.loads(p.read_text()) if p.exists() else None


def main():
    full = {}
    for display, bare, tr in MODELS:
        rows = load_frozen93_rows(display)
        f93 = aggregate_frozen93(rows) if rows else None
        bacq = load_b_acq_summary(tr)
        dphys = load_d_phys_summary(tr)
        full[display] = {
            "requested_bare": bare, "requested_tokenrouter": tr,
            "frozen93_n": len(rows), "frozen93": f93,
            "b_acquisition": bacq, "d_physical": dphys,
        }
    (OUT / "v3_full_aggregate.json").write_text(json.dumps(full, indent=2, default=str))
    print(json.dumps(full, indent=2, default=str)[:2000])
    print("wrote", OUT / "v3_full_aggregate.json")


if __name__ == "__main__":
    main()
