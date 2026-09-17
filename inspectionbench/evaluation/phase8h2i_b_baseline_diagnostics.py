#!/usr/bin/env python3
"""phase8h2i_b_baseline_diagnostics.py — Reporting-layer baseline / degeneracy
diagnostics for Family B-Acquisition (Phase 8H.2I).

Does NOT touch b_acquisition_scoring.py or any scorer definition. This is a
read-only reporting utility: it computes simple-policy (trivial) baselines
alongside actual pilot TDA_post/ADA numbers, and surfaces
acceptable_acquisition_set / UHA-applicability degeneracy so these are never
silently misrepresented as broad discriminative capability measures.

No composite metric is invented; every number below is either an existing
scorer output (read from raw pilot logs) or a mechanically-computed
constant-policy baseline over gold alone.

Usage:
    python scripts/phase8h2i_b_baseline_diagnostics.py
    python scripts/phase8h2i_b_baseline_diagnostics.py --pilot-suffix _b_acq_4_repilot
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
B_FINAL_MANIFEST = REPO_ROOT / "reports" / "benchmark" / "b_acquisition_final_manifest.json"
PILOT_DIR = REPO_ROOT / "reports" / "benchmark" / "b_acquisition_pilot"


def load_b_episodes() -> List[Dict[str, Any]]:
    return json.loads(B_FINAL_MANIFEST.read_text())["episodes"]


# ---------------------------------------------------------------------------
# Trivial-policy baselines (mechanical, computed from gold alone -- never
# from a model response). These are NOT new scoring definitions; they mirror
# exactly what score_b_acquisition_episode already computes, applied to a
# hypothetical constant response instead of a real one.
# ---------------------------------------------------------------------------
def trivial_policy_baseline(episodes: List[Dict[str, Any]], *, always_acquire: bool,
                            always_terminal: str) -> Dict[str, float]:
    n = len(episodes)
    if n == 0:
        return {}
    ada = mar = uar = tda = 0
    for e in episodes:
        g = e["gold"]
        acq_required = g["acquisition_required"]
        if always_acquire == acq_required:
            ada += 1
        if acq_required and not always_acquire:
            mar += 1
        if (not acq_required) and always_acquire:
            uar += 1
        if always_terminal == g["final_terminal_action"]:
            tda += 1
    return {"ADA": round(ada / n, 4), "MAR": round(mar / n, 4), "UAR": round(uar / n, 4),
           "TDA_post": round(tda / n, 4), "n": n}


def majority_class_terminal_baseline(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The single most informative trivial baseline for TDA_post: always
    predict whichever terminal action is most common in gold. Distinct from
    trivial_policy_baseline (which fixes acquire+terminal together) --
    this isolates the terminal-action distribution alone."""
    if not episodes:
        return {}
    c = Counter(e["gold"]["final_terminal_action"] for e in episodes)
    majority_action, majority_count = c.most_common(1)[0]
    return {"distribution": dict(c), "majority_action": majority_action,
           "majority_baseline_TDA_post": round(majority_count / len(episodes), 4),
           "n": len(episodes)}


def per_template_baselines(episodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_t: Dict[str, List[Dict[str, Any]]] = {}
    for e in episodes:
        by_t.setdefault(e["template_id"], []).append(e)
    out = {}
    for t, eps in sorted(by_t.items()):
        out[t] = {
            "majority_class": majority_class_terminal_baseline(eps),
            "always_acquire_always_commit": trivial_policy_baseline(
                eps, always_acquire=True, always_terminal="COMMIT"),
            "always_acquire_always_escalate": trivial_policy_baseline(
                eps, always_acquire=True, always_terminal="ESCALATE"),
            "always_stop_always_commit": trivial_policy_baseline(
                eps, always_acquire=False, always_terminal="COMMIT"),
        }
    return out


# ---------------------------------------------------------------------------
# Degeneracy diagnostics (Step 6): acceptable_acquisition_set cardinality,
# UHA applicability, B-ACQ-4 arm counts. Read-only over the manifest --
# never alters V3.
# ---------------------------------------------------------------------------
def acceptable_set_cardinality_distribution(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    sizes = Counter(len(e["gold"]["acceptable_acquisition_set"]) for e in episodes)
    n_multi = sum(v for k, v in sizes.items() if k > 1)
    return {"cardinality_counts": dict(sorted(sizes.items())),
           "n_episodes": len(episodes),
           "n_with_cardinality_gt_1": n_multi,
           "note": ("ASA is currently mostly single-reference (cardinality<=1 on every "
                    "episode as of this report); a multi-reference (|set|>1) acceptable "
                    "set is future V4 work, not implemented here.")}


def uha_applicability(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    applicable = [e for e in episodes if e["gold"]["acquisition_genuinely_unavailable"]]
    return {"n_total": len(episodes), "n_uha_applicable": len(applicable),
           "by_template": dict(Counter(e["template_id"] for e in applicable)),
           "note": ("UHA is only applicable when gold.acquisition_genuinely_unavailable is "
                    "True; this diagnostic exists so UHA is never reported as a rate over "
                    "all B episodes.")}


def b_acq_4_arm_counts(episodes: List[Dict[str, Any]]) -> Dict[str, int]:
    b4 = [e for e in episodes if e["template_id"] == "B-ACQ-4"]
    return dict(Counter(e["arm"] for e in b4))


# ---------------------------------------------------------------------------
# Join actual pilot results (if present) against the trivial baselines --
# read-only over existing raw JSONL logs, never re-scores.
# ---------------------------------------------------------------------------
def load_pilot_summary(model_safe_name: str, suffix: str) -> Optional[Dict[str, Any]]:
    p = PILOT_DIR / f"b_acq_pilot_summary_{model_safe_name}{suffix}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def report(pilot_suffix: str = "") -> Dict[str, Any]:
    episodes = load_b_episodes()
    out: Dict[str, Any] = {
        "n_episodes": len(episodes),
        "majority_class_terminal_baseline_overall": majority_class_terminal_baseline(episodes),
        "per_template_baselines": per_template_baselines(episodes),
        "acceptable_set_cardinality": acceptable_set_cardinality_distribution(episodes),
        "uha_applicability": uha_applicability(episodes),
        "b_acq_4_arm_counts": b_acq_4_arm_counts(episodes),
    }
    if pilot_suffix:
        models = ["tokenrouter_openai_gpt-5.4-mini", "tokenrouter_qwen3.5-omni-plus",
                 "tokenrouter_google_gemini-3.1-flash-image-preview"]
        joined = {}
        for m in models:
            s = load_pilot_summary(m, pilot_suffix)
            if s:
                joined[m] = s
        out["pilot_results_joined"] = joined
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-suffix", default="",
                    help="join against b_acq_pilot_summary_<model><suffix>.json if present")
    args = ap.parse_args()
    print(json.dumps(report(args.pilot_suffix), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
