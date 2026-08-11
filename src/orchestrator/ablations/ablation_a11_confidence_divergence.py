#!/usr/bin/env python3
"""Ablation A11 — Reasoning-Confidence → Commitment Divergence by
Perception Category.

Answers: which perception-failure categories make a model's LOW-confidence
reasoning flip into an OVERCONFIDENT commit? Generalizes the measured
Gemini-Robotics-ER finding (hedge-saturated reasoning trace → confident
numeric value on a frost-degraded gauge) across every backend that has been
run through the PMC benchmark, at read granularity, with zero new API calls
— it consumes the results files preserved in reports/ablations/inputs/ and
the per-episode traces they point at.

Signals per read:
    u_reason  — expressed uncertainty: the provider's token_entropy channel.
                For gemini-er this is the hedging-language score over the
                model's own visible reasoning; for gemini/tokenrouter it is
                the self-reported uncertainty field; for moondream it is the
                fixed parse-quality heuristic. All are documented per-provider.
    committed — the model claimed gauge_readable=True AND produced a numeric
                value. NOTE: commitment is defined by the ACT, not by the
                self-reported confidence — for providers where
                confidence := 1 − uncertainty by construction (gemini-er),
                a self-report-based definition would be structurally zero.

Metric (docs/AblationBlueprint.md §A11):
    hedged read : u_reason ≥ U_HEDGE (0.5)
    OCD[backend, category] = P(committed | hedged read)
    r_pb[backend]          = point-biserial corr(u_reason, committed) over
                             all reads — calibrated behavior is strongly
                             negative; ≈0 or positive marks the drift regime.

Output: reports/ablations/a11_confidence_divergence.json + markdown tables.
Usage:  python src/orchestrator/ablations/ablation_a11_confidence_divergence.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
from apparatus import is_apparatus_failure  # noqa: E402

INPUTS_DIR = REPO_ROOT / "reports" / "ablations" / "inputs"
REPORTS_DIR = REPO_ROOT / "reports" / "ablations"

U_HEDGE = 0.5


def load_reads() -> list[dict]:
    reads: list[dict] = []
    for results_file in sorted(INPUTS_DIR.glob("*_results.json")):
        results = json.loads(results_file.read_text())
        variant = results.get("prompt_variant", "baseline")
        backend = results["backend"] + ("" if variant == "baseline" else f" [{variant}]")
        for row in results["results"]:
            trace_path = Path(row.get("trace_path") or "")
            if not trace_path.exists():
                continue
            trace = json.loads(trace_path.read_text())
            for rec in trace["records"]:
                if rec.get("kind") != "observation":
                    continue
                rr = rec["read_result"]
                # exclude provider-level call/parse failures — those are
                # apparatus errors, not model abstentions (older traces
                # lack perception_category; kept unless explicitly flagged).
                # P0-3: the vocabulary now lives in orchestrator/apparatus.py so
                # this filter and the grader cannot drift apart. Importing it
                # also picks up "no_answer", which this local tuple omitted —
                # empty-content reads were being counted as honest abstentions.
                # (Re-running A11 across the retained traces after the fix moves
                # only moondream's denominator, 150 -> 120, and its commit-rate
                # is 0.00 either way, so no published figure changes.)
                if is_apparatus_failure(rr):
                    continue
                reads.append({
                    "backend": backend,
                    "scenario_id": row["scenario_id"],
                    "category": row["category"],
                    "gt_readable": row["gauge_readable_gt"],
                    "u_reason": float(rr["token_entropy"]),
                    "raw_confidence": float(rr["raw_confidence"]),
                    "committed": bool(rr["gauge_readable"]) and rr["value"] is not None,
                    "value": rr["value"],
                })
    return reads


def point_biserial(us: list[float], committed: list[bool]) -> float | None:
    if len(set(committed)) < 2 or len(us) < 3:
        return None
    mean_u = statistics.fmean(us)
    std_u = statistics.pstdev(us)
    if std_u == 0:
        return None
    group1 = [u for u, c in zip(us, committed) if c]
    group0 = [u for u, c in zip(us, committed) if not c]
    p = len(group1) / len(us)
    return ((statistics.fmean(group1) - statistics.fmean(group0)) / std_u
            * math.sqrt(p * (1 - p)))


def main() -> int:
    reads = load_reads()
    if not reads:
        print("ERROR: no reads found — preserve benchmark results files into "
              f"{INPUTS_DIR} first (see docs/AblationBlueprint.md §A11).",
              file=sys.stderr)
        return 2

    by_bc: dict[tuple, list[dict]] = defaultdict(list)
    by_b: dict[str, list[dict]] = defaultdict(list)
    for r in reads:
        by_bc[(r["backend"], r["category"])].append(r)
        by_b[r["backend"]].append(r)

    ocd_rows = []
    for (backend, category), rs in sorted(by_bc.items()):
        hedged = [r for r in rs if r["u_reason"] >= U_HEDGE]
        overcommit = [r for r in hedged if r["committed"]]
        ocd_rows.append({
            "backend": backend, "category": category,
            "reads": len(rs), "hedged_reads": len(hedged),
            "overconfident_commits": len(overcommit),
            "OCD": round(len(overcommit) / len(hedged), 4) if hedged else None,
        })

    backend_rows = []
    for backend, rs in sorted(by_b.items()):
        us = [r["u_reason"] for r in rs]
        cs = [r["committed"] for r in rs]
        hedged = [r for r in rs if r["u_reason"] >= U_HEDGE]
        overcommit = [r for r in hedged if r["committed"]]
        r_pb = point_biserial(us, cs)
        backend_rows.append({
            "backend": backend, "reads": len(rs),
            "hedged_reads": len(hedged),
            "OCD_overall": round(len(overcommit) / len(hedged), 4) if hedged else None,
            "commit_rate_on_unreadable_gt": round(
                sum(1 for r in rs if not r["gt_readable"] and r["committed"])
                / max(1, sum(1 for r in rs if not r["gt_readable"])), 4),
            "r_pb_u_vs_commit": round(r_pb, 4) if r_pb is not None else None,
        })

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "a11_confidence_divergence.json"
    out.write_text(json.dumps({
        "ablation": "A11_confidence_divergence",
        "u_hedge_threshold": U_HEDGE,
        "commitment_definition": "claimed gauge_readable AND numeric value emitted",
        "per_backend_category": ocd_rows,
        "per_backend": backend_rows,
        "n_reads_total": len(reads),
    }, indent=2))

    print("### OCD by backend × perception category")
    print("| backend | category | reads | hedged | overconfident commits | OCD |")
    print("|---|---|---|---|---|---|")
    for r in ocd_rows:
        ocd = f"{r['OCD']:.2f}" if r["OCD"] is not None else "—(no hedged reads)"
        print(f"| {r['backend']} | {r['category']} | {r['reads']} | "
              f"{r['hedged_reads']} | {r['overconfident_commits']} | {ocd} |")

    print("\n### Per-backend calibration signature")
    print("| backend | reads | hedged | OCD overall | commit rate on GT-unreadable | r_pb(u, commit) |")
    print("|---|---|---|---|---|---|")
    for r in backend_rows:
        ocd = f"{r['OCD_overall']:.2f}" if r["OCD_overall"] is not None else "—"
        rpb = f"{r['r_pb_u_vs_commit']:+.3f}" if r["r_pb_u_vs_commit"] is not None else "—"
        print(f"| {r['backend']} | {r['reads']} | {r['hedged_reads']} | {ocd} | "
              f"{r['commit_rate_on_unreadable_gt']:.2f} | {rpb} |")
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
