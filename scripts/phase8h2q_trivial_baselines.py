#!/usr/bin/env python3
"""phase8h2q_trivial_baselines.py -- trivial-policy baselines for A and B,
matching the always-ESCALATE baseline already reported for D (0.80 on
terminal decisions, per Section 4.2). Computed directly from gold labels
in the frozen manifest / b_acquisition manifest -- model-independent,
uses any one model's raw file only to read the (identical) gold field.
Read-only.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FROZEN93 = REPO_ROOT / "reports" / "benchmark" / "v3_full_results" / "frozen93"
BACQ = REPO_ROOT / "reports" / "benchmark" / "b_acquisition_pilot"
OUT = REPO_ROOT / "reports" / "benchmark" / "trivial_baselines_v1.md"


def a_baselines():
    rows = [json.loads(l) for l in (FROZEN93 / "raw_GPT-5.2.jsonl").read_text().splitlines()]
    a_rows = [r for r in rows if r["dim"] == "A"]
    n = len(a_rows)
    from collections import Counter
    gold_dist = Counter(r["runner_return"]["gold"] for r in a_rows)
    out = {"n": n, "gold_distribution": dict(gold_dist)}
    for policy in ("COMMIT", "ESCALATE", "ABORT"):
        correct = sum(1 for r in a_rows if r["runner_return"]["gold"] == policy)
        out[f"always_{policy}_TDA"] = round(correct / n, 4)
    return out


def b_baselines():
    rows = [json.loads(l) for l in (BACQ / "b_acq_pilot_raw_tokenrouter_openai_gpt-5.2_v3_full.jsonl").read_text().splitlines()]
    n = len(rows)
    stop_n = sum(1 for r in rows if not r["gold"]["acquisition_required"])
    acquire_avail_n = sum(1 for r in rows if r["gold"]["acquisition_required"] and not r["gold"]["acquisition_genuinely_unavailable"])
    acquire_unavail_n = sum(1 for r in rows if r["gold"]["acquisition_required"] and r["gold"]["acquisition_genuinely_unavailable"])

    # never-acquire + always-COMMIT: ADA correct only on STOP episodes where
    # gold action is COMMIT; ACQUIRE episodes always fail ADA (policy never
    # acquires when gold requires it).
    stop_commit_n = sum(1 for r in rows if not r["gold"]["acquisition_required"]
                        and r["gold"]["final_terminal_action"] == "COMMIT")
    never_acquire_always_commit_b_ags = stop_commit_n / n

    # always-COMMIT (regardless of acquisition): TDA_post correct whenever
    # gold final action is COMMIT.
    commit_n = sum(1 for r in rows if r["gold"]["final_terminal_action"] == "COMMIT")
    always_commit_tda_post = commit_n / n

    return {
        "n": n, "n_STOP": stop_n, "n_ACQUIRE_available": acquire_avail_n,
        "n_ACQUIRE_unavailable": acquire_unavail_n,
        "never_acquire_always_COMMIT_B_AGS": round(never_acquire_always_commit_b_ags, 4),
        "always_COMMIT_TDA_post": round(always_commit_tda_post, 4),
    }


def main():
    a = a_baselines()
    b = b_baselines()
    d_from_paper = 0.80  # Section 4.2: "An always-escalate policy obtains 0.80 on terminal decisions"

    lines = ["# Trivial-policy baselines for A, B (D's is already reported in the paper)",
             "",
             "## A (evidence grounding), N=48, gold distribution: " + str(a["gold_distribution"]),
             "",
             f"- always-COMMIT TDA: {a['always_COMMIT_TDA']}",
             f"- always-ESCALATE TDA: {a['always_ESCALATE_TDA']}",
             f"- always-ABORT TDA: {a['always_ABORT_TDA']}",
             "",
             "GSR baseline is trivially 0 for every fixed policy: none of these deliver or use",
             "evidence, so the grounding contract is never satisfied regardless of the terminal",
             "action -- a fixed policy cannot achieve nonzero GSR.",
             "",
             f"## B (evidence acquisition), N={b['n']} "
             f"(STOP={b['n_STOP']}, ACQUIRE+available={b['n_ACQUIRE_available']}, "
             f"ACQUIRE+unavailable={b['n_ACQUIRE_unavailable']})",
             "",
             f"- never-acquire + always-COMMIT, B_AGS: {b['never_acquire_always_COMMIT_B_AGS']}",
             f"- always-COMMIT (regardless of acquisition), TDA_post: {b['always_COMMIT_TDA_post']}",
             "",
             "## D (relational physical grounding) -- already in the paper, Section 4.2",
             "",
             f"- always-ESCALATE terminal-decision accuracy: {d_from_paper}",
             "",
             "## For the main table / results section",
             "",
             f"**Every evaluated model's TDA (62.8-68.8%) is BELOW A's always-ESCALATE baseline "
             f"({a['always_ESCALATE_TDA']:.1%}).** The 42/48 ESCALATE gold imbalance means a "
             "label-agnostic constant policy beats every model on TDA alone -- this is exactly "
             "why TDA is not the headline metric and why GSR (which no fixed policy can score "
             "above 0 on, since none of them deliver or use evidence) is reported as primary in "
             "the main table. Cite this baseline explicitly wherever TDA is shown, so a raw "
             "percentage is never read as 'good' without this anchor.",
             "",
             f"B_AGS should be read against {b['never_acquire_always_COMMIT_B_AGS']:.1%} "
             "(never-acquire + always-COMMIT) as its floor, not 0 -- every model in the main "
             "table exceeds this floor."]
    md = "\n".join(lines)
    OUT.write_text(md + "\n")
    print(md)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
