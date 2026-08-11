"""v7_label_adjudication.py — V7 scenario semantic validity: measure label
reliability, emit review worksheets, and publish a machine-readable label-status
registry that gates which metrics may be reported.

Discovered during P0-1: the 20-row dev catalog
(``src/orchestrator/data/perception_real.csv``) and the 1,296-row release
catalog (``RobotInspection/shared/perception/perception.csv``) are two
*independent labelings of the same source images*. All 20 shared scenario_ids
resolve to an identical ``source_file``, yet ``gauge_readable`` disagrees on 5
of 20 — in both directions — and ``gauge_value`` on 10 of 20.

Per the Rev-2 plan ("c then a"), that conflict seeds V7 rather than being
silently resolved. This script emits **two strata**, because they answer
different questions:

``conflict``
    The 20 ids where two labelings exist. A *biased* sample by construction —
    it is where disagreement already surfaced. Good for adjudication, invalid
    as a reliability estimate for the corpus.

``random``
    A seeded random sample of query scenarios drawn from the whole release
    catalog. This is the stratum whose κ may be reported as the corpus-level
    V7 figure (GO gate κ ≥ 0.6).

It also writes ``reports/v7/label_status.csv`` — the registry consumed by
``src/evaluation/label_status.py`` so that contested labels are *excluded from
metrics by construction* rather than by a footnote. Fields whose corpus κ falls
below the gate are marked ``contested`` for every scenario in the conflict
stratum until an adjudicator fills ``adjudicated_value``.

Review questions include ``instrument_status``: AOBv2-REAL-018 carries a red
out-of-service tag (停用证 No. IEA762) across the dial face, which neither
catalog records and no schema field can express — an unlabelled operational
condition that dominates the correct action regardless of readability.

Usage::

    python scripts/v7_label_adjudication.py                    # both strata
    python scripts/v7_label_adjudication.py --random-sample 40 --seed 20260811
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
DEV_CSV = REPO_ROOT / "src" / "orchestrator" / "data" / "perception_real.csv"
FULL_CSV = (
    REPO_ROOT.parent
    / "AssetOpsBenchScenarioGeneration"
    / "RobotInspection"
    / "shared"
    / "perception"
    / "perception.csv"
)
CACHE_DIR = REPO_ROOT / "src" / "orchestrator" / "data" / "pmc_cache"
OUT_DIR = REPO_ROOT / "reports" / "v7"

KAPPA_GATE = 0.6

# Ordered by how load-bearing the field is for the benchmark's claims.
AUDIT_FIELDS = [
    "gauge_readable",      # gates every abstention/commit judgement
    "recommended_action",  # the L3 decision label
    "category",            # drives the A12 recovery-rung choice
    "image_role",          # decides whether the row is a test scenario at all
    "gauge_value",         # L1 MAE target
    "asset",               # descriptive; not scored
]

# Which metric each field gates. Consumed by src/evaluation/label_status.py.
FIELD_METRICS = {
    "gauge_readable": "l1_accuracy_f1,l2_commit_rate",
    "gauge_value": "l1_mae",
    "recommended_action": "l3_action_accuracy",
    "category": "a12_recovery_rung",
    "image_role": "scenario_membership",
    "asset": "",
}

# Open questions for the reviewer that no current schema field can express.
# 'instrument_status' exists because a decommissioned gauge makes the correct
# action independent of whether the dial is legible.
REVIEW_QUESTIONS = ["instrument_status", "description_accurate"]

WORKSHEET_FIELDS = [
    "stratum", "scenario_id", "field", "dev_value", "full_value", "source_file",
    "cached_image", "category_full", "description",
    "adjudicated_value", "adjudicator", "rationale",
]


def read_catalog(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {r["scenario_id"].strip(): r for r in csv.DictReader(f)}


def cohens_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Cohen's κ for two categorical labelings of the same items.

    Returns 1.0 when both labelings are constant and identical (κ is undefined
    there — no variance to correct for — and perfect agreement is the honest
    reading for a validity report).
    """
    if not a:
        return float("nan")
    n = len(a)
    labels = sorted(set(a) | set(b))
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    expected = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def _norm(row: Dict[str, str], field: str) -> str:
    return row.get(field, "").strip().lower()


def reliability(dev: Dict, full: Dict, ids: Sequence[str],
                fields: Sequence[str]) -> List[Dict]:
    out = []
    for field in fields:
        a = [_norm(dev[s], field) for s in ids]
        b = [_norm(full[s], field) for s in ids]
        agree = sum(1 for x, y in zip(a, b) if x == y)
        out.append({
            "field": field,
            "n": len(ids),
            "agree": agree,
            "raw_agreement": round(agree / len(ids), 4) if ids else float("nan"),
            "kappa": round(cohens_kappa(a, b), 4),
        })
    return out


def _row(stratum: str, sid: str, field: str, dev_v: str, full_v: str,
         cat: Dict[str, str]) -> Dict[str, str]:
    src = cat.get("source_file", "").strip()
    cached = CACHE_DIR / Path(src).name
    return {
        "stratum": stratum,
        "scenario_id": sid,
        "field": field,
        "dev_value": dev_v,
        "full_value": full_v,
        "source_file": src,
        "cached_image": str(cached) if cached.exists() else "",
        "category_full": cat.get("category", "").strip(),
        "description": cat.get("description", "").strip()[:200],
        "adjudicated_value": "",
        "adjudicator": "",
        "rationale": "",
    }


def conflict_rows(dev: Dict, full: Dict, fields: Sequence[str]) -> List[Dict[str, str]]:
    """One row per (scenario, field) where the two labelings disagree."""
    rows = []
    for sid in sorted(set(dev) & set(full)):
        for field in fields:
            if _norm(dev[sid], field) != _norm(full[sid], field):
                rows.append(_row("conflict", sid, field,
                                 dev[sid].get(field, "").strip(),
                                 full[sid].get(field, "").strip(), full[sid]))
        for q in REVIEW_QUESTIONS:
            rows.append(_row("conflict", sid, q, "", "", full[sid]))
    return rows


def random_rows(full: Dict, fields: Sequence[str], n: int,
                seed: int, exclude: Sequence[str]) -> List[Dict[str, str]]:
    """Seeded random sample of query scenarios — the unbiased V7 stratum.

    Only one labeling exists here, so ``full_value`` carries the catalog label
    and the reviewer supplies the independent second labeling in
    ``adjudicated_value``; κ is then computed between the two.
    """
    pool = sorted(
        s for s, r in full.items()
        if r.get("image_role", "").strip().lower() == "query" and s not in set(exclude)
    )
    picked = random.Random(seed).sample(pool, min(n, len(pool)))
    rows = []
    for sid in sorted(picked):
        for field in list(fields) + REVIEW_QUESTIONS:
            rows.append(_row("random", sid, field, "",
                             full[sid].get(field, "").strip(), full[sid]))
    return rows


def label_status(stats: Sequence[Dict], dev: Dict, full: Dict,
                 fields: Sequence[str]) -> List[Dict[str, str]]:
    """Per (scenario, field) trust registry.

    Three states, because "we found a problem here" and "we have not looked
    here yet" are different claims:

    ``contested``  the two labelings disagree on *this* scenario. Direct
                   evidence; excluded from metrics until adjudicated.
    ``trusted``    two independent labelings agree on this scenario.
    ``pending``    only one labeling exists and the unbiased (random) stratum
                   has not been adjudicated, so corpus reliability for the
                   field is unknown.

    Note what is deliberately *not* done here: the conflict-stratum κ is not
    used to gate scenarios outside that stratum. That stratum is biased by
    construction — it is the set where disagreement already surfaced — so
    projecting its κ onto the corpus would understate reliability everywhere.
    The corpus-level gate comes from the random stratum once
    ``adjudicated_value`` is filled in; until then those pairs are ``pending``.
    """
    rows = []
    for sid in sorted(full):
        for field in fields:
            if sid in dev:
                agrees = _norm(dev[sid], field) == _norm(full[sid], field)
                status = "trusted" if agrees else "contested"
                reason = "dual labeling agrees" if agrees else \
                    "dual-labeling disagreement"
            else:
                status = "pending"
                reason = "single labeling; random stratum not yet adjudicated"
            rows.append({
                "scenario_id": sid,
                "field": field,
                "status": status,
                "reason": reason,
                "gates_metric": FIELD_METRICS.get(field, ""),
            })
    return rows


def _write(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fields", nargs="*", default=AUDIT_FIELDS)
    ap.add_argument("--random-sample", type=int, default=40,
                    help="size of the unbiased V7 stratum (0 disables)")
    ap.add_argument("--seed", type=int, default=20260811)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    dev, full = read_catalog(DEV_CSV), read_catalog(FULL_CSV)
    shared = sorted(set(dev) & set(full))
    same_image = sum(
        1 for s in shared
        if dev[s].get("source_file", "").strip() == full[s].get("source_file", "").strip()
    )
    print(f"shared scenario_ids: {len(shared)}   identical source_file: {same_image}\n")

    stats = reliability(dev, full, shared, args.fields)
    print("CONFLICT STRATUM (biased by construction - adjudication input, NOT a corpus estimate)")
    print(f"{'field':22s} {'n':>4s} {'agree':>7s} {'raw':>8s} {'kappa':>8s}  gate({KAPPA_GATE})")
    print("-" * 66)
    for s in stats:
        gate = "PASS" if s["kappa"] >= KAPPA_GATE else "FAIL"
        print(f"{s['field']:22s} {s['n']:4d} {s['agree']:7d} "
              f"{s['raw_agreement']:8.2%} {s['kappa']:8.3f}  {gate}")

    rows = conflict_rows(dev, full, args.fields)
    if args.random_sample:
        rows += random_rows(full, args.fields, args.random_sample, args.seed, shared)

    _write(args.out_dir / "label_adjudication_worksheet.csv", WORKSHEET_FIELDS, rows)
    _write(args.out_dir / "label_reliability.csv",
           ["field", "n", "agree", "raw_agreement", "kappa"], stats)

    status = label_status(stats, dev, full, args.fields)
    _write(args.out_dir / "label_status.csv",
           ["scenario_id", "field", "status", "reason", "gates_metric"], status)

    n_conf = sum(1 for r in rows if r["stratum"] == "conflict")
    n_rand = sum(1 for r in rows if r["stratum"] == "random")
    blocked = sorted({s["field"] for s in stats if s["kappa"] < KAPPA_GATE})
    contested = sum(1 for r in status if r["status"] == "contested")

    print(f"\nworksheet rows: {n_conf} conflict + {n_rand} random "
          f"(seed={args.seed}) -> {args.out_dir / 'label_adjudication_worksheet.csv'}")
    print(f"label_status: {contested}/{len(status)} (scenario, field) pairs contested")
    if blocked:
        gated = sorted({FIELD_METRICS[f] for f in blocked if FIELD_METRICS.get(f)})
        print(f"\nBELOW GATE: {', '.join(blocked)}")
        print(f"  metrics blocked until adjudicated: {', '.join(gated)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
