"""build_pairs.py — Generate the query/reference pairing table for a perception catalog.

P0-1. The repo shipped a hand-built ``pairs.csv`` covering only the 13-scenario
dev subset (``src/orchestrator/data/perception_real.csv``). The full release
catalog — ``RobotInspection/shared/perception/perception.csv``, 1,296 rows,
856 readable / 440 unreadable — has no pairing table, which is the actual reason
``build_scenarios`` could only ever yield 13 scenarios and why every query row in
the dev subset happened to be ``gauge_readable=false``.

The matching rule is reverse-engineered from the hand-built table and is
validated against it by ``--verify`` (and by
``src/orchestrator/tests/test_build_pairs.py``): all 13 original rows must be
reproduced exactly, including ``match_tier`` and ``ts_diff_seconds``.

Tier priority, best first::

    cat+asset   reference shares both category and asset
    cat_only    reference shares category
    timestamp   nearest reference in time, any category/asset

Within a tier the reference minimising ``|ts_query - ts_reference|`` wins; ties
break on filename so the output is deterministic.

Usage::

    python scripts/build_pairs.py --verify           # regression vs shipped table
    python scripts/build_pairs.py \
        --perception-csv .../shared/perception/perception.csv \
        --out src/orchestrator/data/pairs_full.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "src" / "orchestrator" / "data"
SHIPPED_PAIRS = DATA_DIR / "pairs.csv"
DEV_PERCEPTION_CSV = DATA_DIR / "perception_real.csv"
FULL_PERCEPTION_CSV = (
    REPO_ROOT.parent
    / "AssetOpsBenchScenarioGeneration"
    / "RobotInspection"
    / "shared"
    / "perception"
    / "perception.csv"
)

PAIR_FIELDS = [
    "scenario_id",
    "category",
    "asset",
    "query_source",
    "reference_source",
    "match_tier",
    "ts_diff_seconds",
]

# IMG_20220106_104236.jpg / IMG_20220106_104909_1.jpg -> 2022-01-06 10:49:09
_TS_RE = re.compile(r"(\d{8})_(\d{6})")


def parse_timestamp(source_file: str) -> Optional[datetime]:
    """Recover the capture time encoded in the PMC filename."""
    m = _TS_RE.search(Path(source_file).name)
    if m is None:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


@dataclass(frozen=True)
class Row:
    scenario_id: str
    category: str
    asset: str
    image_role: str
    gauge_readable: bool
    source_file: str
    timestamp: Optional[datetime]

    @property
    def basename(self) -> str:
        return Path(self.source_file).name


def load_rows(perception_csv: Path) -> List[Row]:
    with perception_csv.open(encoding="utf-8-sig", newline="") as f:
        return [
            Row(
                scenario_id=r["scenario_id"].strip(),
                category=r["category"].strip(),
                asset=r["asset"].strip(),
                image_role=r["image_role"].strip().lower(),
                gauge_readable=r["gauge_readable"].strip().lower() == "true",
                source_file=r["source_file"].strip(),
                timestamp=parse_timestamp(r["source_file"]),
            )
            for r in csv.DictReader(f)
        ]


def _best(candidates: Sequence[Row], query: Row) -> Optional[Row]:
    """Nearest in time; deterministic tie-break on basename."""
    usable = [c for c in candidates if c.timestamp is not None]
    if not usable or query.timestamp is None:
        return None
    return min(
        usable,
        key=lambda c: (abs((query.timestamp - c.timestamp).total_seconds()), c.basename),
    )


def match_reference(query: Row, references: Sequence[Row]) -> Optional[tuple]:
    """Return ``(reference_row, match_tier)`` under the tier priority, or None."""
    pool = [r for r in references if r.source_file != query.source_file]
    tiers = (
        ("cat+asset", [r for r in pool if r.category == query.category and r.asset == query.asset]),
        ("cat_only", [r for r in pool if r.category == query.category]),
        ("timestamp", pool),
    )
    for tier, candidates in tiers:
        pick = _best(candidates, query)
        if pick is not None:
            return pick, tier
    return None


def build_pairs(rows: Sequence[Row]) -> List[Dict[str, str]]:
    """One pair per ``image_role=query`` row that resolves a reference.

    References are the clean, readable observations (``image_role`` of
    ``reference`` or ``clean``). A reference is never paired with itself.
    """
    references = [r for r in rows if r.image_role in ("reference", "clean") and r.gauge_readable]
    queries = [r for r in rows if r.image_role == "query"]

    pairs: List[Dict[str, str]] = []
    for q in sorted(queries, key=lambda r: r.scenario_id):
        matched = match_reference(q, references)
        if matched is None:
            continue
        ref, tier = matched
        delta = abs(int((q.timestamp - ref.timestamp).total_seconds()))
        pairs.append(
            {
                "scenario_id": q.scenario_id,
                "category": q.category,
                "asset": q.asset,
                "query_source": q.source_file,
                "reference_source": ref.source_file,
                "match_tier": tier,
                "ts_diff_seconds": str(delta),
            }
        )
    return pairs


def write_pairs(pairs: Sequence[Dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PAIR_FIELDS)
        w.writeheader()
        w.writerows(pairs)


def verify_against_shipped() -> int:
    """Regression gate: the rule must reproduce the hand-built 13-row table."""
    rows = load_rows(DEV_PERCEPTION_CSV)
    generated = {p["scenario_id"]: p for p in build_pairs(rows)}
    with SHIPPED_PAIRS.open(encoding="utf-8-sig", newline="") as f:
        expected = {r["scenario_id"]: r for r in csv.DictReader(f)}

    failures: List[str] = []
    for sid, exp in sorted(expected.items()):
        got = generated.get(sid)
        if got is None:
            failures.append(f"{sid}: not generated")
            continue
        for field in ("query_source", "reference_source", "match_tier", "ts_diff_seconds"):
            if got[field] != exp[field]:
                failures.append(f"{sid}.{field}: expected {exp[field]!r}, got {got[field]!r}")

    extra = sorted(set(generated) - set(expected))
    print(f"shipped rows: {len(expected)}   generated: {len(generated)}")
    if extra:
        print(f"generated but not in shipped table: {extra}")
    if failures:
        print("\nMISMATCHES:")
        for line in failures:
            print("  " + line)
        return 1
    print("OK - all shipped pairs reproduced exactly")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perception-csv", type=Path, default=FULL_PERCEPTION_CSV)
    ap.add_argument("--out", type=Path, default=DATA_DIR / "pairs_full.csv")
    ap.add_argument("--verify", action="store_true",
                    help="regression-check the rule against the shipped 13-row pairs.csv")
    args = ap.parse_args()

    if args.verify:
        return verify_against_shipped()

    if not args.perception_csv.exists():
        print(f"ERROR: perception csv not found: {args.perception_csv}", file=sys.stderr)
        return 2

    rows = load_rows(args.perception_csv)
    pairs = build_pairs(rows)
    write_pairs(pairs, args.out)

    queries = sum(1 for r in rows if r.image_role == "query")
    tiers: Dict[str, int] = {}
    for p in pairs:
        tiers[p["match_tier"]] = tiers.get(p["match_tier"], 0) + 1
    print(f"rows={len(rows)}  queries={queries}  paired={len(pairs)}  -> {args.out}")
    print("match_tier: " + ", ".join(f"{k}={v}" for k, v in sorted(tiers.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
