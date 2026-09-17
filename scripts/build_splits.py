"""build_splits.py — Leak-safe evaluation splits and the coverage matrix.

Splits A-F from the Rev-2 plan. Two properties matter more than the sizes:

**Leak safety is enforced at the image level, not the scenario level.** Each
scenario pairs a query image with a reference image, and references are reused —
one image serves as reference for up to 20 scenarios. Splitting by scenario id
would put the same photograph in both the test and the regenerated set. This
script therefore builds connected components over shared images (query *or*
reference) and assigns whole components to a split. The corpus decomposes into
280 components with the largest at 2.7% of scenarios, so component-level
assignment costs almost no balance.

**Split C ships as two views rather than one down-sampled set.** The natural
distribution is ~54% ``CLEAN_GAUGE``; hiding that would misrepresent operational
reality, while reporting only that would let a class prior masquerade as
capability. Both are emitted and the plan requires reporting both — the gap
between them *is* the measurement of whether a model distinguishes states.

Usage::

    python scripts/build_splits.py
    python scripts/build_splits.py --seed 20260811 --out-dir config/splits
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "src" / "orchestrator" / "data"
FULL_CSV = (
    REPO_ROOT.parent / "AssetOpsBenchScenarioGeneration" / "RobotInspection"
    / "shared" / "perception" / "perception.csv"
)
PAIRS_CSV = DATA_DIR / "pairs_full.csv"
DEV_CSV = DATA_DIR / "perception_real.csv"
INDUSTRIAL_CSV = DATA_DIR / "perception_industrial.csv"
OUT_DIR = REPO_ROOT / "config" / "splits"

#: Candidate source-holdout roots, best first. V5 asks whether the failure
#: patterns transfer to observations from a *different capture source*, so the
#: staged in-repo sample is the weakest option and real facility captures the
#: strongest. Overridable with ASSETOPS_HOLDOUT_DIR.
HOLDOUT_CANDIDATES = [
    # Genuine facility captures (iPhone-sourced, distinct from the PMC corpus).
    Path("/media/adityapachauri/second_drive/external_datasets/real_world_samples/"
         "AssetOps Gauge"),
    # Second independent source; different instrument population.
    Path("/media/adityapachauri/second_drive/external_datasets/real_world_samples/RPM-10K"),
    REPO_ROOT / "src" / "perception" / "data" / "pmc_sample",
]

#: Target proportions for the component-assigned L1 splits. Development is
#: carved out first by id (the dual-labelled scenarios), so these apply to what
#: remains.
TARGETS = {"pilot": 0.11, "test": 0.67, "regenerated": 0.22}

#: Decision vocabulary — the axis most at risk of collapsing.
DECISION_FIELD = "recommended_action"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------
# Leak-safe components
# --------------------------------------------------------------------------

def build_components(pairs: Sequence[Dict[str, str]]) -> Dict[str, int]:
    """Map scenario_id -> component id, connecting scenarios that share an image.

    Both endpoints matter: a photograph used as scenario X's query may be
    scenario Y's reference, and putting X and Y in different splits leaks it.
    """
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    by_image: Dict[str, List[str]] = defaultdict(list)
    for p in pairs:
        for key in ("query_source", "reference_source"):
            by_image[p[key]].append(p["scenario_id"])
    for sids in by_image.values():
        for s in sids[1:]:
            union(sids[0], s)

    roots = sorted({find(p["scenario_id"]) for p in pairs})
    index = {r: i for i, r in enumerate(roots)}
    return {p["scenario_id"]: index[find(p["scenario_id"])] for p in pairs}


def assign_components(components: Dict[str, int], targets: Dict[str, float],
                      seed: int) -> Dict[str, str]:
    """Greedily assign whole components to splits, largest first.

    Largest-first keeps the realised sizes close to target: assigning big
    components last is what makes proportional splitting overshoot.
    """
    members: Dict[int, List[str]] = defaultdict(list)
    for sid, comp in components.items():
        members[comp].append(sid)

    total = len(components)
    quota = {name: frac * total for name, frac in targets.items()}
    filled = {name: 0 for name in targets}

    rng = random.Random(seed)
    order = sorted(members.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    # Shuffle within equal sizes so the assignment is not an artifact of id order.
    buckets: Dict[int, List[Tuple[int, List[str]]]] = defaultdict(list)
    for comp, sids in order:
        buckets[len(sids)].append((comp, sids))
    order = []
    for size in sorted(buckets, reverse=True):
        group = buckets[size]
        rng.shuffle(group)
        order.extend(group)

    out: Dict[str, str] = {}
    for _, sids in order:
        # Whichever split is furthest from its quota, proportionally.
        name = max(quota, key=lambda k: (quota[k] - filled[k]) / max(quota[k], 1))
        for sid in sids:
            out[sid] = name
        filled[name] += len(sids)
    return out


# --------------------------------------------------------------------------
# Views
# --------------------------------------------------------------------------

def balanced_view(ids: Sequence[str], catalog: Dict[str, Dict[str, str]],
                  seed: int, min_class: int = 10) -> List[str]:
    """Down-sample so no decision class can be exploited as a prior.

    The decision axis is balanced *first* and category is spread within it.
    Balancing on (category, decision) cells jointly — the obvious approach —
    actually makes things worse here: the cells are wildly uneven, so a uniform
    cap deletes most of the minority-decision scenarios and drives the majority
    baseline **up**. The caller asserts the resulting baseline is no worse than
    the natural view, which is the property that matters.

    Classes smaller than ``min_class`` (``ESCALATE`` has 6 corpus-wide) are kept
    whole rather than allowed to set the cap for everyone else.
    """
    by_decision: Dict[str, List[str]] = defaultdict(list)
    for sid in ids:
        by_decision[catalog[sid][DECISION_FIELD].strip()].append(sid)
    if not by_decision:
        return []

    big = {d: m for d, m in by_decision.items() if len(m) >= min_class}
    small = {d: m for d, m in by_decision.items() if len(m) < min_class}
    cap = min((len(m) for m in big.values()), default=0)

    rng = random.Random(seed)
    out: List[str] = []
    for members in small.values():
        out.extend(members)
    for decision in sorted(big):
        members = big[decision]
        # Spread the cap across categories: round-robin over category buckets so
        # a single visual category cannot fill an entire decision class.
        buckets: Dict[str, List[str]] = defaultdict(list)
        for sid in sorted(members):
            buckets[catalog[sid]["category"].strip()].append(sid)
        for b in buckets.values():
            rng.shuffle(b)
        picked: List[str] = []
        order = sorted(buckets)
        while len(picked) < cap and any(buckets[c] for c in order):
            for c in order:
                if buckets[c] and len(picked) < cap:
                    picked.append(buckets[c].pop())
        out.extend(picked)
    return sorted(out)


def coverage(ids: Sequence[str], catalog: Dict[str, Dict[str, str]]) -> Dict:
    """Counts along the plan's reportable axes."""
    def tally(field: str) -> Dict[str, int]:
        return dict(Counter(catalog[s][field].strip() for s in ids if s in catalog))

    readable = Counter(
        catalog[s]["gauge_readable"].strip().lower() for s in ids if s in catalog
    )
    actions = tally(DECISION_FIELD)
    majority = max(actions.values()) / len(ids) if ids and actions else 0.0
    return {
        "n": len(ids),
        "category": tally("category"),
        "decision": actions,
        "majority_class_baseline": round(majority, 4),
        "gauge_readable": {"true": readable.get("true", 0), "false": readable.get("false", 0)},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=20260811)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    if not (FULL_CSV.exists() and PAIRS_CSV.exists()):
        print(f"ERROR: need {FULL_CSV} and {PAIRS_CSV} "
              "(run scripts/build_pairs.py first)")
        return 2

    catalog = {r["scenario_id"]: r for r in read_csv(FULL_CSV)}
    pairs = read_csv(PAIRS_CSV)
    dev_ids = sorted({r["scenario_id"] for r in read_csv(DEV_CSV)} & set(catalog))

    # A. Development holds the dual-labelled scenarios, whose ground truth is
    # contested — they must not sit in the test set. Carving them out *by id*
    # leaks: a dev scenario's reference photograph is often another scenario's
    # image, so the picture reappears in C. Components are therefore built over
    # the whole corpus and any component touching a dev scenario is pinned to A
    # in full.
    components_all = build_components(pairs)
    dev_components = {components_all[s] for s in dev_ids if s in components_all}
    dev_ids = sorted(s for s, c in components_all.items() if c in dev_components)

    components = {s: c for s, c in components_all.items() if c not in dev_components}
    assignment = assign_components(components, TARGETS, args.seed)

    split_names = {"pilot": "B_pilot", "test": "C_test", "regenerated": "D_regenerated"}
    splits: Dict[str, List[str]] = defaultdict(list)
    splits["A_development"] = dev_ids
    for sid, name in assignment.items():
        splits[split_names[name]].append(sid)
    for k in splits:
        splits[k] = sorted(splits[k])

    # C ships as two views over the same scenarios.
    view_natural = splits["C_test"]
    view_balanced = balanced_view(view_natural, catalog, args.seed)

    # E. Source holdout: genuinely different capture sources, enumerated from
    # whichever candidate roots are mounted. These images are **unlabelled** —
    # no readability, value or action ground truth exists for them — so V5
    # cannot score accuracy until they are annotated. Recording the gap here
    # keeps "we have real images" from being mistaken for "we validated on real
    # images", which the plan explicitly forbids.
    holdout_sources = []
    for root in [Path(os.environ["ASSETOPS_HOLDOUT_DIR"])] if os.environ.get(
            "ASSETOPS_HOLDOUT_DIR") else HOLDOUT_CANDIDATES:
        if not root.exists():
            continue
        imgs = sorted(p for p in root.rglob("*")
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if imgs:
            holdout_sources.append({
                "root": str(root),
                "n_images": len(imgs),
                "labelled": False,
                "examples": [p.name for p in imgs[:3]],
            })
    holdout_total = sum(s["n_images"] for s in holdout_sources)

    industrial = [r["scenario_id"] for r in read_csv(INDUSTRIAL_CSV)] \
        if INDUSTRIAL_CSV.exists() else []

    # Leak check: no image may appear in two splits.
    img_split: Dict[str, str] = {}
    leaks: List[str] = []
    by_id = {p["scenario_id"]: p for p in pairs}
    for name, ids in splits.items():
        for sid in ids:
            p = by_id.get(sid)
            if not p:
                continue
            for key in ("query_source", "reference_source"):
                prev = img_split.setdefault(p[key], name)
                if prev != name:
                    leaks.append(f"{p[key]}: {prev} vs {name}")

    manifest = {
        "schema": "assetops.splits/1",
        "seed": args.seed,
        "leak_safety": {
            "unit": "connected components over shared query/reference images",
            "n_components": len(set(components.values())),
            "cross_split_image_leaks": len(leaks),
            "examples": leaks[:5],
        },
        "splits": {
            "A_development": {"ids": splits["A_development"],
                              "note": "dual-labelled, contested GT; never in the test set"},
            "B_pilot": {"ids": splits["B_pilot"]},
            "C_test": {
                "note": "fixed test set; never tuned. Two views over the same scenarios.",
                "view_natural_prior": view_natural,
                "view_balanced": view_balanced,
            },
            "D_regenerated": {"ids": splits["D_regenerated"]},
            "E_source_holdout": {
                "sources": holdout_sources,
                "n": holdout_total,
                "labelled": False,
                "blocks": ["V5"],
                "note": ("Images from independent capture sources are available, but "
                         "carry no readability/value/action ground truth. V5 (source "
                         "dependence) is BLOCKED on annotating them; possessing real "
                         "images is not real-world validation."),
            },
            "F_industrial": {"ids": industrial, "n": len(industrial)},
        },
        "coverage": {
            "A_development": coverage(splits["A_development"], catalog),
            "B_pilot": coverage(splits["B_pilot"], catalog),
            "C_test_natural": coverage(view_natural, catalog),
            "C_test_balanced": coverage(view_balanced, catalog),
            "D_regenerated": coverage(splits["D_regenerated"], catalog),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "splits.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"components: {manifest['leak_safety']['n_components']}   "
          f"cross-split image leaks: {len(leaks)}")
    for name in ("A_development", "B_pilot", "C_test", "D_regenerated"):
        ids = view_natural if name == "C_test" else splits[name]
        cov = coverage(ids, catalog)
        print(f"  {name:16s} n={cov['n']:4d}  readable {cov['gauge_readable']['true']:3d}/"
              f"{cov['gauge_readable']['false']:3d}  majority={cov['majority_class_baseline']:.1%}")
    cb = coverage(view_balanced, catalog)
    cn = coverage(view_natural, catalog)
    print(f"  {'C_test[balanced]':16s} n={cb['n']:4d}  readable "
          f"{cb['gauge_readable']['true']:3d}/{cb['gauge_readable']['false']:3d}  "
          f"majority={cb['majority_class_baseline']:.1%}")
    print(f"  {'E_source_holdout':16s} n={holdout_total:4d}  UNLABELLED - V5 blocked")
    for s in holdout_sources:
        print(f"      {s['n_images']:4d}  {s['root']}")
    print(f"\n-> {out}")

    problems = 0
    if leaks:
        print(f"\nFAIL: {len(leaks)} cross-split image leaks")
        problems += 1
    # A "balanced" view that raises the majority baseline is not balanced.
    if cb["majority_class_baseline"] > cn["majority_class_baseline"] + 1e-9:
        print(f"\nFAIL: balanced view majority {cb['majority_class_baseline']:.1%} "
              f"exceeds natural {cn['majority_class_baseline']:.1%}")
        problems += 1
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
