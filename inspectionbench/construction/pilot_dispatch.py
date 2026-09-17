"""pilot_dispatch.py -- Phase 8H.1 multi-model pilot: frozen manifest -> execution units.

Consumes ``reports/ec/phase8h1_pilot_manifest.json`` (SHA-gated) and resolves every
episode to an EXISTING, already-validated runner. Dispatch is keyed on the manifest's
``scenario_template`` string plus ``dimension`` -- NEVER on ``scenario_id``. There is
no scenario-ID-specific branch anywhere in this module.

Nothing here calls a model or touches CouchDB. It is pure structure: load, validate,
group, and (for A/B world-first rows) rebuild the world from the manifest's own
``world_seed`` + ``scenario_parameters`` and assert the normalized hash matches.

Execution unit = the natural call boundary of the underlying runner:
  A            -> 1 unit / episode        run_a_episode(world, arm, model=..., ...)
  B  (R011)    -> 1 unit / regime arm     run_l3_pilot_executed.run_episode(...)
  B  (fm7a)    -> 1 unit / episode        run_a_episode(world, arm, ...)   world.contradiction=True
  C            -> 1 unit / scenario       run_classc_pilot.run(...)
  D            -> 1 unit / scenario       run_classd_pilot.run(...)
  E            -> 1 unit / sequence       run_class_e_pilot.run_sequence(...)  -> 3 episode rows
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

MANIFEST_PATH = REPO_ROOT / "reports" / "ec" / "phase8h1_pilot_manifest.json"
MANIFEST_SHA256 = "d2b48c0b0c9ef19f6c8f6ddd936da098ffd5d0d4023650619f01c1ee043087fe"

EXPECTED_BY_DIM = {"A": 48, "B": 12, "C": 9, "D": 6, "E": 18}
EXPECTED_TOTAL = 93

REGIMES = ("FULL", "PHYSICAL_ONLY", "DIGITAL_ONLY")


class ManifestIntegrityError(Exception):
    """Raised on any manifest SHA / count / structure mismatch. A hard STOP."""


class DispatchError(Exception):
    """Raised when an episode cannot be resolved to a runner. A hard STOP."""


# --------------------------------------------------------------------------
# manifest loading + SHA validation
# --------------------------------------------------------------------------

def manifest_sha256(path: Path = MANIFEST_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manifest(path: Path = MANIFEST_PATH, *, expected_sha: Optional[str] = MANIFEST_SHA256,
                  verify_counts: bool = True) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ManifestIntegrityError(f"manifest not found: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha is not None and digest != expected_sha:
        raise ManifestIntegrityError(
            f"manifest SHA mismatch: expected {expected_sha}, got {digest}. "
            f"The frozen benchmark has changed -- STOP.")
    m = json.loads(path.read_text())
    if verify_counts:
        eps = m.get("episodes", [])
        if len(eps) != EXPECTED_TOTAL:
            raise ManifestIntegrityError(f"episode count {len(eps)} != {EXPECTED_TOTAL}")
        by_dim: Dict[str, int] = {}
        for e in eps:
            by_dim[e["dimension"]] = by_dim.get(e["dimension"], 0) + 1
        if by_dim != EXPECTED_BY_DIM:
            raise ManifestIntegrityError(f"by-dim {by_dim} != {EXPECTED_BY_DIM}")
        ids = [e["episode_id"] for e in eps]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ManifestIntegrityError(f"duplicate episode_ids: {dupes}")
    m["_sha256"] = digest
    return m


# --------------------------------------------------------------------------
# world rebuild (A / B world-first rows only) + hash assertion
# --------------------------------------------------------------------------

def rebuild_world(row: Dict[str, Any]):
    """Rebuild an A or B-fm7a world from the manifest row's own seed + params.
    Asserts the recomputed normalized hash equals the manifest's recorded value.
    Returns the world object, or None for rows that carry no world hash (B/R011,
    C, D)."""
    if row.get("normalized_world_hash") is None:
        return None
    from scenario_gen import (FACTORIAL, sample_world,
                              sample_world_fm7a_contradiction)
    from canonical_identity import normalize_world

    tmpl = row["scenario_template"]
    seed = row["world_seed"]
    params = row.get("scenario_parameters", {})
    label = row["scenario_id"]

    if tmpl == "sample_world/FM-6a":
        cell = next(c for c in FACTORIAL if c.name == params["cell"])
        world = sample_world(seed, cell, asset_id=params["asset"], scenario_id=label)
    elif tmpl == "sample_world_fm7a_contradiction":
        world = sample_world_fm7a_contradiction(
            seed, params["physical_in_band"], asset_id=params["asset"], scenario_id=label)
        world.contradiction = True
    elif tmpl == "sample_sequence":
        from sequence_executor import sample_sequence
        world = sample_sequence(seed, n_episodes=params.get("n_episodes", 3))
    else:
        raise DispatchError(f"no world-rebuild rule for template {tmpl!r} "
                            f"(episode {row['episode_id']})")

    got = normalize_world(world).world_id
    if got != row["normalized_world_hash"]:
        raise ManifestIntegrityError(
            f"{row['episode_id']}: rebuilt world hash {got} != manifest "
            f"{row['normalized_world_hash']} -- STOP.")
    return world


# --------------------------------------------------------------------------
# execution-unit grouping
# --------------------------------------------------------------------------

@dataclass
class ExecUnit:
    """One runner invocation. ``episode_ids`` is what it will produce rows for
    (1 for A/B/C/D, 3 for an E sequence)."""
    unit_id: str
    dim: str
    template: str
    kind: str                        # "a_episode" | "b_incumbent" | "classc" | "classd" | "e_sequence"
    episode_ids: List[str]
    rows: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


#: template string -> execution kind. The ONLY dispatch key besides `dimension`.
_TEMPLATE_KIND = {
    "sample_world/FM-6a": "a_episode",
    "sample_world_fm7a_contradiction": "a_episode",
    "l3_arms/R011 (incumbent, real)": "b_incumbent",
    "run_classc_pilot (real fixture)": "classc",
    "run_classd_pilot (real fixture)": "classd",
    "sample_sequence": "e_sequence",
}


def build_execution_units(manifest: Dict[str, Any]) -> List[ExecUnit]:
    eps = manifest["episodes"]
    units: List[ExecUnit] = []
    e_by_seq: Dict[str, List[Dict[str, Any]]] = {}

    for row in eps:
        tmpl = row["scenario_template"]
        kind = _TEMPLATE_KIND.get(tmpl)
        if kind is None:
            raise DispatchError(
                f"{row['episode_id']}: unknown scenario_template {tmpl!r} -- "
                f"cannot dispatch. STOP.")

        if kind == "e_sequence":
            e_by_seq.setdefault(row["world_id"], []).append(row)
            continue

        if kind in ("a_episode", "b_incumbent"):
            if row["regime"] not in REGIMES:
                raise DispatchError(f"{row['episode_id']}: bad regime {row['regime']!r}")

        units.append(ExecUnit(
            unit_id=row["episode_id"], dim=row["dimension"], template=tmpl,
            kind=kind, episode_ids=[row["episode_id"]], rows=[row]))

    # E: one unit per sequence, episodes ordered by episode_index
    for seq_id, rows in sorted(e_by_seq.items()):
        rows = sorted(rows, key=lambda r: r["scenario_parameters"]["episode_index"])
        idxs = [r["scenario_parameters"]["episode_index"] for r in rows]
        if idxs != list(range(len(rows))):
            raise DispatchError(f"{seq_id}: episode_index set {idxs} not contiguous from 0")
        units.append(ExecUnit(
            unit_id=seq_id, dim="E", template="sample_sequence", kind="e_sequence",
            episode_ids=[r["episode_id"] for r in rows], rows=rows,
            extra={"seed": rows[0]["world_seed"], "n_episodes": len(rows)}))

    covered = sorted(eid for u in units for eid in u.episode_ids)
    manifest_ids = sorted(r["episode_id"] for r in eps)
    if covered != manifest_ids:
        missing = sorted(set(manifest_ids) - set(covered))
        extra = sorted(set(covered) - set(manifest_ids))
        raise DispatchError(f"dispatch coverage gap: missing={missing} extra={extra}")
    return units


def dispatch_coverage(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only report: does every episode resolve, and to what."""
    units = build_execution_units(manifest)
    by_kind: Dict[str, int] = {}
    for u in units:
        by_kind[u.kind] = by_kind.get(u.kind, 0) + 1
    return {
        "n_units": len(units),
        "n_episodes_covered": sum(len(u.episode_ids) for u in units),
        "units_by_kind": by_kind,
        "e_sequences": sorted(u.unit_id for u in units if u.kind == "e_sequence"),
    }


if __name__ == "__main__":
    m = load_manifest()
    print(f"manifest SHA OK: {m['_sha256']}")
    cov = dispatch_coverage(m)
    print(json.dumps(cov, indent=2))
