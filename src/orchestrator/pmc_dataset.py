"""pmc_dataset.py — Loads real PMC gauge-photo scenarios for the orchestrator.

Joins the perception seed catalog with the query/reference resolution table
(pairs.csv) and the raw PMC images (Boston Dynamics-style field photos,
bundled in data_detection.zip). Extracts only the needed image members from
the zip into a local cache — the archive itself is left untouched.

Only scenarios present in the pairs table are used: those are the actual
query/reference test cases (image_role=query in the catalog). Rows with
image_role=reference/clean are clean images used *as* someone else's
reference — not test scenarios themselves.

P0-1 (Rev-2 plan). The default catalog is the **full 1,296-row release**
(``RobotInspection/shared/perception/perception.csv``), not the 20-row dev
subset. This matters for validity, not merely for scale: every query row in
the dev subset happens to carry ``gauge_readable=false``, which leaves L1
accuracy/F1/MAE undefined and lets a policy that abstains unconditionally
score perfectly. The full catalog yields **311 readable / 440 unreadable**
query scenarios, so abstention is measurable against a false-abstention rate.

The dev subset remains available as a fast fixture; see ``DEV_*`` below and
``build_scenarios(perception_csv=DEV_PERCEPTION_CSV, pairs_csv=DEV_PAIRS_CSV)``.

Paths are environment-overridable so the benchmark is runnable off this
machine (P0-2)::

    ASSETOPS_PMC_ZIP          data_detection.zip
    ASSETOPS_PMC_CACHE        extracted-image cache dir
    ASSETOPS_PERCEPTION_CSV   perception catalog
    ASSETOPS_PAIRS_CSV        query/reference pairing table
"""

from __future__ import annotations

import csv
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_DIR = Path(__file__).resolve().parent / "data"

# The 20-row dev subset shipped in-repo: fast, offline-friendly, and used by the
# smoke tests. Not a valid accuracy benchmark on its own (all queries unreadable).
DEV_PERCEPTION_CSV = DATA_DIR / "perception_real.csv"
DEV_PAIRS_CSV = DATA_DIR / "pairs.csv"

# The full release catalog (sibling scenario-generation repo) plus the pairing
# table generated from it by ``scripts/build_pairs.py``.
FULL_PERCEPTION_CSV = (
    Path(__file__).resolve().parents[3]
    / "AssetOpsBenchScenarioGeneration"
    / "RobotInspection"
    / "shared"
    / "perception"
    / "perception.csv"
)
FULL_PAIRS_CSV = DATA_DIR / "pairs_full.csv"


def _env_path(var: str, fallback: Path) -> Path:
    raw = os.environ.get(var, "").strip()
    return Path(raw).expanduser() if raw else fallback


def _default_catalog() -> tuple:
    """Prefer the full release catalog; fall back to the dev subset when the
    sibling repo or its generated pairs table is absent, so a fresh clone still
    runs (with a correspondingly weaker claim)."""
    if FULL_PERCEPTION_CSV.exists() and FULL_PAIRS_CSV.exists():
        return FULL_PERCEPTION_CSV, FULL_PAIRS_CSV
    return DEV_PERCEPTION_CSV, DEV_PAIRS_CSV


_CATALOG_CSV, _PAIRS_CSV = _default_catalog()

DEFAULT_ZIP_PATH = _env_path(
    "ASSETOPS_PMC_ZIP",
    Path("/media/adityapachauri/second_drive/external_datasets/"
         "pmc_gauge_dataset/data_detection.zip"),
)
DEFAULT_PERCEPTION_CSV = _env_path("ASSETOPS_PERCEPTION_CSV", _CATALOG_CSV)
DEFAULT_PAIRS_CSV = _env_path("ASSETOPS_PAIRS_CSV", _PAIRS_CSV)
DEFAULT_CACHE_DIR = _env_path("ASSETOPS_PMC_CACHE", DATA_DIR / "pmc_cache")

_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_numeric(raw: Optional[str]) -> Optional[float]:
    """Extract a leading float from strings like '0.05 MPa', '45°C', 'UNREADABLE'."""
    if raw is None:
        return None
    raw = raw.strip()
    if not raw or raw.upper() == "UNREADABLE":
        return None
    m = _NUMERIC_RE.search(raw)
    return float(m.group()) if m else None


@dataclass
class RealScenario:
    """A single real-image gauge inspection episode (PMC-sourced)."""

    scenario_id: str
    category: str
    asset: str
    location: str
    description: str
    query_image: Path
    reference_image: Path
    iot_value: Optional[float]
    iot_value_raw: str
    gauge_readable_gt: bool
    gauge_value_gt: Optional[float]
    gauge_range_min: Optional[float]
    gauge_range_max: Optional[float]
    gauge_unit: str
    recommended_action: str
    forbidden_actions: List[str]
    trap: str
    match_tier: str = ""
    gauge_bbox: Optional[tuple] = None  # YOLO (cx, cy, w, h) normalized, from
                                        # data_detection/labels/ — grounds the
                                        # agentic-zoom recovery rung
    # Operational state of the instrument itself, independent of whether its
    # dial is legible. Surfaced by the V7 audit: AOBv2-REAL-018 carries a red
    # out-of-service tag (停用证 No. IEA762, 2021-07-05) across the dial, and a
    # perfectly readable value off a decommissioned gauge must not be committed.
    # Neither source catalog records this, so it defaults to "unknown" until the
    # V7 worksheet's `instrument_status` question is adjudicated.
    #   unknown | in_service | out_of_service | calibration_due
    instrument_status: str = "unknown"

    @property
    def gauge_span(self) -> float:
        """Best-effort span; falls back to a generic 100-unit scale when the
        source CSV has UNREADABLE/negative range values (known data-quality
        gap in the seed catalog per Loop1_FinalPlan.md §1.2)."""
        if self.gauge_range_min is None or self.gauge_range_max is None:
            return 100.0
        span = self.gauge_range_max - self.gauge_range_min
        return span if span > 0 else 100.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "category": self.category,
            "asset": self.asset,
            "location": self.location,
            "description": self.description,
            "query_image": str(self.query_image),
            "reference_image": str(self.reference_image),
            "iot_value": self.iot_value,
            "gauge_readable_gt": self.gauge_readable_gt,
            "gauge_value_gt": self.gauge_value_gt,
            "gauge_range_min": self.gauge_range_min,
            "gauge_range_max": self.gauge_range_max,
            "gauge_unit": self.gauge_unit,
            "recommended_action": self.recommended_action,
            "forbidden_actions": self.forbidden_actions,
            "trap": self.trap,
            "instrument_status": self.instrument_status,
        }


def _read_perception_csv(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {row["scenario_id"]: row for row in csv.DictReader(f)}


def _read_pairs_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _extract_members(
    filenames: List[str],
    member_prefix: str,
    zip_path: Path,
    cache_dir: Path,
    required: bool = True,
) -> Dict[str, Path]:
    """Extract only the requested basenames (under member_prefix) from
    data_detection.zip into cache_dir, skipping already-extracted files and
    __MACOSX junk. With required=False, silently skips absent members."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    resolved: Dict[str, Path] = {}
    missing = []
    for name in filenames:
        dest = cache_dir / name
        if dest.exists():
            resolved[name] = dest
        else:
            missing.append(name)
    if not missing:
        return resolved

    if not zip_path.exists():
        raise FileNotFoundError(
            f"PMC archive not found at {zip_path}. Mount the external drive or "
            f"pass --zip to point at data_detection.zip."
        )
    with zipfile.ZipFile(zip_path) as zf:
        member_by_name = {
            Path(m.filename).name: m
            for m in zf.infolist()
            if not Path(m.filename).name.startswith("._")
            and m.filename.startswith(member_prefix)
        }
        for name in missing:
            member = member_by_name.get(name)
            if member is None:
                if required:
                    raise FileNotFoundError(f"{name} not found in {zip_path}")
                continue
            dest = cache_dir / name
            with zf.open(member) as src, dest.open("wb") as out:
                out.write(src.read())
            resolved[name] = dest
    return resolved


def extract_pmc_images(
    filenames: List[str],
    zip_path: Path = DEFAULT_ZIP_PATH,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Dict[str, Path]:
    return _extract_members(filenames, "data_detection/images/", zip_path, cache_dir)


def extract_pmc_labels(
    image_basenames: List[str],
    zip_path: Path = DEFAULT_ZIP_PATH,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Dict[str, Path]:
    """Extract the YOLO label .txt for each image basename (not all images
    have labels; absent ones are skipped)."""
    label_names = [Path(n).stem + ".txt" for n in image_basenames]
    return _extract_members(label_names, "data_detection/labels/", zip_path,
                            cache_dir, required=False)


def _parse_yolo_bbox(label_path: Path) -> Optional[tuple]:
    """First YOLO row -> (cx, cy, w, h), all normalized 0-1."""
    try:
        line = label_path.read_text().strip().splitlines()[0].split()
        return tuple(float(v) for v in line[1:5])
    except (IndexError, ValueError, OSError):
        return None


# Zoom-view margins: level 1 keeps context (bbox x2.5), level 2 is tight
# (bbox x1.4). Crop-zoom is a standoff-reduction PROXY for static photos —
# it cannot add information beyond sensor resolution, which is exactly what
# makes degraded/occluded gauges zoom-immune and the recovery ladder a real
# reasoning problem rather than a free win.
ZOOM_MARGINS = (2.5, 1.4)


def generate_zoom_views(
    query_image: Path,
    gauge_bbox: Optional[tuple],
    cache_dir: Path = DEFAULT_CACHE_DIR,
    min_width: int = 800,
) -> List[Path]:
    """Produce the agentic-zoom view files for one query image:
    <stem>_zoom1.jpg (context crop) and <stem>_zoom2.jpg (tight crop),
    upscaled so the gauge occupies more pixels. Falls back to center crops
    (60% / 35%) when no YOLO bbox exists. Idempotent."""
    from PIL import Image

    views: List[Path] = []
    with Image.open(query_image) as img:
        img_w, img_h = img.size
        for level, margin in enumerate(ZOOM_MARGINS, start=1):
            dest = cache_dir / f"{query_image.stem}_zoom{level}.jpg"
            if dest.exists():
                views.append(dest)
                continue
            if gauge_bbox is not None:
                cx, cy, bw, bh = gauge_bbox
                half_w = bw * margin / 2
                half_h = bh * margin / 2
            else:
                cx, cy = 0.5, 0.5
                frac = (0.60, 0.35)[level - 1]
                half_w = half_h = frac / 2
            left = max(0, int((cx - half_w) * img_w))
            top = max(0, int((cy - half_h) * img_h))
            right = min(img_w, int((cx + half_w) * img_w))
            bottom = min(img_h, int((cy + half_h) * img_h))
            crop = img.crop((left, top, right, bottom))
            if crop.size[0] < min_width:
                scale = min_width / crop.size[0]
                crop = crop.resize((min_width, int(crop.size[1] * scale)),
                                   Image.Resampling.LANCZOS)
            crop.convert("RGB").save(dest, format="JPEG", quality=90)
            views.append(dest)
    return views


def build_scenarios(
    perception_csv: Path = DEFAULT_PERCEPTION_CSV,
    pairs_csv: Path = DEFAULT_PAIRS_CSV,
    zip_path: Path = DEFAULT_ZIP_PATH,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    limit: Optional[int] = None,
) -> List[RealScenario]:
    """Join perception_real.csv (ground truth) with pairs.csv (query/reference
    resolution) and materialize images from the PMC zip. Returns scenarios in
    pairs.csv order (deterministic, matches the user-supplied sample)."""
    catalog = _read_perception_csv(perception_csv)
    pairs = _read_pairs_csv(pairs_csv)
    if limit is not None:
        pairs = pairs[:limit]

    basenames = sorted(
        {Path(p["query_source"]).name for p in pairs}
        | {Path(p["reference_source"]).name for p in pairs}
    )
    local_paths = extract_pmc_images(basenames, zip_path=zip_path, cache_dir=cache_dir)
    label_paths = extract_pmc_labels(
        [Path(p["query_source"]).name for p in pairs],
        zip_path=zip_path, cache_dir=cache_dir)

    scenarios: List[RealScenario] = []
    for pair in pairs:
        row = catalog.get(pair["scenario_id"])
        if row is None:
            continue
        query_name = Path(pair["query_source"]).name
        reference_name = Path(pair["reference_source"]).name
        label_path = label_paths.get(Path(query_name).stem + ".txt")
        gauge_bbox = _parse_yolo_bbox(label_path) if label_path else None
        forbidden = [a.strip() for a in row["forbidden_actions"].split("|") if a.strip()]
        scenarios.append(
            RealScenario(
                scenario_id=row["scenario_id"],
                category=row["category"],
                asset=row["asset"],
                location=row["location"],
                description=row["description"],
                query_image=local_paths[query_name],
                reference_image=local_paths[reference_name],
                iot_value=_parse_numeric(row["iot_value"]),
                iot_value_raw=row["iot_value"],
                gauge_readable_gt=row["gauge_readable"].strip().lower() == "true",
                gauge_value_gt=_parse_numeric(row["gauge_value"]),
                gauge_range_min=_parse_numeric(row["gauge_range_min"]),
                gauge_range_max=_parse_numeric(row["gauge_range_max"]),
                gauge_unit=row["gauge_unit"],
                recommended_action=row["recommended_action"].strip(),
                forbidden_actions=forbidden,
                trap=row["trap"],
                match_tier=pair.get("match_tier", ""),
                gauge_bbox=gauge_bbox,
                instrument_status=(row.get("instrument_status") or "unknown").strip(),
            )
        )
    return scenarios


if __name__ == "__main__":
    for sc in build_scenarios(limit=5):
        print(sc.scenario_id, sc.category, "readable_gt=", sc.gauge_readable_gt,
              "query=", sc.query_image.name)
