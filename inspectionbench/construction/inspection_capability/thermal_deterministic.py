"""thermal_deterministic.py — Part 16A: T0, the *Fixed deterministic
perception baseline*. Real pixels/temperatures -> fixed, never-tuned
thresholds -> a genuine `ThermalObservation` -- no model call, no network,
no randomness. Same downstream contract as thermal_perception.py's VLM
path (T1): identical dataclass, identical OBSERVATION_FIELDS, feeds the
same commit_thermal_decision / score_l3_grounded.

**T0 has two variants and they are NOT interchangeable language.** Which one
runs is a property of the INPUT DATA, decided automatically by attempting a
real radiometric decode -- never chosen by the caller:

  T0a -- non-radiometric (Mendeley BMPs: R049/R050/R088/R089's source
         images have zero FLIR/FFF/EXIF markers, confirmed this pass).
         Operates on FALSE-COLOUR PIXEL INTENSITY. Its thresholds are
         image-intensity/spatial-statistic thresholds. They must NEVER be
         called "temperature thresholds" or "physics thresholds" in any
         report -- there is no temperature in these files to threshold.
         A weak T0a result is evidence about a non-radiometric
         image-statistic baseline, and NOTHING ELSE. It is not evidence
         about thermal/temperature reasoning.

  T0b -- radiometric (REVA, rotating-electromechanical IR -- both verified
         genuinely radiometric via flir_radiometric.py). Operates on REAL
         DEGREES CELSIUS. Its thresholds are legitimately physics-informed.

Both variants share one feature-extraction code path (`_frame_features`);
only the UNITS of the input 2D array differ (intensity counts vs degC),
which is exactly why the threshold CONSTANTS differ and why the two must be
reported under different names. `computed_features["baseline_variant"]` is
"T0a" or "T0b" and `computed_features["radiometric"]` is a bool -- both are
diagnostics, never surfaced in OBSERVATION_FIELDS.

Reused, not reimplemented:
  - flir_radiometric.read_thermal_celsius (Pass 3/16A) for T0b's degC plane.
  - thermal_baselines._quadrant_means (Pass 2) for the spatial-asymmetry
    feature, on whichever unit the input array is already in.
Both are loaded by direct file path (importlib.util.spec_from_file_location),
matching acoustic_perception.py's precedent -- NOT via a sys.path insert,
which previously caused a real cross-module collision bug (see Pass 5's
plan entry) when a directory containing its own main.py was added to
sys.path. Neither src/orchestrator/data/ nor experiments/thermal_grounding/
contains a main.py (verified), but the safe pattern is used uniformly
regardless, since that is exactly the kind of assumption that broke once
already.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

# Relative import, not a sys.path + bare `from thermal_perception import`:
# thermal_deterministic.py is always loaded as a member of the
# inspection_capability PACKAGE (main.py imports it as
# `inspection_capability.thermal_deterministic`), and a bare/sys.path-based
# import of a sibling module creates a SEPARATE module object under a
# different name -- its classes then fail isinstance() against the
# packaged-import version main.py actually uses, even though attribute
# access still works (a real bug this module's own tests caught: fixed
# here rather than left as a subtle isinstance trap).
from .thermal_perception import (  # noqa: E402  -- same dataclass/contract as T1
    OBSERVATION_FIELDS,
    ThermalObservation,
    ImageResolutionError,
    read_image_bytes,
)


def _load_by_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    # Required for `@dataclass` classes under `from __future__ import
    # annotations` (flir_radiometric.FlirCalibration): dataclasses'
    # postponed-annotation resolution looks the module up via
    # sys.modules[cls.__module__], which is empty unless the module is
    # registered here BEFORE exec_module runs.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_REPO_ROOT = Path(__file__).resolve().parents[3]
_flir = _load_by_path(
    "_thermal_deterministic_flir_radiometric",
    _REPO_ROOT / "src" / "orchestrator" / "data" / "flir_radiometric.py",
)
_baselines = _load_by_path(
    "_thermal_deterministic_baselines",
    _REPO_ROOT / "experiments" / "thermal_grounding" / "thermal_baselines.py",
)
read_thermal_celsius = _flir.read_thermal_celsius
FlirRadiometricError = _flir.FlirRadiometricError
_quadrant_means = _baselines._quadrant_means

__all__ = [
    "observe_thermal_deterministic",
    "compute_deterministic_observation",
]

DETERMINISTIC_MODEL_NAME = "deterministic-v1"  # not a model, but fills ThermalObservation.model_name

# ---------------------------------------------------------------------------
# T0a thresholds -- non-radiometric, false-colour PIXEL INTENSITY (0-255
# per-pixel brightest channel, same extraction thermal_baselines.py uses).
# NEVER call these temperature or physics thresholds.
# ---------------------------------------------------------------------------
TAU_INTENSITY_HOTSPOT = 40.0    # (p99 - p50) intensity units
TAU_INTENSITY_ASYMMETRY = 15.0  # quadrant_asymmetry_index intensity units
TAU_INTENSITY_UNIFORM = 25.0    # (p95 - p50) intensity units

# ---------------------------------------------------------------------------
# T0b thresholds -- radiometric, real DEGREES CELSIUS. Legitimately
# physics-informed: a >8C hotspot above the frame's own median, or a
# >6C quadrant spread, is a physically meaningful localized temperature
# rise on a motor housing regardless of ambient (REVA/rotating span very
# different ambients -- ~20C for REVA, ~22-27C for rotating -- so an
# ABSOLUTE degC threshold would be wrong; these are all RELATIVE to the
# frame's own median, same normalization T0a uses on intensity).
# ---------------------------------------------------------------------------
TAU_CELSIUS_HOTSPOT = 8.0
TAU_CELSIUS_ASYMMETRY = 6.0
TAU_CELSIUS_UNIFORM = 5.0


def _frame_features(arr: np.ndarray) -> dict[str, float]:
    """Shared feature extraction -- works identically on an intensity array
    (T0a) or a degC array (T0b); only the caller's choice of threshold
    constants differs afterward."""
    flat = arr.astype(np.float64).ravel()
    p50, p95, p99 = np.percentile(flat, [50, 95, 99])
    quadrants = _quadrant_means(arr)
    quad_values = list(quadrants.values())
    return {
        "p50": float(p50), "p95": float(p95), "p99": float(p99),
        "max": float(flat.max()), "mean": float(flat.mean()),
        "hotspot_margin": float(p99 - p50),
        "uniform_margin": float(p95 - p50),
        "quadrant_means": quadrants,
        "quadrant_asymmetry_index": float(max(quad_values) - min(quad_values)),
    }


def _classify(features: dict[str, float], *, tau_hotspot: float,
             tau_asym: float, tau_uniform: float) -> dict[str, Any]:
    """A-priori thresholds only -- fixed constants in, categorical
    ThermalObservation fields out. Never fit to any label."""
    hotspot_present = features["hotspot_margin"] > tau_hotspot
    spatial_pattern = "asymmetric" if features["quadrant_asymmetry_index"] > tau_asym else "symmetric"
    thermal_pattern = "non_uniform" if features["uniform_margin"] > tau_uniform else "uniform"

    hotspot_location: Optional[str] = None
    if hotspot_present:
        hotspot_location = max(features["quadrant_means"], key=features["quadrant_means"].get)

    # Confidence: a fixed monotone function of margin-to-threshold, not a
    # fitted probability. Saturates at 0.95 so it is never literally certain.
    margin_ratio = features["hotspot_margin"] / max(tau_hotspot, 1e-6)
    evidence_confidence = float(min(0.95, 0.5 + 0.15 * min(margin_ratio, 3.0)))

    return {
        "thermal_pattern": thermal_pattern,
        "hotspot_present": hotspot_present,
        "hotspot_location": hotspot_location,
        "spatial_pattern": spatial_pattern,
        "relative_temperature": (
            f"p99-p50 margin={features['hotspot_margin']:.1f} "
            f"(threshold={tau_hotspot:.1f}); quadrant_asymmetry={features['quadrant_asymmetry_index']:.1f} "
            f"(threshold={tau_asym:.1f})"
        ),
        "observation_quality": 0.9,  # the image itself decoded successfully; this is not a model confidence
        "evidence_confidence": evidence_confidence,
    }


def compute_deterministic_observation(image_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Real decode -> real feature extraction -> a-priori classification.
    Returns (observation_fields, computed_features). Tries a genuine
    radiometric decode first (T0b); falls back to false-colour intensity
    (T0a) ONLY when the file carries no radiometric payload -- this is a
    property of the file, not a caller choice.
    """
    try:
        celsius_plane, calib = read_thermal_celsius(image_path)
        finite = celsius_plane[np.isfinite(celsius_plane)]
        if finite.size == 0:
            raise FlirRadiometricError("radiometric plane decoded but contains no finite values")
        features = _frame_features(np.where(np.isfinite(celsius_plane), celsius_plane, np.nanmedian(finite)))
        obs = _classify(features, tau_hotspot=TAU_CELSIUS_HOTSPOT,
                        tau_asym=TAU_CELSIUS_ASYMMETRY, tau_uniform=TAU_CELSIUS_UNIFORM)
        features["baseline_variant"] = "T0b"
        features["radiometric"] = True
        features["emissivity"] = calib.emissivity
        return obs, features
    except FlirRadiometricError:
        pass  # fall through to T0a -- this file has no radiometric payload

    img = Image.open(image_path)
    intensity = np.array(img.convert("RGB")).max(axis=2).astype(np.float64)
    features = _frame_features(intensity)
    obs = _classify(features, tau_hotspot=TAU_INTENSITY_HOTSPOT,
                    tau_asym=TAU_INTENSITY_ASYMMETRY, tau_uniform=TAU_INTENSITY_UNIFORM)
    features["baseline_variant"] = "T0a"
    features["radiometric"] = False
    return obs, features


def observe_thermal_deterministic(
    image_path: Path, *, reference_path: Optional[Path] = None,
) -> ThermalObservation:
    """Same call shape as thermal_perception.observe_thermal_image (drop-in
    for the T0 arm), but fully deterministic -- no model_name/api_key/
    base_url/temperature/retries/timeout params, because none apply.
    reference_path is accepted for interface parity but unused: T0's
    thresholds are per-frame, not comparative (unlike IMAGE_REFERENCE's VLM
    path, there is no learned notion of "compare to this reference" here).
    """
    raw = read_image_bytes(image_path)
    image_sha256 = hashlib.sha256(raw).hexdigest()
    obs_fields, features = compute_deterministic_observation(image_path)

    return ThermalObservation(
        thermal_pattern=obs_fields["thermal_pattern"],
        hotspot_present=obs_fields["hotspot_present"],
        hotspot_location=obs_fields["hotspot_location"],
        spatial_pattern=obs_fields["spatial_pattern"],
        relative_temperature=obs_fields["relative_temperature"],
        observation_quality=obs_fields["observation_quality"],
        evidence_confidence=obs_fields["evidence_confidence"],
        model_name=f"{DETERMINISTIC_MODEL_NAME}-{features['baseline_variant']}",
        image_path=str(image_path),
        image_bytes_len=len(raw),
        image_sha256=image_sha256,
        reference_image_path=str(reference_path) if reference_path is not None else None,
        raw_response="",  # no model call was made; nothing to log as a raw response
        latency_ms=0,
    )
