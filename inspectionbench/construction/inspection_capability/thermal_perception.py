"""thermal_perception.py — the thermal *interpretation* stage: decode the real
image and describe what it physically shows. Invoked only from
`read_thermal_image` (src/servers/robot/main.py).

This module is the fix for a specific, previously-confirmed gap: prior to
Pass 1, `request_observation(modality="thermal")` resolved a thermal
`ObservationRecord` (asset/inspection/freshness/quality-checked) but only
ever returned its `source_ref` metadata — a path string — never the pixels
it pointed at. No vision provider was wired to consume that path, and the
benchmark question text separately pre-digested the perceptual finding in
prose. The agent could therefore answer correctly without any image ever
being opened.

`observe_thermal_image()` closes that gap by actually opening the file
`source_ref` points at, feeding the *real* bytes through the exact
`_encode_resized()` path `tokenrouter_vision_provider.py` already uses for
gauge images (same PIL.Image.open -> resize -> base64 pipeline; not
reimplemented here), and asking a real VLM what the image shows. It returns
a sha256 of the raw file bytes alongside the observation specifically so a
caller (or a test) can prove decoding actually happened, rather than taking
"status: RESOLVED" on faith.

**Perception is separated from decision (Pass 2).** This module replaces the
*conventional thermal interpretation stage* — the step a radiometric camera's
ROI/alarm logic would perform — and nothing more. It reports what the image
supports (`thermal_pattern`, `hotspot_present`, `hotspot_location`,
`spatial_pattern`, `relative_temperature`, `observation_quality`,
`evidence_confidence`) and deliberately emits **no** `action`, **no**
`verdict`, and **no** dataset `fault_class`. Choosing between COMMIT /
SCHEDULE_MAINTENANCE / SHUTDOWN / ESCALATE is the *industrial operational
decision*, which belongs to the agent reasoning over this observation plus
IoT/enterprise context — not to the perception model. Pass 1 collapsed the
two, which made the benchmark unable to measure the decision at all. The
separation is pinned by a structural test.

Deliberately NOT reused: `TokenRouterVisionProvider` itself. That class is
wired to the ROSClaw `Provider`/`EventBus` async contract and the
gauge-reading prompt/parser pair (`resolve_prompt`, `_parse_json_response`
from `gemini_vision_provider`) — both gauge-specific. Pulling in the full
provider would mean either misusing the gauge prompt for a thermal image or
constructing an unused EventBus just to satisfy its constructor. This module
reuses exactly the one function that generalizes (`_encode_resized`) and its
reference+query two-image block shape, and makes its own synchronous HTTP
call with a thermal-specific prompt.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_ORCH_SRC = Path(__file__).resolve().parents[1]
if str(_ORCH_SRC) not in sys.path:
    sys.path.insert(0, str(_ORCH_SRC))

from tokenrouter_vision_provider import _encode_resized  # noqa: E402

__all__ = [
    "DEFAULT_MODEL",
    "ImageResolutionError",
    "ThermalClassificationError",
    "ThermalObservation",
    "resolve_image_path",
    "read_image_bytes",
    "observe_thermal_image",
]

DEFAULT_MODEL = "z-ai/glm-4.6v"

#: Fields a thermal observation may legitimately contain. Used by
#: main.py's structural test and by ThermalObservation itself -- neither
#: "action", "verdict", nor "fault_class" appear here, on purpose.
OBSERVATION_FIELDS = (
    "thermal_pattern", "hotspot_present", "hotspot_location", "spatial_pattern",
    "relative_temperature", "observation_quality", "evidence_confidence",
)

# Deliberately does NOT restate any of R049/R050's pre-digested findings
# (thermal_profile, hotspot_detected, hotspot_location, fault_class) --
# those were exactly the text the agent used to answer without looking at
# the image. This prompt asks the VLM to determine all of that itself from
# the attached pixels.
#
# Deliberately does NOT ask for a verdict, an action, or any fault-class
# vocabulary (Pass 2): this is the *perception* stage only, mirroring what a
# conventional thermal-camera ROI/alarm module would report -- a physical
# description of the heat pattern, not an operational decision. The agent
# combines this observation with IoT/enterprise context to decide the
# action; folding that decision in here would silently re-create Pass 1's
# monolith.
_THERMAL_PROMPT = """\
You are inspecting a thermal (infrared) image of an induction motor captured \
by a robot's egocentric thermal camera during a physical inspection.

Look at the attached thermal image directly and describe, physically, what \
the temperature distribution shows. Do NOT decide what operational action \
should follow -- only describe the physical evidence.

Known traps: minor warmth at bearing/brush contact points is a normal, \
symmetric artifact and must not be reported as a hotspot. A genuine fault \
hotspot is asymmetric relative to the motor's normal symmetric heat pattern.

Return ONLY a JSON object with exactly these fields:
{
  "thermal_pattern": "uniform|non_uniform",
  "hotspot_present": true|false,
  "hotspot_location": "<location string, or null if none present>"
    e.g. "rotor windings", "fan/ventilation area", "phase winding B",
    "bearing/brush contacts (symmetric, not a hotspot)",
  "spatial_pattern": "symmetric|asymmetric",
  "relative_temperature": "<qualitative description relative to the rest of the motor body, e.g. 'localized region markedly hotter than surrounding stator', 'uniform, no region elevated'>",
  "observation_quality": <0.0-1.0, your confidence that the image itself is clear/legible enough to interpret>,
  "evidence_confidence": <0.0-1.0, your confidence in the physical description above>
}

This is a PERCEPTION report only -- do not include "verdict", "action", or
any fault classification. Describe only what the pixels show."""


class ImageResolutionError(Exception):
    """Raised when a resolved observation's source_ref does not point at a
    real, readable file on disk. Never caught silently -- callers must
    surface this as a failed/UNAVAILABLE result, not fabricate a
    classification."""


class ThermalClassificationError(RuntimeError):
    """Raised when the image WAS successfully decoded (bytes read, sha256
    computed) but the model call itself failed -- e.g. no API key, or a
    network/HTTP error. Carries the decode-proof fields so a caller can
    still report that real pixels were read even though classification
    failed, rather than losing that evidence behind a bare exception."""

    def __init__(self, message: str, *, image_bytes_len: int, image_sha256: str, image_path: str):
        super().__init__(message)
        self.image_bytes_len = image_bytes_len
        self.image_sha256 = image_sha256
        self.image_path = image_path


@dataclass
class ThermalObservation:
    """A physical description of what a thermal image shows -- perception
    output only. No `verdict`, `action`, or `fault_class` field exists here
    on purpose (see module docstring); `OBSERVATION_FIELDS` above is the
    authoritative list this dataclass's own field set must match."""
    thermal_pattern: str            # "uniform" | "non_uniform"
    hotspot_present: bool
    hotspot_location: Optional[str]
    spatial_pattern: str            # "symmetric" | "asymmetric"
    relative_temperature: str
    observation_quality: float
    evidence_confidence: float
    model_name: str
    image_path: str
    image_bytes_len: int
    image_sha256: str
    reference_image_path: Optional[str]
    raw_response: str
    latency_ms: int


def resolve_image_path(source_ref: dict[str, Any]) -> Path:
    """source_ref["value"] is a path like "data/industrial_cache/motor/r030.bmp",
    relative to src/orchestrator/ (see build_observation_records.py, which
    writes these paths relative to the RobotInspection asset root that
    industrial_cache is symlinked alongside). Resolves to an absolute path;
    does NOT check existence -- that is read_image_bytes()'s job, so a
    missing file is reported with one clear error rather than two.
    """
    value = source_ref.get("value") or ""
    p = Path(value)
    return p if p.is_absolute() else (_ORCH_SRC / p).resolve()


def read_image_bytes(path: Path) -> bytes:
    if not path.exists() or not path.is_file():
        raise ImageResolutionError(f"resolved thermal image path does not exist on disk: {path}")
    data = path.read_bytes()
    if not data:
        raise ImageResolutionError(f"resolved thermal image file is empty: {path}")
    return data


def _parse_json_response(raw_text: str) -> dict[str, Any]:
    """Same tolerant extraction gemini_vision_provider/_parse_json_response
    use for chat-completion output that may wrap JSON in prose or a
    ```json fence -- reimplemented minimally here rather than imported,
    since that helper is bound to the gauge-reading field set."""
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in model response: {raw_text[:300]!r}")
    return json.loads(match.group(0))


def observe_thermal_image(
    image_path: Path,
    *,
    reference_path: Optional[Path] = None,
    model_name: str = DEFAULT_MODEL,
    api_key: str = "",
    base_url: str = "",
    temperature: float = 0.2,
    max_retries: int = 4,
    timeout_s: float = 90.0,
) -> ThermalObservation:
    """Actually opens `image_path`, encodes it via the real
    tokenrouter_vision_provider._encode_resized() pipeline, and sends it to
    a live VLM to describe what it physically shows -- perception only, no
    decision (see module docstring). Raises ImageResolutionError if the file
    cannot be read, and ThermalClassificationError if the model call itself
    fails (both left uncaught deliberately -- the caller decides how to
    represent failure, this function never invents an observation).

    `reference_path`, if given, is encoded and attached as a second image
    (reference, then query -- the same block order
    `tokenrouter_vision_provider._call_sync` already uses for its
    reference+query pair) so the VLM can compare against a known-normal
    frame rather than describing the query image in isolation.
    """
    raw = read_image_bytes(image_path)
    image_sha256 = hashlib.sha256(raw).hexdigest()

    try:
        return _observe_decoded(
            image_path, raw=raw, image_sha256=image_sha256, reference_path=reference_path,
            model_name=model_name, api_key=api_key, base_url=base_url, temperature=temperature,
            max_retries=max_retries, timeout_s=timeout_s,
        )
    except ThermalClassificationError:
        raise
    except Exception as exc:
        # The image WAS genuinely decoded (raw/image_sha256 above prove
        # it) -- whatever failed next (missing key, network, HTTP error,
        # unparseable response) is a perception-call failure, not a decode
        # failure. Re-raised carrying the decode proof so the caller can
        # still report it rather than losing that evidence behind a bare
        # exception.
        raise ThermalClassificationError(
            str(exc), image_bytes_len=len(raw), image_sha256=image_sha256, image_path=str(image_path),
        ) from exc


def _observe_decoded(
    image_path: Path, *, raw: bytes, image_sha256: str, reference_path: Optional[Path],
    model_name: str, api_key: str, base_url: str, temperature: float,
    max_retries: int, timeout_s: float,
) -> ThermalObservation:
    api_key = api_key or os.environ.get("TOKENROUTER_API_KEY", "")
    base_url = (base_url or os.environ.get(
        "TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1")).rstrip("/")
    if not api_key:
        raise RuntimeError("TOKENROUTER_API_KEY not set -- cannot observe thermal image")

    # The exact decode/resize/base64 path tokenrouter_vision_provider.py
    # already uses for gauge images -- not reimplemented.
    query_b64 = _encode_resized(image_path)

    content: list[dict[str, Any]] = []
    if reference_path is not None:
        reference_b64 = _encode_resized(reference_path)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{reference_b64}"}})
        prompt_text = (
            "The FIRST image below is a known-normal reference thermal frame of this same "
            "asset class. The SECOND image is the query frame under inspection -- compare "
            "against the reference where useful.\n\n" + _THERMAL_PROMPT
        )
    else:
        prompt_text = _THERMAL_PROMPT
    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{query_b64}"}})
    content.append({"type": "text", "text": prompt_text})

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": 4096,
    }

    def _build_req():
        return urllib.request.Request(
            f"{base_url}/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})

    t0 = time.time()
    req = _build_req()
    last_exc: Optional[Exception] = None
    raw_text = ""
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                d = json.loads(r.read())
            raw_text = d["choices"][0]["message"]["content"] or ""
            break
        except urllib.error.HTTPError as e:  # pragma: no cover - network path
            last_exc = e
            err_text = e.read().decode(errors="ignore")
            if "temperature" in err_text and "deprecated" in err_text:
                payload.pop("temperature", None)
                req = _build_req()
                continue
            if e.code in (429, 500, 502, 503) and attempt < max_retries - 1:
                time.sleep(6 * (attempt + 1))
            else:
                raise RuntimeError(f"TokenRouter HTTP {e.code}: {err_text[:300]}") from e
        except Exception as e:  # pragma: no cover - network path
            last_exc = e
            if attempt < max_retries - 1:
                time.sleep(6 * (attempt + 1))
            else:
                raise
    else:
        raise last_exc  # pragma: no cover

    latency_ms = int((time.time() - t0) * 1000)
    parsed = _parse_json_response(raw_text)

    # Only OBSERVATION_FIELDS are ever read out of the model's response --
    # a model that disobeys the prompt and emits "verdict"/"action"/
    # "fault_class" anyway simply has those keys ignored here. The
    # dataclass has no field to carry them even if we wanted to.
    return ThermalObservation(
        thermal_pattern=str(parsed.get("thermal_pattern", "unknown")),
        hotspot_present=bool(parsed.get("hotspot_present", False)),
        hotspot_location=parsed.get("hotspot_location"),
        spatial_pattern=str(parsed.get("spatial_pattern", "unknown")),
        relative_temperature=str(parsed.get("relative_temperature", "")),
        observation_quality=float(parsed.get("observation_quality", 0.5)),
        evidence_confidence=float(parsed.get("evidence_confidence", 0.5)),
        model_name=model_name,
        image_path=str(image_path),
        image_bytes_len=len(raw),
        image_sha256=image_sha256,
        reference_image_path=str(reference_path) if reference_path is not None else None,
        raw_response=raw_text,
        latency_ms=latency_ms,
    )
