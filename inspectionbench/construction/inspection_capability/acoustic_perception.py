"""acoustic_perception.py — the acoustic *interpretation* stage: decode the
real WAV, compute real spectral features, and describe what the sound
physically shows. Invoked only from `read_acoustic`
(src/servers/robot/main.py). PASS 5 (acoustic un-gating), mirrors
thermal_perception.py's structure and its perception/decision split exactly.

Why features, not raw audio, reach the model: unlike `read_thermal_image`,
which sends real pixels to a vision-capable chat-completions endpoint via
`_encode_resized()`, there is no verified audio-input contract on the
TokenRouter chat-completions API this project already depends on (never
tested, not assumed here). Per the Pass 4 audit's own conclusion about
src/servers/vibration/ — "numpy stays server-side, only compact summaries
reach the LLM ... exactly the right pattern" — this module reuses that
template for acoustic: the WAV is genuinely decoded and FFT'd with real
signal-processing code (`src/servers/vibration/dsp/fft_analysis.py`'s
`compute_fft`, not reimplemented), and the LLM receives the *computed
numbers* (dominant frequency, band energy, RMS level in dB, SNR estimate),
not a picture of a waveform. This is also physically closer to how a real
acoustic-imaging payload (e.g. Fluke SV600) already outputs a spatial/
spectral map rather than a photograph -- the model is being asked to do the
same interpretive step a real acoustic-inspection workflow performs on
already-processed sensor output.

**Perception is separated from decision**, identically to thermal: this
module reports `acoustic_pattern`, `anomaly_present`, `dominant_frequency_hz`,
`band_energy_db`, `snr_estimate_db`, `observation_quality`,
`evidence_confidence` and emits **no** `action`, **no** `verdict`, and
**no** dataset label (MIMII's own normal/abnormal split, or machine
id/type) — pinned by a structural test mirroring
TestPerceptionHasNoDecisionFields.
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
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

_ORCH_SRC = Path(__file__).resolve().parents[1]
if str(_ORCH_SRC) not in sys.path:
    sys.path.insert(0, str(_ORCH_SRC))


def _load_compute_fft():
    """Load src/servers/vibration/dsp/fft_analysis.compute_fft by file path,
    NOT via sys.path + `import dsp.fft_analysis` -- that approach was tried
    first and caused a real bug: it requires putting
    src/servers/vibration/ on sys.path, and since that directory ALSO
    contains its own main.py, every later bare `import main` in this
    process (couchdb_executor.py's `import main as robot`) could then
    resolve to the WRONG main.py depending on sys.path order, silently
    breaking the robot MCP server's own tools. fft_analysis.py has no
    relative/package imports (only numpy + scipy), so it loads cleanly by
    direct file path under a private module name, touching sys.path not at
    all."""
    import importlib.util

    fft_path = (
        Path(__file__).resolve().parents[3] / "src" / "servers" / "vibration" / "dsp" / "fft_analysis.py"
    )
    module_name = "_acoustic_perception_fft_analysis"
    spec = importlib.util.spec_from_file_location(module_name, fft_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load fft_analysis.py from {fft_path}")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec_module: fft_analysis.py has no dataclasses
    # today so this was latent, not observed -- but thermal_deterministic.py
    # hit the real failure mode (dataclasses' postponed-annotation
    # resolution needs sys.modules[cls.__module__] populated) on a sibling
    # file loaded the same way, so this loader is fixed proactively too.
    import sys
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.compute_fft


compute_fft = _load_compute_fft()  # reused, not reimplemented

__all__ = [
    "DEFAULT_MODEL",
    "AudioResolutionError",
    "AcousticClassificationError",
    "AcousticObservation",
    "OBSERVATION_FIELDS",
    "resolve_audio_path",
    "read_audio_bytes",
    "decode_wav",
    "compute_acoustic_features",
    "observe_acoustic",
]

DEFAULT_MODEL = "z-ai/glm-4.6"  # text model -- this stage sends computed numbers, not pixels/audio

#: Fields an acoustic observation may legitimately contain. Mirrors
#: thermal_perception.OBSERVATION_FIELDS -- "anomaly_present" is a
#: perceptual read of the FEATURES (is there something unusual about this
#: spectrum), never MIMII's own ground-truth label, which this module never
#: sees.
OBSERVATION_FIELDS = (
    "acoustic_pattern", "anomaly_present", "dominant_frequency_hz",
    "band_energy_db", "snr_estimate_db", "observation_quality", "evidence_confidence",
)

_ACOUSTIC_PROMPT_TEMPLATE = """\
You are inspecting acoustic sensor evidence from an industrial asset, captured \
by a robot-mounted acoustic payload during a physical inspection.

The raw audio has ALREADY been decoded and spectrally analyzed by real signal \
processing (FFT). You are given the computed features below -- interpret them \
physically. Do NOT decide what operational action should follow -- only \
describe what the evidence shows.

Computed acoustic features (real, from the actual recording):
  duration_s: {duration_s:.2f}
  sample_rate_hz: {sample_rate_hz}
  rms_level_db: {rms_level_db:.1f}
  dominant_frequency_hz: {dominant_frequency_hz:.1f}
  dominant_frequency_magnitude_db: {dominant_freq_mag_db:.1f}
  spectral_centroid_hz: {spectral_centroid_hz:.1f}
  band_energy_db: {band_energy_json}
  crest_factor: {crest_factor:.2f}

Known traps: a single strong low-frequency tone near typical motor line/rotation \
frequencies can be a normal running signature, not a fault -- broadband energy \
increase or a NEW frequency component not explained by the nominal operating \
condition is the more reliable anomaly signal. A high crest factor (>6-8) \
suggests impulsive/transient content (e.g. a leak hiss or impact), not a pure tone.

Return ONLY a JSON object with exactly these fields:
{{
  "acoustic_pattern": "tonal|broadband|impulsive|mixed|quiet",
  "anomaly_present": true|false,
  "dominant_frequency_hz": <float, the frequency you judge most acoustically significant>,
  "band_energy_db": <float, overall energy level relevant to your judgement>,
  "snr_estimate_db": <float, your estimate of signal-to-background ratio>,
  "observation_quality": <0.0-1.0, confidence the recording itself is clear enough to interpret>,
  "evidence_confidence": <0.0-1.0, confidence in the physical description above>
}}

This is a PERCEPTION report only -- do not include "verdict", "action", or
any fault/machine classification. Describe only what the computed evidence shows."""


class AudioResolutionError(Exception):
    """Raised when a resolved observation's source_ref does not point at a
    real, readable WAV file on disk. Never caught silently."""


class AcousticClassificationError(RuntimeError):
    """Raised when the audio WAS successfully decoded and features WERE
    computed but the model call itself failed. Carries the decode-proof
    fields so a caller can still report real evidence was processed."""

    def __init__(self, message: str, *, audio_bytes_len: int, audio_sha256: str, audio_path: str):
        super().__init__(message)
        self.audio_bytes_len = audio_bytes_len
        self.audio_sha256 = audio_sha256
        self.audio_path = audio_path


@dataclass
class AcousticObservation:
    """A physical description of what acoustic evidence shows -- perception
    output only. No `verdict`, `action`, or dataset label field exists here
    on purpose (see module docstring)."""
    acoustic_pattern: str          # "tonal" | "broadband" | "impulsive" | "mixed" | "quiet"
    anomaly_present: bool
    dominant_frequency_hz: float
    band_energy_db: float
    snr_estimate_db: float
    observation_quality: float
    evidence_confidence: float
    model_name: str
    audio_path: str
    audio_bytes_len: int
    audio_sha256: str
    computed_features: dict[str, Any]
    raw_response: str
    latency_ms: int


def resolve_audio_path(source_ref: dict[str, Any]) -> Path:
    """Identical convention to thermal_perception.resolve_image_path:
    relative paths resolve against src/orchestrator/; absolute paths pass
    through untouched (MIMII files live outside the repo, on second_drive,
    like REVA/rotating in Pass 3)."""
    value = source_ref.get("value") or ""
    p = Path(value)
    return p if p.is_absolute() else (_ORCH_SRC / p).resolve()


def read_audio_bytes(path: Path) -> bytes:
    if not path.exists() or not path.is_file():
        raise AudioResolutionError(f"resolved acoustic file does not exist on disk: {path}")
    data = path.read_bytes()
    if not data:
        raise AudioResolutionError(f"resolved acoustic file is empty: {path}")
    return data


def decode_wav(raw: bytes) -> tuple[np.ndarray, int]:
    """Real WAV decode via the stdlib `wave` module (no assumption of a
    heavier audio library) -- returns (mono float32 samples in [-1, 1],
    sample_rate_hz). Raises AudioResolutionError for anything that isn't a
    genuinely readable PCM WAV; never fabricates a signal."""
    import io
    try:
        with wave.open(io.BytesIO(raw), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            frames = wf.readframes(n_frames)
    except Exception as exc:
        raise AudioResolutionError(f"could not decode WAV data: {exc}") from exc

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    if sampwidth not in dtype_map:
        raise AudioResolutionError(f"unsupported WAV sample width: {sampwidth} bytes")
    samples = np.frombuffer(frames, dtype=dtype_map[sampwidth]).astype(np.float64)
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)  # downmix to mono
    max_val = float(2 ** (8 * sampwidth - 1))
    samples = (samples / max_val).astype(np.float32)
    return samples, framerate


def compute_acoustic_features(samples: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Real spectral analysis via compute_fft (reused from the vibration
    DSP module, not reimplemented). All numbers here are genuinely computed
    from the decoded waveform -- nothing here is looked up from a label."""
    if samples.size == 0:
        raise AudioResolutionError("decoded WAV has zero samples")

    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    rms_db = 20.0 * np.log10(max(rms, 1e-12))
    peak = float(np.max(np.abs(samples)))
    crest_factor = float(peak / max(rms, 1e-12))

    fft_result = compute_fft(samples.astype(np.float64), fs=float(sample_rate))
    freqs = fft_result["frequencies"]
    mags_db = fft_result["magnitude_db"]
    mag_linear = fft_result["magnitude"]

    peak_idx = int(np.argmax(mag_linear[1:])) + 1 if len(mag_linear) > 1 else 0  # skip DC
    dominant_freq = float(freqs[peak_idx]) if len(freqs) else 0.0
    dominant_mag_db = float(mags_db[peak_idx]) if len(mags_db) else -120.0

    centroid = (
        float(np.sum(freqs * mag_linear) / max(np.sum(mag_linear), 1e-12))
        if len(freqs) else 0.0
    )

    # Coarse third-octave-ish band split for a compact, model-readable summary.
    band_edges = [(0, 500), (500, 2000), (2000, 8000), (8000, sample_rate // 2)]
    band_energy_db: dict[str, float] = {}
    for lo, hi in band_edges:
        mask = (freqs >= lo) & (freqs < hi)
        band_energy = float(np.sum(mag_linear[mask] ** 2)) if mask.any() else 0.0
        band_energy_db[f"{lo}-{hi}Hz"] = 10.0 * np.log10(max(band_energy, 1e-12))

    return {
        "duration_s": float(len(samples) / sample_rate),
        "sample_rate_hz": int(sample_rate),
        "rms_level_db": rms_db,
        "dominant_frequency_hz": dominant_freq,
        "dominant_freq_mag_db": dominant_mag_db,
        "spectral_centroid_hz": centroid,
        "band_energy_db": band_energy_db,
        "crest_factor": crest_factor,
    }


def _parse_json_response(raw_text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in model response: {raw_text[:300]!r}")
    return json.loads(match.group(0))


def observe_acoustic(
    audio_path: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    api_key: str = "",
    base_url: str = "",
    temperature: float = 0.2,
    max_retries: int = 4,
    timeout_s: float = 90.0,
) -> AcousticObservation:
    """Actually decodes `audio_path`, computes real spectral features, and
    sends those features to a live text LLM to interpret -- perception only,
    no decision. Raises AudioResolutionError if the file cannot be
    read/decoded, and AcousticClassificationError if the model call itself
    fails."""
    raw = read_audio_bytes(audio_path)
    audio_sha256 = hashlib.sha256(raw).hexdigest()
    samples, sample_rate = decode_wav(raw)
    features = compute_acoustic_features(samples, sample_rate)

    try:
        return _observe_features(
            audio_path, raw=raw, audio_sha256=audio_sha256, features=features,
            model_name=model_name, api_key=api_key, base_url=base_url, temperature=temperature,
            max_retries=max_retries, timeout_s=timeout_s,
        )
    except AcousticClassificationError:
        raise
    except Exception as exc:
        raise AcousticClassificationError(
            str(exc), audio_bytes_len=len(raw), audio_sha256=audio_sha256, audio_path=str(audio_path),
        ) from exc


def _observe_features(
    audio_path: Path, *, raw: bytes, audio_sha256: str, features: dict[str, Any],
    model_name: str, api_key: str, base_url: str, temperature: float,
    max_retries: int, timeout_s: float,
) -> AcousticObservation:
    api_key = api_key or os.environ.get("TOKENROUTER_API_KEY", "")
    base_url = (base_url or os.environ.get(
        "TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1")).rstrip("/")
    if not api_key:
        raise RuntimeError("TOKENROUTER_API_KEY not set -- cannot observe acoustic evidence")

    prompt_text = _ACOUSTIC_PROMPT_TEMPLATE.format(
        duration_s=features["duration_s"], sample_rate_hz=features["sample_rate_hz"],
        rms_level_db=features["rms_level_db"], dominant_frequency_hz=features["dominant_frequency_hz"],
        dominant_freq_mag_db=features["dominant_freq_mag_db"],
        spectral_centroid_hz=features["spectral_centroid_hz"],
        band_energy_json=json.dumps({k: round(v, 1) for k, v in features["band_energy_db"].items()}),
        crest_factor=features["crest_factor"],
    )
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": temperature,
        # PASS R3 fix: 2048 was silently too small for z-ai/glm-4.6, a
        # reasoning model whose "reasoning_content" tokens count against
        # max_tokens -- a real prompt (unlike a trivial "Say OK") consumed
        # the entire budget on reasoning and returned content="" with
        # finish_reason="length", which then failed JSON parsing downstream.
        # Measured directly against this prompt: reasoning alone can run
        # long enough that even 4096 occasionally still truncates before any
        # content token is emitted (observed real latency ~88s at 180s
        # timeout for one successful call) -- 8192 gives real headroom
        # rather than chasing the exact threshold empirically.
        "max_tokens": 8192,
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

    return AcousticObservation(
        acoustic_pattern=str(parsed.get("acoustic_pattern", "unknown")),
        anomaly_present=bool(parsed.get("anomaly_present", False)),
        dominant_frequency_hz=float(parsed.get("dominant_frequency_hz", features["dominant_frequency_hz"])),
        band_energy_db=float(parsed.get("band_energy_db", features["rms_level_db"])),
        snr_estimate_db=float(parsed.get("snr_estimate_db", 0.0)),
        observation_quality=float(parsed.get("observation_quality", 0.5)),
        evidence_confidence=float(parsed.get("evidence_confidence", 0.5)),
        model_name=model_name,
        audio_path=str(audio_path),
        audio_bytes_len=len(raw),
        audio_sha256=audio_sha256,
        computed_features=features,
        raw_response=raw_text,
        latency_ms=latency_ms,
    )
