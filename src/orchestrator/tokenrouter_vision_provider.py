"""tokenrouter_vision_provider.py — Tier 1 via any OpenAI-compatible
vision model routed through TokenRouter (https://api.tokenrouter.com/v1).

Same ROSClaw Provider contract and the SAME structured-JSON gauge-reading
prompt as GeminiGaugeVisionProvider (imported, not duplicated), so results
are directly comparable across backends — the only variable is the model.
Credentials come from TOKENROUTER_API_KEY / TOKENROUTER_BASE_URL (same env
vars the repo's src/llm/routers.py already standardizes on).

Images are resized to 800px wide (the same preprocessing DeepMind's own
robotics notebook uses) — the raw PMC photos are 3-4 MB, which both slows
requests and risks per-request payload limits on some routed models.

Epistemic-uncertainty note: same caveats as GeminiGaugeVisionProvider —
OpenAI-compatible chat completions on routed models do not reliably expose
logprobs, so `uncertainty` is the model's self-report, honestly labeled.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spot_assetops_orchestrator import (  # noqa: E402
    Event,
    EventBus,
    Provider,
    ProviderManifest,
    ProviderRequest,
    ProviderResponse,
    TOPIC_INFERENCE_COMPLETED,
)
from gemini_vision_provider import (  # noqa: E402
    _PROMPT_VARIANTS, _parse_json_response, resolve_prompt,
)


def _encode_resized(path: Path, width: int = 800, quality: int = 85) -> str:
    from PIL import Image
    img = Image.open(path)
    if img.size[0] > width:
        img = img.resize((width, int(width * img.size[1] / img.size[0])),
                         Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


class TokenRouterVisionProvider(Provider):
    """Tier 1 provider for any TokenRouter-routed OpenAI-compatible VLM."""

    name = "tokenrouter_vision_provider"
    version = "0.1.0"
    capabilities = ["vlm.gauge_reading_real"]

    def __init__(self, manifest: ProviderManifest, bus: EventBus,
                 query_image: Path, reference_image: Optional[Path],
                 model_name: str, api_key: str = "", base_url: str = "",
                 temperature: float = 0.4, max_retries: int = 4,
                 timeout_s: float = 90.0, prompt_variant: str = "baseline",
                 framing: str = "neutral"):
        super().__init__(manifest)
        self._bus = bus
        self._query_image = query_image
        self._reference_image = reference_image
        self._model_name = model_name
        self._api_key = api_key or os.environ.get("TOKENROUTER_API_KEY", "")
        self._base_url = (base_url or os.environ.get(
            "TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1")).rstrip("/")
        self._temperature = temperature
        self._max_retries = max_retries
        self._timeout_s = timeout_s
        self._query_b64: Optional[str] = None
        self._reference_b64: Optional[str] = None
        self._prompt = resolve_prompt(prompt_variant, framing)
        self.framing = framing
        self.prompt_variant = prompt_variant

    async def load(self) -> None:
        if not self._api_key:
            raise RuntimeError("TOKENROUTER_API_KEY not set")
        self._query_b64 = _encode_resized(self._query_image)
        if self._reference_image is not None:
            self._reference_b64 = _encode_resized(self._reference_image)
        self._healthy = True

    async def unload(self) -> None:
        self._healthy = False

    def set_query_image(self, path: Path) -> None:
        """Swap the query image mid-episode (recovery-ladder rung);
        recomputes the cached resized base64 payload."""
        self._query_image = path
        self._query_b64 = _encode_resized(path)

    def _call_sync(self) -> str:
        content = []
        if self._reference_b64:
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{self._reference_b64}"}})
        content.append({"type": "image_url", "image_url": {
            "url": f"data:image/jpeg;base64,{self._query_b64}"}})
        content.append({"type": "text", "text": self._prompt})
        body = json.dumps({
            "model": self._model_name,
            "messages": [{"role": "user", "content": content}],
            "temperature": self._temperature,
            # thinking-style models (e.g. glm-4.6v) burn budget on hidden
            # reasoning before emitting content; 512 produced finish=length
            # with EMPTY content -> silent parse-failure fallback (verified)
            "max_tokens": 4096,
        }).encode()
        req = urllib.request.Request(
            f"{self._base_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self._api_key}"})
        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self._timeout_s) as r:
                    d = json.loads(r.read())
                return d["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as e:  # pragma: no cover - network path
                last_exc = e
                if e.code in (429, 500, 502, 503) and attempt < self._max_retries - 1:
                    time.sleep(6 * (attempt + 1))
                else:
                    raise RuntimeError(f"TokenRouter HTTP {e.code}: {e.read().decode()[:300]}")
            except Exception as e:  # pragma: no cover - network path
                last_exc = e
                if attempt < self._max_retries - 1:
                    time.sleep(6 * (attempt + 1))
                else:
                    raise
        raise last_exc  # pragma: no cover

    async def infer(self, request: ProviderRequest) -> ProviderResponse:
        import asyncio
        t0 = time.time()
        try:
            raw_text = await asyncio.get_running_loop().run_in_executor(None, self._call_sync)
            parsed = _parse_json_response(raw_text)
        except Exception as exc:
            response = ProviderResponse(
                request_id=request.request_id, provider=self.name,
                capability=request.capability,
                result={"gauge_readable": False, "value": None,
                       "perception_category": "call_or_parse_error"},
                confidence=0.0, evidence=[{"type": "token_entropy", "value": 1.0}],
                latency_ms=int((time.time() - t0) * 1000), status="failed",
                errors=[str(exc)[:300]])
            self._bus.publish(Event(topic=TOPIC_INFERENCE_COMPLETED, source=self.name,
                                    trace_id=request.request_id,
                                    payload={"error": str(exc)[:300]}))
            return response

        uncertainty = float(parsed.get("uncertainty", 0.5))
        result = {
            "value": parsed.get("value"),
            "gauge_readable": bool(parsed.get("gauge_readable", False)),
            "perception_category": parsed.get("perception_category", "unknown"),
            "evidence_text": parsed.get("evidence", ""),
            "standoff_m": 0.0, "gauge_apparent_px": 0,
            "frame_id": f"tr_{self._query_image.stem}",
        }
        response = ProviderResponse(
            request_id=request.request_id, provider=self.name,
            capability=request.capability, result=result,
            confidence=float(parsed.get("confidence", 0.5)),
            evidence=[{"type": "token_entropy", "value": uncertainty,
                      "note": "self-reported uncertainty proxy (no logprobs via "
                              "routed chat completions)"}],
            latency_ms=int((time.time() - t0) * 1000),
            status="ok" if result["gauge_readable"] else "degraded")
        self._bus.publish(Event(
            topic=TOPIC_INFERENCE_COMPLETED, source=self.name, trace_id=request.request_id,
            payload={"request_id": request.request_id, "value": result["value"],
                     "gauge_readable": result["gauge_readable"], "uncertainty": uncertainty,
                     "confidence": response.confidence, "status": response.status,
                     "model": self._model_name}))
        return response
