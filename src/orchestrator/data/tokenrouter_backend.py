"""
TokenRouter (OpenAI-compatible chat completions) backend for whole-window
gauge labeling -- alternative to the Gemini REST backend in
label_windows_vlm.py, for use when the Gemini free-tier quota is exhausted.

Reuses the exact request pattern already proven in
AssetOpsBench/src/orchestrator/tokenrouter_vision_provider.py (multiple
image_url content parts + one text part, Bearer auth, TOKENROUTER_API_KEY /
TOKENROUTER_BASE_URL from .env), but stdlib-only (no openai/httpx dependency)
to match label_windows_vlm.py's zero-extra-install approach.
"""
import base64
import io
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL = "openai/gpt-5.4-mini"
DEFAULT_BASE_URL = "https://api.tokenrouter.com/v1"


def _encode_resized(path: Path, width: int = 800, quality: int = 85) -> str:
    from PIL import Image
    img = Image.open(path)
    if img.size[0] > width:
        img = img.resize((width, int(width * img.size[1] / img.size[0])),
                         Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def load_tokenrouter_credentials(env_file: Path):
    import os
    key = os.environ.get("TOKENROUTER_API_KEY")
    base_url = os.environ.get("TOKENROUTER_BASE_URL", DEFAULT_BASE_URL)
    if key:
        return key, base_url
    if env_file.exists():
        found = {}
        for line in env_file.read_text().splitlines():
            for name in ("TOKENROUTER_API_KEY", "TOKENROUTER_BASE_URL"):
                if line.startswith(f"{name}="):
                    found[name] = line.split("=", 1)[1].strip()
        if "TOKENROUTER_API_KEY" in found:
            return found["TOKENROUTER_API_KEY"], found.get("TOKENROUTER_BASE_URL", DEFAULT_BASE_URL)
    raise RuntimeError("TOKENROUTER_API_KEY not found in environment or .env")


def call_tokenrouter(img_paths, prompt: str, api_key: str, base_url: str = DEFAULT_BASE_URL,
                     model: str = DEFAULT_MODEL, retries: int = 4, timeout_s: float = 90.0) -> dict:
    if isinstance(img_paths, (str, Path)):
        img_paths = [img_paths]

    content = []
    for p in img_paths:
        b64 = _encode_resized(Path(p))
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    content.append({"type": "text", "text": prompt})

    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.4,
        "max_tokens": 4096,  # thinking-style routed models can burn budget on hidden reasoning
    }).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                payload = json.loads(resp.read())
            text = payload["choices"][0]["message"]["content"] or ""
            match = re.search(r"\{[\s\S]+\}", text)
            return json.loads(match.group()) if match else {}
        except urllib.error.HTTPError as e:
            body_text = e.read().decode(errors="replace")
            if e.code in (429, 500, 502, 503) and attempt < retries - 1:
                wait = 10 * (attempt + 1)
                print(f"    [HTTP {e.code}] waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    [HTTP {e.code}] {body_text[:300]}")
                return {}
        except Exception as e:
            print(f"    [error] {e}")
            if attempt < retries - 1:
                time.sleep(5)
            else:
                return {}
    return {}
