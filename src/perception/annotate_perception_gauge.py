"""
Stage 4: VLM pre-annotation + Cross-LLM rephrasing gate for PerceptionGauge-49.

For each generated scenario:
  1. AD-Copilot (Gemini 2.5-flash or Claude) receives the reference+query image pair
     and drafts perception_description, gauge_readable, and recommended_action.
  2. The drafted text is passed through a rephrasing gate (Claude if primary is Gemini,
     Llama-3 via Ollama if Claude) to strip vendor-specific linguistic patterns.
  3. Results are written to metadata.json alongside the images.

Usage:
    python annotate_perception_gauge.py --assets-dir /path/to/assets [--limit N]

Env:
    GOOGLE_API_KEY   — for Gemini 2.5-flash (primary annotator)
    ANTHROPIC_API_KEY — for Claude (rephrasing gate)
"""

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path


# ------------------------------------------------------------------
# VLM clients
# ------------------------------------------------------------------

def _encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def call_gemini(ref_path: Path, query_path: Path, prompt: str, api_key: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    ref_b64   = _encode_image(ref_path)
    query_b64 = _encode_image(query_path)
    contents = [
        {"role": "user", "parts": [
            {"inline_data": {"mime_type": "image/jpeg", "data": ref_b64}},
            {"inline_data": {"mime_type": "image/jpeg", "data": query_b64}},
            {"text": prompt},
        ]},
    ]
    for attempt in range(5):
        try:
            resp = model.generate_content(contents)
            return resp.text
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                time.sleep(15 * (attempt + 1))
            else:
                raise
    return ""


def call_claude(text: str, api_key: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": text}],
    )
    return msg.content[0].text


def call_ollama_llama(text: str, host: str = "http://localhost:11434") -> str:
    import urllib.request, json as _json
    body = json.dumps({"model": "llama3", "prompt": text, "stream": False}).encode()
    req  = urllib.request.Request(f"{host}/api/generate", data=body,
                                   headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return _json.loads(r.read())["response"]


# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------

ANNOTATION_PROMPT = """You are an industrial inspection expert reviewing two images of the same gauge:
- Image 1 (reference): clean, unobstructed view
- Image 2 (query): degraded/obstructed field condition

Answer ONLY in this JSON structure, no other text:
{{
  "gauge_readable": true | false,
  "perception_description": "<one sentence describing the exact visual challenge in the query image>",
  "perception_severity": "low" | "medium" | "high",
  "recommended_action": "CLEAN_GAUGE" | "ROUTE_UPDATE" | "SENSOR_RECALIBRATE" | "ESCALATE" | "COMMIT_READING",
  "reasoning": "<one sentence explaining the recommendation>"
}}

Category hint: {category}
Asset: {asset}
Failure mode: {failure_mode}
"""

REPHRASE_PROMPT = """Rephrase the following industrial inspection description in neutral,
vendor-agnostic technical language. Do not change the factual content.
Remove any model-specific phrasing patterns. Output only the rephrased text, nothing else.

Original: {text}
"""


# ------------------------------------------------------------------
# Annotation logic
# ------------------------------------------------------------------

def annotate_scenario(scenario_dir: Path, gemini_key: str, claude_key: str) -> dict:
    meta_path = scenario_dir / "metadata.json"
    if not meta_path.exists():
        return {}

    with open(meta_path) as f:
        meta = json.load(f)

    ref_path   = scenario_dir / "reference.jpg"
    query_path = scenario_dir / "query.jpg"
    if not ref_path.exists() or not query_path.exists():
        return meta

    # Stage 1: VLM annotation (Gemini primary, Claude fallback)
    prompt = ANNOTATION_PROMPT.format(
        category=meta.get("category", ""),
        asset=meta.get("asset", ""),
        failure_mode=meta.get("failure_mode", ""),
    )

    raw_annotation = ""
    annotator_used = "none"
    if gemini_key:
        try:
            raw_annotation = call_gemini(ref_path, query_path, prompt, gemini_key)
            annotator_used = "gemini-2.5-flash"
        except Exception as e:
            print(f"    [Gemini error] {e}")

    if not raw_annotation and claude_key:
        # Claude text-only fallback (no image pair) — uses existing description
        fallback_prompt = (
            f"Given this industrial gauge scenario:\n"
            f"Asset: {meta.get('asset')}\n"
            f"Category: {meta.get('category')}\n"
            f"Failure mode: {meta.get('failure_mode')}\n"
            f"Description: {meta.get('description')}\n\n"
            f"Output JSON: {{\"gauge_readable\": bool, \"perception_description\": str, "
            f"\"perception_severity\": str, \"recommended_action\": str, \"reasoning\": str}}"
        )
        try:
            raw_annotation = call_claude(fallback_prompt, claude_key)
            annotator_used = "claude-fallback"
        except Exception as e:
            print(f"    [Claude error] {e}")

    # Parse JSON from VLM response
    annotation = {}
    match = re.search(r"\{[^{}]+\}", raw_annotation, re.DOTALL)
    if match:
        try:
            annotation = json.loads(match.group())
        except json.JSONDecodeError:
            pass

    if not annotation:
        # Keep existing metadata values
        annotation = {
            "gauge_readable":         meta.get("gauge_readable") == "true",
            "perception_description": meta.get("description", ""),
            "perception_severity":    "medium",
            "recommended_action":     meta.get("recommended_action", ""),
            "reasoning":              meta.get("trap", ""),
        }

    # Stage 2: Rephrasing gate — strip vendor bias from perception_description
    desc = annotation.get("perception_description", "")
    if desc and claude_key and annotator_used != "claude-fallback":
        try:
            rephrased = call_claude(REPHRASE_PROMPT.format(text=desc), claude_key)
            annotation["perception_description"] = rephrased.strip()
            annotation["rephrased_by"] = "claude-sonnet-4-6"
        except Exception:
            pass
    elif desc:
        try:
            rephrased = call_ollama_llama(REPHRASE_PROMPT.format(text=desc))
            annotation["perception_description"] = rephrased.strip()
            annotation["rephrased_by"] = "llama3-ollama"
        except Exception:
            pass

    # Preserve catalog-authored gold BEFORE any overwrite below, so a scorer can
    # later compare the VLM's draft against it. Only set once: a re-run must not
    # let an already-promoted VLM guess become the new "gold".
    if "gold_gauge_readable" not in meta:
        meta["gold_gauge_readable"] = meta.get("gauge_readable")
    if "gold_recommended_action" not in meta:
        meta["gold_recommended_action"] = meta.get("recommended_action")

    # Merge into metadata
    meta["vlm_annotation"] = {
        "annotator": annotator_used,
        **annotation,
    }
    # Promote key fields to top level for CouchDB ingestion compatibility
    if annotation.get("gauge_readable") is not None:
        meta["gauge_readable"] = str(annotation["gauge_readable"]).lower()
    if annotation.get("recommended_action"):
        meta["recommended_action"] = annotation["recommended_action"]
    if annotation.get("perception_description"):
        meta["description"] = annotation["perception_description"]

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets-dir",
                    default=os.path.join("/media/adityapachauri/second_drive",
                                         "perception_gauge_49", "assets"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--delay", type=float, default=4.0,
                    help="Seconds between Gemini calls (rate-limit headroom)")
    args = ap.parse_args()

    gemini_key = os.environ.get("GOOGLE_API_KEY", "")
    claude_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not gemini_key and not claude_key:
        print("[ERROR] Set GOOGLE_API_KEY or ANTHROPIC_API_KEY before running.")
        return

    assets_path = Path(args.assets_dir)
    scenario_dirs = sorted([d for d in assets_path.iterdir() if d.is_dir()])
    if args.limit:
        scenario_dirs = scenario_dirs[:args.limit]

    print(f"Annotating {len(scenario_dirs)} scenarios …")
    ok = 0
    for d in scenario_dirs:
        print(f"  {d.name} …", end=" ", flush=True)
        meta = annotate_scenario(d, gemini_key, claude_key)
        if meta:
            action = meta.get("recommended_action", "?")
            readable = meta.get("gauge_readable", "?")
            print(f"readable={readable}  action={action}")
            ok += 1
        else:
            print("SKIPPED (no metadata)")
        time.sleep(args.delay)

    print(f"\nAnnotated {ok}/{len(scenario_dirs)} scenarios.")


if __name__ == "__main__":
    main()
