"""run_l3_pilot.py — Six-scenario L3 evidence-dependency pilot.

Apparatus/model pilot, **not** a benchmark result. Six scenarios x two
competence arms cannot resolve a 10 pp effect; the purpose is to establish that
the arms behave as designed against a real model before any powered run is
specified.

Reports **CC and PROC separately and never aggregates them**. That separation is
load-bearing here: in four of the six scenarios the shortcut yields the *same*
verdict as gold, so CC cannot distinguish a grounded answer from an ungrounded
one and PROC is the only discriminating signal. A single blended score would
hide exactly the failure the pilot exists to detect.

Provenance for every run — both repository SHAs, the frozen-config hash, the
preregistration and amendment SHAs, model id and generation settings — is
captured before the first call and written into the results file.

Usage::

    python scripts/run_l3_pilot.py --model openai/gpt-5.4-mini
    python scripts/run_l3_pilot.py --dry-run     # provenance + payloads, no calls
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
SCEN_REPO = REPO_ROOT.parent / "AssetOpsBenchScenarioGeneration"
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import l3_arms as A  # noqa: E402
from l3_scoring import score_l3  # noqa: E402

SYSTEM_PROMPT = """\
You are an industrial inspection agent. You are given an inspection task, the
tools available to you, and any attached evidence sources.

Decide the correct operational action. Reply with EXACTLY this JSON object and
nothing else:
{"verdict": "COMMIT|ESCALATE|ABORT", "reason": "<one sentence>", "pa": <float or null>, "tool_sequence": ["<tool>", ...]}

"tool_sequence" must list, in order, the tools you actually relied on to reach
your decision. Do not list tools you were not offered.
"""

SYSTEM_PROMPT_ID = "l3_pilot/v1"
TEMPERATURE = 0.0
MAX_TOKENS = 1024


def _git_sha(repo: Path) -> Dict[str, Any]:
    def run(*args):
        try:
            return subprocess.run(["git", "-C", str(repo), *args],
                                  capture_output=True, text=True,
                                  timeout=15).stdout.strip()
        except Exception:
            return ""
    return {"path": str(repo), "sha": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain"))}


def _file_sha(path: Path) -> Optional[str]:
    """Blob SHA of a tracked file — pins the exact preregistration text."""
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "hash-object", str(path)],
                              capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        return None


def provenance(model: str) -> Dict[str, Any]:
    try:
        from frozen_config import config_hash
        fc = config_hash()
    except Exception:
        fc = None
    docs = {
        "V1_EvidenceContract.md": REPO_ROOT / "docs" / "V1_EvidenceContract.md",
        "L3_ScenarioClassification.md": REPO_ROOT / "docs" / "L3_ScenarioClassification.md",
        "FM_Crosswalk.md": REPO_ROOT / "docs" / "FM_Crosswalk.md",
    }
    return {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repos": {"AssetOpsBench": _git_sha(REPO_ROOT),
                  "AssetOpsBenchScenarioGeneration": _git_sha(SCEN_REPO)},
        "frozen_config_hash": fc,
        "document_blob_shas": {k: _file_sha(v) for k, v in docs.items()},
        "model": model,
        "generation": {"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
                       "system_prompt_id": SYSTEM_PROMPT_ID},
        "scenarios": list(A.PILOT_SCENARIOS),
        "status": "apparatus/model pilot - NOT a benchmark result",
    }


def call_model(model_id: str, system: str, user_text: str, api_key: str,
               base_url: str, timeout: float = 90.0
               ) -> Tuple[Dict[str, Any], float, Optional[str]]:
    body = json.dumps({
        "model": model_id,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user_text}],
        "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read())
    except Exception as exc:  # noqa: BLE001
        return {"verdict": "CALL_ERROR", "tool_sequence": []}, time.time() - t0, str(exc)
    latency = time.time() - t0
    text = (payload.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    if not text.strip():
        return {"verdict": "NO_ANSWER", "tool_sequence": []}, latency, "empty content"
    clean = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    m = re.search(r"\{.*\}", clean, re.DOTALL)
    if not m:
        return ({"verdict": "PARSE_ERROR", "tool_sequence": []}, latency,
                f"no JSON object: {text[:120]}")
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        return ({"verdict": "PARSE_ERROR", "tool_sequence": []}, latency, str(exc))
    parsed.setdefault("tool_sequence", [])
    return parsed, latency, None


def render_user_text(payload: Dict[str, Any]) -> str:
    return (f"{payload['question'].strip()}\n\n"
            f"Tools available: {', '.join(payload['tools'])}\n"
            f"Attached evidence: {', '.join(payload['attached_evidence']) or 'none'}\n"
            f"Allowed verdicts: {', '.join(payload['allowed_actions'])}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="openai/gpt-5.4-mini")
    ap.add_argument("--base-url", default=os.environ.get(
        "TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1"))
    ap.add_argument("--dry-run", action="store_true",
                    help="capture provenance and render payloads without calling")
    ap.add_argument("--json", type=Path,
                    default=REPO_ROOT / "reports" / "v1" / "l3_pilot_results.json")
    args = ap.parse_args()

    prov = provenance(args.model)
    print("PROVENANCE")
    for name, r in prov["repos"].items():
        print(f"  {name:32s} {r['sha'][:12]} ({r['branch']})"
              f"{'  DIRTY' if r['dirty'] else ''}")
    print(f"  {'frozen_config':32s} {str(prov['frozen_config_hash'])[:12]}")
    for k, v in prov["document_blob_shas"].items():
        print(f"  {k:32s} {str(v)[:12]}")
    print(f"  {'model':32s} {args.model}")
    print(f"  {'generation':32s} temp={TEMPERATURE} max_tokens={MAX_TOKENS} "
          f"prompt={SYSTEM_PROMPT_ID}")
    print(f"  {'scenarios':32s} {', '.join(A.PILOT_SCENARIOS)}\n")

    api_key = os.environ.get("TOKENROUTER_API_KEY", "")
    if not args.dry_run and not api_key:
        print("ERROR: TOKENROUTER_API_KEY not set", file=sys.stderr)
        return 2

    rows: List[Dict[str, Any]] = []
    for sid in A.PILOT_SCENARIOS:
        gold = A.SCENARIOS[sid]["gold"]
        fm = A.SCENARIOS[sid]["fm"]
        for spec in A.arms_for(sid):
            payload = A.render_arm(spec)
            user_text = render_user_text(payload)
            if args.dry_run:
                rows.append({"scenario_id": sid, "fm": fm, "arm": spec.arm_id,
                             "probe": spec.insufficient_evidence_probe,
                             "chars": len(user_text)})
                continue
            resp, latency, err = call_model(args.model, SYSTEM_PROMPT, user_text,
                                            api_key, args.base_url)
            verdict = str(resp.get("verdict") or resp.get("action") or "")
            apparatus_fail = verdict in ("CALL_ERROR", "NO_ANSWER", "PARSE_ERROR")
            scores = (None if apparatus_fail
                      else score_l3(resp, {"fm": fm}, {"action": gold}))
            rows.append({
                "scenario_id": sid, "fm": fm, "gold": gold, "arm": spec.arm_id,
                "probe": spec.insufficient_evidence_probe,
                "verdict": verdict, "tool_sequence": resp.get("tool_sequence", []),
                "reason": resp.get("reason", ""),
                "CC": None if apparatus_fail else scores["CC"],
                "PROC": None if apparatus_fail else scores["PROC"],
                "UDR": None if apparatus_fail else scores.get("UDR"),
                "ODR": None if apparatus_fail else scores.get("ODR"),
                "apparatus_failure": apparatus_fail, "error": err,
                "latency_s": round(latency, 2),
            })
            flag = "APPARATUS-FAIL" if apparatus_fail else ""
            print(f"  {sid} {spec.arm_id:14s} gold={gold:9s} verdict={verdict:12s} "
                  f"CC={rows[-1]['CC']} PROC={rows[-1]['PROC']} "
                  f"({latency:.1f}s) {flag}")

    out = {"schema": "assetops.l3_pilot/1", "provenance": prov, "results": rows}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n-> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
