#!/usr/bin/env python3
"""phase8i6_new_panel_full_run.py -- runs the full evaluation suite (all
6 pools this project has built: frozen-93, B-Acquisition, D-physical,
A-expanded-v1, C-expanded-v1, E-expanded-v1) for the 4 CHANGED primary-
panel models (GPT-6-Astra, Claude Opus 5.5, Gemini 3.1 Pro Preview,
DeepSeek V4 Pro 0813). Qwen3.5-397B-A17B is UNCHANGED and NOT re-run.

Reuses every existing, already-tested runner UNCHANGED -- this script
only orchestrates them for new model labels:
  frozen-93        -> scripts/phase8h1_run_pilot.py        (subprocess, bare id)
  B-Acquisition     -> scripts/phase8h2g_b_acquisition_pilot.py (subprocess, prefixed id)
  D-physical        -> scripts/phase8h2e_dphys_pilot.py     (subprocess, prefixed id)
  A-expanded-v1     -> phase8h2m_a_expanded_full_run.run_model (imported, bare id)
  C-expanded-v1     -> phase8i4_c_expanded_full_run.run_model (imported, bare id)
  E-expanded-v1     -> phase8h2z_e_expanded_full_run.run_model (imported, bare id)

Sequential per model AND per pool (CouchDB write-conflict avoidance,
same established fix as every expansion run this session). Resume-safe
wherever the underlying script already is (C/E-expanded); frozen-93/B/D
are re-run from scratch per model if partially present (they predate
resume support and none of these 4 models has been run before, so this
is a non-issue for a fresh model).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

NEW_MODELS = [
    # (label, bare_id, prefixed_id)
    ("GPT-6-Astra", "openai/gpt-6-astra", "tokenrouter/openai/gpt-6-astra"),
    ("Claude_Opus_5.5", "anthropic/claude-opus-5.5", "tokenrouter/anthropic/claude-opus-5.5"),
    ("Gemini_3.1-Pro-Preview", "google/gemini-3.1-pro-preview", "tokenrouter/google/gemini-3.1-pro-preview"),
    ("DeepSeek_V4_Pro_0813", "deepseek/deepseek-v4-pro-0813", "tokenrouter/deepseek/deepseek-v4-pro-0813"),
]

FROZEN93_DIR = REPO_ROOT / "reports" / "benchmark" / "v3_full_results" / "frozen93"
B_MANIFEST = REPO_ROOT / "reports" / "benchmark" / "b_acquisition_final_manifest.json"


def _run(cmd, env, log_prefix):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    print(f"[{log_prefix}] exit={proc.returncode} elapsed={dt:.0f}s", flush=True)
    if proc.returncode != 0:
        print(proc.stdout[-3000:], flush=True)
        print(proc.stderr[-3000:], flush=True)
    else:
        print(proc.stdout[-1500:], flush=True)
    return proc.returncode == 0


def run_frozen93(label, bare_id, api_key, base_url):
    out_path = FROZEN93_DIR / f"raw_{label}.jsonl"
    if out_path.exists():
        n = sum(1 for _ in open(out_path))
        if n >= 93:
            print(f"[frozen93] {label} already complete ({n}/93), skipping", flush=True)
            return True
    env = dict(os.environ)
    cmd = [sys.executable, "scripts/phase8h1_run_pilot.py",
          "--model", bare_id, "--confirm", "--out", str(out_path)]
    return _run(cmd, env, "frozen93")


def run_b_acquisition(label, prefixed_id, api_key, base_url):
    out_path = REPO_ROOT / "reports" / "benchmark" / "b_acquisition_pilot" / f"b_acq_pilot_raw_{prefixed_id.replace('/', '_')}_v3_full.jsonl"
    if out_path.exists():
        n = sum(1 for _ in open(out_path))
        if n >= 66:
            print(f"[B-acq] {label} already complete ({n}/66), skipping", flush=True)
            return True
    env = dict(os.environ)
    cmd = [sys.executable, "scripts/phase8h2g_b_acquisition_pilot.py",
          "--model", prefixed_id, "--confirm", "--manifest", str(B_MANIFEST),
          "--out-suffix", "_v3_full"]
    return _run(cmd, env, "B-acq")


def run_d_physical(label, prefixed_id, api_key, base_url):
    out_path = REPO_ROOT / "reports" / "benchmark" / "dphys_pilot" / f"dphys_pilot_raw_{prefixed_id.replace('/', '_')}_v3_full.jsonl"
    if out_path.exists():
        n = sum(1 for _ in open(out_path))
        if n >= 70:
            print(f"[D-phys] {label} already complete ({n}/70), skipping", flush=True)
            return True
    env = dict(os.environ)
    cmd = [sys.executable, "scripts/phase8h2e_dphys_pilot.py",
          "--model", prefixed_id, "--confirm", "--out-suffix", "_v3_full"]
    return _run(cmd, env, "D-phys")


def run_a_expanded(label, bare_id, api_key, base_url):
    import phase8h2m_a_expanded_full_run as A
    manifest = json.loads(A.MANIFEST_PATH.read_text())
    rows = manifest["episodes"]
    A.run_model(label, bare_id, rows, api_key, base_url)
    return True


def run_c_expanded(label, bare_id, api_key, base_url):
    import phase8i4_c_expanded_full_run as C
    manifest = json.loads(C.MANIFEST_PATH.read_text())
    scenario_ids = manifest["new_8_this_pool"]
    C.run_model(label, bare_id, scenario_ids, api_key, base_url)
    return True


def run_e_expanded(label, bare_id, api_key, base_url):
    import phase8h2z_e_expanded_full_run as E
    manifest = json.loads(E.MANIFEST_PATH.read_text())
    rows = manifest["episodes"]
    sequences = E.sequences_from_manifest(rows)
    E.run_model(label, bare_id, sequences, api_key, base_url)
    return True


def main() -> int:
    api_key = os.environ.get("TOKENROUTER_API_KEY", "")
    base_url = os.environ.get("TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1")
    if not api_key:
        print("ERROR: TOKENROUTER_API_KEY not set", file=sys.stderr)
        return 1

    only = sys.argv[1] if len(sys.argv) > 1 else None
    pools = sys.argv[2].split(",") if len(sys.argv) > 2 else \
        ["frozen93", "b_acq", "d_phys", "a_expanded", "c_expanded", "e_expanded"]

    for label, bare_id, prefixed_id in NEW_MODELS:
        if only and label != only:
            continue
        print(f"\n{'='*70}\n{label}  (bare={bare_id}, prefixed={prefixed_id})\n{'='*70}", flush=True)

        if "frozen93" in pools:
            run_frozen93(label, bare_id, api_key, base_url)
        if "b_acq" in pools:
            run_b_acquisition(label, prefixed_id, api_key, base_url)
        if "d_phys" in pools:
            run_d_physical(label, prefixed_id, api_key, base_url)
        if "a_expanded" in pools:
            run_a_expanded(label, bare_id, api_key, base_url)
        if "c_expanded" in pools:
            run_c_expanded(label, bare_id, api_key, base_url)
        if "e_expanded" in pools:
            run_e_expanded(label, bare_id, api_key, base_url)

    print("\nAll requested models/pools complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
