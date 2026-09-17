#!/usr/bin/env python3
"""phase8h2j_v3_primary_panel_pilot.py — Small real-API execution-fidelity
pilot for the InspectionBench V3 primary five-model panel (Phase 8H.2J,
implementation Phase 6).

NOT the paper result. This is a fidelity gate: does every primary-panel
model actually execute the real protocol for each of A/B/C/D/E, deliver
real observations, avoid gold leakage, and get scored correctly?

Reuses EXISTING, already-validated execution logic without modification:
  - B: scripts/phase8h2g_b_acquisition_pilot.run_model  (imported directly)
  - D: scripts/phase8h2e_dphys_pilot.run_model           (imported directly)
  - A/C/E: the SAME underlying functions phase8h1_run_pilot.py's own
    dispatcher (_run_unit) calls, applied to a hand-picked small subset of
    the frozen-93 manifest instead of all 93 -- phase8h1_run_pilot.py has
    no subset flag, so this script does NOT modify it; it reuses the
    identical functions directly:
      A: phase8h_live_pilot.run_a_episode
      C: run_classc_pilot.run (with classc_fixtures, exactly as _run_unit does)
      E: run_class_e_pilot.run_sequence

Served-model verification (eval_model_panel.verify_served_model): one cheap
1-token check per model before the real pilot calls, using the SAME
OpenAICompatBackend class every downstream runner uses -- confirms the
router is not silently substituting before spending real pilot tokens.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
OUT = REPO_ROOT / "reports" / "benchmark" / "v3_primary_panel_pilot"
OUT.mkdir(parents=True, exist_ok=True)

import eval_model_panel as EMP  # noqa: E402
import pilot_dispatch as PD  # noqa: E402


def pre_flight_served_model_check(model_cfg: EMP.ModelConfig) -> Dict[str, Any]:
    from llm.openai_compat import OpenAICompatBackend
    backend = OpenAICompatBackend(model_cfg.tokenrouter_id)
    r = backend.generate_with_usage("Reply with exactly: OK", temperature=0.0)
    check = EMP.verify_served_model(model_cfg.tokenrouter_id, r.served_model)
    return {"requested": model_cfg.tokenrouter_id, "served_model": r.served_model,
           "ok": check.ok, "reason": check.reason}


def pick_frozen93_subset(n_per_dim: int = 2) -> Dict[str, List[Dict[str, Any]]]:
    manifest = PD.load_manifest()
    eps = manifest["episodes"]
    by_dim: Dict[str, List[Dict[str, Any]]] = {}
    for e in eps:
        by_dim.setdefault(e["dimension"], []).append(e)
    # A/C/E only here -- B and D come from their own dedicated 66/70-episode
    # manifests (richer, more representative than the frozen-93's small B/D
    # slice), per this task's own instruction to prefer representative
    # episodes without creating a special case.
    picked = {}
    for dim in ("A", "C"):
        rows = by_dim.get(dim, [])
        # deterministic pick: first N by episode_id sort, not random
        picked[dim] = sorted(rows, key=lambda r: r["episode_id"])[:n_per_dim]
    # E: each real unit is a 3-episode SEQUENCE (one run_sequence() call
    # produces all 3) -- picking N rows sorted by episode_id would pick
    # ep0+ep1 of the SAME sequence, causing run_ace_unit to call
    # run_sequence() twice for identical real API work. Pick N DISTINCT
    # sequences' first episode instead: more representative (covers more
    # of E's construct diversity) and avoids the redundant re-run.
    e_rows = by_dim.get("E", [])
    seen_seq: set = set()
    e_picked = []
    for r in sorted(e_rows, key=lambda r: r["episode_id"]):
        seq_id = r["episode_id"].split("::")[1]  # "E::SEQ-5000::ep0" -> "SEQ-5000"
        if seq_id in seen_seq:
            continue
        seen_seq.add(seq_id)
        e_picked.append(r)
        if len(e_picked) >= n_per_dim:
            break
    picked["E"] = e_picked
    return picked


def run_ace_unit(row: Dict[str, Any], *, model: str, api_key: str, base_url: str) -> Dict[str, Any]:
    """Run ONE frozen-93 A/C/E row for real, through the identical
    functions phase8h1_run_pilot.py's _run_unit dispatches to. Returns a
    dict with fidelity-relevant fields extracted from the real runner
    return value.

    `model` MUST be the BARE model id (no "tokenrouter/" prefix) -- unlike
    the B/D-physical runners (which go through OpenAICompatBackend/
    resolve_router_creds and need the prefix to select a router),
    phase8h_live_pilot.run_a_episode / run_classc_pilot.run /
    run_class_e_pilot.run_sequence all call their own `_chat()` helper,
    which POSTs `{"model": model, ...}` directly to TokenRouter via raw
    urllib with NO prefix resolution at all (matching
    phase8h1_run_pilot.py's own bare MODEL_IDS convention). Verified live:
    passing the prefixed id here causes a 100%-reproducible
    `model_not_found` 503 from the gateway ("No available channel for
    model tokenrouter/anthropic/claude-sonnet-4.6") -- a real bug found
    and fixed during this pilot, not gateway flakiness and not a
    benchmark-content issue."""
    model = model.split("tokenrouter/", 1)[-1] if model.startswith("tokenrouter/") else model
    dim = row["dimension"]
    t0 = time.time()
    if dim == "A":
        import phase8h_live_pilot as LP
        world = PD.rebuild_world(row)
        ret = LP.run_a_episode(world, row["regime"], model=model, api_key=api_key, base_url=base_url)
        ret = dict(ret)
        if hasattr(ret.get("metric"), "to_dict"):
            ret["metric"] = ret["metric"].to_dict()
    elif dim == "C":
        import run_classc_pilot as C
        from classc_fixtures import FIXTURES, FixtureSession
        from couchdb_executor import CouchDBExecutor
        ex = CouchDBExecutor()
        sid = row["scenario_id"]
        fx = FIXTURES[sid]
        ex.reset(sid, "FULL", seed=1)
        with FixtureSession(ex._robot.db, fx) as sess:
            problems = sess.verify_applied()
            if problems:
                raise RuntimeError(f"{sid}: FIXTURE NOT APPLIED -> {problems}")
            ret = dict(C.run(model, api_key, base_url, ex, sid,
                            skip_reset=True, fixture_verified=not problems))
    elif dim == "E":
        import run_class_e_pilot as E
        from sequence_executor import SequenceExecutor, sample_sequence
        from couchdb_executor import CouchDBExecutor
        ex = CouchDBExecutor()
        se = SequenceExecutor(ex)
        # deterministic small sequence: use the row's own recorded seed
        units = PD.build_execution_units(PD.load_manifest())
        unit = next(u for u in units if row["episode_id"] in u.episode_ids)
        seq_world = sample_sequence(unit.extra["seed"], n_episodes=unit.extra["n_episodes"])
        full_ret = E.run_sequence(model, api_key, base_url, se, seq_world, unit.extra["seed"])
        ret = {"episodes": full_ret["episodes"], "sequence_id": full_ret["sequence_id"],
               "trace_chain_valid": full_ret.get("sequence_trace_chain_valid")}
    else:
        raise ValueError(dim)
    elapsed = time.time() - t0
    return {"episode_id": row["episode_id"], "dimension": dim, "model": model,
           "elapsed_s": round(elapsed, 2), "gold": row.get("gold_action"), "runner_return": ret}


def run_b_subset(model_id: str, limit: int) -> Dict[str, Any]:
    import phase8h2g_b_acquisition_pilot as B
    episodes = B.load_episodes([REPO_ROOT / "reports/benchmark/b_acquisition_final_manifest.json"])
    # one per template for representativeness, not just the first N
    by_t: Dict[str, List[Dict[str, Any]]] = {}
    for e in episodes:
        by_t.setdefault(e["template_id"], []).append(e)
    subset = [v[0] for _, v in sorted(by_t.items())][:limit] if limit < len(by_t) else \
        [v[0] for _, v in sorted(by_t.items())]
    return B.run_model(model_id, subset, confirm=True, out_suffix="_v3_panel_pilot")


def run_d_subset(model_id: str, limit: int) -> Dict[str, Any]:
    import phase8h2e_dphys_pilot as D
    episodes = D.load_canonical_episodes()
    by_s: Dict[str, List[Dict[str, Any]]] = {}
    for e in episodes:
        by_s.setdefault(e["constraint_stratum"], []).append(e)
    subset = [v[0] for _, v in sorted(by_s.items())][:limit]
    return D.run_model(model_id, subset, confirm=True, out_suffix="_v3_panel_pilot")


def main() -> int:
    import os
    api_key = os.environ.get("TOKENROUTER_API_KEY", "")
    base_url = os.environ.get("TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1")
    if not api_key:
        print("ERROR: TOKENROUTER_API_KEY not set", file=sys.stderr)
        return 2

    ace_subset = pick_frozen93_subset(n_per_dim=2)
    print(f"A/C/E subset: {[f'{k}={len(v)}' for k,v in ace_subset.items()]}")

    results: Dict[str, Any] = {}
    for cfg in EMP.PRIMARY_PANEL:
        print(f"\n=== {cfg.display_name} ({cfg.tokenrouter_id}) ===")
        model_result: Dict[str, Any] = {"model": cfg.display_name, "tokenrouter_id": cfg.tokenrouter_id}

        preflight = pre_flight_served_model_check(cfg)
        model_result["preflight_served_model_check"] = preflight
        print(f"  preflight: served_model={preflight['served_model']!r} ok={preflight['ok']}")
        if not preflight["ok"]:
            model_result["halted"] = True
            model_result["halt_reason"] = f"served-model mismatch: {preflight['reason']}"
            results[cfg.display_name] = model_result
            continue

        ace_results = []
        for dim, rows in ace_subset.items():
            for row in rows:
                try:
                    r = run_ace_unit(row, model=cfg.tokenrouter_id, api_key=api_key, base_url=base_url)
                    ace_results.append(r)
                    print(f"  {dim} {row['episode_id']}: OK ({r['elapsed_s']}s)")
                except Exception as e:  # noqa: BLE001
                    ace_results.append({"episode_id": row["episode_id"], "dimension": dim,
                                        "error": f"{type(e).__name__}: {e}"})
                    print(f"  {dim} {row['episode_id']}: FAILED {type(e).__name__}: {str(e)[:150]}")
        model_result["ace"] = ace_results

        try:
            model_result["b"] = run_b_subset(cfg.tokenrouter_id, limit=3)
            print(f"  B: {model_result['b']['n_episodes']} episodes -> "
                 f"ADA={model_result['b'].get('ADA_mean')} ASA={model_result['b'].get('ASA_mean')}")
        except Exception as e:  # noqa: BLE001
            model_result["b"] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  B: FAILED {type(e).__name__}: {str(e)[:150]}")

        try:
            model_result["d"] = run_d_subset(cfg.tokenrouter_id, limit=3)
            print(f"  D: {model_result['d']['n_episodes']} episodes -> "
                 f"TDA={model_result['d'].get('TDA_mean')} CSA={model_result['d'].get('CS_F1_mean')}")
        except Exception as e:  # noqa: BLE001
            model_result["d"] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  D: FAILED {type(e).__name__}: {str(e)[:150]}")

        results[cfg.display_name] = model_result

    out_path = OUT / "v3_primary_panel_pilot_raw.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
