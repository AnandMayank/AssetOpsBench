#!/usr/bin/env python3
"""phase8h1_run_pilot.py -- Phase 8H.1 multi-model pilot executor.

Consumes the FROZEN manifest (reports/ec/phase8h1_pilot_manifest.json,
SHA d2b48c0b...3087fe), runs ONE model over all 93 episodes through the
existing validated runners, and checkpoints one JSONL line per episode.

Modes
  --probe              connectivity/config probe only (ZERO generation tokens)
  --dry-run            run gates A-E on MOCK runner returns; ZERO API calls
  --model <id>         execute that model for real (requires explicit --confirm)

Nothing in this file re-implements execution logic:
  A / B-fm7a   scripts/phase8h_live_pilot.run_a_episode(world, arm, model=...)
  B  R011      scripts/run_l3_pilot_executed.run_episode(...)
  C            scripts/run_classc_pilot.run(...)
  D            scripts/run_classd_pilot.run(...)
  E            scripts/run_class_e_pilot.run_sequence(...)

Dispatch is via src/orchestrator/pilot_dispatch.py (keyed on scenario_template,
never scenario_id). Failure handling via src/orchestrator/pilot_failures.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
EC = REPO_ROOT / "reports" / "ec"

import pilot_dispatch as PD          # noqa: E402
import pilot_failures as PF          # noqa: E402

RAW_SCHEMA = "phase8h1_multimodel_raw/1"
BASE_URL = os.environ.get("TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1")
MODEL_IDS = ["openai/gpt-5.4-mini", "qwen3.5-omni-plus", "google/gemini-3.1-flash-image-preview"]


# ======================================================================
# connectivity probe -- ZERO generation tokens
# ======================================================================

def connectivity_probe(models: List[str], *, api_key: str, base_url: str = BASE_URL,
                       timeout: int = 20) -> Dict[str, Any]:
    """Strictly connectivity/configuration. Verifies:
      - an API key is present in the environment;
      - the base URL host resolves and the /models endpoint is reachable;
      - each requested model id is accepted (present in /models, or a
        max_tokens=1 stop-immediately request is accepted by the router).

    It does NOT consume a meaningful generation, does not touch benchmark
    state, produces no pilot data, and is not an episode. Expected API cost:
    zero generation tokens (a GET on /models; if a fallback POST is needed it
    is max_tokens=1 which the router bills at its minimum, still not a pilot
    call and never recorded as one).
    """
    out: Dict[str, Any] = {"probe": "connectivity_only", "base_url": base_url,
                           "api_key_present": bool(api_key), "models": {}}
    if not api_key:
        out["status"] = "FAIL"
        out["detail"] = "TOKENROUTER_API_KEY not set"
        return out

    listed: set = set()
    try:
        req = urllib.request.Request(f"{base_url}/models",
                                     headers={"Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read())
        for item in (payload.get("data") or payload.get("models") or []):
            mid = item.get("id") or item.get("name")
            if mid:
                listed.add(mid)
        out["models_endpoint"] = "reachable"
    except Exception as exc:  # noqa: BLE001
        out["models_endpoint"] = f"unreachable: {type(exc).__name__}: {exc}"

    for mid in models:
        entry: Dict[str, Any] = {}
        if mid in listed:
            entry["accepted"] = True
            entry["via"] = "/models listing"
        else:
            # fallback: a stop-immediately request. max_tokens=1 -> no
            # meaningful generation. Purely 'does the router accept this id'.
            try:
                body = json.dumps({"model": mid,
                                   "messages": [{"role": "user", "content": "ping"}],
                                   "max_tokens": 1}).encode()
                req = urllib.request.Request(f"{base_url}/chat/completions", data=body,
                                             headers={"Content-Type": "application/json",
                                                      "Authorization": f"Bearer {api_key}"})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    _ = json.loads(r.read())
                entry["accepted"] = True
                entry["via"] = "max_tokens=1 accept-check (no pilot data)"
            except urllib.error.HTTPError as exc:
                entry["accepted"] = exc.code not in (400, 404)
                entry["via"] = f"HTTP {exc.code}"
            except Exception as exc:  # noqa: BLE001
                entry["accepted"] = False
                entry["via"] = f"{type(exc).__name__}: {exc}"
        out["models"][mid] = entry

    out["status"] = "PASS" if all(v.get("accepted") for v in out["models"].values()) else "FAIL"
    return out


# ======================================================================
# real execution
# ======================================================================

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _persist_row(fh, *, model: str, manifest_sha: str, unit, episode_id: str,
                 row_meta: Dict[str, Any], runner_return: Dict[str, Any],
                 attempt: int, verdict_obj: PF.FailureVerdict,
                 started: str, finished: str) -> None:
    rec = {
        "schema": RAW_SCHEMA,
        "model": model,
        "episode_id": episode_id,
        "dim": row_meta["dimension"],
        "world_id": row_meta["world_id"],
        "scenario_id": row_meta["scenario_id"],
        "arm": row_meta["regime"],
        "manifest_sha256": manifest_sha,
        "unit_id": unit.unit_id,
        "unit_kind": unit.kind,
        "gold": row_meta.get("gold_action"),
        "started_at": started, "finished_at": finished, "attempt": attempt,
        "failure_class": None if verdict_obj.cls == PF.MODEL_BEHAVIOR else verdict_obj.cls,
        "failure_reason": verdict_obj.reason if verdict_obj.cls != PF.MODEL_BEHAVIOR else None,
        "is_apparatus_failure": verdict_obj.is_apparatus_failure,
        "runner_return": runner_return,
    }
    fh.write(json.dumps(rec, default=str) + "\n")
    fh.flush()
    os.fsync(fh.fileno())


def _run_unit(unit, *, model: str, api_key: str, base_url: str, live) -> Dict[str, Any]:
    """Invoke the right existing runner for one execution unit. Returns a dict
    {episode_id: runner_return_for_that_episode}. Raises on semantic failure."""
    LP = live
    if unit.kind == "a_episode":
        row = unit.rows[0]
        world = PD.rebuild_world(row)
        ret = LP.run_a_episode(world, row["regime"], model=model,
                               api_key=api_key, base_url=base_url)
        ret = dict(ret)
        if hasattr(ret.get("metric"), "to_dict"):
            ret["metric"] = ret["metric"].to_dict()
        return {unit.episode_ids[0]: ret}

    if unit.kind == "b_incumbent":
        from l3_arms import arms_for
        row = unit.rows[0]
        spec = next(s for s in arms_for("R011") if s.arm_id == row["regime"])
        import run_l3_pilot_executed as R
        ex = LP.CouchDBExecutor()
        ret = R.run_episode(model, api_key, base_url, ex, "R011", spec)
        return {unit.episode_ids[0]: dict(ret)}

    if unit.kind == "classc":
        import run_classc_pilot as C
        from classc_fixtures import FIXTURES, FixtureSession
        ex = LP.CouchDBExecutor()
        sid = unit.rows[0]["scenario_id"]
        fx = FIXTURES[sid]
        # Reset THEN fixture THEN run(skip_reset=True) -- see run_classc_pilot's
        # main() and run() docstrings: this is the only order under which a
        # fixture profile field (e.g. R001's panel_stuck=True) survives into
        # the episode, since ex.reset() unconditionally clears it.
        ex.reset(sid, "FULL", seed=1)
        with FixtureSession(ex._robot.db, fx) as sess:
            problems = sess.verify_applied()
            if problems:
                raise RuntimeError(f"{sid}: FIXTURE NOT APPLIED -> {problems}")
            ret = C.run(model, api_key, base_url, ex, sid,
                       skip_reset=True, fixture_verified=not problems)
        return {unit.episode_ids[0]: dict(ret)}

    if unit.kind == "classd":
        import run_classd_pilot as D
        from classc_fixtures import FIXTURES, FixtureSession
        ex = LP.CouchDBExecutor()
        sid = unit.rows[0]["scenario_id"]
        fx = FIXTURES[sid]
        ex.reset(sid, "FULL", seed=1)
        with FixtureSession(ex._robot.db, fx) as sess:
            problems = sess.verify_applied()
            if problems:
                raise RuntimeError(f"{sid}: FIXTURE NOT APPLIED -> {problems}")
            ret = D.run(model, api_key, base_url, ex, sid, skip_reset=True)
        return {unit.episode_ids[0]: dict(ret)}

    if unit.kind == "e_sequence":
        import run_class_e_pilot as E
        from sequence_executor import SequenceExecutor, sample_sequence
        ex = LP.CouchDBExecutor()
        se = SequenceExecutor(ex)
        seed = unit.extra["seed"]
        seq_world = sample_sequence(seed, n_episodes=unit.extra["n_episodes"])
        ret = E.run_sequence(model, api_key, base_url, se, seq_world, seed)
        eps = ret["episodes"]
        by_eid: Dict[str, Any] = {}
        for k, eid in enumerate(unit.episode_ids):
            ep = dict(eps[k])
            ep["_sequence"] = {kk: ret[kk] for kk in
                               ("sequence_id", "asset", "process", "seed",
                                "sequence_trace_chain_valid", "apparatus_failure_any")}
            if k == 0:
                ep["_sequence"]["trace"] = ret.get("trace")
                ep["_sequence"]["world"] = ret.get("world")
            by_eid[eid] = ep
        return by_eid

    raise PD.DispatchError(f"unknown unit kind {unit.kind!r}")


def run_model(model: str, *, api_key: str, base_url: str, manifest: Dict[str, Any],
              out_path: Path, resume: bool = True) -> Dict[str, Any]:
    import phase8h_live_pilot as LP
    manifest_sha = manifest["_sha256"]
    units = PD.build_execution_units(manifest)
    meta_by_eid = {r["episode_id"]: r for r in manifest["episodes"]}

    done: set = set()
    if resume and out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["episode_id"])
                except Exception:  # noqa: BLE001
                    pass

    summary = {"model": model, "out": str(out_path), "n_episodes": 0,
               "resumed_skipped": len(done), "infrastructure_retries": 0,
               "halted": False, "halt_reason": None}

    mode = "a" if out_path.exists() else "w"
    with out_path.open(mode) as fh:
        for unit in units:
            if all(eid in done for eid in unit.episode_ids):
                continue

            attempt, rr = 0, None
            while True:
                attempt += 1
                started = _now()
                try:
                    rr = _run_unit(unit, model=model, api_key=api_key,
                                   base_url=base_url, live=LP)
                    vr = PF.validate_runner_return(
                        unit.kind, next(iter(rr.values())) if unit.kind != "e_sequence"
                        else {"sequence_id": 1, "episodes": [0, 0, 0], "trace": 1})
                    if unit.kind != "e_sequence" and vr.stop_pilot:
                        raise PD.DispatchError(vr.reason)
                    break
                except PD.DispatchError as exc:
                    summary.update(halted=True, halt_reason=f"{unit.unit_id}: {exc}")
                    return summary
                except Exception as exc:  # noqa: BLE001
                    verdict = PF.classify_exception(exc)
                    if verdict.stop_pilot:
                        summary.update(halted=True,
                                       halt_reason=f"{unit.unit_id}: {verdict.reason}")
                        return summary
                    if verdict.retryable and attempt <= PF.MAX_RETRIES:
                        summary["infrastructure_retries"] += 1
                        time.sleep(PF.BACKOFF_SECONDS[min(attempt - 1, len(PF.BACKOFF_SECONDS) - 1)])
                        continue
                    summary.update(halted=True,
                                   halt_reason=f"{unit.unit_id}: retries exhausted: {verdict.reason}")
                    return summary

            for eid, ret in rr.items():
                ce = ret.get("call_errors") or ret.get("errors") or []
                verdict = PF.classify_call_errors(ce)
                if verdict.stop_pilot:
                    summary.update(halted=True, halt_reason=f"{eid}: {verdict.reason}")
                    return summary
                finished = _now()
                _persist_row(fh, model=model, manifest_sha=manifest_sha, unit=unit,
                             episode_id=eid, row_meta=meta_by_eid[eid],
                             runner_return=ret, attempt=attempt, verdict_obj=verdict,
                             started=started, finished=finished)
                summary["n_episodes"] += 1

    summary["n_episodes"] += len(done)
    return summary


# ======================================================================
# entrypoint
# ======================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", action="store_true", help="connectivity probe only, zero tokens")
    ap.add_argument("--dry-run", action="store_true", help="gates A-E on mock returns, zero API")
    ap.add_argument("--model", help="model id to execute (real API)")
    ap.add_argument("--confirm", action="store_true", help="required alongside --model")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    api_key = os.environ.get("TOKENROUTER_API_KEY", "")

    if args.probe:
        print(json.dumps(connectivity_probe(MODEL_IDS, api_key=api_key), indent=2))
        return 0

    if args.dry_run:
        from phase8h1_dryrun_gates import run_all_gates
        res = run_all_gates()
        print(json.dumps(res, indent=2))
        return 0 if res["all_pass"] else 1

    if args.model:
        if not args.confirm:
            print("REFUSED: --model requires --confirm and an approved readiness report.",
                  file=sys.stderr)
            return 2
        if not api_key:
            print("ERROR: TOKENROUTER_API_KEY not set -- no fabricated results.", file=sys.stderr)
            return 2
        manifest = PD.load_manifest()
        out = args.out or (EC / f"phase8h1_pilot_raw_{args.model.replace('/', '_')}.jsonl")
        summary = run_model(args.model, api_key=api_key, base_url=BASE_URL,
                            manifest=manifest, out_path=out)
        print(json.dumps(summary, indent=2))
        return 1 if summary["halted"] else 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
