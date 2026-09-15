#!/usr/bin/env python3
"""phase8h2e_dphys_pilot.py — Phase 8H.2E, Phase 11: D-physical model re-pilot.

Evaluates the SAME 3-model roster as the prior 3,939-episode benchmark pilot
(reports/ec/phase8h1_run_pilot.py's MODEL_IDS) against the 70 canonical
D-physical episodes in reports/benchmark/d_physical_manifest.json.

This is a construct-validation re-pilot of the NEW pool only -- it does not
touch the frozen 93-episode manifest, does not re-run A/B/C/D-enterprise/E,
and does not modify the global final_benchmark_manifest.json.

The prompt exposes ONLY agent-visible primitives (robot state, candidate
geometry, mission plan) -- NEVER the oracle's admissible/violated_constraints/
checks fields (Phase 3's evaluator-only boundary, structurally enforced
elsewhere and re-checked here by never reading spec.gold into the prompt).

Modes:
  --probe        connectivity check only, zero generation tokens
  --model <id>   run one model for real (requires --confirm), e.g.
                 tokenrouter/openai/gpt-5.4-mini
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))
sys.path.insert(0, str(REPO_ROOT / "src"))
OUT = REPO_ROOT / "reports" / "benchmark" / "dphys_pilot"
OUT.mkdir(parents=True, exist_ok=True)

import dphys_scoring as S  # noqa: E402
from dphys_generator import RelationalGold, CONSTRAINT_NAMES  # noqa: E402

MODEL_IDS = [
    "tokenrouter/openai/gpt-5.4-mini",
    "tokenrouter/qwen3.5-omni-plus",
    "tokenrouter/google/gemini-3.1-flash-image-preview",
]

MANIFEST_PATH = REPO_ROOT / "reports" / "benchmark" / "d_physical_manifest.json"


def load_canonical_episodes() -> List[Dict[str, Any]]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    eps = manifest["episodes"]
    fp_seen = set()
    canonical = []
    for e in eps:
        if e["fingerprint"] in fp_seen:
            continue
        fp_seen.add(e["fingerprint"])
        canonical.append(e)
    assert len(canonical) == manifest["canonical_final"], (
        f"dedup mismatch: {len(canonical)} != manifest canonical_final={manifest['canonical_final']}")
    return canonical


PROMPT_TEMPLATE = """You are a robot inspection agent. Before dispatching the robot arm to inspect a gauge, you must decide whether the inspection is physically feasible given the robot's current state and the target's geometry.

ASSET: {asset}
ROBOT STATE (from get_battery / get_pose / list_waypoints):
- battery_pct: {battery_pct}
- current pose: available via get_pose
- waypoints: available via list_waypoints, navigate_to

CANDIDATE INSPECTION GEOMETRY (agent-visible, observed at the target):
- candidate standoff distance(s) from the panel: {standoffs} meters
- panel mount height (z): {panel_z} meters
- approach ground slope: {slope} degrees
- estimated obstacle clearance required at this site: {clearance} meters
- panel latch grasp force required: {grasp_force} N
- planned mission: {route_legs} route leg(s), estimated {mission_duration_s} seconds

ROBOT'S KNOWN PHYSICAL ENVELOPE (public specification, not this episode's private state):
- max arm reach: 0.985 m
- arm workspace z-range: 0.0-1.8 m
- max gripper grasp force: 130.0 N
- max arm payload: 11.0 kg
- max stable approach slope: 30.0 degrees
- battery capacity: 605 Wh, locomotion 150W, arm actuation 50W, dock reserve 5%{reach_note}

TASK: Decide whether to DISPATCH the robot to attempt this inspection, or ESCALATE (do not dispatch; a human must resolve it). If multiple candidate standoffs are listed, you may select one to dispatch to.

You must reason about which of the six named physical constraints -- reach, joint_and_collision, clearance, grasp_payload, stability, energy -- are satisfied or violated given the numbers above, before deciding.

Respond with ONLY a JSON object, no other text:
{{
  "terminal_action": "DISPATCH" or "ESCALATE",
  "predicted_violated_constraints": [list of zero or more of: "reach", "joint_and_collision", "clearance", "grasp_payload", "stability", "energy"],
  "predicted_limiting_constraint": name of the single most-binding violated constraint, or "MULTIPLE" if two or more bind simultaneously, or null if none violated,
  "predicted_selected_standoff": the standoff value (meters) you would dispatch to, or null if escalating or only one candidate was given
}}"""


#: Fixed robot manufacturer constant (kinematics/analytic_arm.py::SHOULDER_H),
#: identical for every episode regardless of asset/constraint/margin -- not
#: derived from this episode's world or the oracle's verdict, so disclosing
#: it (and the formula that uses it) reveals no per-episode gold information.
#: See reports/benchmark/dphys_reach_fix_plan.md for the full root-cause
#: analysis: without this fact, the agent has standoff and panel height but
#: no way to combine them into a comparable "required reach" quantity.
_SHOULDER_HEIGHT_M = 0.75
_REACH_NOTE = (
    "\n- arm shoulder height: {shoulder_height} m (required reach = 3D straight-line "
    "distance from the arm shoulder to the target = sqrt(standoff^2 + (panel_z - "
    "shoulder_height)^2); dispatch requires required reach <= max arm reach AND panel_z "
    "within the workspace z-range)"
).format(shoulder_height=_SHOULDER_HEIGHT_M)


def _episode_tests_reach(ep: Dict[str, Any]) -> bool:
    """True iff `reach` is one of the constraints this episode's construct
    actually tests -- single:reach, or any coupled stratum containing reach
    (dual:reach+clearance, dual:reach+stability, dual:energy+reach,
    triple:reach+clearance+stability). CAP-X is intentionally excluded per
    dphys_reach_fix_plan.md's stated scope (its construct is candidate
    selection, not per-constraint attribution)."""
    params = ep["params"]
    if ep["template_id"] in ("T-D-PHYS-REACH", "T-D-PHYS-ENERGY"):
        return params["constraint"] == "reach"
    if ep["template_id"] == "T-D-PHYS-COUPLED":
        return "reach" in params["coupling"]
    return False


def render_prompt(ep: Dict[str, Any]) -> str:
    """Render the agent-visible prompt from an episode record. Reads ONLY
    `params` (the generating parameters) -- never `gold` -- to guarantee the
    oracle's verdict cannot leak into the prompt even by accident."""
    params = ep["params"]
    asset = ep["asset"]
    # Reconstruct the SAME access dict the generator built, to read the
    # agent-visible geometry fields -- without ever touching ep["gold"].
    import dphys_generator as D
    profiles = D.load_asset_profiles()
    verifier_registry = D.load_registry()
    from spot_admissibility_verifier import SpotAdmissibilityVerifier
    verifier = SpotAdmissibilityVerifier(verifier_registry)

    if ep["template_id"] in ("T-D-PHYS-REACH", "T-D-PHYS-ENERGY"):
        access = D._base_access(profiles, asset)
        access.update(D.MARGIN_OVERRIDES[params["constraint"]][params["margin"]](access, verifier))
        standoffs = [params["standoff"]]
    elif ep["template_id"] == "T-D-PHYS-COUPLED":
        access = D._base_access(profiles, asset)
        for c in params["coupling"]:
            access.update(D.MARGIN_OVERRIDES[c]["inadmissible"](access, verifier))
        standoffs = [params["standoff"]]
    else:  # T-D-PHYS-CAP-X
        access = D._base_access(profiles, asset)
        if params["config"] == "mixed_one_admissible" and asset in ("hydraulic_pump_1", "metro_pump_1"):
            access["obstacle_clearance_m"] = 0.65
        elif params["config"] == "all_inadmissible":
            access["obstacle_clearance_m"] = 1.20
        standoffs = params["standoffs"]

    return PROMPT_TEMPLATE.format(
        asset=asset, battery_pct=access.get("battery_pct", 100.0),
        standoffs=standoffs, panel_z=access["panel_pose"]["z"],
        slope=access.get("approach_slope_deg", 0.0),
        clearance=access.get("obstacle_clearance_m", 0.0),
        grasp_force=access.get("latch_grasp_force_N", 0.0),
        route_legs=access.get("route_legs", 1),
        mission_duration_s=access.get("mission_duration_s", 300.0),
        reach_note=(_REACH_NOTE if _episode_tests_reach(ep) else ""),
    )


def parse_response(text: Optional[str]) -> Dict[str, Any]:
    """Best-effort strict JSON extraction. Any parse failure yields an empty
    dict, which the scorer scores as CC=0 / all constraints wrong -- never
    crashes the pilot.

    `text` can be None (an API call that succeeds with no exception but
    returns an empty response body -- observed live on reasoning models
    that exhaust their output budget on hidden reasoning tokens before
    emitting any content, e.g. Qwen3.5-397B-A17B). This was a real gap
    against this function's own documented contract: json.loads(None)
    raised and was caught, but the fallback re.search(pattern, None) was
    not guarded and crashed the entire pilot run. Fixed to normalize None
    to "" up front, matching every other malformed-response case."""
    text = text or ""
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


def connectivity_probe() -> Dict[str, Any]:
    from llm.routers import resolve_router_creds
    out = {"probe": "connectivity_only", "models": {}}
    for mid in MODEL_IDS:
        try:
            creds = resolve_router_creds(mid)
            out["models"][mid] = {"base_url": creds.base_url, "api_key_present": bool(creds.api_key)}
        except Exception as e:
            out["models"][mid] = {"error": str(e)}
    return out


def run_model(model_id: str, episodes: List[Dict[str, Any]], *, confirm: bool,
             out_suffix: str = "") -> Dict[str, Any]:
    if not confirm:
        raise SystemExit("--confirm required alongside --model for a real API run")
    from llm.openai_compat import OpenAICompatBackend
    backend = OpenAICompatBackend(model_id)

    safe_name = model_id.replace("/", "_")
    out_path = OUT / f"dphys_pilot_raw_{safe_name}{out_suffix}.jsonl"
    scores = []
    with out_path.open("w") as fh:
        for ep in episodes:
            prompt = render_prompt(ep)
            t0 = time.time()
            try:
                result = backend.generate_with_usage(prompt, temperature=0.0)
                raw_text = result.text
                usage = {"input_tokens": result.input_tokens, "output_tokens": result.output_tokens}
                error = None
            except Exception as e:
                raw_text, usage, error = "", {}, str(e)
            elapsed = time.time() - t0

            response = parse_response(raw_text) if not error else {}
            gold_dict = ep["gold"]
            gold = RelationalGold(constraints=gold_dict["constraints"], admissible=gold_dict["admissible"],
                                  limiting_constraint=gold_dict["limiting_constraint"],
                                  selected_standoff=gold_dict["selected_standoff"])
            score = S.score_dphys_episode(gold, response)
            row = {
                "model": model_id, "episode_id": ep["episode_id"], "template_id": ep["template_id"],
                "constraint_stratum": ep["constraint_stratum"], "asset": ep["asset"],
                "prompt": prompt, "raw_response": raw_text, "parsed_response": response,
                "error": error, "elapsed_s": round(elapsed, 2), "usage": usage,
                "gold": gold_dict, "score": score.to_dict(),
            }
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
            scores.append(score)
            print(f"  {ep['episode_id']}: CC={score.CC} CSA={score.CSA} LCA={score.LCA}"
                 f"{' ERROR:' + error if error else ''}")

    n = len(scores)
    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {
        "model": model_id, "n_episodes": n,
        "CC_mean": _mean([s.CC for s in scores]),
        "PAC_mean": _mean([s.PAC for s in scores]),
        "UDR_mean": _mean([s.UDR for s in scores]),
        "ODR_mean": _mean([s.ODR for s in scores]),
        "CSA_mean": _mean([s.CSA for s in scores]),
        "CS_precision_mean": _mean([s.CS_precision for s in scores]),
        "CS_recall_mean": _mean([s.CS_recall for s in scores]),
        "CS_F1_mean": _mean([s.CS_F1 for s in scores]),
        "LCA_mean": _mean([s.LCA for s in scores]),
        "out_path": str(out_path),
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--model", help="model id to run for real, e.g. tokenrouter/openai/gpt-5.4-mini")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="smoke-test on only the first N episodes")
    ap.add_argument("--strata", help="comma-separated constraint_stratum values to restrict to "
                                     "(e.g. for a focused repair re-pilot); default: all canonical episodes")
    ap.add_argument("--out-suffix", default="", help="suffix for output filenames, e.g. '_reach_repair', "
                                                      "to avoid overwriting a prior full-pilot run")
    args = ap.parse_args()

    if args.probe:
        print(json.dumps(connectivity_probe(), indent=2))
        return 0

    if args.model:
        episodes = load_canonical_episodes()
        if args.strata:
            wanted = set(args.strata.split(","))
            episodes = [e for e in episodes if e["constraint_stratum"] in wanted]
        if args.limit:
            episodes = episodes[:args.limit]
        print(f"D-physical pilot: {len(episodes)} canonical episodes, model={args.model}")
        summary = run_model(args.model, episodes, confirm=args.confirm, out_suffix=args.out_suffix)
        summary_path = OUT / f"dphys_pilot_summary_{args.model.replace('/', '_')}{args.out_suffix}.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return 0

    print("Nothing to do -- pass --probe or --model <id> --confirm")
    return 1


if __name__ == "__main__":
    sys.exit(main())
