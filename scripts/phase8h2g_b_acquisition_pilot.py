#!/usr/bin/env python3
"""phase8h2g_b_acquisition_pilot.py — Family B-Acquisition prototype model
pilot (Phase 8H.2G, §11).

Genuine 2-turn protocol (per the approved task's instruction to reuse the
SUFF-A1 bounded-interaction pattern), NOT a single-shot prediction:
  Turn 1: model sees the INITIAL evidence state + capability requirement,
          decides {acquire, requested_modality}.
  Turn 2 (only if acquire=True): model is shown the REAL
          request_observation response (RESOLVED with real metadata, or a
          genuine UNAVAILABLE with the real reason) and decides
          {claimed_observation_id, terminal_action}.
  If acquire=False after turn 1: terminal_action is asked for directly,
          no turn 2.

This is a construct-validation prototype pilot of the 16-episode candidate
pool only -- does not touch B-legacy, D-physical, or the global manifest.
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
OUT = REPO_ROOT / "reports" / "benchmark" / "b_acquisition_pilot"
OUT.mkdir(parents=True, exist_ok=True)

import b_acquisition_scoring as S  # noqa: E402
from b_acquisition_generator import BAcquisitionGold  # noqa: E402

MODEL_IDS = [
    "tokenrouter/openai/gpt-5.4-mini",
    "tokenrouter/qwen3.5-omni-plus",
    "tokenrouter/google/gemini-3.1-flash-image-preview",
]
MANIFEST_PATH = REPO_ROOT / "reports" / "benchmark" / "b_acquisition_manifest.json"
SCALED_MANIFEST_PATH = REPO_ROOT / "reports" / "benchmark" / "b_acquisition_scaled_manifest.json"


def load_episodes(manifest_paths: Optional[List[Path]] = None) -> List[Dict[str, Any]]:
    """Default (no argument): the original 16-episode prototype manifest,
    UNCHANGED behavior. Passing multiple paths (e.g. the 16-episode
    prototype + the 50-episode scaled manifest) merges their episode lists
    -- reads only, never regenerates or alters an episode definition.
    Duplicate episode_ids across manifests are rejected (a real error, not
    silently deduped) since the two manifests are supposed to be disjoint
    (verified at generation time -- b_acquisition_scaled_manifest.json
    drops any config that fingerprint-collides with the prototype)."""
    paths = manifest_paths or [MANIFEST_PATH]
    episodes: List[Dict[str, Any]] = []
    seen_ids: set = set()
    for p in paths:
        m = json.loads(p.read_text())
        for ep in m["episodes"]:
            if ep["episode_id"] in seen_ids:
                raise SystemExit(f"duplicate episode_id across manifests: {ep['episode_id']!r} "
                                 f"(from {p})")
            seen_ids.add(ep["episode_id"])
            episodes.append(ep)
    return episodes


TURN1_TEMPLATE = """You are an inspection agent. You have an initial evidence ledger for an inspection capability that has specific evidence requirements.

ASSET: {asset}
CAPABILITY REQUIRED EVIDENCE: {required_evidence}{optional_evidence_line}
CURRENT LEDGER (what you already have):
{ledger_state}

You have ONE tool available: request_observation(modality, max_age_s, quality_threshold) -- resolves one piece of sensor evidence for this asset, or returns UNAVAILABLE if none qualifies. This is the ONLY way to get more evidence; you cannot fabricate a reading.

TASK: Decide whether your current ledger already satisfies the capability's required evidence, or whether you need to acquire more.

Respond with ONLY a JSON object, no other text:
{{
  "acquire": true or false,
  "requested_modality": "<modality name>" or null
}}"""

TURN2_RESOLVED_TEMPLATE = """Your request_observation({modality}) call returned:
{{"status": "RESOLVED", "observation_id": "{observation_id}", "modality": "{modality}", "sensor_metadata": {sensor_metadata}}}

Based on this, make your terminal decision.

Respond with ONLY a JSON object, no other text:
{{
  "claimed_observation_id": "<the observation_id you are basing your decision on>" or null,
  "terminal_action": "COMMIT" or "ESCALATE"
}}"""

TURN2_UNAVAILABLE_TEMPLATE = """Your request_observation({modality}) call returned:
{{"status": "UNAVAILABLE", "reason": "no {modality!r} observations exist for asset {asset!r}"}}

No qualifying observation exists. You must NOT claim to have received one, and you must NOT invent a reading. Decide what to do given this genuine unavailability, following standard recovery policy: if evidence cannot be obtained, escalate.

Respond with ONLY a JSON object, no other text:
{{
  "claimed_observation_id": null,
  "terminal_action": "COMMIT" or "ESCALATE"
}}"""


def _real_initial_ledger_observation(ep: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Re-resolve `ep['initial_modality']` against the REAL, unmodified
    store at quality_threshold=0.0 (i.e. unfiltered by any capability
    requirement) -- this returns the actual underlying evidence the ledger
    was seeded from, including its REAL `quality` field, which the prior
    prompt omitted entirely (the B-ACQ-4 observability gap this fix
    addresses). Reads ONLY asset/scenario_id/modality from the episode
    record -- never `ep["gold"]`, never the arm label ("ambiguous"/
    "unambiguous"), never the capability's own quality_threshold (that
    stays visible separately, in CAPABILITY REQUIRED EVIDENCE, as a
    requirement the model must compare against -- not folded into this
    fact). Returns None if genuinely no record exists at all (distinct
    from "exists but low quality")."""
    import b_acquisition_generator as B
    from b_acquisition_generator import ObservationRequest
    resolver = B._resolver()
    req = ObservationRequest(asset_id=ep["asset"], inspection_id=ep["scenario_id"],
                             modality=ep["initial_modality"], max_age_s=999_999_999,
                             quality_threshold=0.0)
    result = resolver.resolve(req)
    if result.status != "RESOLVED" or result.record is None:
        return None
    return {"modality": ep["initial_modality"], "quality": result.record.quality}


def render_turn1(ep: Dict[str, Any]) -> str:
    cap = ep["capability"]
    required = cap.get("required_evidence", [])
    # Scoped fix: ONLY B-ACQ-4's construct depends on a quality_threshold
    # gate the agent must reason about -- B-ACQ-1/2/3's ledger_state
    # rendering is left BYTE-IDENTICAL to before this fix (verified by
    # test_b_acq_1_2_3_prompts_unchanged_by_the_quality_fix), so their
    # already-validated pilot behavior is not disturbed by this change.
    if ep["template_id"] == "B-ACQ-4":
        # SECOND real finding from live verification of the first fix: the
        # stored `capability` dict is CAP_B_ACQ_RECONCILIATION, a single
        # shared constant with a fixed quality_threshold=0.95 -- identical
        # for BOTH the "ambiguous" and "unambiguous" arms. Gold for the
        # "unambiguous" arm is actually computed against a DIFFERENT,
        # episode-specific threshold (0.5, stored in
        # ep["params"]["quality_threshold"], already present in the
        # existing manifest -- not a new field). Rendering the generic
        # capability dict verbatim would tell the model the SAME
        # requirement (0.95) in both arms, making them indistinguishable
        # even with the real quality value now shown. Substituting the
        # real per-episode threshold here is a pilot-prompt correction
        # only -- it reads an already-stored, already-gold-consistent
        # field, and changes no gold/generator logic whatsoever.
        # Scale-generation note: B-ACQ-4 now has TWO real (asset, modality)
        # pairs (acoustic -- original; thermal/motor_01 -- new, see
        # b_acquisition_scale_generation_report.md Sec 2). The override
        # below was originally hardcoded to "acoustic"; generalized here to
        # whichever modality THIS episode's own params record
        # (ep["params"]["modality"], added at generation time) -- a
        # mechanical widening of the SAME already-validated fix, not a new
        # template behavior. Falls back to "acoustic" for any episode
        # record that predates the "modality" params key (none in the
        # current manifests, kept for safety).
        target_modality = ep["params"].get("modality", "acoustic")
        required = [dict(r) for r in required]
        for r in required:
            if r.get("modality") == target_modality:
                r["quality_threshold"] = ep["params"]["quality_threshold"]
        real_obs = _real_initial_ledger_observation(ep)
        if real_obs is None:
            # Genuinely nothing on record for this modality/asset at all --
            # honest zero count, not the previously hardcoded initial_count.
            ledger_state = f"- {ep['initial_modality']}: 0 observation(s) delivered"
        else:
            # The REAL fact this fix adds: the delivered observation's
            # actual quality value. The model must compare it itself
            # against CAPABILITY REQUIRED EVIDENCE's own stated
            # quality_threshold (shown separately below) -- no verdict, no
            # "sufficient"/"insufficient" label, no gold field is stated
            # here.
            ledger_state = (f"- {real_obs['modality']}: 1 observation(s) delivered "
                            f"(quality={real_obs['quality']})")
    else:
        ledger_state = f"- {ep['initial_modality']}: {ep['initial_count']} observation(s) delivered"

    # Disclosure fix (this pass): the capability contract's own
    # `optional_evidence` field (e.g. CAP_B_ACQ_RECONCILIATION /
    # CAP_B_ACQ_RECONCILIATION_THERMAL's `iot_timeseries` entry) was
    # computed into gold's `required_acquisition` for B-ACQ-4's "ambiguous"
    # arm from the start, but was never rendered into the turn-1 prompt --
    # only `required_evidence` was shown. The agent was therefore asked to
    # select an acquisition target it was never told existed. This is a
    # real, structural capability-contract field (same shape as
    # `required_evidence`, computed by the generator, never gold/label/
    # fault-class derived) -- rendering it is a disclosure fix, not a gold
    # or acceptable-set change. Generic over `cap.get(...)`: B-ACQ-1/2/3's
    # contracts never declare `optional_evidence`, so this is empty for
    # them and their rendered prompt is BYTE-IDENTICAL to before this fix
    # (verified by test_b_acq_1_2_3_prompts_unchanged_by_the_disclosure_fix).
    optional = cap.get("optional_evidence", [])
    optional_evidence_line = (f"\nCAPABILITY OPTIONAL/COMPLEMENTARY EVIDENCE (may resolve "
                              f"ambiguity or reconcile the required evidence; not mandatory "
                              f"on its own): {json.dumps(optional)}") if optional else ""

    return TURN1_TEMPLATE.format(
        asset=ep["asset"], required_evidence=json.dumps(required), ledger_state=ledger_state,
        optional_evidence_line=optional_evidence_line,
    )


def get_real_acquisition_response(ep: Dict[str, Any], requested_modality: Optional[str]
                                  ) -> Dict[str, Any]:
    """Execute the REAL modality the MODEL requested, live against the real
    store -- NEVER gold's `required_acquisition`. This was a real
    environment-fidelity bug, found and fixed this pass: the prior version
    ignored `requested_modality` entirely and always resolved
    `ep["gold"]["required_acquisition"]`, so a model that requested a
    DIFFERENT modality than gold's was shown a turn-2 response falsely
    labeled as the result of its own request -- misleading, not merely a
    scoring inaccuracy (ASA itself was already computed from the model's
    real turn-1 output; the ENVIRONMENT EXECUTION was the unfaithful part).

    `quality_threshold=0.0` for the executed request is the same default
    the real `request_observation` MCP tool itself uses
    (src/servers/robot/main.py's `quality_threshold: float = 0.0` default)
    -- the correct, faithful choice given this pilot's turn-1 schema does
    not ask the model to specify one.

    A falsy/missing `requested_modality` (a malformed turn-1 response) is
    treated as a genuine UNAVAILABLE for an unspecified modality -- there
    is nothing real to resolve, so no observation is fabricated."""
    if not requested_modality:
        return {"status": "UNAVAILABLE", "modality": requested_modality or "<unspecified>"}
    import b_acquisition_generator as B
    resolver = B._resolver()
    from b_acquisition_generator import ObservationRequest
    # acquisition_exclude_ids: same mechanical, label-blind rank-2 selection
    # rule gold was computed against (see _THERMAL_PRESENT_EXCLUDE_RANK1_ID
    # in b_acquisition_generator.py) -- empty for every episode except
    # motor_01's B-ACQ-1/B-ACQ-3 present arm. Without this, a real model
    # requesting "thermal" here would be resolved against the naive rank-1
    # record while gold expects rank-2, reintroducing a gold/live-execution
    # divergence of exactly the kind fixed in the prior fidelity pass.
    exclude_ids = frozenset(ep.get("params", {}).get("acquisition_exclude_ids", []))
    req = ObservationRequest(asset_id=ep["asset"], inspection_id=ep["scenario_id"],
                             modality=requested_modality, max_age_s=999_999_999,
                             quality_threshold=0.0, exclude_ids=exclude_ids)
    result = resolver.resolve(req)
    if result.status == "RESOLVED":
        return {"status": "RESOLVED", "observation_id": result.record.observation_id,
               "modality": requested_modality, "sensor_metadata": result.record.sensor_metadata or {}}
    return {"status": "UNAVAILABLE", "modality": requested_modality}


def parse_json(text: str) -> Dict[str, Any]:
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
    out_path = OUT / f"b_acq_pilot_raw_{safe_name}{out_suffix}.jsonl"

    # Resume support: an interrupted prior run of this exact model/manifest
    # should not re-spend API calls on already-completed episodes (same
    # pattern added elsewhere this session after repeated session-teardown
    # interruptions). Any existing row -- error or not -- counts as done;
    # this script's own retry logic already handles transient failures
    # within an episode, so a persisted row reflects a real outcome.
    done_eids = set()
    resumed_scores = []
    if out_path.exists():
        with out_path.open() as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done_eids.add(rec.get("episode_id"))
                if rec.get("score") is not None:
                    resumed_scores.append(rec["score"])
        if done_eids:
            print(f"Resuming: {len(done_eids)}/{len(episodes)} episodes already complete, skipping those")

    scores = list(resumed_scores)  # dicts, matching score.to_dict() below
    with out_path.open("a") as fh:
        for ep in episodes:
            if ep["episode_id"] in done_eids:
                continue
            gold_dict = ep["gold"]
            gold = BAcquisitionGold(
                initial_evidence_sufficient=gold_dict["initial_evidence_sufficient"],
                acquisition_required=gold_dict["acquisition_required"],
                required_acquisition=gold_dict["required_acquisition"],
                acceptable_acquisition_set=frozenset(gold_dict["acceptable_acquisition_set"]),
                final_terminal_action=gold_dict["final_terminal_action"],
                acquisition_genuinely_unavailable=gold_dict["acquisition_genuinely_unavailable"],
                recovery_policy_on_unavailable=gold_dict["recovery_policy_on_unavailable"],
            )

            t0 = time.time()
            error = None
            turn1_raw = turn2_raw = ""
            actually_delivered_ids = set()
            executed_modality = None  # the modality the ENVIRONMENT actually resolved --
            # for transparency, must equal requested_modality whenever acquire=True (see
            # test_executed_modality_always_matches_requested_modality)
            delivered_observation = None  # the real ObservationRecord.sensor_metadata delivered,
            # or None -- recorded for the fidelity audit log, not used in scoring
            try:
                # --- turn 1 ---
                prompt1 = render_turn1(ep)
                r1 = backend.generate_with_usage(prompt1, temperature=0.0)
                turn1_raw = r1.text
                resp1 = parse_json(turn1_raw)
                acquire = bool(resp1.get("acquire"))
                requested_modality = resp1.get("requested_modality")

                if acquire:
                    real = get_real_acquisition_response(ep, requested_modality)
                    executed_modality = real["modality"]
                    if real["status"] == "RESOLVED":
                        actually_delivered_ids.add(real["observation_id"])
                        delivered_observation = {"observation_id": real["observation_id"],
                                                 "modality": real["modality"],
                                                 "sensor_metadata": real["sensor_metadata"]}
                        prompt2 = TURN2_RESOLVED_TEMPLATE.format(
                            modality=real["modality"], observation_id=real["observation_id"],
                            sensor_metadata=json.dumps(real["sensor_metadata"]))
                    else:
                        prompt2 = TURN2_UNAVAILABLE_TEMPLATE.format(
                            modality=real["modality"], asset=ep["asset"])
                    r2 = backend.generate_with_usage(prompt2, temperature=0.0)
                    turn2_raw = r2.text
                    resp2 = parse_json(turn2_raw)
                    claimed_observation_id = resp2.get("claimed_observation_id")
                    terminal_action = resp2.get("terminal_action")
                else:
                    # no acquisition -- ask for terminal action inline via a
                    # minimal follow-up (still 1 real call, keeping the
                    # protocol's turn count honest: 0 or 1 acquisition turns)
                    prompt2 = ('Given your decision not to acquire more evidence, make your '
                              'terminal decision. Respond with ONLY JSON: '
                              '{"claimed_observation_id": null, "terminal_action": "COMMIT" or "ESCALATE"}')
                    r2 = backend.generate_with_usage(prompt2, temperature=0.0)
                    turn2_raw = r2.text
                    resp2 = parse_json(turn2_raw)
                    claimed_observation_id = resp2.get("claimed_observation_id")
                    terminal_action = resp2.get("terminal_action")
            except Exception as e:
                error = str(e)
                acquire, requested_modality = None, None
                claimed_observation_id, terminal_action = None, None

            elapsed = time.time() - t0
            response = {"acquire": acquire, "requested_modality": requested_modality,
                       "claimed_observation_id": claimed_observation_id,
                       "terminal_action": terminal_action}
            score = S.score_b_acquisition_episode(
                gold, response, actually_delivered_observation_ids=frozenset(actually_delivered_ids))

            row = {
                "model": model_id, "episode_id": ep["episode_id"], "template_id": ep["template_id"],
                "asset": ep["asset"], "arm": ep["arm"],
                "matched_group_id": ep.get("matched_group_id") or ep["params"].get("matched_group_id"),
                "initial_evidence": {"modality": ep["initial_modality"], "count": ep["initial_count"]},
                "turn1_raw": turn1_raw, "turn2_raw": turn2_raw, "response": response,
                # Fidelity audit fields (this pass): what the environment actually executed vs
                # what the model requested, and what was actually delivered -- must always agree
                # when acquire=True (verified by test_executed_modality_always_matches_requested_modality).
                "executed_modality": executed_modality,
                "delivered_observation": delivered_observation,
                "modality_fidelity_ok": (executed_modality == requested_modality) if acquire else None,
                "error": error, "elapsed_s": round(elapsed, 2),
                "gold": gold_dict, "score": score.to_dict(),
            }
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
            scores.append(score.to_dict())
            print(f"  {ep['episode_id']}: ADA={score.ADA} ASA={score.ASA} UHA={score.UHA} "
                 f"TDA_post={score.TDA_post}{' ERROR:' + error if error else ''}")

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {
        "model": model_id, "n_episodes": len(scores),
        "ADA_mean": _mean([s["ADA"] for s in scores]),
        "ASA_mean": _mean([s["ASA"] for s in scores]),
        "MAR_mean": _mean([s["MAR"] for s in scores]),
        "UAR_mean": _mean([s["UAR"] for s in scores]),
        "TDA_post_mean": _mean([s["TDA_post"] for s in scores]),
        "UHA_mean": _mean([s["UHA"] for s in scores]),
        "UHA_n_applicable": sum(1 for s in scores if s["UHA"] is not None),
        "out_path": str(out_path),
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--template", help="restrict to one template_id, e.g. B-ACQ-4")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--manifest", action="append",
                    help="manifest path to load episodes from; pass multiple times to merge "
                         "(default: the original 16-episode prototype manifest only)")
    args = ap.parse_args()

    if args.probe:
        print(json.dumps(connectivity_probe(), indent=2))
        return 0

    if args.model:
        manifest_paths = [Path(p) for p in args.manifest] if args.manifest else None
        episodes = load_episodes(manifest_paths)
        if args.template:
            episodes = [e for e in episodes if e["template_id"] == args.template]
        if args.limit:
            episodes = episodes[:args.limit]
        print(f"B-acquisition pilot: {len(episodes)} episodes, model={args.model}")
        summary = run_model(args.model, episodes, confirm=args.confirm, out_suffix=args.out_suffix)
        summary_path = OUT / f"b_acq_pilot_summary_{args.model.replace('/', '_')}{args.out_suffix}.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return 0

    print("Nothing to do -- pass --probe or --model <id> --confirm")
    return 1


if __name__ == "__main__":
    sys.exit(main())
