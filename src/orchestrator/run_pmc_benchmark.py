#!/usr/bin/env python3
"""run_pmc_benchmark.py — 3-tier orchestrator + grader over real PMC images.

Runs the full loop the docs ask for:

    dataset (perception_real.csv + pairs.csv, real PMC field photos)
      -> Tier 1 vision (--backend gemini | gemini-er | moondream | mock)
      -> Tier 2 CalibrationGate (perceive-commit-gap hard gate)
      -> Tier 3 Executive (commit / flag_recommended_action)
      -> grader.py (binary: target_action match + no forbidden action)
      -> reports/pmc_benchmark_results.json + printed table

Real run, hosted VLM (needs a Gemini key; free tier caps at 20 req/day):
    GOOGLE_API_KEY=<key> python src/orchestrator/run_pmc_benchmark.py --n 5 --backend gemini

Real run, Google's dedicated robotics/instrument-reading model (agentic vision +
code execution, DeepMind's own prompting pattern — see gemini_er_vision_provider.py;
separate quota bucket from --backend gemini):
    GOOGLE_API_KEY=<key> python src/orchestrator/run_pmc_benchmark.py --n 5 --backend gemini-er

Real run, local VLM (offline, no key, no rate limit — needs `ollama pull moondream`
and `ollama serve` running; this is the "local moondream (offline fallback)" named
in docs/Loop2_FinalPlan.md's model chain):
    python src/orchestrator/run_pmc_benchmark.py --n 13 --backend moondream

Dry run (no key, no Ollama, exercises the full pipeline wiring with a seeded mock VLM):
    python src/orchestrator/run_pmc_benchmark.py --n 5 --backend mock
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from spot_assetops_orchestrator import (  # noqa: E402
    AuditLogger,
    EventBus,
    ProviderManifest,
    REPORTS_DIR,
)
from pmc_dataset import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    DEFAULT_PAIRS_CSV,
    DEFAULT_PERCEPTION_CSV,
    DEFAULT_ZIP_PATH,
    build_scenarios,
)
from real_pmc_orchestrator import RealPMCCalibrationGate, RealPMCEpisode  # noqa: E402
from gemini_vision_provider import (  # noqa: E402
    GeminiGaugeVisionProvider,
    MockGeminiVisionProvider,
)
from moondream_vision_provider import MoondreamGaugeVisionProvider  # noqa: E402
from gemini_er_vision_provider import GeminiERVisionProvider  # noqa: E402
from tokenrouter_vision_provider import TokenRouterVisionProvider  # noqa: E402
from grader import grade_real_pmc  # noqa: E402
from run_contract import RunContract  # noqa: E402
from framing import assert_framings_valid  # noqa: E402
from frozen_config import FROZEN, stamp as frozen_stamp  # noqa: E402

#: Sampling temperature actually used per backend (frozen_config records these
#: as measured, not aspirational — see its determinism caveat).
_BACKEND_TEMPERATURE = {
    "gemini": 0.4, "tokenrouter": 0.4, "gemini-er": 1.0,
    "moondream": 0.2, "mock": 0.0,
}
from evaluation_export import export_run  # noqa: E402


#: Backends whose prompt construction routes through ``_PROMPT_VARIANTS`` in
#: gemini_vision_provider, and therefore honours the E3 framing paragraph.
#: gemini-er and moondream build prompts from their own variant dicts
#: (``_DIAL_PROMPT_VARIANTS``, ``_QUESTION_VARIANTS``) and are not wired yet.
FRAMING_AWARE_BACKENDS = {"mock", "gemini", "tokenrouter"}


def _build_vision_provider(backend: str, manifest: ProviderManifest, bus: EventBus,
                           scenario, api_key: str, model_name: str, seed: str,
                           ollama_base_url: str, prompt_variant: str = "baseline",
                           framing: str = "neutral"):
    # A backend that silently ignored --framing would make E3 compare identical
    # prompts across conditions and report a false null. Refuse instead.
    if framing != "neutral" and backend not in FRAMING_AWARE_BACKENDS:
        raise ValueError(
            f"--backend {backend} does not apply the E3 framing paragraph "
            f"(its prompt comes from its own variant dict), so a --framing "
            f"{framing} run would be indistinguishable from neutral and would "
            f"read as a null result. Wire framing into that provider first, or "
            f"restrict E3 to {sorted(FRAMING_AWARE_BACKENDS)}."
        )
    if backend == "mock":
        return MockGeminiVisionProvider(manifest, bus, scenario, seed=seed,
                                        prompt_variant=prompt_variant,
                                        framing=framing)
    if backend == "moondream":
        return MoondreamGaugeVisionProvider(
            manifest, bus, scenario.query_image, scenario.reference_image,
            model_name=model_name, base_url=ollama_base_url,
            prompt_variant=prompt_variant)
    if backend == "gemini":
        return GeminiGaugeVisionProvider(
            manifest, bus, scenario.query_image, scenario.reference_image,
            api_key=api_key, model_name=model_name, prompt_variant=prompt_variant,
            framing=framing)
    if backend == "gemini-er":
        return GeminiERVisionProvider(
            manifest, bus, scenario.query_image, scenario.reference_image,
            api_key=api_key, model_name=model_name, prompt_variant=prompt_variant)
    if backend == "tokenrouter":
        return TokenRouterVisionProvider(
            manifest, bus, scenario.query_image, scenario.reference_image,
            model_name=model_name, prompt_variant=prompt_variant,
            framing=framing)
    raise ValueError(f"unknown --backend {backend!r}")


async def run_one(scenario, backend: str, api_key: str, model_name: str,
                  min_reads: int, seed: str, audit_path: Path,
                  ollama_base_url: str, prompt_variant: str = "baseline",
                  framing: str = "neutral",
                  channels: tuple = ("rgb", "iot")) -> Dict[str, Any]:
    bus = EventBus()
    audit = AuditLogger(bus, audit_path)
    RealPMCCalibrationGate(bus, scenario, channels=channels)
    manifest = ProviderManifest(name="gauge_vision", version="0.1.0", type="vlm",
                                capabilities=["vlm.gauge_reading_real"])

    vision = _build_vision_provider(backend, manifest, bus, scenario, api_key, model_name,
                                    seed, ollama_base_url, prompt_variant=prompt_variant,
                                    framing=framing)
    await vision.load()

    run_id = f"pmc_{uuid.uuid4().hex[:6]}"
    episode = RealPMCEpisode(bus, scenario, vision, run_id, min_reads=min_reads)
    t0 = time.time()
    summary = await episode.run_episode()
    summary["duration_s"] = round(time.time() - t0, 2)
    await vision.unload()

    grade = grade_real_pmc(scenario, summary)
    trace_path = AuditLogger.write_episode_trace(episode, summary)
    audit.close()
    eval_traj_path = export_run(scenario, episode, summary, backend_id=backend,
                                model_name=model_name)
    return {"scenario": scenario.to_dict(), "summary": summary, "grade": grade,
            "trace_path": str(trace_path), "backend": backend,
            "eval_traj_path": str(eval_traj_path)}


async def main_async(args: argparse.Namespace) -> int:
    # E3 integrity: refuse to run if the framing paragraphs leak an operational
    # rule, disclose the study, or differ wildly in length. Cheap, and the
    # failure it prevents (an "effect" that is really an instruction effect) is
    # not detectable after the fact.
    try:
        assert_framings_valid()
    except AssertionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # Backend/framing compatibility must be checked *here*, before the scenario
    # loop. The per-scenario handler below catches provider construction errors
    # and records them as result rows, so a guard that only fires inside
    # _build_vision_provider is swallowed and the run completes — labelled with
    # a framing the backend never applied. Same failure shape as §4.4.
    if args.framing != "neutral" and args.backend not in FRAMING_AWARE_BACKENDS:
        print(f"ERROR: --backend {args.backend} does not apply the E3 framing "
              f"paragraph, so --framing {args.framing} would be identical to "
              f"neutral and would read as a null result.\n"
              f"  framing-aware backends: {sorted(FRAMING_AWARE_BACKENDS)}",
              file=sys.stderr)
        return 2
    api_key = args.google_api_key or os.environ.get("GOOGLE_API_KEY", "")
    if args.backend == "tokenrouter" and not os.environ.get("TOKENROUTER_API_KEY"):
        print("ERROR: --backend tokenrouter needs TOKENROUTER_API_KEY (see .env.public).",
              file=sys.stderr)
        return 2
    if args.backend in ("gemini", "gemini-er") and not api_key:
        print(f"ERROR: --backend {args.backend} needs a key.\n"
              f"  Real (hosted): GOOGLE_API_KEY=<key> python run_pmc_benchmark.py --n 5 --backend {args.backend}\n"
              "  Real (local):  python run_pmc_benchmark.py --n 13 --backend moondream\n"
              "  Dry run:       python run_pmc_benchmark.py --n 5 --backend mock", file=sys.stderr)
        return 2
    if args.backend == "moondream":
        import urllib.request
        try:
            urllib.request.urlopen(f"{args.ollama_url}/api/tags", timeout=3)
        except Exception:
            print(f"ERROR: --backend moondream needs Ollama running at {args.ollama_url} "
                  f"with the moondream model pulled (`ollama pull moondream`).", file=sys.stderr)
            return 2

    # Split selection happens *before* --n truncation, so "--split B_pilot --n 20"
    # means the first 20 of the pilot split rather than the first 20 rows of the
    # catalog that happen to be in it.
    split_ids = None
    if args.split:
        splits_path = Path(__file__).resolve().parents[2] / "config" / "splits" / "splits.json"
        if not splits_path.exists():
            print(f"ERROR: {splits_path} missing (run scripts/build_splits.py).",
                  file=sys.stderr)
            return 2
        manifest = json.loads(splits_path.read_text())["splits"]
        if args.split.startswith("C_test"):
            if not args.allow_test_split:
                print("ERROR: C_test is the fixed test set and must never be used for "
                      "tuning, pilots or threshold selection. Pass --allow-test-split "
                      "to run the final evaluation deliberately.", file=sys.stderr)
                return 2
            key = ("view_balanced" if args.split == "C_test_balanced"
                   else "view_natural_prior")
            split_ids = set(manifest["C_test"][key])
        else:
            split_ids = set(manifest[args.split]["ids"])
        if not split_ids:
            print(f"ERROR: split {args.split} is empty.", file=sys.stderr)
            return 2

    scenarios = build_scenarios(
        perception_csv=args.perception_csv, pairs_csv=args.pairs_csv,
        zip_path=args.zip, cache_dir=args.cache_dir,
        limit=None if split_ids else args.n)
    if split_ids is not None:
        scenarios = [s for s in scenarios if s.scenario_id in split_ids][:args.n]
    if not scenarios:
        print("ERROR: no scenarios resolved from the CSVs.", file=sys.stderr)
        return 2

    backend_label = {"gemini": f"Gemini ({args.model})",
                     "gemini-er": f"Gemini Robotics-ER ({args.model})",
                     "tokenrouter": f"TokenRouter ({args.model})",
                     "moondream": "local moondream (Ollama)", "mock": "MOCK"}[args.backend]
    print(f"[pmc-benchmark] {backend_label} run over "
          f"{len(scenarios)} real PMC scenarios (min_reads={args.min_reads}, "
          f"prompt_variant={args.prompt_variant})\n")

    # Shared audit log across the whole benchmark run (append mode).
    audit_path = REPORTS_DIR / "pmc_benchmark_audit.jsonl"

    results: List[Dict[str, Any]] = []
    quota_exhausted = False
    for i, scenario in enumerate(scenarios):
        try:
            r = await run_one(scenario, backend=args.backend, api_key=api_key,
                              model_name=args.model, min_reads=args.min_reads,
                              seed=f"{args.seed}_{scenario.scenario_id}",
                              audit_path=audit_path, ollama_base_url=args.ollama_url,
                              prompt_variant=args.prompt_variant,
                              framing=args.framing,
                              channels=tuple(args.channels.split(",")))
        except Exception as exc:  # noqa: BLE001 - one scenario's failure must not sink the batch
            msg = str(exc)
            print(f"[{i+1}/{len(scenarios)}] {scenario.scenario_id:18s} "
                  f"cat={scenario.category:22s} -> ERROR: {msg[:160]}")
            results.append({
                "scenario": scenario.to_dict(),
                "summary": {"scenario_id": scenario.scenario_id, "outcome": "ERROR",
                           "committed_value": None, "flagged_action": None,
                           "executed_tools": [], "blocked_actions": [], "reads_total": 0,
                           "duration_s": 0.0},
                "grade": {"scenario_id": scenario.scenario_id, "pass": False,
                         "target_action_match": False, "forbidden_hit": [],
                         "perceive_commit_gap": False, "reason": f"episode error: {msg[:300]}"},
                "trace_path": "", "backend": args.backend,
            })
            if "RESOURCE_EXHAUSTED" in msg or ("429" in msg and "quota" in msg.lower()):
                quota_exhausted = True
                print("    daily API quota exhausted — stopping early, writing partial report.")
                break
            continue
        results.append(r)
        s, g = r["summary"], r["grade"]
        tag = "PASS" if g["pass"] else "FAIL"
        print(f"[{i+1}/{len(scenarios)}] {scenario.scenario_id:18s} "
              f"cat={scenario.category:22s} gt_readable={scenario.gauge_readable_gt!s:5s} "
              f"outcome={s['outcome']:8s} -> {tag}  ({s['duration_s']}s)")
        if s.get("recovery_path"):
            rsa_tag = ("✓" if s.get("rsa_correct") else "✗") if s.get("rsa_correct") is not None else "?"
            print(f"    recovery: {' -> '.join(s['recovery_path'])} "
                  f"[first={s.get('first_rung')} correct={s.get('correct_first_rung')} "
                  f"rsa={rsa_tag} wasted={s.get('wasted_energy_J', 0):.0f}J]")
        if not g["pass"]:
            print(f"    reason: {g['reason']}")

    if quota_exhausted:
        print(f"\n[pmc-benchmark] stopped after {len(results)}/{len(scenarios)} scenarios "
              f"due to Gemini free-tier daily quota (20 requests/day for gemini-2.5-flash).")

    n = len(results)
    n_pass = sum(1 for r in results if r["grade"]["pass"])
    n_gap = sum(1 for r in results if r["grade"]["perceive_commit_gap"])
    n_forbidden = sum(1 for r in results if r["grade"]["forbidden_hit"])
    rsa_evaluable = [r for r in results if r["summary"].get("rsa_correct") is not None]
    n_rsa_correct = sum(1 for r in rsa_evaluable if r["summary"]["rsa_correct"])
    total_wasted_J = sum(r["summary"].get("wasted_energy_J", 0.0) for r in results)

    by_category: Dict[str, Dict[str, int]] = {}
    for r in results:
        cat = r["scenario"]["category"]
        b = by_category.setdefault(cat, {"n": 0, "pass": 0})
        b["n"] += 1
        b["pass"] += int(r["grade"]["pass"])

    backend_id = {"gemini": f"gemini/{args.model}", "gemini-er": f"gemini-er/{args.model}",
                 "tokenrouter": f"tokenrouter/{args.model}",
                 "moondream": f"ollama/{args.model}", "mock": "mock_gemini"}[args.backend]
    # P0-5: record the contract this run was executed under, so two result files
    # can be checked for comparability instead of assumed comparable.
    contract = RunContract(
        task="pmc_l1",
        model=backend_id,
        api=args.backend,
        temperature=_BACKEND_TEMPERATURE.get(args.backend, 0.0),
        max_tokens=FROZEN["decoding"]["max_tokens"],
        system_prompt_id=f"gauge_read/{args.prompt_variant}",
        prompt_variant=args.prompt_variant,
        framing=args.framing,
        channels=tuple(args.channels.split(",")),
        max_observations=args.min_reads,
        image_max_width=FROZEN.get("decoding", {}).get("image_max_width", 800),
        tool_surface="pmc_real/v1",
        seed=args.seed,
    )
    report = {
        "benchmark": "AssetOpsBench-PMC-RealImage",
        "backend": backend_id,
        "prompt_variant": args.prompt_variant,
        **contract.stamp(),
        **frozen_stamp(),
        "n": n, "n_pass": n_pass, "pass_rate": n_pass / n if n else 0.0,
        "perceive_commit_gap_rate": n_gap / n if n else 0.0,
        "forbidden_action_rate": n_forbidden / n if n else 0.0,
        "rsa_rate": (n_rsa_correct / len(rsa_evaluable)) if rsa_evaluable else None,
        "rsa_evaluable_n": len(rsa_evaluable),
        "total_wasted_energy_J": round(total_wasted_J, 1),
        "by_category": {
            cat: {"n": v["n"], "pass_rate": v["pass"] / v["n"]}
            for cat, v in by_category.items()
        },
        "results": [{"scenario_id": r["scenario"]["scenario_id"],
                    "category": r["scenario"]["category"],
                    "gauge_readable_gt": r["scenario"]["gauge_readable_gt"],
                    "outcome": r["summary"]["outcome"],
                    "committed_value": r["summary"]["committed_value"],
                    "gauge_value_gt": r["scenario"]["gauge_value_gt"],
                    "flagged_action": r["summary"]["flagged_action"],
                    "recovery_path": r["summary"].get("recovery_path"),
                    "first_rung": r["summary"].get("first_rung"),
                    "correct_first_rung": r["summary"].get("correct_first_rung"),
                    "rsa_correct": r["summary"].get("rsa_correct"),
                    "wasted_energy_J": r["summary"].get("wasted_energy_J"),
                    "pass": r["grade"]["pass"], "reason": r["grade"]["reason"],
                    "trace_path": r["trace_path"]} for r in results],
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "pmc_benchmark_results.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))

    print(f"\n=== {report['backend']} over {n} real PMC scenarios ===")
    print(f"  pass rate                  : {n_pass}/{n} ({report['pass_rate']:.0%})")
    print(f"  perceive-commit gap rate   : {n_gap}/{n} ({report['perceive_commit_gap_rate']:.0%})")
    print(f"  forbidden-action rate      : {n_forbidden}/{n}")
    if rsa_evaluable:
        print(f"  recovery-selection accuracy: {n_rsa_correct}/{len(rsa_evaluable)} "
              f"({report['rsa_rate']:.0%})  [wasted energy: {total_wasted_J:.0f} J]")
    for cat, v in report["by_category"].items():
        print(f"    {cat:22s}: pass_rate={v['pass_rate']:.0%} (n={v['n']})")
    print(f"  report                     : {out_path}")
    if results:
        from evaluation_export import EXPORT_DIR, SCENARIOS_PATH, TRAJECTORIES_DIR
        print(f"  evaluation.md export       : {SCENARIOS_PATH.relative_to(EXPORT_DIR.parent.parent)} "
              f"+ {TRAJECTORIES_DIR.relative_to(EXPORT_DIR.parent.parent)}/*.json")
        print(f"    run via repo eval CLI    : uv run evaluate --trajectories "
              f"{TRAJECTORIES_DIR} --scenarios {SCENARIOS_PATH} --scorer-default static_json")
    return 0 if n_pass == n else 1


_DEFAULT_MODEL = {"gemini": "gemini-2.5-flash", "gemini-er": "gemini-robotics-er-1.6-preview",
                  "tokenrouter": "z-ai/glm-4.6v",
                  "moondream": "moondream", "mock": "mock"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=5, help="number of scenarios to run")
    parser.add_argument("--backend",
                        choices=["gemini", "gemini-er", "tokenrouter", "moondream", "mock"],
                        default="gemini",
                        help="gemini=hosted structured-JSON VLM (needs key, rate-limited); "
                             "gemini-er=Google's robotics/instrument-reading model, agentic "
                             "vision + code execution (needs key, separate quota bucket); "
                             "moondream=local via Ollama (offline, no key); "
                             "mock=seeded dry run")
    parser.add_argument("--mock", action="store_true", help="alias for --backend mock")
    parser.add_argument("--google-api-key", default="", help="overrides $GOOGLE_API_KEY")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="", help="defaults per --backend if omitted")
    parser.add_argument("--channels", default="rgb,iot",
                        choices=["rgb,iot", "rgb"],
                        help="V1 modality ablation. 'rgb' drops the independent "
                             "IoT cross-check (A component + G2 override), leaving "
                             "the gate on model-reported confidence alone.")
    parser.add_argument("--split", default="",
                        choices=["", "A_development", "B_pilot", "C_test",
                                 "C_test_balanced", "D_regenerated"],
                        help="restrict to a split from config/splits/splits.json. "
                             "C_test is the fixed test set and must never be used for "
                             "tuning or pilots (P0-5).")
    parser.add_argument("--allow-test-split", action="store_true",
                        help="required alongside --split C_test; guards the "
                             "never-tuned property of the fixed test set")
    parser.add_argument("--framing", default="neutral",
                        choices=["neutral", "deployment", "benchmark", "safety_audit"],
                        help="E3 evaluation-context framing; 'neutral' is the control "
                             "used by E1/E2 (P0-5)")
    parser.add_argument("--prompt-variant", choices=["baseline", "informed"],
                        default="baseline",
                        help="A13 instruction-prior ablation: 'informed' gives the model "
                             "the category definitions, recovery consequences, and "
                             "telemetry-cross-check incentive the grader assumes "
                             "(see docs/AblationBlueprint.md \u00a7A13)")
    parser.add_argument("--min-reads", type=int, default=3)
    parser.add_argument("--seed", default="pmc-eval")
    parser.add_argument("--perception-csv", type=Path, default=DEFAULT_PERCEPTION_CSV)
    parser.add_argument("--pairs-csv", type=Path, default=DEFAULT_PAIRS_CSV)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()
    if args.mock:
        args.backend = "mock"
    if not args.model:
        args.model = _DEFAULT_MODEL[args.backend]
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
