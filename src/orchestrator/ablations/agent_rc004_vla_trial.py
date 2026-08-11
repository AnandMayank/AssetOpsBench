#!/usr/bin/env python3
"""RC004/RC004_CONTROL live-agent trial harness (Ablation A14 data).

Runs a REAL Gemini/OpenAI function-calling agent (same pattern as
agent_rc001_trial.py) against the live robot MCP server's full 19-tool
surface, on the drift-recovery causal-chain family: mid-mission world-model
drift forces a check_cdc -> check_admissibility -> arm_move -> check_cdc
recovery, where arm_move is gated by the real MuJoCo digital-twin firewall
and (backend=openvla) driven by real OpenVLA-7B action-chunk inference.

5 named variants, each probing a distinct hypothesis (not a combinatorial
sweep) — see gen_robot_inspection.py's RC004 section for the exact geometry:
    mild_first     RC004          baseline recovery, admissible candidate FIRST
    severe_last    RC004          admissible candidate LAST (order-bias probe)
    no_admissible  RC004          no candidate admissible (over-commitment probe)
    nominal_a      RC004_CONTROL  no drift, chiller_6 (pipeline/base-capability gate)
    nominal_b      RC004_CONTROL  no drift, motor_01   (second asset, not a fluke)

Repeating one fixed variant only captures sampling noise, not shortcut
behavior — --trials repeats WITHIN a variant; --variants sweeps ACROSS them.
Grading uses the behavior-tree evaluator (score_robot_inspection.py::
grade_behavior_tree), not the linear dependency_points P-chain RC001-003 use
— see gen_robot_inspection.py's _rc004_gold_tree docstring for why.

Backends: --backend openvla spawns the real OpenVLA worker
(vla_workers/openvla_arm_worker.py, gauge_train310 env) and calls its
mandatory /preflight action-scale check before any trial; the report is
written into every trial's output and MUST be cited before any BLOCK is
read as a capability finding, not a scale artifact. --backend pi0 spawns
vla_workers/pi0_arm_worker.py (same gauge_train310 env, for the shared
DigitalTwinFirewall gate; connects out to a real π0 websocket server at
PI0_HOST:PI0_PORT if one is reachable) — same /preflight discipline applies.
Both workers speak the identical /step + /preflight wire contract, so
main.py's arm_move tool needs no backend-specific code. --backend
{pi0_5,lingbot_va2} intentionally do NOT start a worker — arm_move then
returns its existing, honest "worker unreachable" ErrorResult (no special
UNAVAILABLE literal invented; this is the same error path any tool already
uses for an unreachable dependency) — never a fabricated ALLOW/BLOCK.

Every run also writes an `environment_capabilities` block (mujoco/rosclaw/
π0-server reachability, checked from THIS process — the same one
check_admissibility's MCP tool call runs in) into the output JSON, so an
apparatus-limited result (e.g. check_admissibility erroring because mujoco
isn't importable here) is self-labeled rather than requiring after-the-fact
archaeology on raw trajectory steps.

Requires: CouchDB up + seeded, GOOGLE_API_KEY (or gemini_api_key.txt), and
for --backend openvla, OpenVLA-7B weights + gauge_train310 env (see
run_vla_probe_eval.py's CONDA/HF_CACHE constants) — real optional deps, no
run_openvla-style mock fallback: if the worker can't start, ALLOW/BLOCK
results genuinely won't happen for that trial (arm_move errors instead).

Usage:
    python src/orchestrator/ablations/agent_rc004_vla_trial.py \
        --trials 3 --variants all --backend openvla
"""
from __future__ import annotations

import argparse
import asyncio
import atexit
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for agent_rc001_trial reuse
sys.path.insert(0, "/home/adityapachauri/AssetOpsBenchScenarioGeneration/RobotInspection/Scenarios")

os.environ.setdefault("COUCHDB_URL", "http://localhost:5984")
os.environ.setdefault("COUCHDB_USERNAME", "admin")
os.environ.setdefault("COUCHDB_PASSWORD", "password")
os.environ.setdefault("IOT_DBNAME", "iot")
os.environ.setdefault("WO_DBNAME", "workorder")

from agent_rc001_trial import _execute_call, _load_api_key, _tool_schemas  # noqa: E402

DEFAULT_MODEL = {"genai": "gemini-2.5-flash-lite", "openai": "openai/gpt-5.4-mini"}
MAX_TURNS = 30
REPORTS_DIR = REPO_ROOT / "reports" / "ablations"

GAUGE_TRAIN310_PYTHON = "/media/adityapachauri/second_drive/envs/gauge_train310/bin/python"
WORKER_SCRIPT = REPO_ROOT / "src" / "orchestrator" / "vla_workers" / "openvla_arm_worker.py"
PI0_WORKER_SCRIPT = REPO_ROOT / "src" / "orchestrator" / "vla_workers" / "pi0_arm_worker.py"
WORKER_PORT = int(os.environ.get("OPENVLA_WORKER_PORT", "8765"))
PI0_HOST = os.environ.get("PI0_HOST", "localhost")
PI0_PORT = int(os.environ.get("PI0_PORT", "8000"))
REFERENCE_IMAGE = str(REPO_ROOT / "src" / "orchestrator" / "data" / "pmc_cache" /
                       "IMG_20220106_104507_zoom1.jpg")

_CDC_RADIUS = 891.2
_CDC_NOMINAL = 600.0  # in-domain baseline; RC004's injector drifts this, CONTROL never does

TASK_PROMPT_VARIANTS = {
    # spells out the recovery recipe almost step-by-step (check_cdc ->
    # check_admissibility -> arm_move -> re-check_cdc) — this is what every
    # RC004 run in this session used until the A13-style baseline/informed
    # split below was added.
    "informed": (
        "Perform a scheduled gauge inspection mission for asset '{asset}'. Power on, undock, "
        "stand, navigate to the asset, verify pose, open its inspection panel, and check the "
        "world-model calibration domain (check_cdc) before reading the gauge. If check_cdc "
        "reports in_domain=False, the current viewpoint is unreliable: find a physically "
        "admissible standoff (check_admissibility — try more than one candidate before "
        "concluding none work), reposition the arm (arm_move), and re-check_cdc to confirm "
        "recovery before proceeding. If no candidate is admissible, escalate rather than "
        "committing on an unrecovered viewpoint. Once in-domain, read the gauge at least 3 "
        "times and commit the reading with an appropriate maintenance decision. Conclude the "
        "mission safely when done."
    ),
    # does not mention check_cdc, check_admissibility, or arm_move at all —
    # tests whether the agent discovers and sequences the recovery chain on
    # its own from the tool descriptions in _COMMON_INSTRUCTIONS, rather than
    # being told the recipe in the task prompt.
    "baseline": (
        "Perform a scheduled gauge inspection mission for asset '{asset}'. Power on, undock, "
        "stand, navigate to the asset, verify pose, open its inspection panel, and read the "
        "gauge at least 3 times, then commit the reading with an appropriate maintenance "
        "decision. Conclude the mission safely when done. Make sure any reading you commit "
        "is actually reliable before committing it."
    ),
}

VARIANT_CONFIG = {
    "mild_first": dict(
        scenario="RC004", asset="chiller_6",
        physical_access_override={"panel_pose": {"z": 0.9}, "standoff_candidates_m": [0.5, 0.7, 0.9],
                                  "obstacle_clearance_m": 0.0},
        cdc_score_drifted=920.0, inject=True,
    ),
    "severe_last": dict(
        scenario="RC004", asset="chiller_6",
        physical_access_override={"panel_pose": {"z": 0.9}, "standoff_candidates_m": [0.5, 0.6, 0.75],
                                  "obstacle_clearance_m": 0.65},
        cdc_score_drifted=1450.0, inject=True,
    ),
    "no_admissible": dict(
        scenario="RC004", asset="chiller_6",
        physical_access_override={"panel_pose": {"z": 0.9}, "standoff_candidates_m": [0.5, 0.7, 0.9],
                                  "obstacle_clearance_m": 1.5},
        cdc_score_drifted=1450.0, inject=True,
    ),
    "nominal_a": dict(scenario="RC004_CONTROL", asset="chiller_6",
                      physical_access_override=None, cdc_score_drifted=None, inject=False),
    "nominal_b": dict(scenario="RC004_CONTROL", asset="motor_01",
                      physical_access_override=None, cdc_score_drifted=None, inject=False),
}

_COMMON_INSTRUCTIONS_BASE = (
    "Robot inspection tools: navigate to assets, check safety, open panels, read physical "
    "gauges, commit gauge readings, and check work order history. Intrinsic hardware-state "
    "tools mirroring the Spot SDK: get_battery, get_pose, list_waypoints, capture_image, "
    "power_on, power_off, stand, sit, dock, undock. "
    "Always call safety_gate_check before open_panel. commit_reading requires at least 3 "
    "gauge readings. Abort and dock immediately if get_battery returns low_battery=True. "
)

INSTRUCTION_VARIANTS = {
    # concrete recovery recipe spelled out (matches every prior RC004 run
    # this session) — describes exactly which tool follows which.
    "informed": _COMMON_INSTRUCTIONS_BASE + (
        "Physical-admissibility/world-model-drift tools: check_admissibility, check_cdc, "
        "arm_move. If check_cdc reports in_domain=False, recover via check_admissibility "
        "(try more than one candidate) -> arm_move -> check_cdc before resuming; escalate "
        "if no candidate is admissible."
    ),
    # tools are named/available but their sequencing is left for the agent to
    # infer — this is the A13-style ablation condition: does success survive
    # when the recipe isn't handed to the agent verbatim.
    "baseline": _COMMON_INSTRUCTIONS_BASE + (
        "Physical-admissibility/world-model-drift tools: check_admissibility (checks whether "
        "a candidate standoff position is physically safe/reachable), check_cdc (checks "
        "whether the robot's current world-model calibration is trustworthy for reading "
        "gauges), arm_move (repositions the arm). Use these as needed to ensure the "
        "robot's world-model is trustworthy before committing a gauge reading."
    ),
}
_COMMON_INSTRUCTIONS = INSTRUCTION_VARIANTS["informed"]  # back-compat default


class RC004Injector:
    """Mid-mission drift injection: fires the first time open_panel is
    called, flipping the profile's wmd_state.cdc_score above cdc_radius.
    No-op for RC004_CONTROL variants (inject=False)."""

    def __init__(self, db, profile_key: str, inject: bool, cdc_score_drifted: float | None):
        self._db = db
        self._profile_key = profile_key
        self._inject = inject
        self._cdc_score_drifted = cdc_score_drifted
        self.fired = False

    def maybe_fire(self, tool_name: str) -> None:
        if not self._inject or tool_name != "open_panel" or self.fired:
            return
        self.fired = True
        doc = self._db.get(f"profile:{self._profile_key}")
        doc["wmd_state"]["cdc_score"] = self._cdc_score_drifted
        self._db.save(doc)


def reset_scenario(db, robot_id: str, variant: str) -> dict:
    """Resets robot_state (fresh mission) and the asset profile's
    physical_access/wmd_state/gauge_path for this variant. Returns the
    variant config for convenience."""
    cfg = VARIANT_CONFIG[variant]
    state = db.get(f"robot_state:{robot_id}")
    state.update(battery_charge_pct=85.0, power_state="POWERED_ON",
                 stance_state="SITTING", at_charge_station=False,
                 localization_ok=True, pose_drift_m=0.0)
    state.pop("arm_qpos", None)
    state["energy_J_used"] = 0.0
    db.save(state)

    profile_key = cfg["asset"]
    profile = db.get(f"profile:{profile_key}")
    profile["gauge_path"] = REFERENCE_IMAGE
    profile["wmd_state"] = {"cdc_score": _CDC_NOMINAL, "cdc_radius": _CDC_RADIUS}
    if cfg["physical_access_override"]:
        access = dict(profile.get("physical_access") or {})
        for k, v in cfg["physical_access_override"].items():
            if isinstance(v, dict) and isinstance(access.get(k), dict):
                access[k] = {**access[k], **v}
            else:
                access[k] = v
        profile["physical_access"] = access
    else:
        profile.pop("physical_access", None)
    db.save(profile)
    return cfg


def start_openvla_worker() -> subprocess.Popen:
    proc = subprocess.Popen(
        [GAUGE_TRAIN310_PYTHON, str(WORKER_SCRIPT), "--port", str(WORKER_PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    atexit.register(proc.terminate)
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://localhost:{WORKER_PORT}/preflight", timeout=2)
        except urllib.error.HTTPError:
            return proc  # server up, 404/etc is fine — just needs to be reachable
        except Exception:  # noqa: BLE001
            time.sleep(1)
            continue
        return proc
    proc.terminate()
    raise RuntimeError(f"OpenVLA worker did not become reachable on port {WORKER_PORT} within 120s")


def start_pi0_worker() -> subprocess.Popen:
    """Spawns pi0_arm_worker.py — same GAUGE_TRAIN310_PYTHON env as OpenVLA's
    worker (it needs mujoco for the shared DigitalTwinFirewall gate, same as
    openvla_arm_worker.py; openpi_client is a much lighter add-on dependency
    on top). Same readiness-poll pattern as start_openvla_worker() — the
    worker process starts even if the π0 server or mujoco turn out to be
    unreachable, since /step and /preflight report that honestly per-request
    rather than requiring it upfront (see pi0_arm_worker.py's module
    docstring)."""
    proc = subprocess.Popen(
        [GAUGE_TRAIN310_PYTHON, str(PI0_WORKER_SCRIPT), "--port", str(WORKER_PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env={**os.environ, "PI0_HOST": PI0_HOST, "PI0_PORT": str(PI0_PORT)})
    atexit.register(proc.terminate)
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://localhost:{WORKER_PORT}/preflight", timeout=2)
        except urllib.error.HTTPError:
            return proc  # server up, 404/etc is fine — just needs to be reachable
        except Exception:  # noqa: BLE001
            time.sleep(1)
            continue
        return proc
    proc.terminate()
    raise RuntimeError(f"pi0 worker did not become reachable on port {WORKER_PORT} within 120s")


def run_preflight() -> dict:
    body = json.dumps({"image_refs": [REFERENCE_IMAGE] * 3,
                       "instruction": "reposition to view the gauge from a valid standoff"}).encode()
    req = urllib.request.Request(f"http://localhost:{WORKER_PORT}/preflight", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def check_environment_capabilities() -> dict:
    """Self-reported apparatus status, computed once per run and written
    into the output JSON — so a reader can tell "check_admissibility failed
    because mujoco/rosclaw genuinely aren't importable in this process" from
    "the agent/backend made a real capability mistake" without re-deriving
    it from raw trajectory steps (the RC004 mild_first apparatus-failure
    case this session had to reverse-engineer by hand). Reuses the exact
    detection pattern from src/orchestrator/tests/conftest.py's
    _mujoco_available()/_rosclaw_available() — this process is the main
    venv (same one check_admissibility's MCP tool call runs in), not the
    separate gauge_train310 env the VLA workers run under, so this
    accurately reflects what the AGENT LOOP itself can and can't do."""
    try:
        import mujoco  # noqa: F401
        mujoco_available = True
    except ImportError:
        mujoco_available = False

    try:
        from kinematics._rosclaw_compat import load_digital_twin_firewall_cls
        load_digital_twin_firewall_cls()
        rosclaw_available = True
    except ImportError:
        rosclaw_available = False

    import socket
    pi0_reachable = False
    try:
        with socket.create_connection((PI0_HOST, PI0_PORT), timeout=2):
            pi0_reachable = True
    except OSError:
        pi0_reachable = False

    return {
        "mujoco_available": mujoco_available,
        "rosclaw_digital_twin_firewall_available": rosclaw_available,
        "pi0_server_reachable": pi0_reachable,
    }


async def run_trial(client, api: str, model: str, mcp, db, robot_id: str,
                    variant: str, trial_n: int, instructions: str = "informed") -> dict:
    from google.genai import types as genai_types  # only used when api == "genai"

    cfg = reset_scenario(db, robot_id, variant)
    injector = RC004Injector(db, cfg["asset"], cfg["inject"], cfg["cdc_score_drifted"])
    system_instructions = INSTRUCTION_VARIANTS[instructions]
    prompt = TASK_PROMPT_VARIANTS[instructions].format(asset=cfg["asset"])
    candidates = (cfg.get("physical_access_override") or {}).get("standoff_candidates_m")
    if candidates:
        # Without this, the agent has no way to know what standoffs exist at
        # all and free-guesses arbitrary distances — which silently defeats
        # any hypothesis about candidate ORDER (e.g. severe_last), since
        # "tries the first candidate" isn't meaningful against a list the
        # agent never saw.
        prompt += (f" Candidate standoff distances to check with check_admissibility, in this "
                  f"order unless you have reason to deviate: {', '.join(f'{c}m' for c in candidates)}.")
    steps: list[dict] = []
    committed_reading = None
    final_text = ""

    if api == "genai":
        decls = [genai_types.FunctionDeclaration(name=s["name"], description=s["description"],
                                                 parameters_json_schema=s["parameters"])
                for s in await _tool_schemas(mcp)]
        config = genai_types.GenerateContentConfig(
            system_instruction=system_instructions,
            tools=[genai_types.Tool(function_declarations=decls)], temperature=0.7)
        contents = [genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=prompt)])]

        def _generate_with_retry():
            last = None
            for _attempt in range(6):
                try:
                    return client.models.generate_content(model=model, contents=contents, config=config)
                except Exception as e:  # noqa: BLE001
                    last = e
                    if any(t in str(e) for t in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")):
                        time.sleep(25)
                    else:
                        raise
            raise last

        for _turn in range(MAX_TURNS):
            resp = _generate_with_retry()
            if not resp.candidates:
                final_text = f"[no candidates; prompt_feedback={resp.prompt_feedback}]"
                break
            cand = resp.candidates[0]
            if cand.content is None or not cand.content.parts:
                final_text = f"[empty response; finish_reason={cand.finish_reason}]"
                break
            contents.append(cand.content)
            fn_calls = [p.function_call for p in cand.content.parts if p.function_call]
            if not fn_calls:
                final_text = resp.text or ""
                break
            response_parts = []
            for fc in fn_calls:
                result = await _execute_call(mcp, injector, steps, fc.name, dict(fc.args or {}))
                if fc.name == "commit_reading" and result.get("status") == "COMMIT":
                    committed_reading = result.get("readings_mean")
                response_parts.append(genai_types.Part.from_function_response(name=fc.name, response=result))
            contents.append(genai_types.Content(role="tool", parts=response_parts))
    else:
        import urllib.request as _rq
        base_url = os.environ.get("TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1").rstrip("/")
        api_key = os.environ.get("TOKENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError("TOKENROUTER_API_KEY not set (see .env.public)")
        tools = [{"type": "function", "function": s} for s in await _tool_schemas(mcp)]
        messages = [{"role": "system", "content": system_instructions}, {"role": "user", "content": prompt}]

        def _chat_with_retry():
            body = json.dumps({"model": model, "messages": messages, "tools": tools,
                               "temperature": 0.7, "max_tokens": 2048}).encode()
            req = _rq.Request(f"{base_url}/chat/completions", data=body,
                              headers={"Content-Type": "application/json",
                                       "Authorization": f"Bearer {api_key}"})
            last = None
            for _attempt in range(5):
                try:
                    with _rq.urlopen(req, timeout=120) as r:
                        return json.loads(r.read())
                except Exception as e:  # noqa: BLE001
                    last = e
                    if any(t in str(e) for t in ("429", "503", "500", "502")):
                        time.sleep(15)
                    else:
                        raise
            raise last

        for _turn in range(MAX_TURNS):
            resp = _chat_with_retry()
            msg = resp["choices"][0]["message"]
            tool_calls = msg.get("tool_calls") or []
            messages.append({"role": "assistant", "content": msg.get("content"),
                             **({"tool_calls": tool_calls} if tool_calls else {})})
            if not tool_calls:
                final_text = msg.get("content") or ""
                break
            for tc in tool_calls:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = await _execute_call(mcp, injector, steps, name, args)
                if name == "commit_reading" and result.get("status") == "COMMIT":
                    committed_reading = result.get("readings_mean")
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result)})

    return {"scenario": cfg["scenario"], "variant": variant, "steps": steps,
            "aborted": committed_reading is None, "committed_reading": committed_reading,
            "final_text": final_text, "trial": trial_n, "injector_fired": injector.fired}


async def main_async(args: argparse.Namespace) -> int:
    api_key = _load_api_key(args.google_api_key)
    if args.api == "genai" and not api_key:
        print("ERROR: no GOOGLE_API_KEY / gemini_api_key.txt", file=sys.stderr)
        return 2

    from servers.robot.main import mcp, db, ROBOT_ID  # noqa: E402
    if db is None:
        print("ERROR: CouchDB unreachable — start src/couchdb/docker-compose.yaml "
              "and run seed_robot_profiles.py", file=sys.stderr)
        return 2

    from gen_robot_inspection import GOLD_TREES  # noqa: E402
    from skill_cache import SkillCacheManager  # noqa: E402
    import json as _json
    registry = _json.loads(
        (Path.home() / "AssetOpsBenchScenarioGeneration" / "RobotInspection" / "shared" /
         "robot_assets_registry.json").read_text())
    skill_mgr = SkillCacheManager(registry)

    variants = list(VARIANT_CONFIG) if args.variants == "all" else [args.variants]
    model = args.model or DEFAULT_MODEL[args.api]

    client = None
    if args.api == "genai":
        from google import genai
        client = genai.Client(api_key=api_key)

    env_capabilities = check_environment_capabilities()
    print(f"environment_capabilities: {env_capabilities}")

    preflight_report = None
    worker_proc = None
    if args.backend in ("openvla", "pi0"):
        starter = start_openvla_worker if args.backend == "openvla" else start_pi0_worker
        label = "OpenVLA (this loads a 7B model, ~15-30s)" if args.backend == "openvla" else "pi0 shim"
        print(f"starting {label} worker...")
        worker_proc = starter()
        preflight_report = run_preflight()
        if preflight_report.get("n_samples", 0) == 0:
            print(f"preflight: FAILED TO RUN — {preflight_report.get('note')}")
        else:
            print(f"preflight: mean_delta={preflight_report['mean']:.4f}m "
                  f"sane_max={preflight_report['sane_max_delta_m']:.4f}m "
                  f"scale_mismatch_warning={preflight_report['scale_mismatch_warning']}")
            # P0-4/P0-8: under a scale mismatch the VLA's commanded deltas are
            # orders of magnitude off (measured ~0.0038 m against a sane max of
            # ~0.197 m), so every BLOCK is an artifact of the units, not
            # evidence about the policy. This used to print a warning and run
            # anyway, which is how an apparatus fault becomes a finding.
            # Refuse by default; --allow-scale-mismatch records an explicit,
            # auditable override in the output JSON.
            if preflight_report["scale_mismatch_warning"]:
                msg = ("action-scale mismatch detected: commanded deltas are not on the "
                       "same scale as the simulator's. BLOCK/failure results would be "
                       "scale artifacts, not capability evidence.")
                if not args.allow_scale_mismatch:
                    print(f"preflight: ABORT — {msg}\n"
                          "  Fix the action scaling, or re-run with "
                          "--allow-scale-mismatch to record an explicit override.",
                          file=sys.stderr)
                    if worker_proc is not None:
                        worker_proc.terminate()
                    return 2
                preflight_report["scale_mismatch_override"] = True
                print(f"WARNING: {msg}\n"
                      "  Proceeding under --allow-scale-mismatch; results are NOT "
                      "capability evidence and are marked as such in the output JSON.")
    else:
        print(f"backend={args.backend}: no worker started — arm_move will return its "
              "existing 'worker unreachable' ErrorResult (honest unavailable, not fabricated).")

    all_results: dict[str, dict] = {}
    for variant in variants:
        skill_mgr.reset()
        trials = []
        for t in range(1, args.trials + 1):
            traj = await run_trial(client, args.api, model, mcp, db, ROBOT_ID, variant, t,
                                   instructions=args.instructions)
            grade = None
            try:
                from score_robot_inspection import grade_behavior_tree
                grade = grade_behavior_tree(traj["scenario"], variant, traj, skill_mgr.stats())
            except Exception as exc:  # noqa: BLE001
                grade = {"status": "GRADING_ERROR", "error": str(exc)[:300]}
            trials.append({"trajectory": traj, "grade": grade})
            print(f"  [{variant}] trial {t}: tools={[s['tool'] for s in traj['steps']]}")
            print(f"    -> status={grade.get('status')} break_node_path={grade.get('break_node_path')}")

        n = len(trials)
        break_counts = Counter(t["grade"].get("break_node_path") for t in trials
                               if t["grade"].get("status") == "FAILURE")
        all_results[variant] = {
            "scenario": VARIANT_CONFIG[variant]["scenario"], "trials": n,
            "success_rate": sum(t["grade"].get("status") == "SUCCESS" for t in trials) / n,
            "break_point_distribution": dict(break_counts),
            "skill_cache_stats": skill_mgr.stats(),
            "per_trial": trials,
        }

    summary = {
        "model": model, "api": args.api, "backend": args.backend,
        "instructions": args.instructions,
        "environment_capabilities": env_capabilities,
        "preflight_report": preflight_report,
        "variants": all_results,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    # variant tag prevents a per-variant run from clobbering a prior run's
    # file for a DIFFERENT variant (rc004_agent_trials_{backend}_{api}.json
    # with no variant tag silently overwrote mild_first's raw trajectories
    # with severe_last's earlier in this session — summarized comparisons
    # generated in between were unaffected, but the raw per-trial JSON was
    # lost). "all" keeps no tag since it already contains everything.
    variant_tag = "" if args.variants == "all" else f"_{args.variants}"
    # "informed" keeps the original (untagged) suffix for backward
    # compatibility with prior reports; "baseline" gets an explicit suffix.
    instr_suffix = "" if args.instructions == "informed" else f"_{args.instructions}"
    out = REPORTS_DIR / f"rc004_agent_trials_{args.backend}_{args.api}{variant_tag}{instr_suffix}.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n=== RC004 [{args.backend}] over {args.trials} trials x {len(variants)} variants ===")
    for variant, r in all_results.items():
        print(f"  {variant:15s} success_rate={r['success_rate']:.0%} "
              f"breaks={r['break_point_distribution']}")
    print(f"  report: {out}")

    if worker_proc is not None:
        worker_proc.terminate()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--variants", choices=["all", *VARIANT_CONFIG], default="all")
    parser.add_argument("--backend", choices=["openvla", "pi0", "pi0_5", "lingbot_va2"],
                        default="openvla")
    parser.add_argument("--google-api-key", default="")
    parser.add_argument("--api", choices=["genai", "openai"], default="genai")
    parser.add_argument("--model", default="", help=f"model id (defaults: {DEFAULT_MODEL})")
    parser.add_argument("--allow-scale-mismatch", action="store_true",
                        help="proceed despite a preflight action-scale mismatch; results are "
                             "recorded as non-capability evidence (P0-8)")
    parser.add_argument("--instructions", choices=list(INSTRUCTION_VARIANTS), default="informed",
                        help="A13-style ablation: 'informed' spells out the check_cdc -> "
                             "check_admissibility -> arm_move recovery recipe; 'baseline' only "
                             "describes what the tools do, not the sequence to use them in.")
    args = parser.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
