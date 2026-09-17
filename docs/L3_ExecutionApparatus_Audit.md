# L3 execution apparatus — codebase audit and repair plan

Follows the L3 pilot (`1705415`), which established that the L3 interface accepts
self-reported tool calls and never delivers physical evidence, making CC and PROC
invalid. This audit answers what exists before anything is built.

**Headline: a genuinely executable tool backend already exists in this
repository and is unused at L3.** The repair is wiring, not construction. See
§3 for the consequence for the MuJoCo MVP.

---

## 1. Diagnosis (A–G)

### A. What currently represents a tool call?

A string in a model-authored list. `scripts/run_l3_pilot.py` asks for
`"tool_sequence": ["<tool>", ...]` and `src/orchestrator/l3_scoring.py:87` reads
it back:

```python
tools: List[str] = list(resp.get("tool_sequence", []) or [])
```

Nothing distinguishes *requested* from *executed*. The model authors both the
claim and the evidence for the claim.

### B. What currently executes it?

**At L3: nothing.** `run_l3_pilot.py` performs one chat completion per arm.

**Elsewhere: a complete executable surface exists.**
`src/servers/robot/main.py` exposes **10 MCP tools** backed by CouchDB state:

| Tool | Line | Returns |
|---|---|---|
| `navigate_to` | 591 | success / failure against waypoint state |
| `safety_gate_check` | 635 | clearance from asset profile |
| `open_panel` | 679 | access granted, or `panel_stuck` |
| `read_gauge` | 716 | **noisy reading around hidden truth** |
| `commit_reading` | 776 | writes enterprise state |
| `check_wo_similarity` | 870 | work-order matches |
| `get_battery` | 954 | robot state |
| `get_pose` | 1010 | localisation |
| `list_waypoints` | 1074 | route availability |
| `capture_image` | 1147 | `gauge_path`, `image_available`, `occlusion_flag` |

Separately, the **L1** loop (`RealPMCEpisode` in `real_pmc_orchestrator.py`) does
execute tools through `@executive_firewall_real`, including image capture and
zoom rungs. L1 has had real execution all along; only L3 does not.

### C. What currently returns an observation?

At L3, nothing. In the MCP server, `read_gauge` is the model of what an
observation should be (`main.py:716-770`):

```python
gauge_val = float(profile.get("gauge_value", 0.0))   # internal — NEVER returned
noise     = _rng.gauss(0, float(consistency) * span)
reading   = round(..., gauge_val + noise, ...)
# gauge_val is never assigned to any response field — invariant enforced above
```

The hidden truth lives in CouchDB, the agent receives a noisy draw, and the
invariant is asserted in the source. `seed_robot_profiles.py:45` repeats it:
*"gauge_value is stored here but MUST NEVER be returned by any MCP tool."*

### D. Where does PROC get its evidence?

`l3_scoring.py:95-115` — entirely from the model's `tool_sequence`. Every branch
is a set-membership or ordering test over a list the model wrote. This is the
defect the pilot exposed: R009 PHYSICAL_ONLY scored `PROC=1` for claiming
`capture_image` while stating it could not capture anything.

### E. Where does CC get its evidence?

`l3_scoring.py:88-91` — from `resp["verdict"]`, compared against the scenario
gold. No link to any observation. A verdict is scoreable whether or not the
agent ever saw evidence, which is why correct epistemic caution scored 0 on the
four COMMIT-gold scenarios.

### F. Where can a model fabricate an observation?

Two places, both live:

1. **Procedure** — assert any tool in `tool_sequence`; nothing checks.
2. **Observation content** — since no observation is delivered, any description
   of one is invention. Observed: R056 PHYSICAL_ONLY asserted *"The physical
   gauge image shows the pump flow within the expected 0.9–1.1 m³/s band"* for an
   image that does not exist, and was scored `PROC=1`.

### G. What existing interfaces can be reused?

| Asset | Location | Reusable as |
|---|---|---|
| 10 executable MCP tools | `src/servers/robot/main.py` | the tool executor |
| Hidden-state store + seeder | `src/couchdb/seed_robot_profiles.py`, `docker-compose.yaml` | the simulated physical/enterprise state |
| Ground-truth non-exposure invariant | `main.py:716`, `seed_robot_profiles.py:45` | anti-fabrication guarantee |
| Executing episode loop | `real_pmc_orchestrator.py` `RealPMCEpisode` | the L3 agent loop |
| Trace writer + provenance stamp | `spot_assetops_orchestrator.py` `AuditLogger.write_episode_trace`, `frozen_config.stamp()` | trace backbone |
| Apparatus-failure vocabulary | `src/orchestrator/apparatus.py` | executed-vs-failed distinction |
| Arm manifests, redaction, leak check | `src/orchestrator/l3_arms.py` | modality masking (needs lifting to tool level) |
| Run contract | `src/orchestrator/run_contract.py` | version pinning |

Environment: Docker running, `mujoco 3.10.0` installed, `rosclaw` present.
**CouchDB is currently down** — `curl $COUCHDB_URL` returns nothing, so every MCP
tool would return `ErrorResult(error="IoT database unavailable")`.

---

## 2. What is actually missing

Only three things, none of which is a simulator:

1. **An L3 agent loop that executes tools.** A multi-turn loop that offers the
   MCP tools, executes each requested call, returns the real result, and appends
   an immutable trace event. `RealPMCEpisode` is the working precedent.
2. **Execution-grounded scoring.** PROC read from executed trace events rather
   than `tool_sequence`; CC gated on an evidence-provenance chain.
3. **Tool-level modality masking.** `l3_arms.py` masks the *prompt*; the arm must
   withhold the *tool*, so `read_iot` is absent from the executor in
   PHYSICAL_ONLY rather than merely redacted from prose.

---

## 3. Design decision requiring a steer — MuJoCo MVP

The instruction specifies a MuJoCo MVP as the first backend (one pump, one gauge,
one camera, one IoT signal, deterministic reset). **That already exists in
CouchDB form**: four asset profiles with hidden `gauge_value`, `panel_stuck`,
waypoints, robot state, work orders, and a deterministic seeder with
`--dry-run`/`--verify`.

Building MuJoCo first would:

- duplicate a working hidden-state store and tool surface;
- delay the validity fix by days for no gain in validity — the pilot's defect is
  *self-report versus execution*, which CouchDB-backed tools already solve;
- introduce a second state model to keep consistent with the first.

What MuJoCo genuinely adds is **physical admissibility and real rendering** —
reach, clearance, viewpoint feasibility, actual pixels. Those matter for FM-14…
FM-20 and for a real `capture_image`, but none is required to make the six pilot
scenarios measurable. Note `capture_image` currently returns a `gauge_path`, not
an image; delivering actual pixels to a VLM is the one place a renderer would
change the measurement, and it can be staged.

**Recommendation: Stage 1 = wire L3 through the existing MCP executor
(hours, unblocks the six scenarios). Stage 2 = MuJoCo behind the same
`ToolExecutor` interface and trace schema, for admissibility and rendering.**
The interface is designed for both from the start, so Stage 2 is a backend swap.

This does not alter any preregistered construct — gold labels, thresholds,
scenario semantics, taxonomy and metric names are untouched. It changes only the
order of implementation, which is why it is raised rather than actioned.

---

## 4. Implementation plan (file paths)

### Stage 1 — execution-grounded L3

| # | File | Change |
|---|---|---|
| 1 | `src/orchestrator/tool_executor.py` *(new)* | `ToolExecutor` protocol: `available_tools()`, `execute(call) -> ToolResult`, `reset(seed)`. `ToolResult` carries `requested/executed/status/observation_id/observation_hash/payload`. |
| 2 | `src/orchestrator/backends/couchdb_backend.py` *(new)* | Implements `ToolExecutor` over `src/servers/robot/main.py`. Deterministic reset via `seed_robot_profiles.py`. |
| 3 | `src/orchestrator/execution_trace.py` *(new)* | Append-only trace with the seven states: `REQUESTED · EXECUTED · SUCCEEDED · OBSERVATION_DELIVERED · OBSERVATION_USED · DECISION · ACTION_EXECUTED`. Never collapsed. |
| 4 | `src/orchestrator/l3_arms.py` | Add `masked_tools(arm)`; masking moves from prose to executor. Prose redaction stays as defence in depth. |
| 5 | `src/orchestrator/l3_scoring.py` | `score_l3(..., trace=)`. PROC from `EXECUTED ∧ OBSERVATION_DELIVERED`. CC gains a grounded/ungrounded split — **existing names preserved**, no blended score. |
| 6 | `src/orchestrator/l3_integrity.py` *(new)* | Fabrication and coherence checks, reported **separately** from CC/PROC: `fabricated_observation`, `fabricated_procedure`, `verdict_reason_incoherence`. |
| 7 | `scripts/run_l3_pilot.py` | Multi-turn execute loop replacing single-shot; provenance gains simulator SHA, backend id, evaluator version. |
| 8 | `scripts/l3_execution_preflight.py` *(new)* | Blocks API spend unless all 15 acceptance tests pass. |
| 9 | `src/orchestrator/tests/test_l3_execution.py` *(new)* | The 15 acceptance properties. |

### Stage 2 — MuJoCo (after Stage 1 passes)

`src/orchestrator/backends/mujoco_backend.py`, same `ToolExecutor` interface and
trace schema; adds rendered `capture_image` and admissibility checks via the
existing `SpotAdmissibilityVerifier`.

### Scoring change, stated precisely

CC keeps its name and its meaning (*action matches gold*). What is added is a
**separate** provenance predicate, so the preregistered metric is not redefined:

- `CC` — unchanged.
- `CC_grounded` — `CC ∧ required observation was delivered ∧ decision cites it`.
- Reported alongside, never blended. On the current pilot, R009 PHYSICAL_ONLY
  would keep `CC=1` and gain `CC_grounded=0`.

**This is the one point that touches preregistered terminology.** It adds a
metric rather than altering `CC`. Flagged for approval before implementation.

---

## 5. Proposed commit sequence

1. `docs`: this audit (no code).
2. `feat`: `ToolExecutor` + `ExecutionTrace` + tests (no backend).
3. `feat`: CouchDB backend + deterministic reset + tests.
4. `feat`: tool-level modality masking + leak tests.
5. `feat`: execution-grounded PROC, `CC_grounded`, integrity checks + tests.
6. `feat`: execution preflight blocking API spend.
7. `feat`: multi-turn pilot runner with extended provenance.
8. `run`: six-scenario reference pilot, unchanged scenarios.
9. `docs`: results and apparatus-validity comparison.

---

## 6. Open blocker

**CouchDB is not running.** `docker-compose -f src/couchdb/docker-compose.yaml up -d`
then `python src/couchdb/seed_robot_profiles.py --verify` is required before any
backend work can be tested. Docker is available and idle.
