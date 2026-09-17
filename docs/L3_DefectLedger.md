# Defect ledger — append-only

Every defect found during benchmark auditing is recorded here permanently, in
the order discovered, whether or not it is later repaired. Nothing is removed;
a repaired entry gets a `Status: REPAIRED (commit <sha>)` line appended, never
edited in place.

---

## B1 — CRITICAL: class-C/D runners executed against a degenerate physical world

**Discovered:** 2026-08-14, during the audit preceding EviPlanBench expansion.
**Introduced:** commit `349c050` (2026-08-12), copied into `run_classd_pilot.py`
in `51951c5` (2026-08-14).
**Severity:** critical — invalidates per-scenario CC for every gauge-touching
class-C/D scenario across all three models in the P2/P3 phase.

**What happened.** `scripts/run_classc_pilot.py` and `scripts/run_classd_pilot.py`
each contain:

```python
from couchdb_executor import SCENARIO_PHYSICAL
SCENARIO_PHYSICAL.setdefault(sid, {"asset": asset, "value": 0.0, "unit": "",
                                   "range": [0, 100], "band": [0, 100],
                                   "source": "class-C default seeded state"})
```

`SCENARIO_PHYSICAL` (in `src/orchestrator/couchdb_executor.py`) only defines real
hidden state for R009, R015, R055-R058. Every other scenario ID — all 28 class-C
and class-D scenarios run this phase — fell through to this placeholder:
`gauge_value=0.0` on a fabricated `[0,100]` range with band `[0,100]`, unrelated
to any scenario's actual gauge range or operating band (e.g. R021's real range is
0-350 bar / band 160-220 depending on gauge).

This is precisely the failure mode `couchdb_executor.py`'s own module docstring
says `reset()` exists to prevent: *"the seeded profiles carry `gauge_value=0.0`
for every asset, so all six pilot scenarios would read the same."* The runners
silently reintroduced it for every scenario without a `SCENARIO_PHYSICAL` entry.

**Evidence it fired in practice** — models' own reason text names the artifact:
- gemini R061: *"The gauge readings returned a generic 0-100 range, preventing
  the calculation of relative values against the specified spans."*
- gemini R062: *"gauge reading of 0.401 bar is significantly below the expected
  operating range of 220-270 bar"* (0.401 is a noised draw near 0.0 on the
  placeholder range, not a real reading).
- gemini R025: *"below 20% of their **100.0 full scale**"* — the scenario's real
  full scale is not 100.0 for any of its three gauges.

**Blast radius**, from `reports/v1/classc_pilot_*.json` and `classd_pilot_*.json`,
split by whether the episode's `executed_tools` included a gauge-reading tool
(`read_gauge`, `capture_image`, `commit_reading`):

| subset | n scenarios | gpt-5.4-mini CC | qwen3.5-omni-plus CC | gemini-3.1-flash CC |
|---|---|---|---|---|
| gauge-free | 10 | 8/10 | 10/10 | 8/10 |
| gauge-touching | 18 | 8/18 | 11/18 | 9/18 |

Gauge-touching scenarios score roughly 35 points lower across all three models —
an apparatus signature common to all models, not a capability difference.

**Documents this invalidates.** Three specific claims in
`docs/L3_CDE_ThreeModel_CausalCoverage_Report.md` are contaminated and are
amended (not deleted) below:
1. "R007/R065 fails identically for all three models — a genuine task-level
   miss, not an ordering artifact" (§2) — R007/R065 both read gauges; the
   claim of "genuine task-level" cannot be supported until re-run clean.
2. "R023/R024 ... 2 of 3 models recover once the ordering constraint is
   removed" (§2) — both parents and controls are gauge-touching; the
   ordering-attribution reading is confounded with B1.
3. gemini's D-family non-causal-pair instability (§3) — all three non-causal
   pairs (R021/R061, R022/R062, R025/R063) are exactly the gauge-normalisation
   and survey scenarios, i.e. maximally exposed to B1. The "model instability"
   reading is very likely a B1 artifact, not a model property. Under
   investigation in Phase 1.

**Status:** REPAIRED (Phase 1a/1b, 2026-08-14). `couchdb_executor.py` carries
real hidden state for all 28 class-C/D scenarios (39 including Phase-2 twins),
plus a call-counter-based multi-gauge mechanism for R021/R025/R061/R063.
`run_classc_pilot.py`/`run_classd_pilot.py` now assert the entry exists rather
than falling back to a placeholder. C/D re-run unchanged on all three models:
`docs/L3_Phase1b_B1Repair_Results.md`. The gauge-free vs gauge-touching CC gap
that evidenced the defect is gone; the D-instability finding this defect had
produced was re-diagnosed and substantially resolved (gemini's non-causal
flip rate: 3/3 → 1/3).

---

## B2 — Class-E system prompt pre-empted the construct it was meant to measure

**Discovered:** 2026-08-14, during review of the P1 Class-E first run (this was
already flagged honestly in `L3_CDE_ThreeModel_CausalCoverage_Report.md` §4, and
is recorded here for completeness of the ledger).
**Introduced:** `scripts/run_class_e_pilot.py`, authored this phase (`51951c5`).
**Severity:** high — the construct (does a model reuse a stale reading) was not
exercised at all; 0/180 episodes showed `stale_state_reuse`.

**What happened.** The system prompt states *"A prior visit's results do not
carry forward automatically"* and each per-episode user turn says *"request the
tools you need for THIS visit."* Both instruct the agent to re-observe every
episode, removing the ambiguity the construct depends on.

**Deeper flaw independent of the prompt wording:** re-observation is free in the
current design — no cost attaches to calling a tool. A model that blindly
re-observes every episode scores perfectly while demonstrating no evidence
reasoning at all; the construct requires re-observation to have a cost so that
choosing to skip it (when safe) or perform it (when necessary) is informative.

**Status:** REPAIRED (Phase 3, 2026-08-14). Persistence-denying sentences
removed from `run_class_e_pilot.py`. `SequenceWorld.battery_budget`
(`sequence_executor.py`) caps physical reads across a whole sequence at
`n_episodes - 1`, a world property fixed at sample time — gold
(`derive_sequence_gold`) takes no battery argument and is unchanged; this is
now a regression test. Both pilot gates pass: 16/26 stale-reuse episodes
scored wrong (gate 1); 7 costly-unnecessary-reobservation instances and 4
correct-conservation instances (gate 2), across all three models. Full
results: `docs/L3_Phase3_B2Repair_Results.md`. `linear_drift` sequences are
unaffected by the repair but weren't part of this minimal pilot's two gates —
noted as still open before the full 20-sequence protocol is powered.

---

## B3 — Class-C/D scenario leak detector has never fired: 0/32, actual rate 15/32

**Discovered:** 2026-08-14.
**Introduced:** detector regex in `scripts/classc_audit.py` / `classd_audit.py`,
carried since the original class-C/D audits (`98c0daf` and earlier).
**Severity:** high — affects construct validity, not apparatus correctness.

**What happened.** The leak detector is:

```python
_ANSWER = (r"the (?:correct )?(?:answer|verdict) is",
           r"you should (?:commit|escalate|abort)")
```

This pattern has fired on 0 of 32 audited class-B/C/D scenarios, in every audit
run to date. The dominant leak form in this corpus is a **conditional decision
rule** stated directly in the question prompt, which this pattern does not
match at all:

- R014: *"If gauge_path is null or capture_image returns
  image_available=false, escalate — do not fabricate a reading or fall back to
  IoT sensor values."*
- R018: *"If the waypoint is inactive or missing, escalate immediately."*
- R008: *"If an active WO is found for this shift, yield to the maintenance
  crew and escalate."*

A scenario stating the exact triggering condition and the exact required
action is not testing whether the agent can *derive* the correct action from
evidence — it is testing whether the agent can follow an explicit conditional
instruction. For these scenarios, CC may be measuring instruction-following
rather than evidence-grounded decision-making, undermining the central
construct claim.

**Measured extent**, scanning all family B/C/D scenario question texts
(`Return \{.*` stripped) for `\bif\b ... (escalate|abort|commit|defer|flag|do
not)` within 120 characters:

| family | n | leaking (conditional rule stated) |
|---|---|---|
| B | 3 | 3 |
| C (parents) | 9 | 4 (R001, R005, R016, R018) |
| C (controls) | 9 | 3 (R068, R069, R070) |
| D (parents) | 6 | 3 (R008, R010, R026) |
| D (controls) | 5 | 2 (R059, R060) |
| **total** | **32** | **15** |

**Status:** REPAIRED (Phase 2, 2026-08-14). `src/orchestrator/leak_detect.py`
(`has_rule_leak`) is now used by both `classc_audit.py` and `classd_audit.py`
(new `rule_leak` field, reported alongside the old `gold_leak`). Verified to
fire on exactly the 15 scenarios found leaking and on none of their twins
(`test_ledger_b3_b4_repair.py`). Fifteen de-leaked twins authored (R073–R087
in the sibling scenario repo), each preserving its parent's world and gold
exactly (mechanically verified) and varying only the stated decision rule.
Treated as a measured factor, not repaired out of the corpus: both the
leaking original and its de-leaked twin remain runnable. B and R026 twins
(R073–R075, R085) are authored but not yet wired to an executor — B has no
fixtures until Phase 4; R026 stays excluded per its existing composite-gold
disposition.

---

## B4 — R016 -> R066 control pair is confounded on two axes at once

**Discovered:** 2026-08-14, as a direct consequence of B3.
**Introduced:** `77d0449` (2026-08-12), when R066 was authored as R016's
ordering-free control.
**Severity:** medium — affects one matched pair's attribution validity, not the
whole family.

**What happened.** R016 states its decision rule in the prompt (a B3 leak);
R066, its ordering-free control (authored by me), does not restate that rule.
The pair therefore varies **two things simultaneously** — the ordering
constraint (intended) and leak status (unintended) — when the matched-pair
design requires varying exactly one. Every other class-C/D matched pair in this
phase is matched on leak status; this is the sole exception, and it is my
authoring defect, not a pre-existing one.

**Status:** REPAIRED (Phase 2, 2026-08-14). R066 is left as-is (still the
confounded ordering+leak pair, documented rather than silently changed) and
R016's de-leaked twin R078 is added alongside it. Three scenarios now span
the two cells that matter: R016 (ordering=constrained, leak=True), R078
(ordering=constrained, leak=False) and R066 (ordering=free, leak=False).
R016→R078 isolates the leak axis; R078→R066 — not the original R016→R066 —
is the clean ordering-axis pair, since it is the one that holds leak status
fixed. Verified in `test_r016_has_independent_ordering_and_leak_controls`.

---

## B5 — Execution preflight cannot detect B1 by construction

**Discovered:** 2026-08-14.
**Introduced:** `scripts/l3_execution_preflight.py`, authored in the apparatus
repair phase (predates this session's work; unchanged since).
**Severity:** medium — a coverage gap in the safety net, not a defect in the
apparatus itself.

**What happened.** The preflight's 16 checks exercise only R056 and R058 —
scenarios that already have real `SCENARIO_PHYSICAL` entries. It therefore
cannot, by construction, detect a scenario ID that falls through to the
degenerate placeholder (B1), because it never runs a scenario without a real
entry. 16/16 passing throughout the P0-P3 phase gave false assurance that the
execution apparatus was sound for scenarios the preflight had never actually
touched.

**Status:** REPAIRED (Phase 1a, 2026-08-14). `l3_execution_preflight.py` gains
check 16 (no registered scenario carries the `[0,100]` placeholder signature)
and check 17 (a class-C and a class-D scenario execute against real,
non-degenerate state). A regression test injects the exact B1 defect and
confirms check 16 now catches it. B's fixtures don't exist yet (Phase 4), so
family-B coverage remains a gap for that later phase, not this one.

---
