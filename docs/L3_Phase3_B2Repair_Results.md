# Phase 3 -- Class-E repair and minimal pilot (ledger B2)

The first Class-E run (Phase 1) showed 0/180 `stale_state_reuse` across three
models -- the system prompt instructed re-observation directly, pre-empting
the failure mode the family exists to measure (`docs/L3_DefectLedger.md`,
entry B2). This phase repairs the prompt and the deeper flaw it was masking
(re-observation was free), then runs the minimal pilot the plan's two
acceptance gates require before Class E can be powered.

**No historical Class-E result, gold label, or frozen metric changed.**
Family A, C, D untouched this phase. Full data:
`reports/v1/class_e_pilot_*_v2.json`.

---

## 1. What changed

1. **Prompt de-contamination.** Removed *"a prior visit's results do not
   carry forward automatically"* and every *"for THIS visit"* qualifier from
   `run_class_e_pilot.py`. Nothing now tells the agent whether or how state
   persists between visits.
2. **Re-observation given a real cost.** `SequenceWorld.battery_budget`
   (`sequence_executor.py`) caps physical reads (`read_gauge`/`capture_image`)
   across the *whole* 3-episode sequence at `n_episodes - 1` = 2. This is a
   **world property fixed at sample time** -- the same for every world of a
   given length, drawn before any label exists -- so which calls succeed
   depends only on the agent's own choices, never on gold.
3. **Gold is unchanged.** `derive_sequence_gold(world, episode)` takes no
   battery argument and was not touched; its signature and purity are now a
   regression test (`test_gold_signature_has_no_battery_or_model_parameter`).
   What changed is only whether a *fresh* observation is deliverable, not what
   the correct verdict is.
4. **Two new diagnostic fields**, both pure functions of the world and the
   trace's delivery record -- never of what the model chose:
   `reobservation_was_necessary` (did band membership actually change since
   the last delivered observation) and `unnecessary_reobservation` (the model
   read again when it didn't need to). Neither feeds CC, PROC or ORDERING.
5. **Pilot scope narrowed to `stationary` and `step`** processes
   (`--processes`), matching the two acceptance gates below.
   `linear_drift` sequences are unaffected and remain valid; they are simply
   outside this minimal pilot.

## 2. Acceptance gates

Both required before Class E can be powered.

**Gate 1 (drift/step) -- does stale evidence reuse cost correctness?**
26 stale-reuse episodes observed across the three models; **16 of them scored
CC=0** -- reusing an invalidated reading produced a wrong decision, not just
a technically-stale-but-harmless one. **PASS.**

**Gate 2 (stationary) -- both directions of the failure present?**
- *Unnecessary re-observation costs correctness*: 7 instances where a model
  read again with nothing to gain, and either that episode or a later one in
  the same sequence failed as a direct result (see the worked example below).
- *Correct reuse*: 4 instances (all gpt-5.4-mini) where a model skipped an
  unneeded re-read and was still right.

Both halves present. **PASS.**

## 3. A worked example (qwen3.5-omni-plus, SEQ-00001, stationary)

```
ep0  read (necessary=True)   -> COMMIT, correct.        battery 1/2
ep1  read (necessary=False)  -> COMMIT, correct.         battery 2/2  <- unnecessary
ep2  no read (budget spent)  -> "battery depleted for    battery 2/2  <- CC=0
                                  this visit" -> ABORT
```

The model re-observed at episode 1 even though nothing had changed since
episode 0 (a wasted read), which exhausted its budget before episode 2 -- at
which point it correctly recognised it had no fresh evidence and declined
(ABORT) rather than fabricating a reading, but the gold action was COMMIT, so
it scores wrong. **gemini-3.1-flash reproduces the identical pattern on the
same sequence, independently** (different gauge readings, different reason
text, same structural failure) -- not a caching artifact, a shared behavioural
tendency the repair now makes visible.

## 4. Results

| Model | sequences | episodes | CC | stale_state_reuse | unnecessary_reobservation | apparatus failures |
|---|---|---|---|---|---|---|
| gpt-5.4-mini | 8 | 24 | 17/24 | 10 | 2 | 0 |
| qwen3.5-omni-plus | 8 | 24 | 15/24 | 8 | 3 | 0 |
| gemini-3.1-flash | 8 | 24 | 16/24 | 8 | 3 | 0 |

By process:

| Model | stationary CC | step CC |
|---|---|---|
| gpt-5.4-mini | 6/6 | 11/18 |
| qwen3.5-omni-plus | 4/6 | 11/18 |
| gemini-3.1-flash | 4/6 | 12/18 |

gpt-5.4-mini is the only model with a perfect stationary score and shows all
4 instances of correct budget-conserving reuse -- provisionally the most
evidence-conservative of the three on this construct, opposite of its
floor-limited stationary result in the contaminated Phase-1 run (2/6, later
attributable to the prompt telling every model to always re-observe, which
this model apparently followed most literally).

## 5. Integrity

`pytest src/orchestrator/tests`: 339 passed, 1 skipped (10 new tests,
`test_ledger_b2_repair.py`, covering: battery budget is a deterministic world
property; blind re-observation never fits the budget while reading-once
always does; gold's signature has no battery/model parameter and its output
never varies; the executor actually fails the read past budget; the budget
persists across episode boundaries rather than resetting; non-physical tools
are never battery-gated). `l3_execution_preflight.py`: 18/18. No gold label,
frozen metric, or historical scenario/result changed.

## 6. What remains open

- Only `stationary` and `step` were exercised under the repaired apparatus.
  `linear_drift` needs its own pass before Class E's full 20-sequence protocol
  is powered, since a continuously-changing value may need a different
  budget/necessity rule than the single-crossing case tested here.
- The minimal pilot's n=8 per model is not a powered sample; the gates were
  binary existence checks (does the failure mode occur at all), not effect-size
  estimates.
