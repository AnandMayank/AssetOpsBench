# Phase 1b — C/D re-run after the B1 repair, and the D-instability diagnosis test

Re-runs `docs/L3_DefectLedger.md` entry B1 was written against: class C (18
scenarios) and class D (10 scenarios, R026 still excluded — see
`run_classd_pilot.py`'s docstring), unchanged scenario set, same three models,
after the placeholder-world repair (Phase 1a, commit `d5cc0b8` and `29ea23a`).
**No scenario, gold label or metric changed between v1 and v2 — only the hidden
physical state each episode actually ran against.**

Full data: `reports/v1/classc_pilot_*_v2.json`, `classd_pilot_*_v2.json`.

---

## 1. Did the gauge-free vs gauge-touching gap close?

The B1 ledger entry's evidence was a ~35pp CC gap between scenarios that did
and did not call a gauge tool, present identically across all three models —
an apparatus signature, not a capability difference. After repair:

| subset | n | gpt-5.4-mini | qwen3.5-omni-plus | gemini-3.1-flash |
|---|---|---|---|---|
| gauge-touching | 8 | 6/8 | 6/8 | 6/8 |
| gauge-free | 20 | 10/20 | 17/20 | 14/20 |

The gap is gone, and if anything now runs the other direction (gauge-touching
scenarios score *higher* on average). This is consistent with B1 being real:
under the placeholder, gauge-touching scenarios were being actively misled by
a fabricated `[0,100]` reading; with real state removed as a confound, they no
longer underperform. (The gauge-touching/gauge-free split itself also shifted —
8 vs 20 rather than 18 vs 10 — because which tools a model chooses to call is
itself model behaviour, not fixed by scenario ID; this is expected.)

## 2. Overall CC, v1 (contaminated) vs v2 (repaired)

| family | model | v1 | v2 |
|---|---|---|---|
| C (18) | gpt-5.4-mini | 9 | 8 |
| C (18) | qwen3.5-omni-plus | 13 | 15 |
| C (18) | gemini-3.1-flash | 11 | 12 |
| D (10) | gpt-5.4-mini | 7 | 8 |
| D (10) | qwen3.5-omni-plus | 8 | 9 |
| D (10) | gemini-3.1-flash | 6 | 8 |

Movements are modest and mixed in direction (gpt's C total actually drops by
one) — consistent with B1 being real noise, not a bias that inflated every
model's score in the same direction. No headline claim rests on these totals
changing in a particular direction; they are reported for the record.

## 3. The D-instability diagnosis — resolved, prediction confirmed

`docs/L3_CDE_ThreeModel_CausalCoverage_Report.md`'s original finding was that
gemini flipped its verdict on all 3 non-causal D pairs (R021/R061, R022/R062,
R025/R063) while gpt and qwen flipped on none. The plan (§D) predicted this was
**(B) an evaluator artifact**, specifically B1: all three non-causal pairs are
exactly the multi-gauge scenarios maximally exposed to the placeholder defect,
and gemini's own v1 reason text named it directly ("a generic 0-100 range").

**Non-causal flip count, v1 -> v2:**

| model | v1 | v2 |
|---|---|---|
| gpt-5.4-mini | 0/3 | 0/3 |
| qwen3.5-omni-plus | 0/3 | 1/3 |
| gemini-3.1-flash | **3/3** | **1/3** |

**Prediction confirmed.** Gemini's flip count drops from 3/3 to 1/3 once the
apparatus stops feeding it a fabricated gauge scale. The remaining flip
(R025 -> R063) is **not new** — it flipped in v1 too (parent wrong, control
right, same direction both times) — and R025's own construct (a
forbidden-action / survey-only compliance check) is orthogonal to the
gauge-normalisation confound B1 introduced, so there is no reason to expect
the repair to touch it.

**One new flip appeared**: qwen now flips on R022/R062 (COMMIT gold; qwen
gets the parent wrong — "only a generic 'chiller_6' target was available" —
and the control right). R022/R062 is the **waypoint-routing** pair, which
never touches a gauge at all; this flip is unrelated to B1 and is a genuine
model behaviour on a scenario the repair should not have touched. Recorded,
not investigated further this phase (§D's plan already scoped further D
controls as conditional on flips persisting after repair — this one instance
does not on its own justify building them).

**Revised reading of §3 (`L3_CDE_ThreeModel_CausalCoverage_Report.md`)**: the
original claim that gemini's D-family answers are "not reliably driven by the
enterprise factor" is **not supported** by the repaired data — 2 of gemini's
3 apparent instabilities were an apparatus artifact. What remains (1 flip each
for qwen and gemini, on different pairs, 0 for gpt) is ordinary per-scenario
variation, not a model-wide instability pattern.

## 4. One apparatus note, not a new ledger entry

qwen's R025 call hit a transient empty-response failure on the first pass
(`verdict=""`, no error captured) — a single API-layer hiccup, not reproduced
on immediate retry (same model, same seed, same prompt → clean `ESCALATE`,
`CC=1`). The retry result is the one recorded in
`classd_pilot_qwen35omniplus_v2.json`; the original failure is noted in that
file's `notes` field rather than silently dropped.

## 5. Integrity

`pytest src/orchestrator/tests`: 301 passed, 1 skipped (8 new tests from
Phase 1a, `test_ledger_b1_b5_repair.py`). `l3_execution_preflight.py`: 18/18,
including the two new checks that would have caught B1. No scenario, gold
label, frozen metric or historical result changed.
