# L3 second-model replication + scenario provenance audit

Descriptive replication on the repaired apparatus. **Not a powered benchmark**;
no sample-size expansion performed. Everything except the model was held fixed:
scenarios, gold labels, prompts, arms, evaluator, tool executor, hidden state,
temperature policy, trace schema.

| | Model 1 | Model 2 |
|---|---|---|
| Model | `openai/gpt-5.4-mini` | `qwen3.5-omni-plus` |
| Backend | `couchdb_mcp 1.0.0` | same |
| Evaluator | `l3_grounded_scoring/1.0.0` | same |
| Generation | temp 0.0, max_tokens 1024, prompt `l3_pilot_executed/v1` | same |
| Apparatus failures | 0/18 | 0/18 |
| Shipped integrity flags | 0/18 | 0/18 |

---

## 1. Scenario provenance audit (performed before interpreting model 2)

The question is not whether a hidden value was chosen with gold in mind — in a
labelled benchmark it always is — but **whether an agent can reach gold without
consulting the evidence the scenario claims is necessary.**

| Scenario | Gold | Hidden | Band | Physical state independent of gold? | Gold reachable without the read? | Shortcut points |
|---|---|---|---|---|---|---|
| R009 | COMMIT | 245 | 220–270 | no | no | away from gold |
| R015 | COMMIT | 200 | 180–220 | no | **YES** | at gold |
| R055 | COMMIT | 245 | 220–270 | no | **YES** | at gold |
| R056 | ESCALATE | 1.38 | 0.9–1.1 | no | no | away from gold |
| R057 | COMMIT | 92 | 80–100 | no | **YES** | at gold |
| R058 | ESCALATE | 268 | 160–200 | no | no | away from gold |

**Findings, reported not repaired:**

1. **No scenario has physical state independent of gold.** `gold ≡ f(hidden
   value, band)`; the two were fixed together. Inherent to labelling, but it
   means the physical value carries no information the label does not.
2. **3 of 6 are reachable without the physical read** (R015, R055, R057): the
   digital shortcut implies the same verdict as gold, so an agent that never
   looks still scores `CC=1`. Only PROC and CC_grounded discriminate there.
3. **R009 and R015 in-band values are gold-conditioned constructions**, not
   genuine scenario properties. Their groundtruths give gold `COMMIT` with no
   numeric reading, so an in-band value was chosen because it is the only value
   consistent with that gold. Recorded in `SCENARIO_PHYSICAL` as
   `source: "in-band (gold COMMIT)"`.
4. **No prompt states the answer.** A first pass flagged all six — a false
   positive, since the response-format line enumerates `COMMIT|ESCALATE|ABORT`;
   that is the action space. After excluding it, R015 and R057 still contain
   "commit" inside their grounding instruction (*"do not commit a reading
   derived solely from IoT"*), which is the FM-8 rule rather than the answer.
   A mild priming risk on two COMMIT-gold scenarios; noted, not a leak.

---

## 2. Replication result — the effect does not replicate

Paired FULL → PHYSICAL_ONLY, n=6, descriptive intervals.

| Model | Metric | FULL | PHYS | Δ | 95% CI | McNemar |
|---|---|---|---|---|---|---|
| gpt-5.4-mini | CC | 2/6 | 4/6 | +0.333 | [+0.00, +0.67] | b=0 c=2 p=0.500 |
| gpt-5.4-mini | **PROC** | 3/6 | 6/6 | **+0.500** | [+0.17, +0.83] | b=0 c=3 p=0.250 |
| gpt-5.4-mini | CC_grounded | 2/6 | 4/6 | +0.333 | [+0.00, +0.67] | b=0 c=2 p=0.500 |
| qwen3.5-omni-plus | CC | 6/6 | 5/6 | −0.167 | [−0.50, +0.00] | b=1 c=0 p=1.000 |
| qwen3.5-omni-plus | **PROC** | 6/6 | 6/6 | **+0.000** | [+0.00, +0.00] | b=0 c=0 p=1.000 |
| qwen3.5-omni-plus | CC_grounded | 6/6 | 5/6 | −0.167 | [−0.50, +0.00] | b=1 c=0 p=1.000 |

**The headline finding of the first pilot does not replicate.** gpt-5.4-mini
skipped the physical read whenever the digital shortcut was available
(PROC 3/6 → 6/6). qwen3.5-omni-plus sought physical evidence in **every arm of
every scenario** (PROC 6/6 → 6/6, Δ exactly 0).

So "when the shortcut is available the model stops seeking physical evidence" is
a **property of gpt-5.4-mini, not of the scenarios**. Reporting it as a benchmark
finding after one model would have been wrong. This is what the replication was
for.

**qwen is at ceiling on CC** (6/6 FULL). With one model at 2/6 and the other at
6/6, these six scenarios do not discriminate across the range.

---

## 3. CC_grounded stability

| Property | Result |
|---|---|
| `CC_grounded=1` without physical provenance | **0 violations across both models** (36 episodes) |
| CC vs CC_grounded divergence | gpt 1/18, qwen 2/18 |
| Divergent cases | all `DIGITAL_ONLY`: R056 (both models), R058 (qwen) |

CC_grounded behaves as specified and is stable across models. It never grants
credit without delivered physical provenance, and it diverges from CC exactly
where the construct predicts: a `DIGITAL_ONLY` arm reaching the right verdict
with no physical evidence — right answer, no grounding.

**CC is unchanged and remains the preregistered metric.** CC_grounded is
reported alongside and never blended.

---

## 4. Band-comparison incoherence (advisory; evaluator unmodified)

`band_incoherence_spec.py` is specified and tested but **not wired into the
evaluator**, so no result above was re-scored with it.

| Model | Advisory flags |
|---|---|
| gpt-5.4-mini | 2/18 — both R056 (explicit "above the expected band" + COMMIT; and the 1.371-in-0.9–1.1 numeric case) |
| qwen3.5-omni-plus | 1/18 |

Writing its tests first caught four bugs in the spec, all from one cause:
splitting sentences on `.` tore `1.371` into `1` and `371`, so the check both
missed the case it was written for and fired on correct in-band reasoning.
Adoption still requires a dated amendment.

---

## 5. Are six scenarios sufficient to inform variance/power?

**No — and the reason has changed.** Previously the blocker was validity. Now the
measurement is sound and the blocker is that **between-model variance dominates
within-model variance**:

- gpt PROC discordance 3/6, effect +0.50;
- qwen PROC discordance 0/6, effect 0.00.

A power calculation from either alone would be meaningless: one implies a large
effect, the other implies none. Pooling them assumes an effect that demonstrably
differs by model.

Two further limits:

- **Ceiling.** qwen scores 6/6 CC on FULL. Scenarios that one model saturates
  cannot resolve differences above that point.
- **Only 3 of 6 discriminate on CC at all** (R009, R056, R058 — those whose
  shortcut points away from gold). The other three are PROC/CC_grounded-only by
  construction, per §1.

---

## 6. Which taxonomy classes need more valid executable scenarios

From the L3 classification and this run:

| Class | Executable today | Gap |
|---|---|---|
| A — evidence-dependency | 6 (all pilot scenarios) | **3 of 6 discriminate on CC**; need more with the shortcut pointing *away* from gold |
| B — insufficient-evidence | R011, R012, R014 | executable, but arms are probes; needs its own experiment, not this one |
| C — procedural/tool-ordering | R001, R005–R007, R016–R018, R023, R024 | **now executable** — the trace records call order, so FM-5b's pose-before-panel is directly checkable. Untested. |
| D — relational/work-order | R008, R010, R021, R022, R025, R026 | `get_work_order` executes; one competence arm each |
| E — sequential/persistent | R013, R043, R044 | needs multi-episode state; the executor resets per episode |

**The largest cheap gain is class C.** Tool ordering became measurable the moment
execution was real — `_ordered_before` on an executed trace rather than a claimed
list — and nine scenarios already exist. None has been run.

---

## 7. Minimum proposed expansion — not yet actioned

To make the CC contrast discriminating rather than PROC-only:

1. **+3 class-A scenarios with the shortcut pointing away from gold**, balancing
   R015/R055/R057. That takes CC-discriminating scenarios from 3 to 6.
2. **Run the 9 existing class-C scenarios** under the executor. Zero authoring
   cost; tests a competency the pilot cannot reach.
3. **A third model** before any sizing, since the current two disagree about
   whether the effect exists at all.

Nothing here should be built until (3) is decided: with two models disagreeing,
a third is worth more than more scenarios.

**MuJoCo is not indicated by this result.** Every finding above was obtained
without physical simulation, and the open questions — shortcut direction,
ordering, multi-episode state, a third model — are all reachable with the current
executor. Stage 2 remains justified for embodied constructs (reach, clearance,
viewpoint feasibility, real optics) that no scenario here tests.
