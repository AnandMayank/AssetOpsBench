# L3 evidence-dependency pilot — results

**Apparatus/model pilot, not a benchmark result.** 18 calls, 0 apparatus
failures. The headline finding is about the apparatus, not the model.

## Provenance

| Item | Value |
|---|---|
| AssetOpsBench | `3b63e82f14e6` (`phase0-apparatus-integrity`), working tree dirty (unrelated pre-existing changes) |
| AssetOpsBenchScenarioGeneration | `36e3104e0bd7` (`main`), working tree dirty (unrelated pre-existing changes) |
| frozen_config | `64d7cfb7dbfa` |
| `V1_EvidenceContract.md` (blob) | `a0c50d56adab` |
| `L3_ScenarioClassification.md` (blob) | `468d29888c8f` |
| `FM_Crosswalk.md` (blob) | `74ec5ea19dd8` |
| Model | `openai/gpt-5.4-mini` via TokenRouter |
| Generation | temperature 0.0, max_tokens 1024, system prompt `l3_pilot/v1` |
| Scenarios | R009, R015, R055, R056, R057, R058 |

## Results — CC and PROC reported separately, never combined

| Scenario | Gold | FULL CC | FULL PROC | PHYS_ONLY CC | PHYS_ONLY PROC | DIGITAL_ONLY CC | DIGITAL_ONLY PROC |
|---|---|---|---|---|---|---|---|
| R009 | COMMIT | 0 | 0 | 1 | 1 | 0 | 0 |
| R015 | COMMIT | 0 | 0 | 1 | 1 | 0 | 0 |
| R055 | COMMIT | 1 | 1 | 1 | 1 | 0 | 0 |
| R056 | ESCALATE | 1 | 1 | 0 | 1 | 1 | 0 |
| R057 | COMMIT | 0 | 1 | 1 | 1 | 0 | 0 |
| R058 | ESCALATE | 1 | 1 | 1 | 1 | 1 | 0 |

`DIGITAL_ONLY` is an insufficient-evidence probe, not a competence arm; its
column is reported for completeness and excluded from the paired comparison.

### Paired difference, competence arms only (n = 6)

| Signal | FULL | PHYSICAL_ONLY | Δ | Descriptive 95% CI | McNemar |
|---|---|---|---|---|---|
| CC | 3/6 | 5/6 | +0.333 | [−0.33, +0.83] | b=1, c=3, p=0.625 |
| PROC | 4/6 | 6/6 | +0.333 | [+0.00, +0.67] | b=0, c=2, p=0.500 |

Majority-class baseline **67%** (4 COMMIT / 2 ESCALATE) — sanity reference only.

Intervals are descriptive. At n=6 nothing here is a statistical claim, and none
is made.

---

## The finding: gold is not derivable from the agent's actual inputs

**8 of 18 responses explicitly state the evidence was never delivered.**
Representative, R057 FULL:

> "No physical gauge image from capture_image was provided, so a committed
> verification reading cannot be made from history or IoT data alone."

The L3 probe interface is **single-shot text with self-reported tool sequences**.
No tool is executed and no image is delivered. Every one of the six scenarios has
a gold determined by a physical gauge value (245 bar, 1.38 m³/s, 92 °C, 268 bar)
that the agent never receives. It cannot read the gauge because there is no gauge
to read.

Three consequences, each of which invalidates part of the measurement:

**1. CC penalises correct epistemic caution and rewards fabrication.** For the
four COMMIT-gold scenarios, the only way to score CC=1 is to commit a reading
the agent was never given. The model that says "no image was provided, escalate"
— the epistemically correct response — scores 0. In R056 PHYSICAL_ONLY the model
scored PROC=1 while asserting *"The physical gauge image shows the pump flow
within the expected 0.9–1.1 m³/s band"*, a fabricated reading of a
non-existent image. **The pilot rewards precisely the FM-8 failure it exists to
detect.**

**2. PROC measures claimed procedure, not grounding.** R009 PHYSICAL_ONLY
reported `tool_sequence: ["capture_image"]` and scored PROC=1 while its own
reason states *"I cannot physically verify the gauge from the provided evidence
in this interface."* Self-reported tool use cannot distinguish a call that
happened from one that was narrated.

**3. One verdict/reason incoherence.** R009 PHYSICAL_ONLY emitted
`verdict: COMMIT` with a reason arguing the case "should be escalated for on-site
reading", and scored **CC=1** — a correct score for an internally contradictory
answer. (An earlier automated pass flagged two; R058 PHYSICAL_ONLY was a
regex false positive and is coherent.)

### The apparent arm effect is an artifact

PHYSICAL_ONLY outscoring FULL by +0.33 on both signals looks like an
evidence-dependency effect. It is not. In the FULL arm the model has an IoT value
and correctly refuses to substitute it, escalating — scored wrong against a
COMMIT gold. In PHYSICAL_ONLY it has nothing, and in three cases commits anyway.
**The arm that scores better is the one where the model had less information and
guessed.** No evidence-dependency conclusion can be drawn.

---

## Assessment: can this variance size a powered L3 benchmark?

**No — and increasing N would not help.** The blocker is validity, not power.

A sample-size calculation presumes the measurement is unbiased and that noise
shrinks with N. Here the measurement is systematically inverted for four of six
scenarios: CC rewards ungrounded commitment and penalises justified abstention.
Running 200 scenarios would estimate that inverted quantity precisely.

This is the same failure the L1 arm had, one level up. At L1 the RGB-only arm
could not emit COMMIT because the threshold forbade it. Here the agent cannot
emit a *grounded* COMMIT because the evidence is never delivered. Both were
invisible to the metrics and visible in the traces.

### What must be true before a powered L3 run is specified

1. **Tool execution, or delivered evidence.** Either the harness executes
   `capture_image` and returns a gauge observation, or the scenario ships the
   physical reading as an attached observation. Until then no L3 scenario whose
   gold depends on a physical value is answerable.
2. **PROC verified, not self-reported.** Grounding must be read from executed
   calls in a trace, not from a `tool_sequence` field the model writes.
3. **A verdict/reason coherence check**, since the scorer currently accepts a
   verdict contradicted by its own justification.
4. **Re-run this pilot unchanged** afterwards. Its value is as a fixed reference
   point: the same six scenarios, same provenance, same separated CC/PROC
   reporting.

Only after (1)–(3) can observed variance be used for a size calculation, and the
calculation should use `evaluation.paired_stats.min_pairs_for_effect` with the
realised discordance rather than an assumed one.

### What the pilot did establish

- The apparatus runs end to end: 18/18 calls returned parseable JSON, 0
  apparatus failures, provenance captured for both repositories.
- Arm rendering, redaction and per-arm scoring work as designed — the arms
  genuinely differ and no withheld quantity leaked.
- Separating CC from PROC was decisive. A blended score would have shown
  PHYSICAL_ONLY ahead on both and read as a clean positive result.
- The scenario set is not the problem and does not need redesigning; the
  execution interface does.
