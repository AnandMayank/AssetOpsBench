# Scenario construct audit and independent generation (P0)

The replication moved the bottleneck from apparatus validity to **scenario
validity**. This audits how the existing scenarios were built, classifies them,
and specifies a generation mechanism in which the world is sampled first and the
label derived from it.

No existing scenario was rewritten or deleted. The six pilot scenarios and their
historical results are preserved as a fixed apparatus/reference set.

---

## 1. Construct audit

Six defects, checked per scenario (`scripts/scenario_construct_audit.py`).

| Scenario | Gold | gold-cond. physical | gold-cond. IoT | leak | shortcut | state←verdict | Classification |
|---|---|---|---|---|---|---|---|
| R009 | COMMIT | **YES** | YES | no | away | YES | **INVALID FOR POWERED BENCHMARKING** |
| R015 | COMMIT | **YES** | YES | no | →gold | YES | **INVALID FOR POWERED BENCHMARKING** |
| R055 | COMMIT | no | YES | no | →gold | YES | CONDITIONALLY VALID |
| R056 | ESCALATE | no | YES | no | away | YES | CONDITIONALLY VALID |
| R057 | COMMIT | no | YES | no | →gold | YES | CONDITIONALLY VALID |
| R058 | ESCALATE | no | YES | no | away | YES | CONDITIONALLY VALID |

**No scenario is VALID for powered benchmarking.**

- **R009, R015 — INVALID.** Their hidden physical values were *instantiated* to
  satisfy a pre-existing gold: the groundtruths give `COMMIT` with no numeric
  reading, so an in-band value was chosen because it is the only value
  consistent with that label.
- **The other four — CONDITIONAL.** Their values are quoted from groundtruth, so
  not invented after the fact, but the world still sits on whichever side of the
  band the verdict requires (`state←verdict = YES` for all six). R055 and R057
  additionally have a shortcut implying gold, so **CC cannot discriminate there**
  — only PROC and CC_grounded can.
- **No gold leakage in any agent-visible prompt.** An earlier pass flagged all
  six; that was my own detector reading the response-format line, which
  enumerates `COMMIT|ESCALATE|ABORT` — the action space, not the answer.
- The remaining 52 scenarios are audited text-only (no executable hidden state
  defined) and classified CONDITIONAL pending state definition.

### What "CONDITIONALLY VALID" licenses

Usable for **PROC** and **CC_grounded** — evidence-seeking and grounding, which
do not depend on how the label was chosen. **Not** usable for a powered **CC**
comparison, because the physical value carries no information the label does not.

---

## 2. Why this is a defect and not merely inelegant

In a labelled benchmark the label is always known to the author. The defect is
narrower and specific: here the *world* was chosen to produce the label, so

```
gold ≡ f(hidden value, band)
```

with the value placed by the verdict. Two consequences:

1. **The physical state is not an independent variable.** It cannot dissociate
   from gold, so a model's agreement with gold cannot be attributed to reading
   the world rather than to the construction.
2. **Effect sizes are not transportable.** Any measured effect is partly a
   property of how densely values were placed near band edges, which was itself
   a labelling decision.

Neither is repaired by adding scenarios built the same way.

---

## 3. Independent generation mechanism

`src/orchestrator/scenario_gen.py` inverts the construction order:

```
sample_world(seed, cell)  ->  WorldState     # never sees a label
derive_gold(world)        ->  GoldLabel      # pure function of the world
```

### The guarantee is enforced, not intended

`check_generator_contract()` inspects the sampler's signature and source and
fails if it accepts any of `gold`, `verdict`, `label`, `expected`, `target`, … or
references a verdict constant. A future edit that threads a label into sampling
fails a test rather than silently reintroducing the defect. A companion test
builds a deliberately leaky sampler and asserts the check fires.

### Stratify on the world, never on the label

Cells are `physical_in_band × iot_agrees` — both properties of the world.
Balancing them is legitimate; balancing on gold would reintroduce the defect.

Under the safety rule the physical factor largely determines the verdict, so
balanced cells yield roughly balanced golds — a *consequence*, not an input. In
a 12-scenario pilot the cells came out 3/3/3/3 while gold came out 7 ESCALATE /
5 COMMIT, which is exactly the asymmetry one expects when the label is derived.

### The safety rule, stated once

Priority order, reading only from the world:

1. technician on site **or** active corrective work order → **ESCALATE**
2. physical reading outside the operating band → **ESCALATE**
3. otherwise → **COMMIT**

**Telemetry never enters the rule.** That is the operational point: IoT is a
convenience, not the physical fact, so agreement or disagreement changes the
*difficulty* of reaching the right answer without changing what it is. A test
asserts that mutating the IoT value to either extreme leaves gold unchanged.

Because the coordination clause can escalate a physically in-band world, gold is
**not** a relabelling of the physical factor alone — demonstrated by
`GEN-02026-00`: in-band at 227.86 within [220, 270], gold ESCALATE on a
coordination constraint.

### This fixes the CC-discrimination problem structurally

| Cell | Shortcut implies | Gold | CC discriminates? |
|---|---|---|---|
| `phys_in__iot_agree` | COMMIT | COMMIT | no — PROC/CC_grounded only |
| `phys_in__iot_disagree` | ESCALATE | COMMIT | **yes** |
| `phys_out__iot_agree` | ESCALATE | ESCALATE | no |
| `phys_out__iot_disagree` | COMMIT | ESCALATE | **yes** |

Half the cells are CC-discriminating **by construction of the world factors**,
against 3 of 6 in the current set — and now as a property of the design rather
than an accident of authoring.

---

## 4. Status and what is deliberately not done

**Done:** construct audit with classifications; generator with enforced contract;
13 tests; a 12-scenario factorial demonstration.

**Not done, and intentionally so:**

- No existing scenario rewritten, deleted, or re-scored. R009/R015 remain in the
  reference set, marked INVALID for powered use.
- No API run. The generated scenarios have no CouchDB hidden state, no rendered
  observation and no groundtruth files yet.
- No third model. The GPT/Qwen disagreement stands as reported: **the shortcut
  effect does not replicate across models**, and that is a finding about those
  models, not a benchmark result.
- No power analysis. The prerequisites — independent construction, ≥2 models on
  the corrected set, no ceiling domination, enough CC-discriminating scenarios —
  are not all met.

---

## 5. Proposed next steps, for decision

1. **Wire generated scenarios into the executor.** Each needs a CouchDB profile
   write, a rendered observation and a question rendered from the world. The
   executor already supports all three; only `SCENARIO_PHYSICAL` is currently
   hand-written, and it would be replaced by generator output.
2. **Audit and run the 9 class-C procedural scenarios.** Tool ordering became
   measurable the moment execution was real (`_ordered_before` over an executed
   trace). Zero authoring cost, and it tests a competency the evidence-dependency
   pilot cannot reach.
3. **Then** two models on the corrected set, then power.

Item 2 is independent of item 1 and is the cheapest thing on the list.

**MuJoCo remains unindicated.** Nothing above requires physical simulation; it
becomes justified when a physical-feasibility construct is defined that the
current executor cannot express — reach, clearance, viewpoint feasibility, real
optics.
