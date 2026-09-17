# Three-model replication and pre-benchmark phase report

Same 12 generated worlds, three models. Worlds and golds verified **identical**
across all three runs. 108 episodes, **0 apparatus failures, 0 fabrication
flags, 0 coherence flags**. Not a powered benchmark; no sizing committed.

---

## 1. Third-model results

| Model | Arm | CC | CC_grounded | PROC | CC − CC_g |
|---|---|---|---|---|---|
| gpt-5.4-mini | FULL | 7/12 | 4/12 | 7/12 | **3** |
| | PHYSICAL_ONLY | 8/12 | 8/12 | 11/12 | 0 |
| | DIGITAL_ONLY | 4/12 | 0/12 | 0/12 | 4 |
| qwen3.5-omni-plus | FULL | 11/12 | 11/12 | 12/12 | **0** |
| | PHYSICAL_ONLY | 11/12 | 11/12 | 12/12 | 0 |
| | DIGITAL_ONLY | 7/12 | 0/12 | 0/12 | 7 |
| gemini-3.1-flash | FULL | 9/12 | 9/12 | 12/12 | **0** |
| | PHYSICAL_ONLY | 10/12 | 10/12 | 12/12 | 0 |
| | DIGITAL_ONLY | 4/12 | 0/12 | 0/12 | 4 |

**Answer to the (a)/(b)/(c) question: (c), shading into (b).** The grounding gap
in the FULL arm is **3/12 for gpt-5.4-mini and 0/12 for both other models**.
Qwen and Gemini executed a physical read in *every* FULL episode (PROC 12/12), so
their correct decisions were always grounded. Only one of three contemporary
models exhibits the gap.

**Apparatus sanity held uniformly:** DIGITAL_ONLY gives PROC 0/12 and
CC_grounded 0/12 for all three. That separation is guaranteed by masking and is
**not a model finding** — it demonstrates the diagnostic works.

Model selection was on neutral grounds: a third distinct vendor family, image-
capable (the arms require pixel consumption). `glm-4.6v` was excluded on a
purely technical ground — its documented truncation at low token budgets would
have produced apparatus failures rather than model signal, and the budget is
frozen at 1024.

### What this licenses, and what it does not

**Supported:** decision correctness and evidence-grounded correctness are
empirically separable, and the size of the separation is model-dependent —
ranging from 0 to 3 of 12 on identical worlds.

**Not supported:** any universal claim that agents stop seeking physical
evidence when a digital shortcut exists. Two of three models did the opposite,
without exception.

---

## 2. Composite verdict interface (proposal)

Full specification in `docs/L3_CompositeVerdict_and_ClassE_Design.md`. Summary:
`verdict` remains a string for single-asset scenarios and becomes
`{asset: verdict}` only where a manifest declares multiple assets.
`CC_composite` requires every asset to match; `CC_partial` is reported
separately, never blended. R026's gold is **not** changed.

Flattening is demonstrably lossy: ESCALATE erases a correct commit, COMMIT
erases the safety abort, ABORT erases both.

**One ruling needed before implementation:** whether `CC_composite` counts as
"CC" for a composite scenario or as a fourth metric. I propose the former — the
same predicate over a scenario that happens to have several assets — but it is
the only point touching preregistered terminology.

---

## 3. Class-E design

Specified in the same document. Sequence-scoped reset; committed readings with
provenance persist; the drift process is drawn **before any label**; gold derives
from the sequence prefix; a sequence-level hash chain extends the existing trace.

Stale-state failure is scored with **existing metrics**: an agent citing a t0
observation at t2 earns CC but not CC_grounded, because grounding requires an
observation from the current episode. Ten acceptance tests specified. No new
headline metric, no MuJoCo.

---

## 4. Scenario coverage / validity matrix

### Generated worlds (seed 2026)

| Cell | n | Gold | Shortcut implies | CC-discriminating |
|---|---|---|---|---|
| `phys_in__iot_agree` | 3 | 2 COMMIT, 1 ESCALATE | COMMIT | 1/3 |
| `phys_in__iot_disagree` | 3 | 3 COMMIT | ESCALATE | **3/3** |
| `phys_out__iot_agree` | 3 | 3 ESCALATE | ESCALATE | 0/3 |
| `phys_out__iot_disagree` | 3 | 3 ESCALATE | COMMIT | **3/3** |

Cells 3/3/3/3; assets 3/3/3/3; gold 7 ESCALATE / 5 COMMIT; **CC-discriminating
7/12 (58%)**; majority-class baseline 58%; causal factor 9 physical / 3
coordination; **no gold leakage; no gold-conditioned generation.**

### By taxonomy family

| Family | Scenarios | Causal factors | Discriminating arms | Validity | Remaining issue |
|---|---|---|---|---|---|
| **A evidence-dependency (generated)** | 12 worlds | physical band, coordination | FULL, PHYSICAL_ONLY | **VALID** | coordination thin (3/12); no procedural factor |
| A (hand-authored R055–R058) | 4 | physical band | PROC/CC_g only for 2 | CONDITIONAL | state←verdict |
| A (R009, R015) | 2 | — | — | **INVALID** | gold-conditioned; reference only |
| **C procedural** | 9 | ordering, robot/panel state | ordering from trace | **VALID** | coverage 47%; ordering and coverage must stay separate |
| **D relational** | 5 of 6 | work order, human presence | verdict + ordering | **VALID** | R026 blocked on composite interface |
| **E sequential** | 3 | temporal drift | — | **NOT MEASURABLE** | per-episode reset |

---

## 5. Recommended primary metrics

| Family | Primary | Secondary / diagnostic |
|---|---|---|
| A evidence-dependency | **CC** (preregistered) and **PROC** | CC_grounded |
| C procedural | **ordering satisfied** (from executed trace) | procedure coverage, CC |
| D relational | **CC** | ordering, CC_grounded |
| E sequential | *(undetermined — not yet measurable)* | CC_grounded per episode |

**CC_grounded stays a diagnostic.** Three models now show it separating from CC
in exactly one of them. That is enough to establish it *measures something real
and distinct*, and not enough to make it a headline metric. Promoting it now
would rest a primary metric on a single model's behaviour.

**CC is unchanged and remains preregistered.**

---

## 6. Minimum benchmark scale and rationale

**No commitment yet, and the reason is now precise rather than provisional.**

On identical worlds, CC in the FULL arm ranges **7/12 → 11/12 across models** —
a 33 pp spread from model identity alone. Within a model, n=12 gives a standard
error near 14 pp. **Between-model variance exceeds within-model variance**, so
scenario count is not the binding constraint.

What a sizing calculation needs, and does not yet have:

- a *stable* effect. The grounding gap is 3, 0, 0 — a distribution with most mass
  at zero. Sizing against its mean would size against an artifact of which models
  were sampled.
- a decision on whether models are pooled. They should not be: their effects
  differ qualitatively, not just in magnitude.

**Provisional shape, not a commitment:** per-model reporting with ≥40 generated
worlds per family (≈24 CC-discriminating at the current 58% rate, enough for
descriptive per-cell breakdowns), and models treated as a random factor rather
than pooled. That number should be recomputed once the effect distribution is
better characterised.

---

## 7. Is a fourth model necessary?

**It depends which question is primary, and the answer differs.**

- *"Does the gap exist?"* — **answered.** One of three contemporary models shows
  it, on independently generated worlds, with a validated apparatus. A fourth
  adds nothing.
- *"How common is it?"* — **yes, and it is now the binding constraint.** One in
  three is an estimate with essentially no precision. Three or four more models
  would move that from an anecdote to a rate.

Given the reframed central question — whether conventional decision accuracy can
*overestimate* grounded safety — the existence result already carries it, and
prevalence is the natural follow-up. **More models are worth more than more
scenarios**, which inverts the usual benchmark instinct and follows directly from
the variance decomposition above.

---

## 8. Does MuJoCo add a missing construct?

**No — not for anything currently under test.** Every construct measured in this
phase (evidence dependency, grounding, procedural ordering, relational
coordination) executed without physical simulation, and class-E drift is a change
in a scalar hidden state the CouchDB layer already represents.

MuJoCo becomes justified only for constructs the current executor **cannot
express**: arm reach, workspace clearance, stance stability, viewpoint
feasibility, and genuine camera optics (occlusion, glare, focus) as opposed to a
rendered dial. Those map to FM-14…FM-20, none of which is in the current
experiment. Adding it now would expand apparatus without expanding measurable
constructs.

---

## Framing note

The failed universal claim — *"agents stop seeking physical evidence when digital
shortcuts exist"* — is not the finding and should not frame the paper. Two of
three models sought physical evidence in every single episode.

What the phase supports is narrower and more defensible:

> Conventional operational decision accuracy can overestimate evidence-grounded
> safety, and whether it does is a property of the model rather than of the task.

The apparatus that makes this measurable — executed tools, delivered
observations, trace-grounded PROC, and CC_grounded as a separate diagnostic — is
the contribution that survives regardless of how the prevalence question lands.
