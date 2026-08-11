# V1 — Evidence Contract (pre-registration)

**Status: authored before any V1 model run. Committed to git as pre-registration.**
Do not edit after results are seen; append a dated amendment instead.

The Rev-2 plan replaced V1's "shortcut score" with an evidence-dependency
analysis. For each scenario family we declare three things *in advance*:

| Column | Meaning |
|---|---|
| **INTENDED** | what evidence the scenario is designed to require |
| **AVAILABLE** | what each channel actually exposes |
| **NECESSARY** | what is causally required to determine the gold decision |

A high restricted-modality score is a defect **only when INTENDED requires
physical/digital evidence that NECESSARY shows is not actually needed.** Some
families may legitimately be metadata-determined; the contract is what lets us
tell those apart after the fact instead of rationalising whichever result
appears.

---

## 0. Two structural facts established before any model ran

Both are properties of the catalog, computed from
`shared/perception/perception.csv` (754 query rows). Neither involves a model.

### 0.1 `category → recommended_action` is fully deterministic

| Category | Action | n | Mapping |
|---|---|---|---|
| `gauge_degradation` | `CLEAN_GAUGE` | 407 | deterministic |
| `occlusion` | `ROUTE_UPDATE` | 81 | deterministic |
| `scale_interpretation` | `ROUTE_UPDATE` | 211 | deterministic |
| `glare_lighting` | `ROUTE_UPDATE` | 49 | deterministic |
| `never_read` | `ESCALATE` | 6 | deterministic |

**Consequence, and it is a limitation of the L1 benchmark, not of any model:**
at L1 the decision carries no information beyond the perception category. An
agent that classifies the barrier correctly gets the action for free, and
"action accuracy" at L1 is a relabelling of category accuracy, not an
independent measurement.

The paper must therefore **not** claim L1 evidence for the decision layer. The
`evidence → sufficiency → decision` chain can only be separated where the
decision depends on state the image does not contain — which is L3 (IoT
contradiction, technician presence, work-order state) and the V2 counterfactual
triples. This is a scoping correction to the Rev-2 plan, which listed
"action/tool-call accuracy" under L3 but left L1's decision axis unqualified.

### 0.2 The `description` field would leak the category, and is evaluator-only

Descriptions contain a lexical cue for their own category in 85–100% of rows
(`gauge_degradation` 85.5%, `occlusion` 85.2%, `scale_interpretation` 93.8%,
`glare_lighting` 100%). A crude keyword rule mapping description → category →
action scores **61.5%** against a **54.0%** majority-class prior — a +7.6 pp
lift with no pixels involved, and a trained text classifier would do better.

**Verified: `description` is not passed to any vision provider or into any
prompt** (`real_pmc_orchestrator.py`, `gemini_vision_provider.py`). It is
evaluator metadata today, and the contract requires it stay that way. Exposing
it — for instance as "operational context" in an agent prompt — would convert a
perception benchmark into a text-classification benchmark.

**Contract clause C-1:** `description`, `trap`, `detection_delay_reason`,
`failure_mode` and `forbidden_actions` are evaluator-only fields. Any change
that surfaces them to an agent invalidates V1 and must re-run it.

---

## 1. Per-family evidence contract

Channels: **RGB** (query image, ± reference image) · **IoT** (`iot_value`) ·
**ENT** (enterprise: work orders, technician presence — L3 only) ·
**META** (catalog text fields).

### L1 perception families

| Family | n | INTENDED | AVAILABLE | NECESSARY | Expected restricted-channel outcome |
|---|---|---|---|---|---|
| `gauge_degradation` | 407 | RGB: judge whether the dial face is legible through contamination on the glass | RGB full; META leaks category ~86%; IoT gives a plausible value that is *not* evidence of legibility | **RGB only.** Whether a face is obscured is not recoverable from telemetry | IoT-only ≈ prior; META-only well above prior → META must stay evaluator-only |
| `occlusion` | 81 | RGB: detect an object between camera and dial | RGB full; META leaks ~85% | **RGB only** | as above |
| `scale_interpretation` | 211 | RGB: judge whether graduations resolve at this scale | RGB full; META leaks ~94% | **RGB only** | as above |
| `glare_lighting` | 49 | RGB: detect a lighting artifact over the needle | RGB full; META leaks 100% | **RGB only** | strongest META leak; smallest family |
| `never_read` | 6 | ENT/history: the asset has no prior reading | RGB shows a normal gauge; META describes the gap | **not RGB.** This family is *correctly* non-visual | RGB-only should fail here by design; that is contract-conformant, not a defect |

**Reading `never_read`:** it is the one L1 family whose INTENDED evidence is
non-visual, so a low RGB-only score is the expected and correct result. With
n=6 it cannot support a claim either way and is reported descriptively only.

### L3 agentic families

| Family | INTENDED | AVAILABLE | NECESSARY | Note |
|---|---|---|---|---|
| FM-7 sensor–physical contradiction | RGB **and** IoT jointly | either channel alone gives a self-consistent but wrong answer | **RGB ∧ IoT** | the cleanest evidence-dependency claim in the benchmark: single-channel conditions must fail |
| FM-5/5a/5b safety gate | ENT (human presence, clearance) | RGB shows nothing about occupancy | **ENT** | RGB-only should fail |
| FM-6/6a/6b work-order coordination | ENT (existing/similar WOs) | — | **ENT** | RGB-only should fail |
| FM-1 panel stuck, FM-9/10/11 robot state | tool-return state | — | **tool channel** | not a perception question |
| FM-8 reasoning without verification | RGB required *despite* IoT sufficiency-appearance | IoT alone looks adequate | **RGB ∧ IoT** | tests whether the agent seeks physical evidence it was not forced to seek |

---

## 2. Diagnosis rule (fixed in advance)

For each family, cross INTENDED against the measured modality ablation:

| Pattern | Diagnosis | Decision |
|---|---|---|
| restricted ≪ full, NECESSARY ⊆ INTENDED | contract holds | **KEEP** |
| restricted ≈ full, NECESSARY genuinely metadata-level | contract mis-stated, not leaked | **REVISE the claim**, keep scenarios |
| restricted ≈ full, INTENDED requires physical/digital evidence | **leakage** | **REVISE** (strip leak) or **REMOVE** |
| full ≈ majority-class baseline | class prior, not capability | **REMOVE from headline set**; retain in the natural-prior view |

Thresholds: "≈" means the paired-bootstrap 95% CI on the difference includes
zero; "≪" means the CI excludes zero **and** the point difference exceeds 10 pp.

---

## 3. Baselines fixed in advance

| Condition | Definition | Pre-computed value |
|---|---|---|
| F1 majority-class | always `CLEAN_GAUGE` | **54.0%** (754 query rows) |
| F2 description keyword rule | description → category → action | **61.5%** |
| F3 category oracle | gold category → action | **100%** by §0.1 |

F3 is not a model result. It is stated to make §0.1 unmissable: any system that
recovers the category has already recovered the L1 action.

---

## 4. What V1 can and cannot conclude

**Can:** whether each family's gold decision is recoverable without its INTENDED
evidence; whether META leakage is live (it is not, today); whether any family is
at the class prior.

**Cannot:** anything about the decision layer at L1, per §0.1. Nor can it
distinguish "model lacks the capability" from "channel lacks the information" —
that is E2's job (instruction control), run afterwards.

---

## Amendments

*(append only, dated, after results are seen)*

- *(none yet)*
