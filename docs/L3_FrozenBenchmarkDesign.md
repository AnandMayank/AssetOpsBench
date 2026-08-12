# Frozen benchmark design — P3, P4, P5, P6

Produced after the three-model cross-model pilot. **No powered run performed; no
sample size committed.** Historical results are unchanged.

---

# P3 — Frozen scenario / taxonomy audit

## Suitable for powered evaluation

| ID | Family | World factors | Gold derivation | Required evidence | Required procedure | Shortcut | Primary | Secondary | Validity | Causal cells | Powered? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `GEN-*` (generator) | A evidence-dependency | physical in/out band × IoT agrees/disagrees; coordination state | `derive_gold(world)`, world sampled first | physical read | navigate→observe | IoT value | CC, PROC | CC_grounded, integrity | **VALID** | 4 (+coordination) | **yes** |
| R001 | C procedural | panel stuck | groundtruth sequence | panel access attempt | escalate after repeated failure | — | ORDERING, PROC | CC | VALID | 1 | yes |
| R005 | C procedural | technician + active WO | groundtruth | enterprise read | yield before access | — | ORDERING, PROC | CC | VALID | 1 | yes |
| R006, R007 | C procedural | nominal | groundtruth | pose | `get_pose` before `open_panel` | — | ORDERING | CC, PROC | VALID | 1 | yes |
| R016 | C procedural | battery < threshold | groundtruth | battery read | abort and dock | — | ORDERING, PROC | CC | VALID | 1 | yes |
| R017 | C procedural | localisation failed | groundtruth | pose read | abort | — | ORDERING, PROC | CC | VALID | 1 | yes |
| R018 | C procedural | waypoint inactive | groundtruth | waypoint read | escalate, do not navigate | — | ORDERING | CC | VALID | 1 | yes |
| R023 | C procedural | gauge view obstructed | groundtruth | image | reposition before re-read | zoom | ORDERING, PROC | CC | VALID | 1 | yes |
| R024 | C procedural | constrained battery | groundtruth | battery read | budget the route | — | ORDERING, PROC | CC | VALID | 1 | yes |
| R008 | D relational | active corrective WO | groundtruth | work-order read | yield | — | CC | ORDERING | VALID | 1 | yes |
| R010 | D relational | high-similarity prior WO | groundtruth | similarity read | follow recommendation | — | CC | ORDERING | VALID | 1 | yes |
| R021 | D relational | three gauges, differing spans | gold-answer JSON | 3 gauge reads | normalise before comparing | raw-value compare | CC | ORDERING | VALID | 1 | yes |
| R022 | D relational | from-to routing | groundtruth | waypoint reads | routed traversal | — | CC | ORDERING | VALID | 1 | yes |
| R025 | D relational | survey-only constraint | gold-answer JSON | 3 gauge reads | **no** `commit_reading` | commit | CC | ORDERING | VALID | 1 | yes |
| `SEQ-*` (generator) | E sequential | drift process × episode index | `derive_sequence_gold(world, k)` | fresh read **per episode** | re-observe before acting | prior reading | CC, PROC | CC_grounded (stale-state) | **VALID, untested on models** | 3 processes | not yet |

## Excluded from powered evaluation

| ID | Family | Reason | Disposition |
|---|---|---|---|
| **R009, R015** | A | hidden physical value instantiated to satisfy a pre-existing gold | reference set only; results preserved unchanged |
| **R055–R058** | A | world placed on the band side the verdict requires (`state←verdict`); R055/R057 shortcut points at gold | reference set only |
| **R026** | D | gold `PARTIAL` — composite per-asset verdict | **eligible once the composite interface is adopted**; gold unchanged |
| R011, R012, R014 | B insufficient-evidence | ablation arms are probes, not competence arms | separate experiment |
| R013, R043, R044 | E | hand-authored; superseded by generated sequences for powered use | reference |

Nothing excluded was repaired.

---

# P4 — Primary paper estimands

| ID | Estimand | Definition | Role | Rationale |
|---|---|---|---|---|
| **E1** | per-model operational performance | CC, PROC per family | **PRIMARY** | the benchmark's core claim; both are preregistered/frozen |
| **E4** | procedural safety | ORDERING from executed traces | **PRIMARY** | a distinct construct CC cannot express — R016/R017 scored CC=1 while violating ordering |
| **E2** | evidence-grounding gap | CC − CC_grounded, per model | **SECONDARY** | real and separable, but observed in 1 of 3 models; reporting it as primary would rest the paper on one model |
| **E3** | cross-model heterogeneity | variation of E2 across models | **SECONDARY — and the paper's most interesting result** | the finding *is* the heterogeneity, not the effect |
| **E5** | sequential/state consistency | per-episode CC_grounded; stale-state reuse rate | **DIAGNOSTIC** until class E has run on models | apparatus validated, construct unmeasured |

**E2/E3 are secondary jointly and deliberately.** The honest framing is not "there
is a grounding gap" but "conventional decision accuracy *can* overestimate
grounded safety, and whether it does is a model property." That is an existence
plus heterogeneity claim, which secondary status supports and primary status
would overstate.

---

# P5 — Power and design proposal

## Variance decomposition, from the three-model pilot

| Source | Estimate | Basis |
|---|---|---|
| **Between-model** (FULL CC) | 7/12, 9/12, 11/12 → **33 pp spread** | identical worlds, three models |
| **Within-model** (n=12) | SE ≈ 14 pp | binomial at p≈0.75 |
| **Grounding gap E2** | 3, 0, 0 of 12 | most mass at zero |

**Between-model variance exceeds within-model variance.** Adding scenarios
shrinks only the smaller term.

## Models: fixed targets, not random units

The three models differ *qualitatively* — two never skip evidence, one does.
Treating them as exchangeable draws from a population would license pooling, and
pooling a 3/12 gap with two 0/12 gaps produces a mean that describes no model.

**Recommendation: models are fixed evaluation targets. Report per model, never
pooled.** E3 is then a descriptive comparison across a named set, which is what
the data supports.

This has a direct consequence: **there is no single N.** Sizing is per model per
family, and the cross-model claim is qualitative.

## Proposed frozen protocol

| Element | Value | Rationale |
|---|---|---|
| Family A | **40 generated worlds** per model | ≈23 CC-discriminating at the observed 58% rate; ±8 pp within-model at p≈0.75 |
| Family C | **9 scenarios** (all) | fixed set; ORDERING is per-scenario, not sampled |
| Family D | **5 scenarios** (+R026 on adoption) | fixed set |
| Family E | **20 sequences × 3 episodes** per model | 60 episodes; enough to see stale-state reuse if present |
| Arms | FULL, PHYSICAL_ONLY (competence); DIGITAL_ONLY (apparatus sanity) | DIGITAL_ONLY never enters a model claim |
| Models | **≥5**, named and frozen | 3 gave 1 positive; 5 distinguishes "rare" from "one-off" |
| Repeats | 1 per world at temperature 0 | non-determinism is small; budget goes to models |
| Intervals | per-model bootstrap CI on paired differences; **no pooled CI** | pooling assumes exchangeability that is absent |
| MDE | ~15 pp within model at N=40 | honest given n; not sufficient for a 5 pp claim, and none is made |

**Total ≈ 40 + 9 + 5 + 60 = 114 episodes per model per arm-set**, ~340 calls per
model. At five models that is ~1,700 calls — tractable.

## What would change this

If a fourth and fifth model both showed a nonzero gap, E2 could move to primary
and models could plausibly become random units. **That decision should be made
after the data, not before.**

---

# P6 — MuJoCo justification

**Not required for any construct currently in the benchmark.** Every family
above executes on the CouchDB/MCP apparatus, including class-E drift, which is a
change in a scalar hidden state.

Constructs the current executor **cannot** represent, all mapping to
FM-14…FM-20 and none presently under test:

| Construct | Why the executor cannot express it |
|---|---|
| arm reach / joint limits | no kinematic model; `open_panel` is a state flag |
| workspace clearance / collision | no geometry |
| stance stability on a slope | no contact dynamics |
| viewpoint feasibility | `capture_image` renders a dial directly; there is no camera pose, so "can this gauge be seen from here" is unaskable |
| real optical constraints | glare, focus, motion blur and genuine occlusion are absent from a drawn dial |

The last two are the strongest case: the recovery-ladder construct (RF-M1,
agentic zoom vs reposition) is currently approximated by a `panel_stuck` flag
rather than an actual obstructed line of sight.

**Recommendation: MuJoCo remains Stage 2, justified only when FM-14…FM-20 or a
genuine viewpoint-feasibility construct enters the benchmark.** Adding it now
would expand the apparatus without expanding what can be measured.
