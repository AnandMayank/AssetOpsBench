# Causal coverage matrix — after the minimal contrastive repairs

Nine scenarios added (R059–R067). **No historical scenario, result, frozen metric
definition or preregistered value was modified.** No model or API call was made
in producing this document. The powered benchmark has not been run.

---

## 0. What was added, and what was rejected

| Item | Decision |
|---|---|
| 5 class-D enterprise-off partners (R059–R063) | **added** |
| 4 class-C ordering controls (R064–R067) | **added** |
| Composite structured verdict interface | **adopted** in `l3_grounded_scoring.py`; CC semantics unchanged |
| Raise A-generator coordination rate 25% → 40% | **rejected** — see §4 |

---

## 1. The matrix

| Family | Causal factor | Contrastive partner | Gold reachability | Shortcut | Attribution valid? | Independent mechanisms | Scenarios |
|---|---|---|---|---|---|---|---|
| **A** evidence-dependency (`GEN-*`) | physical value in/out band; coordination state | **2×2 factorial**, generated: physical × IoT, both cells populated | derived from the world by `derive_gold`; reachable only via `capture_image`/`read_gauge` in 7/12 | IoT value (misleading in 7/12) | **yes** — factorial, gold never supplied | 2 (+coordination override) | 12 pilot / 40 protocol |
| **C** procedural — ordering | required call order given a precondition | **4 matched controls**: R006→R064, R007→R065, R016→R066, R017→R067. Same asset, same `robot_state`, same gold; ordering constraint removed | from groundtruth; reachable in both members | none | **yes for the 4 controlled**; the other 5 remain single-cell | 9 | 9 + 4 = **13** |
| **D** relational — enterprise | active WO / technician / similarity / survey constraint | **5 matched pairs**, enterprise factor off: R008→R059, R010→R060 (**causal**, gold flips to COMMIT); R021→R061, R022→R062, R025→R063 (**non-causal**, gold unchanged) | from groundtruth; reachable in both members | none | **yes** — both halves present | 5 | 5 + 5 = **10** (+R026) |
| **D** composite | per-asset human presence | none | `{chiller_6: ABORT, motor_01: COMMIT}` | none | partial — measurable now that the interface is adopted, still single-cell | 1 | 1 (R026) |
| **E** sequential (`SEQ-*`) | drift process × episode index | **stationary control** vs step / linear_drift | `derive_sequence_gold(world, k)` from the world | prior-episode reading | **yes** — stationary arm is the control | 3 processes | 20 sequences × 3 episodes |

### Why both halves of the D contrast matter

A pair where gold always flips would confirm nothing: it would be consistent with
"the enterprise factor changed the world" *and* with "the control is simply a
different scenario." R061/R062/R063 hold the factor's removal fixed while gold
stays put, so a model that changes its answer there is responding to surface
variation rather than to the causal factor. Two causal + three non-causal is the
minimum that makes "the enterprise factor moved the decision" falsifiable.

### Why the C controls preserve gold

Ordering controls vary *only* the ordering requirement. Asset, precondition
(`robot_state`) and gold are identical to the parent, verified mechanically in
`test_contrastive_repairs.py`. If gold changed, an ordering failure could not be
separated from a decision failure. Both precondition kinds are covered —
R064/R065 nominal, R066/R067 fault-carrying — so ordering is not confounded with
condition recognition.

---

## 2. Verification of the nine new scenarios

`src/orchestrator/tests/test_contrastive_repairs.py`, 47 tests, all passing.

| Property | How it is checked | Result |
|---|---|---|
| Not gold-conditioned | asset drawn from `ASSETS`; physical envelope from the asset spec, never from the label; controls inherit the parent's world | **pass** — no new hidden value chosen to produce a verdict |
| No answer leak | agent-visible `question.txt` scanned for verdict-revealing phrasing | **pass** 9/9 |
| Not a duplicate | normalised question text compared against the parent and against every other control | **pass** — all distinct |
| Well formed | `question.txt`, `groundtruth.txt`, `manifest.json`; `provenance.control_for` names the parent | **pass** 9/9 |
| Causality matches the declared pair | gold changed ⟺ factor declared causal | **pass** 5/5 |
| World preserved | asset and `robot_state` identical to parent | **pass** 9/9 |

One correction made during verification: the test helper initially read gold from
an `Expected verdict:` line only, which R021 and R025 do not use — they state gold
in the gold-answer JSON. That made both look as though their control had flipped
gold when nothing had. The helper now reads both layouts. This is the same
single-layout parser fault that produced false verdicts in the earlier class-D
audit; the parser, not a scenario, was wrong in both cases.

---

## 3. Integrity suite — zero model/API calls

| Suite | Result |
|---|---|
| `pytest src/orchestrator/tests` | **247 passed** |
| `scripts/l3_execution_preflight.py` | **16/16 passed** |
| `scripts/l3_evidence_audit.py` | **6/6 pass** |
| `scripts/scenario_construct_audit.py` | ran; reasons unchanged from the frozen record |
| Generator reproduction | 12-world seed-2026 set **byte-identical** to the committed pilot |
| `check_generator_contract()` | clean — generator still cannot see a label |

Nothing in the historical record moved.

---

## 4. Coordination-rate audit (the gate on the 25% → 40% change)

The instruction was to change the rate only if 40% has domain or construct
justification, and not if it exists only for statistical balance.

**Finding:** `ACTIVE_WO_RATE = 0.25` and `TECHNICIAN_PRESENT_RATE = 0.15` were
introduced in commit `66b0d2e` as bare literals, with no comment, citation or
domain source. The only real work-order data available is three rows, which would
imply ~67% — a figure no one would adopt from n=3.

So 25% is not defended. But 40% is not defended either: it has no operational
source, and its stated motive was to thicken the coordination cell from 3/12.
That is optimising the benchmark for statistical convenience, which is the thing
the instruction rules out.

**Decision: rejected.** Both rates are retained at their existing values and
**frozen as named generator parameters** with the rationale recorded in-code, so
the number is now explicit rather than an undocumented literal. The principled
fix for the thin coordination cell is stratification on the coordination factor —
sampling to fill the cell without misrepresenting a base rate — which is a design
change to make deliberately, not a knob to turn before a powered run.

---

## 5. State after the repairs

| Family | Scenarios | Attribution |
|---|---|---|
| A | 12 pilot / 40 protocol | valid — factorial |
| C | 13 (9 + 4 controls) | valid for R006/R007/R016/R017; R001, R005, R018, R023, R024 remain single-cell |
| D | 10 (5 + 5 controls) + R026 | valid — matched pairs with both causal and non-causal halves |
| E | 20 sequences | valid — stationary control; **still unrun on models** |

The remaining known gaps, stated rather than closed:

- **Five class-C scenarios still have no control** (R001, R005, R018, R023, R024).
  Ordering attribution holds for the four that were controlled; for these five a
  failure cannot be separated from scenario difficulty.
- **R026 is a single composite cell** — now expressible, not yet contrastable.
- **Class E has never been run against a model.** Its apparatus is validated by
  the ten acceptance tests; the construct is unmeasured.

**Stopping here as instructed.** No powered benchmark, no MuJoCo, no additional
model calls.
