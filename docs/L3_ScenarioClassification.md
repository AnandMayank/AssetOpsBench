# L3 scenario classification and the minimum evidence-dependency pilot

Produced after the L3 preflight returned 2/7 READY. The preflight's value was
negative-space: it showed the existing L3 scenarios are not one experimental
class, and that forcing modality ablations onto all of them would manufacture
invalid comparisons. This document classifies what exists and proposes the
smallest set of new scenarios that yields a valid pilot.

Source of truth throughout is `manifest.json` + `groundtruth.txt`; see
`docs/FM_Crosswalk.md` for why the scenario CSV is not cited.

---

## 1. Classification of existing L3 scenarios

| Class | Meaning | Scenarios | Modality ablation valid? |
|---|---|---|---|
| **A. Evidence-dependency** | gold determined by one channel while another is present as a shortcut; gold survives withholding the shortcut | **R009** (FM-6a), **R015** (FM-8) | **yes** — these are the 2 READY |
| **B. Insufficient-evidence** | gold requires *both* channels, so withholding either makes it unreachable | R011 (FM-7a), R012 (FM-7b), R014 (FM-7d) | no — ablation yields an unanswerable task, not a weaker agent |
| **C. Procedural / tool-ordering** | gold turns on the *order* of calls, not on which evidence exists | R001, R005, R006, R007, R016, R017, R018, R023, R024 | no — no evidence axis to ablate |
| **D. Relational / work-order** | gold determined by enterprise state | R008, R010, R021, R022, R025, R026 | partial — one competence arm only; withholding WO removes the gold |
| **E. Sequential / persistent** | gold depends on history across readings | R013 (FM-7c), R043, R044 (FM-23) | not as a *modality* ablation; history is a third axis |

**The distinguishing property of class A**, and the reason only it supports the
experiment: the gold action is reachable from the *restricted* channel alone,
while the withheld channel offers a tempting shortcut. That makes the restricted
arm a competence measurement rather than an unanswerable task. R009's gold is
COMMIT from a fresh gauge image with IoT present as a substitute; R015's is the
same with history/IoT as the substitute.

Classes B–E remain valuable — they are simply *different experiments*
(escalation-under-missing-evidence, procedural compliance, relational
coordination, drift detection) and must not be pooled into an
evidence-dependency claim.

---

## 2. Why 6 READY cannot be reached from the existing set

- Class B cannot be converted without changing gold labels, which the frozen
  contract forbids.
- Class C has no evidence axis at all.
- Class D gives one competence arm per scenario; a second would require gold to
  survive removal of work-order state, which by construction it cannot.
- Class E needs a temporal axis, not a modality axis.

Reaching the gate therefore requires **new scenarios of class A**, not
reinterpretation of existing ones.

---

## 3. Minimum new-scenario proposal — 4 scenarios

Four is the minimum: 2 existing (R009, R015) + 4 new = 6 READY.

Every proposal uses an **existing FM code and competency**, the existing action
interface `{COMMIT, ESCALATE, ABORT}`, and the class-A template: *gold
determined by the physical read; a digital channel present as a shortcut.*

Gold is deliberately balanced 2 COMMIT / 2 ESCALATE. An all-COMMIT pilot would
reproduce the L1 class-prior problem, where a model that always commits scores
perfectly.

| ID | FM | Competency | Asset | Gold | Physical evidence | Digital shortcut |
|---|---|---|---|---|---|---|
| **N1** | FM-6a | relational | chiller_6 | COMMIT | gauge legible, in range | IoT **agrees** — tempts skipping the read |
| **N2** | FM-6a | relational | metro_pump_1 | ESCALATE | gauge legible, **out of range** | IoT reads nominal — tempts committing "normal" |
| **N3** | FM-8 | epistemic | motor_01 | COMMIT | gauge legible, in range | 30-day historical average available |
| **N4** | FM-7c | relational | hydraulic_pump_1 | ESCALATE | gauge legible, **out of range** | history shows the value as a routine outlier |

N2 and N4 are the load-bearing pair: gold is ESCALATE and is determined by the
*physical* reading, so withholding the digital channel leaves gold reachable
while removing the shortcut that would produce the wrong answer.

### 3.1 Required proofs, per proposal, before implementation

Each must be demonstrated by `scripts/l3_preflight.py` — not asserted — before
any scenario file is written.

| Proof obligation | How it is discharged | N1 | N2 | N3 | N4 |
|---|---|---|---|---|---|
| full arm has a valid gold action | gold parses from `groundtruth.txt`; scorer returns CC=1 for it | ✓ | ✓ | ✓ | ✓ |
| each restricted arm has an expressible action | all three verdicts round-trip through the live scorer | ✓ | ✓ | ✓ | ✓ |
| gold and non-gold both reachable | CC=1 for gold, CC=0 for ≥1 other, in **every** arm | ✓ | ✓ | ✓ | ✓ |
| withholding changes the information | rendered payload differs from FULL at `payload_signature` level | ✓ | ✓ | ✓ | ✓ |
| no field leaks the withheld evidence | `reconstructible()` returns empty for the withheld quantity | ✓ | ✓ | ✓ | ✓ |
| not structurally forced by the verifier | gold reachable **and** ≥1 non-gold reachable ⇒ not forced | ✓ | ✓ | ✓ | ✓ |
| belongs to an existing FM/competency | FM-6a, FM-8, FM-7c — all have scoring branches already | ✓ | ✓ | ✓ | ✓ |

**Design rule that makes these dischargeable**, and the one the existing class-B
scenarios violate: *the digital value must never be the sole determinant of
gold.* If it is, withholding it makes gold unreachable and the arm degenerates
into an insufficient-evidence probe.

### 3.2 Authoring constraints

1. The IoT value must appear in the question prose **only as context**, never as
   the quantity gold depends on — otherwise redaction changes the answer rather
   than removing a shortcut.
2. The physical reading must not be stated in prose at any point; it comes from
   `capture_image`. (`reconstructible()` currently checks only withheld *digital*
   quantities, precisely because no physical value appears in prose today. A
   scenario that stated one would defeat the check — see the asserted limitation
   in `test_reconstruction_check_is_asymmetric_and_says_so`.)
3. IoT history must use the existing schema: per-asset records keyed on
   `asset_id` + `timestamp` in `shared/iot/<asset>.json`, matching the
   AssetOpsBench CouchDB `iot` convention. No new schema.
4. Work orders, where referenced, use the existing Maximo field set in
   `shared/work_order/workorders.csv`.

---

## 4. Resulting pilot, if the four are built and pass

| Scenario | FM | Gold | FULL | PHYSICAL_ONLY | DIGITAL_ONLY |
|---|---|---|---|---|---|
| R009 | FM-6a | COMMIT | competence | competence | probe |
| R015 | FM-8 | COMMIT | competence | competence | probe |
| N1 | FM-6a | COMMIT | competence | competence | probe |
| N2 | FM-6a | ESCALATE | competence | competence | probe |
| N3 | FM-8 | COMMIT | competence | competence | probe |
| N4 | FM-7c | ESCALATE | competence | competence | probe |

12 competence observations per model (6 scenarios × 2 arms) and 6
insufficient-evidence probes. Gold balance 4 COMMIT / 2 ESCALATE — still
COMMIT-leaning, so the majority-class baseline (67%) must be reported alongside,
exactly as split C does for L1.

**This is a pilot, not a leaderboard.** Six paired scenarios cannot resolve a
10 pp effect; per `min_pairs_for_effect`, they support only descriptive
reporting. Its purpose is to establish that the arms behave as designed on real
models before any scaled run is proposed.

---

## 5. What is not proposed

- No new FM codes, competencies, metrics or thresholds.
- No reinterpretation of class B–E scenarios to inflate n.
- No new scenarios for FM-13, FM-20, or bare FM-5/6/7, whose absence is a
  coverage gap recorded in `docs/FM_Crosswalk.md` but is not on the critical path
  for this experiment.
