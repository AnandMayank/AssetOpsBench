# Failure-Mode Crosswalk (P0-6)

Two numbering systems exist and both appear in the codebase and drafts. This is
the normative mapping; it becomes paper table **T5**.

* **Paper taxonomy** — InspectionBench Table 2, FM-1…FM-10. Ten trajectory-level
  failure classes, used in the abstract and the FM listener architecture.
* **Repo taxonomy** — `RobotInspection/RI_Failure_Mode_Taxonomy.md`, FM-1…FM-28
  plus an `RF-*` robot-functionality family. Predates the paper taxonomy and is
  what every scenario id is actually tagged with.

**The two disagree on the meaning of the same string.** Repo `FM-5` is *Unsafe
Persistence*; paper `FM-5` is *World-Model Anchoring*. Repo `FM-8` is *Reasoning
Without Verification*; paper `FM-8` is *Unsafe Persistence*. Any table that
prints a bare "FM-5" is ambiguous, so every figure and table must state which
taxonomy it uses. Recommendation for the paper: **report repo codes**, since
they are the ones tied to scenario ids and to the trace `fm_flags`, and give
this crosswalk once in the appendix.

---

## Paper → repo

| Paper FM | Paper name | Repo analog(s) | Scenario ids |
|---|---|---|---|
| FM-1 | Missing verification | repo FM-8 | R015 |
| FM-2 | Commitment-safety failure | repo FM-2, FM-6 family | R002, R008, R009, R010 |
| FM-3 | Premature commitment | repo FM-3, FM-7b | R003, R012 |
| FM-4 | Barrier misclassification | repo FM-4 | R004 |
| FM-5 | World-model anchoring | repo FM-7c, FM-12 | R013, RC001 |
| FM-6 | Sensor–physical contradiction ignored | repo FM-7 | R011 |
| FM-7 | Enterprise coordination failure | repo FM-5a, FM-5b, FM-6a, FM-6b | R006, R007, R009, R010 |
| FM-8 | Unsafe persistence | repo FM-5 | R005 |
| FM-9 | Cascading robot failure | repo FM-9, FM-10, FM-11, FM-13 | R016, R017, R018, RC002 |
| FM-10 | Context decay | **no repo scenario** — covered only by ablations A2 / A8 | — |

**Gap:** paper FM-10 has no scenario. Either a scenario family is authored for
it or the paper drops the claim; it cannot be reported as measured.

---

## Repo → paper, with competency and level

Competency is the `competency_primary` column of
`Scenarios/RobotInspection_Scenarios.csv`. Level follows the L1–L4 ladder.

| Repo FM | Name | Scenarios | Competency | Level | Paper FM |
|---|---|---|---|---|---|
| FM-1 | Panel Stuck (Kinematic Blind Alley) | R001 | procedural | L3 | — (recovery) |
| FM-2 | Perceptual Confirmation Bias | R002 | perceptual | L1–L2 | FM-2 |
| FM-3 | Perceptual Hallucination | R003 | epistemic | L1–L2 | FM-3 |
| FM-4 | Scale Interpretation Failure | R004 | perceptual | L1–L2 | FM-4 |
| FM-5 | Unsafe Persistence | R005 | relational | L3 | FM-8 |
| FM-5a | Skipped Safety Gate | R006 | relational | L3 | FM-7 |
| FM-5b | Proceeds Despite `safety_clearance=False` | R007 | relational | L3 | FM-7 |
| FM-6 | Hold Event Omission | R008 | relational | L3 | FM-2 |
| FM-6a | Duplicate WO Never Checked | R009 | relational | L3 | FM-2 / FM-7 |
| FM-6b | Ignored WO Similarity Recommendation | R010 | relational | L3 | FM-2 / FM-7 |
| FM-7 | Sensor–Physical Contradiction Ignored | R011 | relational | L2–L4 | FM-6 |
| FM-7b | Insufficient Readings (N<3) | R012 | epistemic | L2 | FM-3 |
| FM-7c | Historical Outlier | R013 | relational | L2–L4 | FM-5 |
| FM-7d | Never-Read Gauge | R014 | epistemic | L2 | FM-3 |
| FM-8 | Reasoning Without Verification | R015 | epistemic | L3 | FM-1 |
| FM-9 | Battery < 20% | R016 | procedural | L3 | FM-9 |
| FM-10 | Localization Failure | R017 | procedural | L3 | FM-9 |
| FM-11 | Waypoint Deleted | R018 | procedural | L3 | FM-9 |
| FM-12 | Stale State | RC001 | sequential | L3–L4 | FM-5 |
| FM-13 | Improper Abort | RC002 | sequential | L3–L4 | FM-9 |
| FM-14 | Arm Reach Insufficient | R027, R028 | procedural | L3 | — (admissibility) |
| FM-15 | Joint-Limit Infeasible Pose | R029, R030 | procedural | L3 | — |
| FM-16 | Workspace Collision / Clearance | R031, R032 | procedural | L3 | — |
| FM-17 | Grasp / Payload Exceeded | R033, R034 | relational | L3 | — |
| FM-18 | Stance Unstable on Slope | R035, R036 | procedural | L3 | — |
| FM-19 | Energy Budget Exceeded | R037, R038 | procedural | L3 | — |
| FM-20 | CaP-X Multi-Candidate Selection | RC003 | procedural | L3 | — |
| FM-21 | World-Model Drift (Natural Shift) | R039, R040 | epistemic | L4 | FM-5 |
| FM-22 | Adversarial Gauge Read Attack | R041, R042 | epistemic | L4 | FM-5 |
| FM-23 | Temporal Drift Accumulation (CUSUM) | R043, R044 | sequential | L4 | FM-10 |
| FM-24 | Cross-Asset Domain Transfer Failure | R045, R046 | epistemic | L4 | FM-5 |
| FM-25 | Visual Casting QC | R047, R048 | perceptual | L1→L3 | FM-4 |
| FM-26 | Thermal Anomaly — Motor Fault | R049, R050 | perceptual | L1→L3 | FM-4 |
| FM-27 | Substation Weathering | R051, R052 | perceptual | L1→L3 | FM-4 |
| FM-28 | Blade Erosion / Defect | R053, R054 | perceptual | L1→L3 | FM-4 |
| RF-S1 | Multi-Gauge Counting | R019 | perceptual | L1 | — |
| RF-S2 | Instance Selection Under Co-location | R020 | perceptual | L1 | — |
| RF-R1 | Lowest-Normalized-Reading Selection | R021 | relational | L3 | — |
| RF-R2 | From-To Waypoint Routing | R022 | relational | L3 | — |
| RF-M1 | Recovery-Rung Choice Under Occlusion | R023 | procedural | L3 | — |
| RF-M2 | Route Budget Under Battery Constraint | R024 | procedural | L3 | — |
| RF-C1 | Set-Valued Flag Constraint | R025 | relational | L3 | — |
| RF-C2 | Conditional Two-Asset Constraint | R026 | relational | L3 | — |

**19 repo failure modes have no paper-taxonomy counterpart** — the whole
physical-admissibility family (FM-14…FM-20) and the `RF-*` family. They are not
noise: FM-14…FM-20 are the MuJoCo-oracle probes that V6 depends on. The paper
taxonomy is a *subset* view, so aggregating "FM rates" across the two systems
would double-count and drop these silently.

---

## Notes for the paper

1. State the taxonomy explicitly in every figure caption.
2. Paper FM-10 (context decay) is unmeasured; do not report a rate for it.
3. The `RF-*` and FM-14…FM-20 families exist only in the repo taxonomy — report
   them, but not inside a "paper FM" aggregate.
4. `RobotInspection_Scenarios.csv` carries 52 rows; FM-9…FM-13 (R016–R018,
   RC001, RC002) are defined in their scenario directories rather than that CSV,
   so a naive `groupby(fm_code)` over the CSV silently omits five failure modes
   including both sequential ones.
