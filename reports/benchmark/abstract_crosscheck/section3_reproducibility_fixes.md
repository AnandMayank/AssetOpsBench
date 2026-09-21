# Section 3 reproducibility fixes — InspectionBench (ICLR 2027 draft, `InspectionBench-1.pdf`)

Status: **written, not yet pushed.** Scope: Sections 3.1–3.3 (tool/execution-backend subsections
included per your later addition), plus the abstract/intro sentences that quote Section-3-derived
numbers. Line numbers are the PDF's own margin numbers. Every fact below was re-opened against the
cited file this pass (four parallel read-only investigations; citations not re-verified a second
time are marked so explicitly). No repo code, manifest, or paper file was modified. Nothing has been
committed or pushed — that happens only if/when you ask.

Conventions: **[MAIN]** → main text. **[APPENDIX]** → appendix. `[FILL AFTER FREEZE]` → a number
that only exists after a repo action (R1–R5) is done; do not submit with any placeholder left.
**REPO** = a repo action is required first. **TEXT** = wording only, fixable today.

---

## Repo actions, in the order that unblocks the text (do these first)

| # | Action | Unblocks |
|---|---|---|
| R1 | Commit `src/perception/` (currently untracked). Restore or drop the `enhance_backgrounds.py` FLUX step. Seed the numpy RNG (`generate_perception_gauge_49.py` currently reads unseeded global numpy state). Point `OUTPUT_ROOT` somewhere that exists and re-run. Write a frozen manifest (generator version, seed, per-image counts, SHA256 checksums). | Item 3 |
| R2 | Fix the Trace A units mismatch (a 0–1 gauge fraction compared against IoT = 150 PSI, `perception_real.csv:9`; span fallback at `pmc_dataset.py:137-144`). Re-run Trace A and Trace B under one fixed `min_reads` on the same day. | Item 1 |
| R3 | Build a real facility-image catalog (image, asset, viewpoint, gauge class) or drop the unverifiable 116/viewpoint claims down to the 45 images actually on disk. | Item 4 |
| R4 | Decide whether `evidence_provenance_class` should be added to the benchmark manifests (it currently lives only in `observation_records.json`, not in any canonical episode manifest). | Item 7 |
| R5 | Adopt `templates_inventory.csv` (drafted in Item 6 below) as a shipped artifact; it is the only thing that makes "14 templates" / "52 variation identifiers" / "600 seeded instances" auditable at all. | Item 6 |

---

## BLOCKING

### 1. Figure 3 / Trace A–B (lines 216–243, 301–319) — [MAIN] REPO+TEXT

**Current text.** L232: `commit_reading() Blocked: inconsistent evidence`. L232-234/309-313: Trace B
shows `commit_reading() ×3` as "still untrusted." L241-243: "The evaluator reads delivered evidence,
acquisition path, and procedural compliance alongside the terminal outcome."

**Problem.** Both traces are real `gpt-5.4-mini` PMC-track runs — `reports/ec/phase8h_figures/qa_demo/`
`pmc_bcb631` (Trace A) and `pmc_6a6f42` (Trace B), byte-identical to their exports in
`reports/evaluation_export/trajectories/`.
- Trace A's block reason is `hard gate: G2_iot` (IoT-disagreement gate; read-consistency C≈0.999), not
  "inconsistent evidence." Root cause traced to a units mismatch: a 0–1 gauge-fraction reading compared
  against IoT = 150 PSI (`perception_real.csv:9`), with a fallback span of 100 when the declared range is
  `UNREADABLE` (`pmc_dataset.py:137-144`) — `|0.85 − 150| / 100 ≈ 1.49 > 0.85` (the gate threshold).
- Trace B has **2** blocked commits after the zoom, not 3. The run's own summary scores the zoom as the
  *wrong* recovery rung (`correct_first_rung: "reposition"`, `rsa_correct: false`), which the paper's
  framing as a successful recovery contradicts.
- Neither trace's tool sequence is agent-chosen: `real_pmc_orchestrator.py:384-437` runs a fixed
  navigate→read×N→commit→recover script; the model answers only the `read_gauge` vision calls.
- The two traces were recorded a month apart (Trace A: 2026-07-12; Trace B: 2026-08-11) under
  different `min_reads` settings, so "same world, two rollouts" is not literally true of the stored
  data.

**Fix [MAIN], after R2:**
> Figure 3: One PMC-track scenario (AOBv2-REAL-008), one world, two acquisition policies, both run
> with gpt-5.4-mini as the gauge perceiver under one configuration (min_reads = `[FILL AFTER FREEZE]`).
> Both end at ROUTE.UPDATE. Trace A commits after repeated direct readings and is blocked by the
> IoT-agreement gate; Trace B changes view before re-reading. The acquisition sequence is scripted by
> the track; the model supplies only the gauge reading.

**Fix [MAIN], lines 301-319 lead-in:**
> In the PMC track the tool sequence is a fixed acquisition policy (navigate, read, optionally zoom,
> commit or flag); the model acts only as the perceiver. Trace A: `read_gauge` returns 0.85, 0.80, 1.00
> (conf. 0.70); `commit_reading` is blocked by the IoT-disagreement gate (G2_iot). Trace B: `read_gauge`
> is ambiguous; `commit_reading` is blocked; `agentic_zoom` then `read_gauge` returns
> `[FILL AFTER FREEZE]`; `commit_reading` is blocked `[FILL AFTER FREEZE: N]` more times.

**Fallback if you do not regenerate:** relabel Figure 3 "Illustrative PMC-track run," drop "one world,
two rollouts," and state the traces were recorded separately.

---

### 2. The four-stage evaluator does not exist in code (lines 240-243, 314-319) — [MAIN] TEXT

**Current text.** "The evaluator reconstructs each one in four stages: what evidence reached the
agent, which inspection actions were executed, whether the required steps were performed, and
whether the final action was correct."

**Problem.** No single code path implements these four named stages. PMC runs are graded by
`grader.py:grade_real_pmc` (:33-88): target-action match, no forbidden label, no perceive-commit gap,
run validity. The synthetic A–E families are scored by TDA/GSR (`l3_grounded_scoring.py`). The two
scorers never call each other (confirmed: `real_pmc_orchestrator.py`, `run_pmc_benchmark.py`,
`evaluation_export.py`, `pmc_dataset.py` import neither `execution_trace` nor `metric_contract`).

**Fix [MAIN] — split into two scorers, with exact definitions:**
> **Two scorers.** The synthetic A–E families are scored by trace-based scoring: terminal decision
> accuracy (TDA) and grounded success rate (GSR). The PMC real-image track is scored separately by a
> grader that reports pass rate, the perceive–commit gap, and recovery-rung accuracy (RSA).
>
> **TDA** = `CC`: `int(action_norm == gold_norm)` (`l3_grounded_scoring.py:194`, aliased in
> `metric_contract.py:60-69`). **GSR** = `CC_grounded`: TDA holds **and** an OBSERVATION_DELIVERED
> event exists for the required modality **and** the trace's hash chain verifies **and** the response
> cites the observations it used **and** the acted-on asset matches **and** no forbidden action was
> taken (`l3_grounded_scoring.py:101-153`, exact conjunction at :141-143). In the released data the
> last three conjuncts are close to vacuous (`metric_contract.py:35-43`), so in practice GSR differs
> from TDA almost entirely through the evidence-delivery and hash-chain conditions.
>
> **PMC commit rule.** A reading commits at score ≥ τ_commit = 0.82 and escalates for confirmation at
> score ≥ τ_escalate = 0.65, where score = 0.35·C + 0.35·A + 0.30·H — C is read consistency
> (`1 − σ/span` over repeated reads), A is agreement with IoT telemetry (`1 − |μ − IoT|/span`), and H
> is a fixed prior of 0.5 (`spot_assetops_orchestrator.py:534-548`, `real_pmc_orchestrator.py:157-183`).
> Two hard gates block commitment regardless of score: readable-rate < 0.5 across reads, or
> |μ − IoT| / span > 0.85 (`real_pmc_orchestrator.py:60-64, 185-197`).

**[APPENDIX] note.** The PMC traces carry only a `payload_digest` (`spot_assetops_orchestrator.py:958-964`),
not a hash chain. Appendix B.2's tamper-evident chain claim (`execution_trace.py:46-48, 98-108, 163-173`,
re-verified) applies to the synthetic A–E traces only; scope that sentence explicitly.

---

### 3. PerceptionGauge "1,296 scenes, 856 readable / 440 unreadable" (lines 196-201) — [MAIN] REPO+TEXT

**Problem.** `src/perception/generate_perception_gauge_49.py` is uncommitted; its declared output
directory does not exist on the mounted drive; it composites existing photos with PIL
(`Image.alpha_composite`, `GaussianBlur`, `ImageDraw`, `resize`, `ImageEnhance`) rather than
procedurally rendering; the FLUX inpainting step it depends on (`enhance_backgrounds.py`) is not in
the repo; and it reads the unseeded global numpy RNG inside `apply_gauge_degradation`, so `--seed`
does not make it fully reproducible. The 49-row seed CSV
(`AssetOpsBenchScenarioGeneration/scenarios_data/shared/catalog/perception.csv`) splits 19 readable /
30 unreadable — it does not scale to 856/440 by any integer multiple. The only "1,296" figure
findable anywhere in the repo is `reports/v7/label_status.csv`'s **real-photograph** catalog
(AOBv2-REAL-0001–1296), a different, unrelated dataset that `docs/BenchmarkSpec_Observability.md:24`
explicitly describes as having "zero procedural-render bias."

**15 asset classes (verbatim, `perception.csv` / `ASSET_GAUGE_TYPE` dict,
`generate_perception_gauge_49.py:48-64`):** Rolling-element bearings, Gears/gearboxes, Induction
motors, PMSM/synchronous motors, Centrifugal pumps, Reciprocating compressors, Hydraulic systems,
Lithium-ion batteries, Wind-turbine drivetrains, Gas turbines, Power transformers, HVAC chillers, HVAC
AHUs, CNC machine tools, Robotic manipulators.

**Fix [MAIN], after R1:**
> **Image-transformed scenes.** We composite `[FILL AFTER FREEZE: N]` scenes (`[FILL]` readable,
> `[FILL]` unreadable) from a 49-row seed matrix over 15 asset classes (bearings, gearboxes, induction
> and PMSM motors, centrifugal pumps, reciprocating compressors, hydraulic systems, Li-ion batteries,
> wind-turbine drivetrains, gas turbines, power transformers, HVAC chillers and AHUs, CNC machine
> tools, robotic manipulators) and six perception categories: gauge degradation, IoT–physical
> contradiction, occlusion, never-read assets, scale interpretation, glare or lighting distortion.
> Scenes are produced by PIL compositing (blur, occlusion masks, resize, glare, lighting) over base
> imagery from SyncG (Deng et al., 2026), together with internally curated and real facility
> photographs. Generation uses seed `[FILL]`, generator version `[FILL]`, released with a frozen
> manifest and per-image checksums.

Do not use "procedurally rendered" anywhere else in the paper — search the full text for it.
DialBench and `real_world_samples` carry no citation anywhere in the repo; either cite them properly
or mark them as internally curated, uncited imagery in the appendix.

**Fallback if you do not re-run:** delete 856/440/1,296 entirely; say "49 seed scenarios, each
composited under `[N]` transformations."

---

### 4. Facility images "116 = 75/37/4, four viewpoints, four grounded assets" (lines 188-195) — [MAIN] REPO+TEXT

**Problem.** Only 45 JPEGs exist on disk (`second_drive/external_datasets/real_world_samples/AssetOps
Gauge/Gauge/`, 44 "Medium Gauge" + 1 "Multiple Gauge," iPhone captures per `scripts/build_splits.py:53`),
with no viewpoint field and no image-to-asset mapping. `reports/paper_audit/datasets_table.md:53-56,68-69`
already flags 116 as an open, unreconciled number (only 20 of ~1,344 PMC images are wired into
`perception_real.csv`).

**Fix, after R3 (if the catalog reproduces 116):**
> An on-site session with a domain expert produced `[116]` images (`[75]` pressure and temperature
> gauges, `[37]` standalone temperature instruments, `[4]` dyed-coolant spill proxies), catalogued in
> `facility_catalog.csv` with asset, viewpoint (straight-on, 15°, 30°, oblique), and gauge class per
> image.

**If it does not reproduce:** state the verifiable count (45 images) and drop the viewpoint sentence
unless every image carries a viewpoint tag.

**Setting claim (folds in your Item F.6):** the intro's "collected at operating data-center
facilities" is unsupported — no artifact anywhere ties these images or the four grounded assets to a
data center. The four assets (a chiller, two pumps, a motor) read as a mechanical/utility plant, not
an IT data-center floor. Recommend "industrial facility" pending confirmation from whoever ran the
on-site session, and remove "data-center" from the intro until confirmed.

---

## HIGH

### 5. `[are/are not]` (line 356) — [MAIN] TEXT

**Fix:**
> Of the 93-episode pilot, 12 legacy episodes (fm7a ×9, R011 ×3) are retained in the release and
> **are** counted in the 4,075; of these, only 3 (the R011 episodes) are scored by the current
> dispatcher — the 9 fm7a episodes carry `scenario_id=None` and are never dispatched
> (`paper_results_v1.json`, data-quality note). Pilot scenarios R001, R005 and R006 are `dimension="C"`
> fixtures from the same 93-episode pool, **not** B-legacy, and are **not** part of the 4,075 at all
> (the 112 canonical C episodes come from a disjoint T-C-* set in `cd_4000_final_manifest.json`).
> Correct the paper's list to name only fm7a and R011 as retained-and-counted legacy templates.

---

### 6. "14 templates", "52 variation identifiers", "600 seeded instances" (lines 346-357) — [MAIN] REPO+TEXT

**Problem, fully resolved this pass.**
- **"14 templates" excludes the 4 T-D-PHYS-* templates that Figure 5's own D column (n=70) scores.**
  14 = 4 B-ACQ + 7 T-C + 3 T-D-enterprise (198 episodes); adding the 4 D-physical templates gives
  **18** templates covering 268 episodes. Given the paper reports D=70 in Figure 5, "14" is internally
  inconsistent with its own results table — use 18, or scope "14" explicitly to
  "enterprise/procedural/acquisition templates" only.
- **"52 variation identifiers" does not reproduce under any tested rule.** Stripping asset name and
  seed from every family's world/scenario identifier (uniformly applied) gives **61**: A 4 + E 1 + C 7
  + D-enterprise 3 + D-physical 18 + B-Acquisition 26 + B-legacy 2. Grouping B-Acquisition by
  `matched_group_id` instead of scenario-id gives 50 (49 with asset stripped from the group id). No
  artifact in the repo defines "variation identifier" or contains a 52-row grouping (grepped across
  all `.py/.md/.json/.csv`). **Recommend stating 61, with the exact rule above, as the correct number**
  — or 50 if B is reported by matched group instead of scenario id. Do not use 52 without defining
  which rule produces it.
- **"600 seeded instances each" is correct only pre-dedup, and only for A's episode count, not its
  seed count.** Each A cell is 200 distinct seeds × 3 regimes = 600 episodes (not 600 seeds).
  Post-dedup the four cells are 597/591/600/600 episodes (12 episodes removed by the fingerprint
  dedup). Suggested wording: "200 seeded worlds per cell, each realised under 3 evidence regimes
  (600 episodes per cell before fingerprint dedup, 597–600 after)."
- **B-ACQ-3 is confirmed structurally identical to B-ACQ-1** (same 6 patterns, 12 episodes,
  `b_acquisition_generator.py:484-505`), but **UHA is not exclusive to B-ACQ-3** — `UHA` is non-null on
  6 B-ACQ-1 episodes and 6 B-ACQ-3 episodes; the reported `UHA_n_applicable=12` spans both templates,
  not "the 12 B-ACQ-3 episodes" as the paper's phrasing implies. Correct this nuance if UHA is
  discussed.
- **45-episode dedup, resolved with new numbers:** 42 fingerprint groups cover 87 of 3,840 A/E
  episodes (39 groups of 2, 3 groups of 3); collapsing to one representative removes 45. Per family:
  **A −12, E −33** — duplicates always occur as complete matched triples, so 15 whole worlds disappear
  (4 A worlds, 11 E sequences). The dedup is asserted in the merge manifest but not materialised: the
  source file `reports/benchmark/final_benchmark_manifest.json` still ships all 3,840 rows, so a naive
  consumer will count 4,120, not 4,075. **Flag for the release: either materialise the deduped file, or
  document the fingerprint-collapse rule prominently enough that a reader reproduces 3,795, not 3,840.**

**Fix [MAIN]:**
> Three granularities appear in this paper: 18 scenario templates at the design level (1 evidence-
> agreement A/E construct, 4 acquisition B templates, 7 procedural C templates, 3 enterprise D
> templates, 4 physical-feasibility D templates — see `templates_inventory.csv`); 61 variation
> identifiers obtained by grouping frozen-manifest world/scenario identifiers after removing asset
> name and per-instance seed; and 4,075 compiled episodes. The evidence-agreement A/E family holds
> four patterns crossed with IoT agreement or disagreement, each instantiated with 200 seeds under 3
> evidence regimes (600 episodes per pattern before a 45-episode cross-family fingerprint dedup, 597–
> 600 after). ... The acquisition B family holds BACQ1, BACQ2, BACQ3 and BACQ4; BACQ3 reuses the
> BACQ1 construction and adds scoring on the UHA (unavailable-handling accuracy) metric.

**`templates_inventory.csv` (draft below, 22 rows summing to 4,075 on `in_4075=yes` rows; ship as a
released artifact — this is what makes 14/18/52/61/600 auditable at all):**

```csv
template_id,family,n_patterns,pattern_ids,regimes_x_seeds,n_episodes_canonical,in_4075,scorer
sample_world/derive_gold,A,4,"GENA-phys_in__iot_agree|GENA-phys_in__iot_disagree|GENA-phys_out__iot_agree|GENA-phys_out__iot_disagree","3 regimes (FULL/PHYSICAL_ONLY/DIGITAL_ONLY) x 800 seeds (50000-50799; 200 per cell)",2388,yes,l3_grounded_scoring.score_l3_grounded (GSR/TDA)
sample_sequence/derive_sequence_gold,E,1,SEQ,"3 temporal slots (ep0/ep1/ep2) x 480 seeds (51000-51479)",1407,yes,l3_grounded_scoring.cc_grounded (CC_grounded)
sample_world_fm7a_contradiction,B-legacy,1,B-fm7a,"3 regimes x 3 seeds (4000-4002)",9,yes,l3_scoring (b_incumbent dispatch; NOT scored - scenario_id=None)
l3_arms/R011,B-legacy,1,R011,"3 regimes x 1 fixture world",3,yes,l3_scoring (b_incumbent; the only 3 B-legacy episodes actually scored)
B-ACQ-1,B-Acquisition,6,"GEN-BACQ1-{absent,present} x {thermal(implicit),acoustic,workorder_history}","3 matched groups x 4 assets; seeds 70000+",12,yes,b_acquisition_scoring.score_b_acquisition_episode (ADA/ASA/TDA_post/UHA -> B_AGS)
B-ACQ-2,B-Acquisition,14,"GEN-BACQ2-{iot,acoustic}_n{3,5,8}_{sufficient,insufficient} (12) + GEN-BACQ2-k1 + GEN-BACQ2-k3","6 matched groups (acoustic/iot x n=3,5,8) x 4-8 episodes",36,yes,b_acquisition_scoring (ADA/ASA/TDA_post -> B_AGS)
B-ACQ-3,B-Acquisition,6,"GEN-BACQ3-{absent,present} x {thermal(implicit),acoustic,workorder_history}","3 matched groups x 4 assets (structurally identical to B-ACQ-1)",12,yes,b_acquisition_scoring (UHA arm, shared with B-ACQ-1)
B-ACQ-4,B-Acquisition,2,"GEN-BACQ4-ambiguous|GEN-BACQ4-unambiguous","3 matched groups (acoustic-chiller_6, acoustic-hydraulic_pump_1, thermal-motor_01) x 2; metro_pump_1 absent",6,yes,b_acquisition_scoring (cross-modal reconciliation)
T-C-BATTERY_ABORT,C,1,GEN-BATTERY_ABORT,"4 assets x 9 reps; seeds 30000-30035",36,yes,ordering.procedural_coverage / ordering_satisfied
T-C-LOCALIZATION_ABORT,C,1,GEN-LOCALIZATION_ABORT,"4 assets x 6 reps; seeds 30056-30079",24,yes,ordering.procedural_coverage
T-C-ROUTE_BUDGET,C,1,GEN-ROUTE_BUDGET,"4 assets x 6 reps; seeds 30088-30111",24,yes,ordering.procedural_coverage
T-C-PANEL_STUCK,C,1,GEN-PANEL_STUCK,"4 assets x 2 reps; seeds 30036-30043",8,yes,ordering.procedural_coverage
T-C-WO_GATE,C,1,GEN-WO_GATE,"4 assets x 2 reps; seeds 30044-30051",8,yes,ordering.procedural_coverage
T-C-WAYPOINT_ESCALATE,C,1,GEN-WAYPOINT_ESCALATE,"4 assets x 2 reps; seeds 30080-30087",8,yes,ordering.procedural_coverage
T-C-PRECEDENCE,C,1,GEN-PRECEDENCE,"4 assets x 1 rep; seeds 30052-30055",4,yes,ordering.procedural_coverage
T-D-WO_GATE,D-enterprise,1,GEN-D_WO_GATE,"4 assets x 2 reps; seeds 30112-30119",8,yes,l3_scoring (CC_terminal)
T-D-WO_SIMILARITY,D-enterprise,1,GEN-D_WO_SIMILARITY,"4 assets x 2 reps; seeds 30120-30127",8,yes,l3_scoring (CC_terminal)
T-D-ROUTE_DEPENDENCY,D-enterprise,1,GEN-D_ROUTE_DEPENDENCY,"4 assets x 1 rep; seeds 30128-30131",4,yes,l3_scoring (CC_terminal)
T-D-PHYS-COUPLED,D-physical,7,"GEN-DPHYS-COUPLED-{reach+clearance,reach+stability,clearance+stability,grasp_payload+energy,energy+reach,joint_and_collision+stability,reach+clearance+stability}","4 assets x 7 dual/triple strata; seeds 60000-62011",28,yes,dphys_scoring.score_dphys_episode (CSA/CC/PAC/CS-F1/LCA)
T-D-PHYS-REACH,D-physical,4,"GEN-DPHYS-{reach-inadmissible,reach-comfortable,clearance-inadmissible,joint_and_collision-inadmissible}","4 assets x 4 single-constraint strata",16,yes,dphys_scoring.score_dphys_episode
T-D-PHYS-ENERGY,D-physical,4,"GEN-DPHYS-{energy-inadmissible,stability-inadmissible,grasp_payload-inadmissible,grasp_payload-comfortable}","4 assets x 4 single-constraint strata",16,yes,dphys_scoring.score_dphys_episode
T-D-PHYS-CAP-X,D-physical,3,"GEN-DPHYS-CAPX-{comfortable_baseline,all_inadmissible,mixed_one_admissible}","4/4/2 episodes",10,yes,dphys_scoring.score_dphys_episode
```
(22 templates, 4,075 episodes, 61 patterns — sums re-verified: 2388+1407+9+3+12+36+12+6+36+24+24+8+8+8+4+8+8+4+28+16+16+10 = 4,075.)

**On "1,280 grounded worlds" (abstract, line 094) — new item, no prior audit coverage:** 1,280 = 800 A
matched groups + 480 E sequences, and is (a) **A/E-only** — it ignores every C/D/B-Acquisition/
D-physical world entirely — and (b) **pre-dedup**: post-dedup it is 1,265 (15 whole matched
groups/sequences lost). Counting distinct world identifiers across *all* 4,075 episodes gives
**1,537** (1,265 A/E + 132 C/D-enterprise + 70 D-physical + 66 B-Acquisition + 4 B-legacy). "World"
also means structurally different things per family: in A/E one world realises ~3 episodes (one per
regime/slot); in C, D, and B-Acquisition, one world realises exactly 1 episode. **Fix [MAIN]:** either
scope "1,280 grounded worlds" explicitly to "the A/E evidence-grounding pool" (and note it is 1,265
after dedup), or replace it with 1,537 and define "world" as counted here. Do not present 1,280 as
covering the full 4,075.

---

### 7. Provenance classes (lines 210-215, 206) — [MAIN] REPO+TEXT

**Problem.** `evidence_provenance_class` is persisted, but only on the 24,601 rows of
`observation_records.json` — **absent from every canonical episode manifest**. Its distribution:
L1 (same-asset capture) = **0**; L2 (asset-class replay) = 9,768; L3 (diagnostic-only) = 510;
NOT_APPLICABLE_DIGITAL_ENTERPRISE (IoT/work-order/geometry) = 14,323. `source_id` is present on only
50 of 66 B-Acquisition episodes; every other record uses `source_ref.dataset_lineage` instead.

**Fix [MAIN]:**
> Every observation record carries an `evidence_provenance_class`: L1 same-asset capture, L2
> asset-class replay, L3 diagnostic-only, or not-applicable for digital-enterprise records (IoT,
> work-order, geometry). Across the 24,601 records the distribution is L1 = 0, L2 = 9,768, L3 = 510,
> not-applicable = 14,323. **No L1 (same-asset) physical evidence exists in this release**; grounded-
> success claims therefore rest on class-level replay, never a same-instrument capture.
>
> *(line 206)* Each record carries a `source_ref.dataset_lineage` provenance tag; `source_id` is
> additionally present on the 50 of 66 B-Acquisition episodes that were hand-curated.

**Reconcile before submitting:** lines 202-204 give 14,313 IoT + 9,757 acoustic + 515 thermal + 7
geometry + 6 RGB + 3 work-order = 24,601, but the class counts above are not a clean 1:1 remap
(N/A = 14,323 vs 14,313+7+3=14,323 ✓ actually reconciles; L3 = 510 vs 515 thermal + 6 RGB = 521 does
**not** — 11 thermal records land in L2 rather than L3). Produce the modality × class crosstab as an
appendix table before publishing either sentence, so the two paragraphs don't silently disagree.

---

### 8. `Appendix ??` for RGB and geometry (line 208) — [MAIN] short + [APPENDIX] table, TEXT

**Fix in text (line 207-208):** "...for the four grounded assets (Patel et al., 2026); Appendix
`[X]` documents the remaining RGB and geometry records."

**Appendix content:**
- **RGB (6):** Kaggle casting-defect dataset (R047–48), Zenodo 7884270 (R051–52), GitHub Blade30
  (R053–54). All L3, all on non-grounded assets (`pump_impeller_1`, `transformer_substation_1`,
  `turbine_blade_1`).
- **Geometry (7):** static fixture lookups from `robot_assets_registry.json`, one per registry asset.
- **Image mirror:** cite `amaan784/hpml-final-project` as the mirror repo used to fetch these images
  (`download_industrial_images.py:6-12,27,69-73,128-138`), alongside proper citations for the three
  original datasets.
- **Asset-type note for 3.1:** "The inspection-asset registry lists six asset types (chiller, motor,
  pump, impeller, transformer, turbine) across seven asset instances: four grounded assets used in the
  canonical episodes, plus three replay-only assets (impeller, transformer, turbine) that appear only
  in the RGB/geometry records above, never in a scored episode." State the "4 grounded + 3 replay-only"
  split explicitly — see Item C for why the abstract's unscoped "six asset types" claim is contradicted.

---

### 9. `Figure ??` snow gauge (line 252) — [MAIN] TEXT

**Fix:** scenario **AOBv2-REAL-002** (`data_detection/images/test/IMG_20220106_104236.jpg`).
> Figure `[X]` shows a snow-obstructed gauge whose dial stays visible; annotators disagreed on the
> obstruction type (snow vs frost), which we retain as a contested label rather than resolving it
> silently.
(Its dual labels are contested — same class of case as the already-pushed AOBv2-REAL-018 appendix
example.)

---

### 10. Annotation, verification, and adjudication (lines 246-262) — [MAIN] TEXT, folds in your Item E

**Problem, now fully resolved.**
- `label_windows_vlm.py` (multi-frame burst, matches the paper's "up to four evenly spaced frames"
  description) defaults to TokenRouter `gpt-5.4-mini`, not `gemini-2.5-flash`, and its output
  (`labels_vlm.json`) is consumed only by video-rendering byproduct scripts — **never** by any
  canonical manifest or `reports/v7/*` artifact.
- `src/perception/reverse_label_real_images.py` (single-image, 16-field prompt, `gemini-2.5-flash`)
  writes `perception_real.csv`/`perception_real_gt.csv`, which **is** read by the actual evaluation
  pipeline (`pmc_dataset.py`, `run_pmc_benchmark.py`, `grader.py`, `scripts/v7_label_adjudication.py`,
  `scripts/build_pairs.py`, `scripts/build_splits.py`). **This is the real pipeline; Appendix A should
  transcribe this prompt, not the multi-frame burst one, if it currently does the latter.**
- **No genuine human domain-expert adjudication exists anywhere in this benchmark's construction.**
  `"sme_adjudicated"` is a code-internal taxonomy label meaning *hand-authored by the generator's
  builder*, not "independently reviewed" (`scenario_contract.py:32-34,68`). Gold is always
  code/generator-derived (`derive_gold()`, never reads an external reviewer's input). One file states
  this explicitly: `reports/ec/phase8h1_c_gold_decisions_self_adjudicated.md:3-4` — **"Status:
  SELF-ADJUDICATED BY CLAUDE, NOT HUMAN-REVIEWED."** Evidence requirements and C-family tool sequences
  are hardcoded in generator scripts (`cd_generator.py:113-128`), also never reviewed. The 396-row
  adjudication worksheet (`reports/v7/label_adjudication_worksheet.csv`) has an empty
  `adjudicated_value` column in all 396 rows. No 3-reviewer / 30-scenario / 5-point-scale study
  artifact exists anywhere (grepped repeatedly across three sessions' worth of search).
- What genuinely exists: dual labeling of 20 photo-annotation scenarios (not benchmark gold), 18 of
  which have at least one contested field; per-field kappa from 0.14 (asset) to 0.77 (category); 0
  adjudicated rows.

**Fix [MAIN] — abstract clause "...adjudicated by domain experts where needed":**
> Gold operational decisions are derived deterministically from the frozen world state by a fixed rule
> function; two evidence tracks additionally map gold from an external dataset's own labels (MIMII,
> induction-motor thermal). No gold record, evidence requirement, or tool sequence in the current
> release was reviewed or adjudicated by an independent domain expert.

**Fix [MAIN] — Section 3.1's "A domain expert then accepts, corrects, or rejects every record...":**
> Records are either derived deterministically from world state or hand-authored during construction.
> A 396-row dual-labeling worksheet exists for the separate photograph-annotation track; its
> adjudicated-value column is empty in all 396 rows, and no completed independent domain-expert review
> of gold, evidence requirements, or tool sequences was found in this release.

**Fix [MAIN] — Section 3.3's "operational-gold and evidence-requirement adjudication and
tool-sequence validation" (paraphrased, ~line 259):** replace "adjudication"/"validation" with
"automated self-consistency tests" (the real mechanism: `test_scenario_contract.py`,
`test_cd_generator.py` — code checking generator output against generator-authored fixtures, not
third-party review).

**Fix [MAIN] — dual-label numbers, replacing the unsupported 3-reviewer sentence:**
> **Dual labeling and agreement.** Two annotators independently labeled 20 scenarios; 18 of 20 have at
> least one contested field. Per-field Cohen's kappa ranges from 0.14 (asset identity) to 0.77
> (category), with gauge_readable at 0.47 and recommended_action at 0.74. None of the contested labels
> have been adjudicated in this release; annotated fields are advisory, and operational gold comes
> from the scenario compiler, not from annotation.

**[APPENDIX]:** the already-pushed AOBv2-REAL-018 decommissioned-gauge case, as a worked example of a
contested label that turned out to reveal a real schema gap rather than annotator noise.

---

## MEDIUM

### 11. Tools (3.3, lines 366-371) — [MAIN] counts + [APPENDIX] catalog, TEXT

**Facts, re-verified this pass.** `couchdb_executor.TOOLSET` has **25** tools (recount matches the
prior pass's own list length, correcting an even-earlier "24" estimate). Across MCP servers: robot
(**29** tools, all new for InspectionBench, commit `d77c732`/2026-06-14), wo (14), vibration (8), tsfm
(6), iot (4), fmsr (3), utilities (3) — 38 reused from AssetOpsBench. **Total 67 tools across 7
servers: 29 new, 38 reused.** `agentic_zoom` exists only in the PMC track
(`real_pmc_orchestrator.py:9,80,92,345-362,414`), never in TOOLSET or any MCP server.

**Spot SDK provenance for the 29 new robot tools** (`src/servers/robot/main.py`; no `bosdyn` import
anywhere — every tool *simulates* the named Spot SDK call against CouchDB state, confirmed by grep):

| Category | Tools | Spot SDK call simulated |
|---|---|---|
| Hardware-state wrappers (10) | `get_battery`, `get_pose`, `list_waypoints`, `capture_image`, `power_on`, `power_off`, `stand`, `sit`, `dock`, `undock` | RobotState `battery_states`/`kinematic_state`; GraphNav `download_graph`; ImageClient `get_image_from_sources`; PowerClient; RobotCommandClient `synchro_stand/sit_command`; DockingClient `blocking_dock/undock` |
| Inspection/enterprise, new (6) | `navigate_to`, `safety_gate_check`, `open_panel`, `read_gauge`, `commit_reading`, `check_wo_similarity` | — (no direct Spot SDK analog; enterprise-integration layer) |
| Multimodal perception (6) | `read_thermal_image`, `commit_thermal_decision`, `read_vibration`, `read_acoustic`, `commit_acoustic_decision`, `escalate` | — |
| Fleet/capability (4) | `list_robots`, `select_robot`, `select_capability`, `request_observation` | — (Inspection Capability Framework) |
| Physical-admissibility (3) | `check_admissibility`, `check_cdc`, `arm_move` | MuJoCo digital twin, not Spot SDK |

Note: `gauge_value` is never returned to the agent in any tool response (`main.py:8-10`) — a real,
checkable invariant worth stating in the reproducibility appendix.

**Fix [MAIN]:**
> The tool surface has 67 tools across seven MCP servers: 29 new robot-facing tools, of which 10 are
> 1:1 simulated wrappers of Spot SDK hardware-state calls (RobotState, GraphNav, ImageClient,
> PowerClient, RobotCommandClient, DockingClient) and 19 are new inspection/enterprise/perception/
> admissibility tools with no direct Spot SDK analog, plus 38 tools reused from AssetOpsBench.
> Episodes on the CouchDB path expose a 25-tool subset (`couchdb_executor.TOOLSET`). `agentic_zoom` is
> available in the PMC track only. Appendix `[X]` lists every tool and its Spot SDK correspondence,
> so a re-implementer can rebuild the Spot-facing subset against the real SDK.

**[APPENDIX]:** the full 67-tool catalog table above, plus the mission sequence already documented in
`main.py:46-50` ("power_on → undock → stand → list_waypoints → get_battery → navigate_to → get_pose →
open_panel → read_gauge (×≥3) → check_wo_similarity → commit_reading/raise WO → navigate_to (dock) →
sit → dock → power_off") as the reproducible reference usage recipe.

---

### 12. Execution backends, ROSClaw and the MuJoCo oracle (lines 372-377) — [MAIN] TEXT

**Problem.** "VLA" is used without being defined. "MuJoCo-backed admissibility oracle" (Section 4,
D-physical paragraph) is stated without saying which of the six constraint checks are actually
MuJoCo-grounded versus analytic-envelope checks, and ROSClaw's role is not explained at all.

**The oracle, precisely** (`spot_admissibility_verifier.py:73-203`, checked in this fixed order):
1. **Reach** — `arm_reach_m = 0.985` m, workspace z ∈ [0, 1.8] m (declared envelope,
   `robot_assets_registry.json`).
2. **Joint limits + collision** — **the only genuinely MuJoCo-simulated check**: a real physics
   simulation of the Spot+arm MJCF model (`rosclaw/e-urdf-zoo/spot/scene_arm.xml`) at the derived
   reaching pose.
3. **Clearance** — standoff ≥ declared obstacle clearance (analytic).
4. **Grasp/payload** — latch force ≤ 130 N, payload ≤ 11 kg (analytic).
5. **Stability** — slope ≤ 30°, CoM shift ≤ footprint half-length (analytic).
6. **Energy** — (locomotion + arm power) × duration ≤ battery minus 5% dock reserve (analytic).

Only check 2 is MuJoCo; checks 1 and 3–6 compare against the declared kinematic/energy envelope in
`robot_assets_registry.json`. The reach violation in the paper's own Table 3 example (0.55 m standoff
called "inadmissible") is driven by **panel height**, not standoff distance: the generator raises the
gauge to z=1.7 m for that episode, giving `hypot(0.55, 1.7−0.75) ≈ 1.10 m > 0.985 m` — the standoff
value is identical in the admissible sibling episode. This is worth a one-line clarification since
"0.55 m standoff, inadmissible" reads as a minimum-distance violation but is in fact a maximum-reach
one driven by target height.

**ROSClaw's actual role** (`reports/paper_audit/rosclaw_audit.md`, re-verified): a *supporting*
component only. Its `DigitalTwinFirewall` supplies the MuJoCo model-loading and collision code that
check 2 calls, via `_rosclaw_compat.py`'s graceful-fallback import. **The core CouchDB episode path
has zero ROSClaw dependency** — ROSClaw does not mediate navigation, perception, robot state, or any
scored tool call; it is used only inside the admissibility verifier and the separate VLA firewall
below.

**How D-physical gold is derived.** `dphys_generator.py` calls the same verifier. The set of violated
constraints becomes `gold.constraints`; `limiting_constraint` is the single violated constraint, or
`"MULTIPLE"` when ≥2 are violated simultaneously, or `None` when all six are satisfied. This is why
LCA is scored on `applicable_n=24` (70 − 32 MULTIPLE − 14 None), independently re-derived twice this
session and confirmed exact. The oracle's output is evaluator-side only and is never shown to the
agent.

**Separate VLA firewall** (`digital_twin_firewall.py:75-125`, gates the `arm_move` tool only): joint
limits over 19 DoF, collision, wrist-z workspace bound, and transition energy at a placeholder 0.5
rad/s. Returns ALLOW/BLOCK only (MODIFY not implemented — a TODO in the code). Reported separately
from the core episode path, as the paper already states.

**Fix [MAIN]:**
> A vision-language-action (VLA) track can additionally pass structured actions through a safety
> layer built on ROSClaw (Zhao et al., 2026) and MuJoCo (Todorov et al., 2012). The admissibility
> oracle checks six constraints — reach, joint limits/collision, clearance, grasp/payload, stability,
> energy — against a declared kinematic envelope; only the joint-limit and collision check is a real
> MuJoCo physics simulation, the remaining five are analytic comparisons against the registry's
> declared envelope. ROSClaw supplies the MuJoCo model and collision code for this check only; it does
> not mediate navigation, perception, robot state, or any tool call on the core episode path. The
> oracle's violated-constraint set becomes the D-physical gold label; it is evaluator-side and never
> shown to the agent. A separate VLA-track firewall gates only the `arm_move` tool with its own
> joint/collision/workspace/energy checks and returns ALLOW or BLOCK; neither firewall judges evidence
> sufficiency or operational correctness.

**[APPENDIX]:** the six-check table with thresholds above, and the gold-derivation rule
(`limiting_constraint ∈ single constraint | "MULTIPLE" | None`, and how that maps to the
`applicable_n=24` LCA denominator).

---

### 13. "Supplementary source" pointers (lines 261-262, 370-371, 4.1) — [MAIN] TEXT

Replace each vague "supplementary source" mention with a named artifact:

| Sentence topic | Real artifact |
|---|---|
| Dataset coverage, eligibility, scorer validation | `inspectionbench/README.md` (model-eligible subset, reproducibility, provenance/licensing sections); `src/orchestrator/eval_eligibility.py`; `inspectionbench/scripts/verify_integrity.py` |
| Construction | `inspectionbench/docs/CONSTRUCTION.md` |
| Scorer validation and execution-trace checks | `docs/InspectionBench_MetricSpec.md` §E (denominators), §G (golden trace suite); `inspectionbench/tests/test_canonical_identity.py`, `test_generator_integrity.py`, `test_dphys_scoring.py`, `test_b_acquisition_scoring.py` |
| Counts | `reports/paper/final_benchmark_statistics.md` |
| Tool catalog | new appendix table, Item 11 |
| Authoring/adjudication protocol | **no artifact exists** — remove this claim entirely (see Item 10) |

---

## NEW ITEMS (your A–F)

### A. Benchmark composition table — see Item 6 (merged; `templates_inventory.csv` is the deliverable)

### B. Evaluated subset vs 4,075 — [MAIN] TEXT

**Selection rule, resolved.** `src/orchestrator/eval_eligibility.py:compute_eligibility()`: eligible =
93 (frozen historical pilot pool) + 66 (B-Acquisition) + 70 (D-physical) = **229**. Of these, **148
are inside the 4,075** (12 B-legacy + 66 B-Acquisition + 70 D-physical); **81 are outside it** (the
pilot pool's A48 + C9 + D-enterprise6 + E18, which are earlier, separately-seeded fixtures retained
only for run-to-run comparability, not canonical members of the 4,075). The remaining 3,846 canonical
episodes carry only generation-time admission artifacts and have never been dispatched to a model.

**Figure 5 is produced by `scripts/phase8h2l_main_table_build.py`; every number traced exactly:**
- **A = 43 for DeepSeek/Qwen** because A's primary metric is GSR over the *answered-only* denominator
  (episodes with a non-empty terminal verdict), not a hard zero over all 48
  (`compute_a()`, `phase8h2l_main_table_build.py:95-124`). DeepSeek's 5 empty verdicts are output
  truncation; Qwen's are a 2/3 truncation/infrastructure split.
- **Claude A = N/A** because `compute_a()` hard-excludes Claude below `n_evaluable < 10`
  (2/48 valid verdicts): 44/46 empty-verdict cases are the model re-emitting a tool-call object at the
  mandatory decision turn instead of the required verdict schema — a genuine protocol-compliance
  failure, not truncation or infrastructure error (confirmed: `finish_reason=stop`, zero API errors on
  every affected episode). The independently re-run validation gives TDA=2/48=0.0417, GSR=0/48=0.0.
- **The † on D (DeepSeek/Qwen)** = `VALID_WITH_CAVEAT`: 42/70 and 16/70 responses truncated at the
  fixed output-token cap; the primary value shown is still the frozen full-set result, never silently
  replaced by the smaller answered-subset (n=28/54 respectively).
- **The ‡ on Qwen's B=0.712 has no definition anywhere in the repo.** `markers: []`, `status: VALID`
  in every artifact checked. This needs the paper's authors to either define what the ‡ means or
  remove it — it cannot be reconstructed from the release artifacts.
- **E's n varies (8/14/17/18/17)** because `CC_grounded` is excluded from the mean when a sequence
  slot's required modality doesn't apply, rather than being scored zero — all 5 models run the same
  6 sequences × 3 slots = 18 nominal, but how far each model progresses through each sequence differs.
  Claude's 8/18 shares the same step-2 protocol-noncompliance root cause as its A exclusion.

**Fix [MAIN], one paragraph:**
> Of the 4,075 canonical episodes, 229 are eligible for live-model evaluation under the runners that
> exist today: the 66 B-Acquisition and 70 D-physical episodes, canonical members of the 4,075, plus
> the frozen 93-episode historical pilot pool (A 48, B 12, C 9, D-enterprise 6, E 18), of which only
> the 12 B-legacy episodes are themselves inside the 4,075. The remaining 3,846 canonical episodes
> carry generation-time admission artifacts only and have never been dispatched to a model. Within the
> evaluated set, denominators are per-cell: A reports GSR over episodes with a valid non-empty
> terminal verdict (48/48 for GPT-5.2 and Mistral; 43/48 for DeepSeek and Qwen; Claude Sonnet 4.6 is
> excluded outright at 2/48 valid verdicts, a protocol-compliance failure in which the model re-emits
> tool calls at the mandatory decision turn rather than a verdict); B and D report the frozen full set
> at n=66 and n=70, with truncation-affected cells flagged but never replaced by their answered
> subsets; C is n=9; E's denominator is the number of sequence slots for which grounding is defined
> (8–18 of 18 nominal slots across 6 sequences), with inapplicable slots excluded rather than scored
> zero.

### C. Metric definitions and run settings — [MAIN] TEXT

**All five Figure 5 columns now pinned to code, confirmed exact:**
- **A** = GSR (`CC_grounded`), defined in Item 2 above.
- **B** = `B_AGS`, a branch-conditional composite (`scripts/phase8h2l_main_table_build.py:127-149`):
  STOP → ADA∧TDA_post; ACQUIRE+available → ADA∧ASA∧TDA_post; ACQUIRE+unavailable → ADA∧ASA∧UHA. This
  is **distinct from** ADA/ASA/TDA_post reported individually in the prose (0.349–0.727 for TDA_post
  is separately correct and verified) — the paper must name `B_AGS` explicitly wherever it describes
  Figure 5's B column, since it is not any one of ADA, ASA, or TDA_post alone.
- **C** = `procedural_coverage` (`len(required∩executed)/len(required)`, `ordering.py:47-61`), the
  Phase-11-corrected successor to the older `ordering_satisfied`. Matches Figure 5 to 3 decimals.
- **D** = `CSA` (exact constraint-set match, `dphys_scoring.py:74-117`). Figure 7's separate LCA bar
  is a companion diagnostic, not a re-plot of the same number.
- **E** = `CC_grounded` restricted to thermal (FM-26) episodes, with the correct clean-subset
  denominators — though `paper_results_v1.json`'s top-level records mislabel these rows `n=18` even
  when the value is computed on 8/14/17; a real, reproducible bug in that file (Figure 5 itself is
  unaffected, it already uses the correct n's).

**Table 5's "agreeing" (243.173 vs 247.179) is a generation-time Gaussian draw, not a checked
tolerance:** IoT = physical + 𝒩(0, σ=1%·span), never re-verified after sampling
(`scenario_gen.py:184-187`); `derive_gold()` never reads telemetry at all. A rare large draw would
still be labeled "agree." Worth a one-line caveat if the paper implies a hard tolerance band.

**Table 3's reach puzzle is resolved** (also covered in Item 12): the 0.55 m standoff is identical in
both the admissible and inadmissible sibling episodes; the violation is driven by the generator
raising panel height to 1.7 m for that episode, giving a 3D reach distance of ≈1.10 m against a 0.985
m maximum — not a too-close/minimum-standoff violation.

**Run settings, all five models:** `tokenrouter/{anthropic/claude-sonnet-4.6, openai/gpt-5.2,
deepseek/deepseek-v4-pro, mistralai/mistral-medium-3-5, qwen/qwen3.5-397b-a17b}`, all at
temperature=0.0, max_output_tokens=2048, single run per episode (`paper_figure_spec_v1.md:97`: "no
claim of statistical significance... single run per model throughout"), run on 2026-09-14
(`started_at` field, all 5 raw jsonl files). **`top_p` is not recorded anywhere in the codebase —
report as NOT FOUND, do not guess a value.**

**Bootstrap CIs, world-clustered:** already computed and stored for B's ADA and D's CC/CSA/CS-F1, but
not for A/C/E (small n). World-clustered CIs are computable from the raw jsonl (`world_id` is present
on every row); an illustrative one-off check (GPT-5.2, A, n=48, 16 distinct world_ids) gives a 95% CI
of roughly [0.29, 0.56] around the 0.4375 point estimate. This is not written into any paper file —
flagged as computable if the authors want to add uncertainty bands to Figure 6.

### D. Evaluation process paragraph — see Item 2 (merged; the "two scorers" framing is the deliverable)

### E. Annotation/adjudication contradiction — see Item 10 (merged)

### F. Consistency errors elsewhere in the paper — [MAIN] TEXT

1. **Abstract's "22.9–46.5 percentage points" does not reproduce from anything.** The correct 4-model
   range (matching Section 4's own body text: 22.9/40.0/23.0/16.2) is **16.2–40.0pp**. Using Claude's
   corrected re-validated gap (TDA=0.0417, GSR=0.0 → 4.17pp), the true 5-model range is **4.2–40.0pp**.
   No computation across TDA spread, GSR spread, 4-model, or 5-model ranges reaches exactly 46.5 (the
   closest coincidence is GSR max−min = 45.8, not the same quantity). **Fix the abstract to 16.2–40.0pp
   (if reporting only the 4 models with a valid A score) or 4.2–40.0pp (if including Claude's excluded-
   but-computable result with a footnote explaining the exclusion).**
2. **"Pass rates are nearly identical" has no matching A–E metric anywhere in the code or results
   files** — the only `pass_rate` field found belongs to the separate PMC track
   (`reports/pmc_benchmark_results.json`). Either rescope this sentence to the PMC track explicitly, or
   rewrite it against a real, named A–E metric — do not leave it as an unattributed claim.
3. **"Appendix A.1" is a dangling reference** — the paper's own structure has Appendix A (annotation
   prompt) and Appendix B (compiled examples), no A.1. Fix or remove the citation.
4. **Table 4's tool list has no error** — confirmed directly against `cd_generator.py:118-124`: the
   4-tool ABORT-branch sequence (`list_waypoints → get_battery → sit → dock`, `capture_image`
   explicitly forbidden) is exactly right. No fix needed here; consider adding "(ABORT branch)" to the
   table caption since the same template's COMMIT branch requires a different 6-tool set.
5. **Figure 5's B column plots `B_AGS`, not ADA/ASA/TDA_post** — see Item C above. ADA, ASA, and
   TDA_post are all individually correct as separately stated in the prose (1.000/1.000/1.000/0.985/
   0.939; 0.933/0.933/0.933/0.911/0.844; 0.349–0.727); the paper just needs to name `B_AGS` explicitly
   for the Figure 5 column so readers don't assume it's one of those three.
6. **"Collected at operating data-center facilities" is unsupported** — see Item 4. Recommend
   "industrial facility" pending confirmation.
7. **Reproducibility/license paragraph, drafted for review (not yet inserted anywhere):**
   > Code in this repository is released under Apache 2.0 (`LICENSE`). Benchmark episode content and
   > metadata inherit the same license; individual observation records sourced from third-party
   > datasets retain their own licenses as recorded in each record's `dataset_lineage`/`source_license`
   > field — notably MIMII (Zenodo 3384388, CC-BY-SA-4.0) and the Mendeley induction-motor thermal
   > dataset (`doi.org/10.17632/pymtdhfzbj.1`, CC-BY-4.0, archive hash-pinned in
   > `build_reva_manifest.py`). No anonymized-repository pointer or author/institution anonymization
   > statement currently exists in the codebase. No record of facility-photo collection consent or
   > release was found in this repository; confirm with whoever conducted the on-site collection
   > session before claiming permission in the paper.

---

### 14. Section 3.3 / Figure 4 — the InspectionBench execution platform, block by block — [MAIN] TEXT

**Current text (lines 324-339, Figure 4 caption + surrounding paragraphs):** "Figure 4: The
InspectionBench environment. Agents receive alerts and asset context, invoke tools to acquire
inspection evidence, and use returned observations to make operational decisions." followed by
"Agent-visible episode," "Tool organization," and "Execution backends" (Items 2, 11, 12 above cover
the tool/backend content already; this item organizes that content explicitly against Figure 4's five
labeled blocks and the platform description the intro/Sec 3.3 currently under-specifies).

Figure 4 itself draws five blocks — **(i) Inspection Trigger** (real-world alert/dispatch), **(ii)
AssetOpsBench Data Model** (Asset/System, Component, FMEA Failure, Work Order, Maintenance History),
**(iii) Agent & Tools** (the LLM-based agent issuing `exe_cmd(tool_name, params)` calls, e.g.
`navigate_to`, `read_gauge`, `analyze_image`, `plan_inspection`, against example tool groups TSFM/
FMSR/Spot), **(iv) Physical Execution** (Server/Environment State: robot state, gauge reading,
industrial environment, MuJoCo safety firewall, API requests), **(v) Observations & Decision**
(visual/thermal/telemetry observations, updated asset state, operational decision) — plus a bottom
feedback loop labeled "Parameterization (mutation/crossover) | Manipulation checks | Feasibility
validation → New World/Template." Each block now has an audited, citable real-system correspondence
except the feedback loop, which does not.

**(i) Inspection Trigger.** No dedicated "alert generator" module was located in this audit pass; the
trigger is implicit in each episode's task prompt (rendered by `scenario_gen.render_question`), which
already encodes the alert condition (out-of-band reading, contradiction, work order, etc.) as text.
State this plainly rather than implying a separate alerting subsystem exists.

**(ii) AssetOpsBench Data Model.** This is the CouchDB substrate `couchdb_executor.py` reads from:
`profile:{asset_id}` documents (asset/gauge state), `robot_state:{robot_id}` documents, and the
`workorder` database (`check_wo_similarity`, `main.py:1004`). The FMEA/failure-mode layer is exposed
through the `fmsr` MCP server (3 tools, Item 11). This block is accurately depicted.

**(iii) Agent & Tools.** This is the tool surface audited in Item 11: **67 tools across seven MCP
servers** (robot, wo, vibration, tsfm, iot, fmsr, utilities), of which **29 are new** for
InspectionBench (10 Spot-SDK hardware-state wrappers, 6 new inspection/enterprise tools, 6 multimodal
perception tools, 4 fleet/capability tools, 3 physical-admissibility tools) and **38 are reused**
from AssetOpsBench. Episodes on the scored CouchDB path expose a 25-tool subset
(`couchdb_executor.TOOLSET`, masked per evidence regime by `mask_tools(TOOLSET, withheld)` —
`couchdb_executor.py:439-440`). The example tool names in Figure 4's block (iii) — `navigate_to`,
`read_gauge`, TSFM `get_forecasting`/`get_model_catalog`, FMSR `get_failure_mode`/`get_mapping` — are
all real, confirmed tool names; add a citation to Figure 4 here and state the 67/29/38/25 counts
explicitly, since the current text never gives readers a total tool-surface size.

**(iv) Physical Execution.** Two genuinely distinct backends exist, and the paper's "Execution
backends" paragraph (lines 372-377) already gestures at this but should name both explicitly, citing
Figure 4's "Server (Environment State)" and "MuJoCo Safety Firewall" boxes directly:
- **The scored path**: `couchdb_executor.py`'s CouchDB-document simulator. This is what produced every
  number in Figure 5/6/7 — confirmed zero ROSClaw/MuJoCo dependency on this path
  (`reports/paper_audit/rosclaw_audit.md`).
- **The separate VLA track**: `arm_move`/`check_admissibility`, gated by the MuJoCo-grounded
  admissibility oracle and the digital-twin firewall (Item 12's six-check table). This is what Figure
  4's "MuJoCo Safety Firewall" box depicts, and it is reported separately from the core episode path,
  as the paper already states — but the current text never clarifies that Figure 4 is showing *both*
  paths in one diagram. Add a sentence distinguishing them under the figure.

**(v) Observations & Decision.** This is the `observation_delivery` boundary the paper already
describes (line 396: "Failed calls and withheld observations remain part of the execution record but
cannot establish that the corresponding evidence was received") — the same mechanism GSR's
`observation_delivered` conjunct checks (Item 2). Cite Figure 4's block (v) at that sentence, since it
is the visual depiction of exactly this delivery/decision boundary.

**The bottom feedback loop ("mutation/crossover... feasibility validation → New World/Template") does
NOT correspond to any implemented mechanism.** Repo-wide search: "crossover" appears only as test
jargon for identity-protection tests (`test_canonical_identity.py:118`, checking that a mutation
"crossover" cannot silently corrupt `world_id` — not a generative operator), and no
`feasibility_valid`/`feasibility_check` function exists anywhere. The real generative mechanism is the
deterministic world-to-scenario compiler already described in 3.2 (`scenario_gen.sample_world` +
fixed cell/asset/seed parameterization, with two invariant checks: asset identity/condition/access
match, and gold recomputation match) — not an evolutionary mutation/crossover search loop. **Fix: either
remove this loop from Figure 4, or relabel it to describe the real compiler mechanism (deterministic
parameterization + the two invariant checks already in 3.2's "Scenario compiler" paragraph), and do
not imply a genetic-algorithm-style generation process that isn't in the codebase.**

**Fix [MAIN], replacing/extending lines 324-339 and the "Tool organization"/"Execution backends"
paragraphs:**
> Figure 4 depicts the full execution platform an episode runs on. An inspection trigger (i) is
> encoded directly in the episode's task prompt; the agent reasons over the AssetOpsBench data model
> (ii) — asset/component state, FMEA failure modes, and work-order/maintenance history stored in
> CouchDB — and acts through a 67-tool surface across seven MCP servers (iii): 29 tools built for
> InspectionBench (10 of which are 1:1 simulated wrappers of Spot SDK hardware-state calls; see
> Appendix `[X]`) and 38 reused from AssetOpsBench. Scored episodes execute against a CouchDB
> document-state simulator with no robot-middleware dependency (iv); a separate vision-language-action
> track additionally routes structured actions through a MuJoCo-grounded physical-admissibility
> firewall (Appendix `[X]`, Item 12), reported apart from the core episode path. Tool results and
> evidence-bearing observations (v) are delivered to the agent only when the underlying call actually
> succeeds and returns evidence; this delivery boundary is what the grounded-success metric (GSR)
> checks beyond bare terminal-decision correctness (Section 4.1).

**[APPENDIX]:** the full 67-tool catalog (Item 11), the six-constraint oracle table (Item 12), and an
explicit note that the world-generation feedback loop in Figure 4 should be read as the deterministic
compiler of Section 3.2, not an evolutionary search process.

---

## Small release-hygiene items spotted (not paper-text fixes, but worth doing before release)

- `GEN-BACQ2-k1` / `GEN-BACQ2-k3` (4 episodes) use a naming convention inconsistent with the other 16
  B-ACQ-2 patterns — likely prototype-era leftovers worth cleaning up.
- `reports/benchmark/final_benchmark_manifest.json` still ships all 3,840 pre-dedup A/E rows while
  `final_benchmark_manifest_v3.json` asserts 3,795 canonical — the dedup is reproducible but not
  materialised; a naive consumer of the release will count 4,120, not 4,075.
- `anthropic_Key.txt` and `gemini_api_key.txt` at the repo root were checked this session: **both are
  properly gitignored, never tracked, never appear in git history.** No action needed — confirmed safe.
- `paper_results_v1.json`'s on-disk LCA values are still the OLD n=70 figures, not the Phase-11-
  corrected n=24 ones (explicitly noted as "not yet applied" in `d_physical_accounting.json`). If the
  paper cites LCA from that file directly, it will get the stale numbers.

---

## Pre-submission checklist

- [ ] R1–R5 done; every `[FILL AFTER FREEZE]` replaced; search the PDF for "FILL", "??", "[are/are
      not]", and the literal string "[14]".
- [ ] Every number in Section 3 traces to a file path re-opened this pass (all citations above were).
- [ ] Search "procedurally rendered", "supplementary source", "data-center", and "46.5" across the
      *entire* paper, not only Section 3/abstract.
- [ ] Figure 3 traces regenerated from one config; caption matches the JSON.
- [ ] Lines 202–204 modality counts and Item 7's class counts reconciled in one crosstab before both
      sentences ship.
- [ ] `templates_inventory.csv` shipped as a release artifact; 14/18/52/61/600/1,280/1,537 all resolve
      to a number that traces to it.
- [ ] `git status` in the repo shows only the intended additions (`src/perception/`, frozen manifests,
      `templates_inventory.csv`, `facility_catalog.csv`) — nothing else changed.
