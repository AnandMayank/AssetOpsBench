# InspectionBench V3 — Abstract-Ready Fact Sheet (VERIFIED FACTS ONLY)

Generated 2026-09-18 from direct inspection of canonical manifests, scoring code, and
evaluation artifacts on branch `inspectionbench-v3-release` (working tree). Every
number below is traceable to a source file listed alongside it. Where the audit could
not verify a number, it is marked NOT FOUND rather than estimated. This file does not
edit or replace the paper; it is the input a future rewrite should draw from.

## BENCHMARK SCALE

- **4,075 canonical episodes** (`final_canonical_total`, `reports/benchmark/final_benchmark_manifest_v3.json`), computed as 4,009 (pre-existing) + 66 (newly merged B-Acquisition). The naive sum of the displayed `family_counts` dict is 4,120 — do not present that sum as the total; the 45-episode gap is a documented A/E cross-family duplicate-fingerprint dedup (42 duplicate groups covering 87 episodes, collapsed to 42 representatives, net -45).
- **4 canonical grounded industrial assets** (NOT 6): `chiller_6`, `hydraulic_pump_1`, `metro_pump_1`, `motor_01` — the exact, complete set defined in `src/orchestrator/scenario_gen.py`'s `ASSETS` registry, and confirmed as the only assets appearing in every family manifest checked (A/E, C/D-enterprise, D-physical, B-Acquisition).
- **Grounded worlds**: no current artifact reproduces "19." The A/E pool has 1,280 unique `matched_group_id`/`world_id` values (800 A + 480 E); the frozen-93 historical pilot pool has 41 unique `world_id` / 26 unique `normalized_world_hash`. "19" appears only in a pre-V3 audit document describing a different, earlier benchmark state.
- **Scenario templates**: no discrete 93-template registry was found. Only 14 discretely-named templates exist in the artifacts checked (4 B-ACQ-* + 7 T-C-* + 3 T-D-*), covering 198 of 4,075 episodes; A/E/B-legacy (3,877 episodes) are cell-parameterized, not built from a named template registry. "93" in this repo's authoritative artifacts denotes the frozen historical pilot EPISODE pool size, not a template count.
- **Rendered scenes**: "1,296" is REAL and structurally confirmed (1,296 unique scenario IDs in `reports/v7/label_status.csv`) but describes **real field photographs**, not procedurally rendered scenes (`docs/BenchmarkSpec_Observability.md`: "Zero procedural-render bias"). Calling it "1,296 rendered scenes" is a category error. Only a 20-row dev subset of this catalog is present in-repo; the full 1,296-row table is an external release artifact.

## GENERATION

- World-to-scenario compiler exists and is quantified: A/E pool `episode_count_generated=3840`, `admitted=3840`, `rejected=0`, `duplicate_fingerprints=42` (verified). C/D pool: `generated=132`, `admitted=132`, `rejected=0`, `duplicate_fingerprints=0`. D-physical: 88 pre-dedup candidates → 70 canonical (18 removed).
- Matched-generation: A has 800 matched groups × 3 evidence regimes (FULL/PHYSICAL_ONLY/DIGITAL_ONLY) = 2,400 episodes; E has 480 matched sequences × 3 temporal episodes = 1,440; B-Acquisition has 15 matched groups across 66 episodes. Gold-invariance was independently spot-checked on one worked A example (`matched_world_validation.json`) and holds (`COMMIT` invariant across all 3 regime members); NOT independently re-verified for all 1,280 groups given time constraints. C/D-enterprise/D-physical do NOT use matched-group construction at all.

## EVIDENCE

- Modalities found with real substrate: visual/gauge (`capture_image`,`read_gauge`), IoT/telemetry (`read_iot`, `iot_timeseries`), acoustic, thermal, work-order/maintenance history. "Pressure" is not a separately tagged modality (folded into gauge readings on `bar`-unit assets).
- Provenance: TWO distinct, non-unified mechanisms exist — (1) an `L1/L2/L3/NOT_APPLICABLE_DIGITAL_ENTERPRISE/SIMULATED_UNCLASSIFIED` evidence-source tier defined in code (`src/orchestrator/world.py`) but NOT found persisted in any current V3 canonical manifest (0 matches across 4 manifests grepped); (2) generation-time lineage fields (`world_id`, `matched_group_id`, `generator_version`, `fingerprint`) present on effectively all 4,075 episodes, plus an explicit `source_provenance` block on 50/66 B-Acquisition episodes only.
- Agent-acquirable evidence: `decisive_delivered` tool lists confirm evidence access varies by regime (e.g. DIGITAL_ONLY drops `capture_image`/`open_panel`/`read_gauge`).

## EVALUATION

- Capabilities A (Evidence Grounding), B (Evidence Acquisition), C (Procedural Grounding), D (Relational Physical Grounding, split D-enterprise/D-physical), E (Temporal Grounding).
- Canonical N: A=2,400 (nominal)/3,795(A+E deduped), E=1,440, B-legacy=12, B-Acquisition=66, C=112, D-enterprise=20, D-physical=70.
- **Model-evaluated N = 229 total** (93 frozen historical + 66 B-Acquisition + 70 D-physical), independently reproduced via `src/orchestrator/eval_eligibility.py`. By dimension: A=48, B=78, C=9, D=76, E=18. Only B-Acquisition (66/66) and D-physical (70/70) reach 100% canonical coverage; A reaches 2.0%, C reaches 8.0%, E reaches 1.25% of their nominal pools.
- 5 models: Claude Sonnet 4.6, GPT-5.2, DeepSeek V4 Pro, Mistral Medium 3.5, Qwen3.5-397B-A17B.

## VALIDATION

- No SME/human-reviewer process for the 4,075-episode benchmark's own construction (worlds/scenarios/gold) was found anywhere in the repository.
- The only human/dual-labeling reliability data found concerns the separate 1,296-photograph perception catalog: 20 scenarios dual-labeled (kappa: gauge_readable=0.4681, recommended_action=0.7368, category=0.7561, image_role=0.7647, gauge_value=0.177, asset=0.1404) — explicitly flagged in the source script's own docstring as a biased, non-corpus-representative sample. 0 of 396 sampled adjudication-worksheet rows have a filled `adjudicated_value`. The corpus-level (random-stratum, n=40) reliability statistic the process intends to compute was NOT FOUND on disk.

## RESULTS

- Claude Sonnet 4.6's A-family result: TDA=0.0208 (1/48), GSR=0.0, gap=2.1pp — driven by 46/48 (95.8%) STEP-2 verdict-schema noncompliance (a protocol-compliance finding, distinct from grounding capability; GPT-5.2 is 48/48 compliant on the identical protocol).
- **CRITICAL PROCESS FINDING**: a documented Phase-11 remediation pass (`reports/benchmark/phase11_offline_remediation_report.md`) computed corrected values for C's primary metric (procedural_coverage, which INVERTS the C ranking versus the old `ordering_satisfied` metric — DeepSeek goes from rank 1 (0.889) to rank 5/worst (0.176)), D-physical's LCA (denominator corrected from n=70 to applicable_n=24, changing every model's value, independently re-derived and confirmed by this audit: 70−32(MULTIPLE)−14(None)=24), and a Mistral B-Acquisition MAR/UAR extraction bugfix — but these corrections were **verified only in scratch and never written** to the on-disk `reports/benchmark/paper_results_v1.json` / `capability_profile_v1.json` (confirmed this pass: 0 occurrences of `n_excluded` in the current on-disk file). Any number pulled from these files today is the STALE, pre-correction value. An abstract/paper rewrite must use the Phase-11-corrected values (recomputed in this audit's `result_number_crosscheck.csv`), not the on-disk file, for C and D-physical LCA claims.
- D-physical always-ESCALATE baseline = 0.8 (56/70 infeasible episodes, independently re-verified). GPT-5.2 leads most D-physical metrics; DeepSeek (42/70 rows) and Qwen (16/70 rows) show substantial output-token-cap truncation (fixed `max_tokens=2048` cap in `src/llm/openai_compat.py`/`litellm.py`) that should be disclosed alongside their D-physical numbers, though the frozen full-set result remains primary (not excluded).

## LIMITATIONS

- Effective grounded-world count for A/E is 1,280 (matched-group instances), not a small number like 19 — if the abstract wants a small "environment" count, it must define and cite a different unit than what A/E's own `world_id` field provides.
- Small-N capabilities: C (n=9 evaluated of 112 canonical, 8.0%), E (n=18 of 1,440, 1.25%, and model-dependent valid-n as low as 8/18 for Claude), D-enterprise (n=6 of 20, 30%), A (n=48 of 2,400 nominal / 3,795 deduped, ~2.0/1.3%).
- Truncation: DeepSeek and Qwen show material D-physical output-truncation (60% and 22.9% of rows respectively) under a fixed 2,048-token cap; this is a shared infrastructure constraint, not model-specific except in how often each model hits it.
- Figure 5 (enterprise-gate agent-vs-environment trace): NOT FOUND as a named artifact in this repository; cannot be described or cited without locating the underlying trace data first.
- Provenance-tier (L1/L2/L3) coverage across the full 4,075-episode benchmark: NOT FOUND as a persisted, countable field; only conceptually defined in code.
