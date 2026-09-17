# InspectionBench V3 — Abstract Fact-Audit: Narrative Summary

Scope: read-only fact audit of construction and evaluation artifacts under
`/home/adityapachauri/AssetOpsBench` (branch `inspectionbench-v3-release`, working tree),
performed 2026-09-18. No paper, docs, manifest, gold, prompt, or scoring-code file was
modified. All outputs live under `reports/benchmark/abstract_crosscheck/`.

## 1. No single current abstract exists

Exhaustive search (see `current_abstract.txt`) found no file containing a verbatim
"Abstract" co-located with the headline numbers 4,075/19/93/6. The nearest material is a
PRE-V3 audit (`reports/paper_audit/`, dated 2026-09-15, before the V3/4,075-episode
merge existed) describing a different, earlier state of the benchmark ("93 registered
scenario templates, of which 19 have a populated WORLD"). This audit therefore treated
the task's four headline numbers as the claim set under test.

## 2. Headline numbers: what verified, what didn't

- **4,075 canonical episodes**: VERIFIED, with an important caveat. The manifest's own
  displayed `family_counts` sum to 4,120, not 4,075 — a 45-episode gap explained,
  in-manifest, by an A/E cross-family duplicate-fingerprint dedup (42 duplicate groups
  covering 87 episodes, collapsed to 42, net -45). Both figures are documented and
  internally consistent once the dedup is understood; an abstract must never present
  `family_counts` as directly summable.
- **19 grounded worlds**: CONTRADICTED / UNSUPPORTED as a current V3 fact. No V3
  artifact reproduces 19. The A/E pool has 1,280 unique world/matched-group instances;
  the frozen-93 historical pool has 41 unique `world_id` / 26 unique
  `normalized_world_hash`. "19" only appears in the stale pre-V3 audit, describing
  POPULATED TEMPLATES, a different unit, in a benchmark state that predates the current
  4,075-episode total.
- **93 scenario templates**: CONTRADICTED. Only 14 discretely-named templates were
  found in the repository (4 `B-ACQ-*` + 7 `T-C-*` + 3 `T-D-*`), covering 198 of 4,075
  episodes. The number 93 in current authoritative artifacts (`reports/ec/
  phase8h1_pilot_manifest.json`) denotes the size of the frozen historical 5-model
  **evaluated episode pool**, not a scenario-template registry — a likely source of
  conflation in any draft that reused this number as a template count.
- **six industrial asset types**: CONTRADICTED. The canonical asset registry
  (`src/orchestrator/scenario_gen.py`'s `ASSETS` dict) has exactly 4 entries
  (`chiller_6`, `hydraulic_pump_1`, `metro_pump_1`, `motor_01`), and this exact 4-asset
  set is the only one found in every family manifest checked (A/E, C/D-enterprise,
  D-physical, B-Acquisition). At most 3 informal classes (chiller/pump/motor) if the two
  pumps are treated as one class.

**Bottom line: of the four headline numbers, only the canonical episode count (4,075)
survives verification as-is (with the dedup caveat above); the other three (19, 93, 6)
are each contradicted by direct inspection of the ground-truth registries/manifests and
need correction in any rewritten abstract.**

## 3. Single most important discrepancy found

Beyond the headline-number corrections, the most consequential finding is a **process
discrepancy, not a construction discrepancy**: `reports/benchmark/
phase11_offline_remediation_report.md` documents a completed, scratch-verified
remediation pass that (a) corrects C's primary metric from `ordering_satisfied` to
`procedural_coverage` — which **inverts the model ranking** (DeepSeek goes from
panel-leading, 0.889, rank 1, to worst, 0.176, rank 5); (b) corrects D-physical's `LCA`
denominator from n=70 to the true `applicable_n=24` (independently re-derived by this
audit from `d_physical_accounting.json`: 70 − 32 `MULTIPLE` − 14 `None` = 24 exactly),
changing every model's LCA value, not just its label; and (c) fixes an isolated
extraction bug in Mistral's B-Acquisition MAR/UAR. **These corrections were verified
only in scratch and were never written to the on-disk result artifacts** —
`reports/benchmark/paper_results_v1.json` still contains the pre-correction values as of
this audit (confirmed: 0 occurrences of the `n_excluded` field the corrected version
should carry). Any number pulled from that file today, for C or D-physical LCA
specifically, is stale. This must be resolved (by committing the Phase-11 corrections,
or by re-deriving them fresh) before any result number from those two families is cited
in a rewritten abstract or Section 4.

## 4. Other notable findings

- Claude Sonnet 4.6's poor A-family score (TDA=0.0208, GSR=0.0) is substantially a
  **protocol-compliance** finding (46/48 = 95.8% of episodes get a tool-call-shaped
  response instead of the required terminal verdict schema; GPT-5.2 is 48/48 compliant
  on the identical protocol) — a live diagnostic re-run ruled out truncation and
  infrastructure error as causes. This should be reported as a distinct disclosed
  finding, not folded silently into a single "grounding" number.
- DeepSeek (42/70, 60%) and Qwen (16/70, 22.9%) show substantial D-physical output
  truncation under a fixed, shared `max_tokens=2048` cap (`src/llm/openai_compat.py`,
  `src/llm/litellm.py`) — an infrastructure constraint, disclosed as a diagnostic, not
  used to exclude the frozen primary result.
- "1,296 rendered scenes" is a category error: 1,296 is a real, structurally confirmed
  count (`reports/v7/label_status.csv`, 1,296 unique scenario IDs) but of **real field
  photographs**, explicitly documented as having "zero procedural-render bias" — not
  procedurally rendered scenes. The genuinely rendered artifact in this repo
  (PerceptionGauge-49) is a much smaller, differently-shaped 49-row seed matrix.
- The L1/L2/L3 evidence-provenance tier system is real in code (`src/orchestrator/
  world.py`) but was not found persisted in any of the 4 current V3 canonical
  manifests grepped (0 matches for `evidence_provenance` in all 4) — an abstract
  claiming benchmark-wide "explicit provenance" should scope that claim to the
  generation-time lineage mechanism (which is genuinely near-universal) rather than to
  this tier system (whose current V3 coverage is NOT FOUND).
- No SME/human-reviewer process was found for the 4,075-episode benchmark's own
  construction. The only human/dual-labeling reliability data found (kappa values in
  `reports/v7/label_reliability.csv`) concerns the separate, unrelated 1,296-photograph
  perception catalog, is based on an explicitly-disclaimed biased n=20 sample (not the
  intended corpus-representative n=40 random-stratum sample, whose reliability
  statistic was never computed), and has zero completed human adjudications on disk.

## 5. Full list of NOT FOUND items (genuine search performed, not located)

- A canonical current abstract text file.
- A literal "19 grounded worlds" count reproducible against current V3 artifacts.
- A 93-item scenario-template registry (only 14 discretely-named templates found).
- Evidence for "six" industrial asset types (found: exactly 4).
- Exact model inference date, top_p, context limits, tool-call turn-budget constant,
  retry count, system/user/tool-schema version strings (`model_protocol_audit.csv`).
- Truncation classification (vs. other invalid-answer causes) for A (DeepSeek/Qwen
  denominator shortfalls), B-Acquisition, C, D-enterprise, and E (`truncation_audit.csv`).
- Per-class precision/recall, balanced accuracy, macro-F1, and confusion matrix for
  D-physical; MuJoCo oracle state variables, thresholds/tolerances, and an enumerated
  oracle-checks list (`d_physical_accounting.json`).
- A Figure 5 enterprise-gate trace artifact (agent-attempted vs. environment-blocked
  commitment) — nothing in the repository is labeled as InspectionBench's own Figure 5
  for this purpose (`figure5_trace_audit.json`).
- A corpus-level (random-stratum, n=40) inter-rater reliability statistic for the
  perception photograph catalog — computed sample exists but its kappa was never
  written to disk (`annotation_validation_audit.json`).
- Per-episode world enumeration beyond aggregate unique-ID counts, and full per-world
  capability-family/asset breakdowns for C, D-enterprise, and D-physical
  (`world_inventory.csv`, `canonical_count_reconstruction.json`).
- A `robot_assets_registry.json` asset-type registry file (referenced by code comments
  in `world.py` but not present in this repository — likely lives in a sibling working
  directory not audited this pass).

## 6. Files created (26 total, all under `reports/benchmark/abstract_crosscheck/`)

current_abstract.txt, abstract_claim_inventory.csv, canonical_count_reconstruction.json,
canonical_count_reconstruction.csv, world_inventory.csv (+ world_inventory_summary.json),
scenario_inventory.csv, asset_inventory.csv, evidence_modality_inventory.csv,
provenance_audit.json, provenance_tier_counts.csv, compiler_audit.json,
matched_world_validation.json, matched_world_validation.csv,
annotation_validation_audit.json, annotation_validation_counts.csv,
rendered_scene_audit.json, capability_coverage.csv, evaluated_trajectory_accounting.csv,
model_protocol_audit.csv, truncation_audit.csv, d_physical_accounting.json,
figure5_trace_audit.json, result_number_crosscheck.csv, ABSTRACT_FACT_SHEET_FINAL.md,
abstract_sentence_recommendations.md, abstract_crosscheck_report.md (this file).

No frozen benchmark, manifest, scoring/runner code, or paper/docs file was modified.
