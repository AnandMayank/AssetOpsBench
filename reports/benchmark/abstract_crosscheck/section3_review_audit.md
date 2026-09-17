# Section 3 Review Audit — Grounded Answers to Reviewer Comments

Generated 2026-09-18 on branch `inspectionbench-v3-release`, extending the prior
abstract fact-check (`reports/benchmark/abstract_crosscheck/ABSTRACT_FACT_SHEET_FINAL.md`,
`provenance_audit.json`, `compiler_audit.json`, `scenario_inventory.csv`). This document
does not edit any paper file — no canonical `.tex`/`.md` paper source exists in this
repository (the paper exists only as an external PDF). It is a grounded answer key for a
human editor to use when rewriting Section 3 elsewhere. Full item-by-item detail is in
the companion `section3_review_checklist.csv` (item_id column matches references below).

## A. Executive summary

Section 3, as marked up, has one structural problem and one credibility problem. The
structural problem: the paper explains formal machinery (world_id hashing, typed-template
instantiation, evidence-regime realizations) before ever showing a reader what a "world"
or a "scenario" concretely looks like, so Figure 5's Trace A/B comparison lands on an
unprepared reader — this is the reviewer's most repeated complaint, in four independent
places (R19/R25/R27/R31, "Qwen" page, typed instruction #9, #14). The credibility problem
is narrower but sharper: several of the paper's headline structural numbers ("19 grounded
worlds," "93 scenario templates," "six industrial asset types") do not correspond to any
artifact in the current 4,075-episode V3 benchmark (restated from the prior audit in
Section D below), and one factual claim the reviewer suspected was wrong ("6 RGB
visual-defect records") is actually verified correct on direct inspection — the review
markup contains both real errors and at least one false alarm, and both need to be
resolved with evidence, not assumption. This pass closes two items the prior audit left
open: the HPML repository (github.com/amaan784/hpml-final-project) is directly and
verifiably the source of some of the paper's replayed thermal/RGB evidence (Section B),
and the L2-asset-class-replay-vs-actual-asset question now has a concrete code answer via
MIMII's `proxy_for_asset` field (Section D). The single highest-value deliverable is the
full worked Sec 3.2 example in Section C, built from one real, currently-in-benchmark
episode.

## B. Missing citations & sourcing

**PerceptionGauge (15-class, 49-row seed matrix).** Source imagery is drawn from three
named pools in `src/perception/generate_perception_gauge_49.py` (lines 35-38): SyncG,
DialBench, and a `real_world_samples` directory. SyncG has a real, already-used citation
elsewhere in this repository's own docs: `docs/Loop2_FinalPlan.md` line 28 cites it as
**"SyncG (Scientific Data 2026)"** — a synthetic gauge-image dataset paper — and this is
the citation that belongs at the PerceptionGauge sentence. DialBench and
`real_world_samples` carry **no citation anywhere in this repository** (grepped `docs/`
and `src/`); they should either be traced to a real citable source or explicitly described
as internally curated, uncited imagery — do not silently present them as if citable.

The 15 asset classes (verbatim, `ASSET_GAUGE_TYPE` dict, same file, lines 48-64): Rolling-
element bearings; Gears/gearboxes; Induction motors; PMSM/synchronous motors; Centrifugal
pumps; Reciprocating compressors; Hydraulic systems; Lithium-ion batteries; Wind-turbine
drivetrains; Gas turbines; Power transformers; HVAC chillers; HVAC AHUs; CNC machine
tools; Robotic manipulators. This directly answers the reviewer's Figure-2 complaint that
data-center assets (AHU, chiller) are never named — they are two of these 15 classes and
should be named in the caption or lead sentence.

**Other external datasets confirmed actually used in code**, each with a real,
citable source: MIMII (`src/orchestrator/data/build_mimii_manifest.py`; Zenodo 3384388,
CC-BY-SA-4.0; pump+fan machine types only, mapped to `hydraulic_pump_1`/`metro_pump_1`
and `chiller_6` via an explicit non-identity `PROXY_ASSET` field); REVA induction-motor
thermal imagery (`build_reva_manifest.py`; Mendeley `pymtdhfzbj` v2, CC-BY-4.0, archive
SHA256 pinned in the script); a rotating-electromechanical-system dataset
(`build_rotating_manifest.py`; Dataverse DATA2500 / *Scientific Data* 2026, condition
codes taken verbatim from the dataset's own Readme). The task prompt's named sources
`thermal_hv_equipment`, `pmc_gauge_dataset`, and `inspecsafe_v1` were **not found** under
those exact filenames in `src/orchestrator/data/` this pass — only
`src/orchestrator/pmc_dataset.py` (likely the PMC/1,296-photo real-image track) and
`reports/ec/phase8h1_inspecsafe_multimodal_audit.{md,csv}` / `scripts/phase8h1_inspecsafe_audit.py`
exist under similar names; a future pass should open these before citing them by name.

**hpml-final-project (github.com/amaan784/hpml-final-project) — directly relevant,
not a false lead.** `src/orchestrator/data/download_industrial_images.py` explicitly
sources labelled pump/motor images from "HPML repo amaan784/hpml-final-project at
data/pumps/" (line 6), with a documented clone instruction and a `--hpml-dir` CLI flag.
`README.md` line 262 already credits this repository as a community AssetOpsBench
contribution ("Visual Inspection Agent for AssetOpsBench... 22 hand-authored visual
inspection scenarios... Amaan Sheikh et al., Columbia University"). `hpml_scenario_id`
fields appear directly in `build_observation_records.py`, in `observation_records.json`'s
records, in `perception_industrial.csv`'s header, and in a real B-Acquisition model
trajectory (`reports/benchmark/b_acquisition_pilot/b_acq_pilot_raw_tokenrouter_qwen3.5-omni-plusfinal_gate.jsonl`,
`"hpml_scenario_id": 503`). This repository is the direct source of some of the paper's
515 thermal + 6 RGB replayed records and should be cited exactly where the reviewer asked
— bullet 3 ("Replayed operational records").

**The "6 RGB visual-defect records" factual-error check — resolved, no error found.**
Direct enumeration of `src/orchestrator/inspection_capability/data/observation_records.json`'s
24,601 `observations` by `modality` gives: `iot_timeseries`=14,313, `acoustic`=9,757,
`thermal`=515, `physical_access_geometry`=7, `rgb_visual_defect`=6,
`workorder_history`=3 — matching the paper's breakdown and its 24,601 total **exactly**.
The 6 RGB records are labelled external-dataset images (from
`src/orchestrator/data/perception_industrial.csv`'s `thermal_fault` rows, re-tagged
`rgb_visual_defect`), per `build_observation_records.py`'s own docstring. The reviewer's
suspicion ("originally we do not have RGB") is unfounded; recommend adding one clause
noting these 6 come from labelled dataset images rather than live robot capture, to
preempt this exact doubt in future review.

## C. The concrete worked example required for Sec 3.2

This is constructed end-to-end from one real episode currently in the canonical V3
benchmark: **`A::GENA-phys_in__iot_agree-chiller_6-050000::FULL`**, plus its two
matched-regime siblings, all sharing `matched_group_id=MG-A-GENA-phys_in__iot_agree-chiller_6-050000`
(source: `inspectionbench/manifests/a_e_pool_manifest.json`).

**1. Canonical asset state (the grounded world W).** The asset is `chiller_6`, a fixed
entry in the generator's asset registry (`src/orchestrator/scenario_gen.py`,
`ASSETS["chiller_6"] = AssetSpec("chiller_6", unit="bar", gauge_range=(0.0, 400.0),
operating_band=(220.0, 270.0))`). `sample_world(seed=50000, cell="phys_in__iot_agree")`
draws two real numeric readings for this world, never seeing a label:
`physical_value = 243.173` bar (inside the operating band) and `iot_value = 247.179` bar
(agreeing with the physical reading, per the `phys_in__iot_agree` cell). No technician is
present and no work order is active (`technician_present=False`,
`active_work_order=False`). This is the world; `world_id = "GENA-phys_in__iot_agree-chiller_6-050000"`,
`world_hash = "d310e93d13ca94b0"` (the SHA-style hash of the normalized world_id string —
this is the concrete instance the abstract `world_id = H(normalize(W_id))` formula in
Sec 3.2 currently gives no example of).

**2. Gold label (derived, never authored backward from a target).**
`derive_gold(world)` (same file) applies a fixed 3-rule priority order that reads only
from the world, never from any desired answer: (1) technician present or active work
order → ESCALATE; (2) physical reading outside the operating band → ESCALATE;
(3) otherwise → COMMIT. Here, no technician/work order, and `243.173` is inside
`[220, 270]`, so `gold_terminal_action = "COMMIT"`, `rule_applied = "physical reading
inside operating band"`. **Telemetry never enters the rule** — the IoT value's role is
only to change how hard it is to reach the right conclusion, not what the right
conclusion is. This directly grounds the paper's earlier evidence-organization
discussion of L1 (same-asset physical evidence) vs digital telemetry: the digital reading
here is genuinely present and agrees with the physical one, but is deliberately excluded
from the ground-truth rule.

**3. Evidence-access condition (the shared scenario's three regime instantiations).**
The same world_id is compiled into three sibling episodes differing only in which tools
are withheld (`src/orchestrator/tool_executor.py`'s `TOOL_MODALITY` dict classifies every
tool as `physical`, `digital`, `enterprise`, `robot`, `thermal`, or `acoustic`;
`mask_tools()` removes every tool whose modality is withheld):

| condition_id | `decisive_delivered` tools (from the real manifest) | withheld |
|---|---|---|
| FULL | `capture_image, get_battery, get_pose, list_waypoints, navigate_to, open_panel, read_gauge, read_iot` (8) | none |
| PHYSICAL_ONLY | `capture_image, get_battery, get_pose, list_waypoints, navigate_to, open_panel, read_gauge` (7) | `read_iot` (digital) |
| DIGITAL_ONLY | `get_battery, get_pose, list_waypoints, navigate_to, read_iot` (5) | `capture_image, open_panel, read_gauge` (physical) |

All three carry `gold_terminal_action = "COMMIT"` — this is the concrete instance of
"the same terminal action can result from different inspection trajectories" that Figure
5's caption asserts abstractly and that the reviewer wants explained before Trace A/B are
named. This IS the answer to the reviewer's own suggested figure ("left side: a world W
with one gauge and a surface; right side: a scenario"): the world is item 1 above; the
scenario is any one row of this table.

**4. Required observation and acceptable action.** Under FULL or PHYSICAL_ONLY, the agent
can call `read_gauge`/`capture_image` and observe `physical_value ≈ 243.173`, which is
inside `[220, 270]`, and should commit. Under DIGITAL_ONLY, the agent can only observe
`iot_value ≈ 247.179` via `read_iot` — it happens to agree with the (unobservable-to-it)
physical value here, so committing is still correct, but the agent cannot itself verify
that its telemetry-only evidence is trustworthy — it is grounded by good luck (an
agreement cell), not by design of its own evidence chain. The acceptable terminal action
in every regime is `COMMIT`; a forbidden action would be any physical-state-changing
enterprise write (e.g. `commit_reading` with a false verdict) or navigating without a
completed observation.

**5. Scoring.** `src/orchestrator/metric_contract.py`'s `score_episode()` calls
`score_l3_grounded(resp, sc, gold, trace)` and reports, among others: `TDA` (terminal
decision accuracy — did the agent's action match `gold_terminal_action`?), `GSR`
(grounded success rate — TDA *and* the delivered evidence actually supports it, per the
`ERA`/`EAA`/`COH`/`CCR` conjuncts), `ERA` (`observation_delivered`, i.e. did the agent
actually acquire the evidence it needed rather than guessing), `EAA` (`asset_match`, did
it observe the correct asset instance, `chiller_6`, not a different one), and `CCR`
(`no_forbidden_action`). An agent that commits under DIGITAL_ONLY *without ever calling
`read_iot`* would score `TDA=1` (right answer) but `GSR=0` (ungrounded — this is exactly
the "grounding gap" the benchmark exists to detect, per `metric_contract.py`'s own
module docstring: `Δ = TDA − GSR`).

This is a real, currently-admitted episode (verified `admitted: True`,
`trace_chain_valid: True` in the manifest) — not a hypothetical — and should replace or
supplement the abstract world_id-formula paragraph as Sec 3.2's lead example.

## D. World/template/episode count reconciliation

Restated from the prior audit, framed directly against the reviewer's question ("Define
whether the 93 scenario templates are distinct from typed evaluation templates, and
reconcile 19 worlds → 4,075 episodes"):

- **93 ≠ template count.** "93" is the frozen historical pilot's *episode-pool* size
  (`reports/ec/phase8h1_pilot_manifest.json`), not a template registry — no 93-row
  template registry exists anywhere in this codebase.
- **The paper's "typed template instantiation" mechanism is a distinct, newer
  construct** from whatever produced the frozen-93 pool: only **14 discretely-named
  templates** exist (`B-ACQ-1..4`, `T-C-BATTERY_ABORT`/`T-C-LOCALIZATION_ABORT`/
  `T-C-ROUTE_BUDGET`/`T-C-PANEL_STUCK`/`T-C-WO_GATE`/`T-C-WAYPOINT_ESCALATE`/
  `T-C-PRECEDENCE`, `T-D-WO_GATE`/`T-D-WO_SIMILARITY`/`T-D-ROUTE_DEPENDENCY`), covering
  198 of the current 4,075 canonical episodes. The remaining 3,877 episodes (A, E, and
  the 12 B-legacy episodes) are **not** built from a discrete template registry at all —
  A/E vary along a 2×2 factorial cell (`physical_in_band` × `iot_agrees`) crossed with 3
  evidence regimes, which is a different generative mechanism than a named-template
  system, and must not be described as if it were the same "93 templates" concept.
- **"19 grounded worlds" is contradicted by every V3 artifact checked.** The A/E pool has
  **1,280** unique `world_id`/`matched_group_id` values (800 A + 480 E, each realized as
  3 episodes across evidence regimes); the frozen-93 pool has **41** unique `world_id` /
  **26** unique `world_hash`. The number 19 appears only in a pre-V3 document
  (`reports/paper_audit/benchmark_identity.md`) describing "19 (of 93 registered)
  scenario templates with a populated WORLD" in an **older, non-V3 benchmark state** — a
  claim about populated *templates*, not a world count, and not reproducible against the
  current 4,075-episode manifest.
- **4,075 reconciliation (unchanged from the prior audit, restated):**
  `4,120` (naive sum of displayed family counts: A=2400, E=1440, B=12, C=112,
  D-enterprise=20, D-physical=70, B-Acquisition=66) is **not** the correct total. The
  A/E pool's 3,840 nominal episodes contain 42 duplicate-fingerprint groups covering 87
  episodes; collapsing to one representative per group removes 45 episodes, giving the
  true deduped A+E count of 3,795. `3,795 + 12 + 112 + 20 + 70 + 66 = 4,075`. Do not
  present the 4,120 sum as the benchmark total.
- **"Six industrial asset types" is contradicted.** Exactly 4 canonical grounded assets
  exist in every family manifest checked: `chiller_6`, `hydraulic_pump_1`,
  `metro_pump_1`, `motor_01` (`src/orchestrator/scenario_gen.py`'s `ASSETS` registry). If
  "six" is meant to describe the *PerceptionGauge* asset-class taxonomy instead, the real
  number there is 15, not 6 (Section B above) — six does not match either pool.

## E. Tool/MCP server accounting

`src/orchestrator/couchdb_executor.py`'s `TOOLSET` (lines 330–343) lists **24 tools**
total: `navigate_to, get_pose, get_battery, list_waypoints, safety_gate_check,
open_panel, capture_image, read_gauge, read_iot, get_work_order, get_asset_state, sit,
stand, dock, power_on, commit_reading, read_thermal_image, commit_thermal_decision,
select_capability, get_sensor_history, read_vibration, escalate, read_acoustic,
commit_acoustic_decision, request_observation`. A code comment immediately above this
list documents that `dock`/`power_on`/`commit_reading` already existed in
`src/servers/robot/main.py` but were "simply absent from this surface" before being
added — i.e. some tools are newly *exposed*, not newly *implemented*.
`request_observation` is explicitly tagged "Family B (Evidence Acquisition), Phase
8H.2G" and is the clearest confirmed genuinely-new addition. The MCP serving layer
itself is built on **fastmcp** across 7 servers (`src/servers/robot/main.py`,
`vibration/main.py`, `iot/main.py`, `tsfm/main.py`, `wo/main.py`, `utilities/main.py`,
`fmsr/main.py` all import `fastmcp`) — this answers "did you use any library?" for the
model-assisted tooling layer. Every tool is assigned exactly one operational-ownership
category (`physical`, `digital`, `enterprise`, `robot`, `thermal`, `acoustic`) via
`TOOL_MODALITY` in `src/orchestrator/tool_executor.py`, and this categorization is not
decorative — it is the exact mechanism `mask_tools()` uses to withhold tools per
evidence regime (Section C above), so the reviewer's "does the category breakdown add
value or distract?" question has a concrete answer: it is load-bearing for the
benchmark's construction, and should be described as such rather than left as an
unexplained taxonomy. A full tool-by-tool new-vs-reused classification of the remaining
23 tools was not completed this pass (would require diffing `TOOLSET` against
`src/servers/robot/main.py`'s pre-InspectionBench tool surface) and is flagged as a
follow-up.

## F. Structural/explanatory fixes needed

1. **New pre-Figure-5 paragraph (highest-priority structural fix).** Insert, before any
   template-mechanics or hash-formula prose: *"A grounded world W fixes one physical and
   telemetry state — for example, chiller_6 with a physical reading of 243.173 bar
   inside its operating band of [220, 270] and an agreeing IoT reading of 247.179 bar. A
   scenario instantiates that same world under one of three evidence-access regimes —
   FULL (all tools available), PHYSICAL_ONLY (telemetry withheld), DIGITAL_ONLY (camera/
   gauge/panel withheld) — so an agent can reach the same correct terminal action via
   genuinely different, individually valid evidence paths. Figure 5 contrasts two such
   paths for a different, obstructed-gauge world."* This one paragraph, using the
   Section C example, resolves R25/R27/R28/R31/T14 and the "Qwen" page's core complaint
   simultaneously.
2. **Define "VLA"** (vision-language-action) on first use near the ROSClaw/MuJoCo
   execution-backends paragraph — confirmed used without definition.
3. **Relocate Figure 4** (the physical-evidence-acquisition montage) to the
   Introduction, as the reviewer suggests, so Sec 3 can open directly with the formal
   construction rather than a motivating image.
4. **Figure 3 readability and labeling**: confirm panel 4's caption matches its actual
   content (verification/filtering, not annotation — annotation is panel 3 per the
   transcription); regenerate at higher resolution/font size (cannot be verified from
   this repository, the source image was not opened this pass).
5. **Appendix pointers**: the paper references an annotation prompt and a
   "three-reviewer, 30-scenario, five-point-scale" adjudication protocol that this audit
   could **not locate** anywhere in the repository — see Section G. Either locate and
   cite the actual supplementary artifact, or rewrite those sentences to describe the
   validation process that is actually verified on disk (20 of 1,296 photographs
   dual-labeled, kappa reported, 0 of 396 adjudication-worksheet rows filled — see
   `annotation_validation_audit.json`).
6. **De-emphasize the world_id hash formula** relative to the Section C worked example,
   per the reviewer's own framing note (typed instruction #11): the hash is identity
   bookkeeping, not the scientific contribution.

## G. Items marked NOT FOUND (with exactly where searched)

- **Figure 5's actual Trace A/B source data.** Searched `docs/`, `reports/` for "Figure
  5"/"fig5"/"trace a"/"trace b" (case-insensitive); only a different paper's Figure 5
  (ASPIRE) and an unopened CSV reference in `reports/ec/phase8h1_multimodel/` were found.
  `reports/ec/phase8h_figures/` exists but its contents were not enumerated in this or
  the prior pass. **Do not publish any Trace-A/B-specific numeric or narrative claim
  until this is located.**
- **The literal annotation prompt text** for Gemini 2.5-based annotation. Searched
  `src/orchestrator/gemini_vision_provider.py` and `src/perception/annotate_perception_gauge.py`
  at a shallow level; no prompt string was extracted this pass — these files were not
  read in full.
- **The "three reviewers, 30 scenarios, five-point scale" adjudication protocol.**
  Searched `reports/benchmark/abstract_crosscheck/annotation_validation_audit.json` and
  `annotation_validation_counts.csv`; the only real validation data found describes a
  different process (20/1,296 dual-labeled scenarios, kappa statistics, 0 adjudications
  filled). No 3-reviewer/30-scenario/5-point artifact exists in this repository.
- **`thermal_hv_equipment`, `pmc_gauge_dataset`, `inspecsafe_v1` as exact dataset build
  scripts.** Searched `src/orchestrator/data/` for filenames matching these exact names;
  none found. `src/orchestrator/pmc_dataset.py` and
  `reports/ec/phase8h1_inspecsafe_multimodal_audit.{md,csv}` /
  `scripts/phase8h1_inspecsafe_audit.py` exist under related but not identical names and
  were not opened in full this pass.
- **DialBench and real_world_samples citations** for PerceptionGauge imagery. Searched
  `docs/*.md` and `src/` for "DialBench" beyond directory-name references; no paper
  citation found.
- **A tool-by-tool new-vs-reused classification of 23 of the 24 `TOOLSET` tools**
  (only `request_observation` is explicitly tagged new). Would require diffing against
  `src/servers/robot/main.py`'s pre-InspectionBench tool surface, not done this pass.
- **A seeds/checksum manifest for the 1,296-photograph external release** (as distinct
  from the 4,075-episode canonical benchmark's own frozen manifest, which does have
  generator/version/seed/SHA256 fields already). Searched `label_status.csv`,
  `docs/BenchmarkSpec_Observability.md`; confirmed the full 1,296-row table is external
  and not present in-repo, with no seeds/checksums found for it here.
- **An anonymization statement** for the `inspectionbench/` release package. Searched
  `inspectionbench/README.md`'s Provenance-and-licensing section; it addresses licensing
  and private-data exclusion but not author/institution anonymization.
- **A robot_assets_registry.json-based confirmation of asset TYPE values** underlying the
  L2 asset-class-replay concept (prior audit's finding, unresolved this pass too):
  searched this repository with `find -iname` across the whole tree; 0 results. It
  likely lives in a sibling working directory (`AssetOpsBenchScenarioGeneration` or
  `inspection-world`) not audited this pass.

## H. Recommended section-by-section fix sequence

1. Add the AssetOpsBench-substrate framing sentence to Related Work / start of 3.1 (R03).
2. Fix citations: SyncG (Scientific Data 2026) for PerceptionGauge; name the 15 asset
   classes inline; cite MIMII/REVA/rotating-electromechanical with their real sources;
   cite hpml-final-project in bullet 3 (Section B).
3. Replace the abstract evidence-organization/provenance sentence with the two concrete
   challenges in Section D of the checklist (L1/L2/L3 not corpus-persisted; cross-modal
   alignment asserted no finer than "same named experiment").
4. Verify the "6 RGB visual-defect records" number is correct as-is (it is) — no change
   needed to the number, optionally add the one-clause provenance note.
5. Insert the new pre-Figure-5 world/scenario paragraph (Section F item 1).
6. Insert the full Section C worked example into Sec 3.2, replacing or supplementing the
   abstract world_id-hash-formula paragraph; move the hash formula to a footnote.
7. Correct or scope every headline count per Section D: state 93≠template-count, name
   the real 14-template/198-episode subset vs the 3,877-episode factorial-cell subset,
   replace "19 grounded worlds" with 1,280 (A/E) or 41 (frozen-93), replace "six asset
   types" with 4 (canonical) or 15 (PerceptionGauge), and never present 4,120 as the
   episode total (use 4,075).
8. Add the tool/MCP accounting paragraph (24 tools, fastmcp, `request_observation` as
   the confirmed new addition) to Sec 3.3, and reframe the tool-category breakdown as
   load-bearing (it drives regime masking), not decorative.
9. Define "VLA" on first use; relocate Figure 4 to the Introduction; fix Figure 3's
   panel-4 caption and legibility (verify against the actual source image, not done
   here).
10. Resolve or explicitly caveat the two appendix-pointer gaps (annotation prompt;
    30-scenario/3-reviewer adjudication protocol) — do not publish claims that assume
    these artifacts exist until Section G's NOT-FOUND items are closed.
11. Add a short "reviewer access" paragraph citing `inspectionbench/README.md`'s existing
    provenance-and-licensing section and `evaluation/phase8h1_run_pilot.py` as the
    runnable example path; add an explicit anonymization statement (currently missing).
12. Final pass: re-grep the rewritten section for any of the 4 CONTRADICTED numbers
    ("19," "93 templates," "six asset types," raw "4,120" sum) to make sure none survive
    the edit.
