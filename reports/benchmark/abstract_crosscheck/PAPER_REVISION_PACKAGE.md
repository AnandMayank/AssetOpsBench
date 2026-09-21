# InspectionBench Paper Revision Package
Consolidated from `reports/benchmark/abstract_crosscheck/` (full 26-file audit) for handing to a paper-editing assistant. Every number here has a source file cited; nothing is invented. Full backing detail lives in the individual audit files in this directory.

---

## 1. THE FOUR HEADLINE CLAIMS — verdicts

| Abstract claim | Verdict | Corrected fact |
|---|---|---|
| "4,075 canonical episodes" | ✅ VERIFIED (with caveat) | 4,075 = 4,009 (pre-existing) + 66 (B-Acquisition). Do **not** sum the displayed `family_counts` dict — it totals 4,120; the 45-episode gap is an A/E cross-family duplicate-fingerprint dedup (42 groups, 87 episodes → 42 representatives). |
| "19 grounded worlds" | ❌ CONTRADICTED | No current V3 artifact defines 19 of anything. A/E pool has 1,280 unique `matched_group_id`/`world_id` instances (800 A + 480 E); the frozen-93 historical pool has 41 unique `world_id` / 26 unique `world_hash`. "19" traces to a **stale pre-V3 document** describing an earlier benchmark state and a different unit. |
| "93 scenario templates" | ❌ CONTRADICTED | Only **14** discretely-named templates exist (4 B-ACQ-*, 7 T-C-*, 3 T-D-*), covering 198/4,075 episodes. "93" is actually the size of the frozen historical **evaluated-episode pool** — the likely source of the conflation. |
| "six industrial asset types" | ❌ CONTRADICTED | Exactly **4** canonical assets, confirmed identical across every family manifest: `chiller_6`, `hydraulic_pump_1`, `metro_pump_1`, `motor_01`. (At most 3 if pumps are grouped into one class: chiller / pump / motor.) |

## 2. THE MOST IMPORTANT FINDING — a process gap, not a construction gap

`reports/benchmark/phase11_offline_remediation_report.md` documents a **completed** remediation pass that was **verified in scratch but never written to disk**. `paper_results_v1.json` / `capability_profile_v1.json` still hold the stale, pre-correction numbers (confirmed: 0 occurrences of `n_excluded` in the current file). Anyone — including a paper-editing tool — pulling numbers from those files today gets outdated values. The three corrections:

1. **C's primary metric (`procedural_coverage`) inverts the model ranking.** Old metric (`ordering_satisfied`): DeepSeek ranked #1 (0.889). Corrected metric: DeepSeek ranks **last** (0.176). Full corrected order: GPT-5.2 (0.576) > Mistral (0.391) > Qwen (0.372) > Claude (0.354) > DeepSeek (0.176).
2. **D-physical LCA denominator corrected from n=70 to applicable_n=24** (independently re-derived and confirmed: 70 − 32 MULTIPLE − 14 None = 24 exactly). Every model's LCA value changes, most dramatically Qwen (0.3036→0.5, +65% relative) and GPT-5.2 (0.5714→0.7083).
3. **A Mistral B-Acquisition MAR/UAR extraction bug**, fixed: MAR 0.0152→0.0606, UAR 0.0303→0.0. All other B-Acquisition numbers (Mistral and the other 4 models) are unaffected.

## 3. RESULT-NUMBER CROSSCHECK (on-disk vs. corrected)

| Claim | On-disk (stale) | Phase-11 corrected (authoritative) |
|---|---|---|
| Claude A TDA / GSR | EXCLUDED/null in `paper_results_v1.json` | **TDA=0.0208 (1/48), GSR=0.0** — driven by 46/48 (95.8%) step-2 verdict-schema noncompliance, a protocol-compliance finding, not a grounding-capability finding (GPT-5.2 is 48/48 compliant on the identical protocol) |
| DeepSeek A GSR | 0.1458, n=48 | 0.1628, n=43 (numerator unchanged: both = 7.0 episodes) |
| Qwen A GSR | 0.3958, n=48 | 0.4419, n=43 |
| C procedural coverage | see ranking inversion above | GPT-5.2 0.576 / Mistral 0.391 / Qwen 0.372 / Claude 0.354 / DeepSeek 0.176 |
| D-physical LCA (n=24) | Claude 0.375 / GPT-5.2 0.5714 / DeepSeek 0.2679 / Mistral 0.3571 / Qwen 0.3036 (all at n=70) | Claude 0.5417 / GPT-5.2 0.7083 / DeepSeek 0.2917 / Mistral 0.375 / Qwen 0.5 |
| Mistral B-Acq MAR/UAR | 0.0152 / 0.0303 | 0.0606 / 0.0 (confirmed extraction bug, not a denominator choice) |
| Model-evaluated N total | 229 (93+66+70) | unchanged — VERIFIED, both agree |
| Eligible-by-dimension | A=48, B=78, C=9, D=76, E=18 | unchanged — VERIFIED |

## 4. CORRECTED ABSTRACT SENTENCE CANDIDATES (pick one, do not blend numbers from different candidates)

**Candidate 1 — scale-first, most conservative:**
> InspectionBench is an execution-grounded benchmark for agentic industrial inspection comprising 4,075 canonical episodes spanning five evaluation capabilities (evidence grounding, evidence acquisition, procedural grounding, relational physical grounding, and temporal grounding) across 4 canonical industrial assets (a chiller, two pumps, and a motor). Episodes are produced by a deterministic world-to-scenario compiler with matched-group construction (e.g. 800 A-family worlds × 3 evidence-delivery regimes, holding gold decisions invariant while varying what evidence is delivered to the agent). Of the 4,075 canonical episodes, 229 have been evaluated against a panel of 5 frontier models; the remaining episodes are scripted, generation-time constructs with no model execution against them to date.

**Candidate 2 — modality/provenance-forward, more cautious:**
> InspectionBench comprises 4,075 canonical, matched-generation inspection episodes over 4 grounded industrial assets, requiring agents to acquire visual, IoT/telemetry, acoustic, thermal, and maintenance-record evidence — under an explicit, generation-time lineage provenance scheme — before committing to a terminal operational decision. A frozen 2-turn protocol and a panel of 5 frontier models were used to evaluate 229 of these episodes; we find that the reported grounding gap for at least one evaluated model is substantially attributable to terminal-verdict protocol-compliance behavior rather than grounding accuracy alone, and that at least one capability's model ranking inverts under a corrected procedural-coverage metric versus the metric first used to report it.

**Candidate 3 — narrowest, safest:**
> InspectionBench is a benchmark of 4,075 canonical inspection episodes, built by a deterministic compiler from a small, fixed set of 4 grounded industrial assets, with matched-group construction that holds gold decisions fixed while varying delivered evidence across regimes. A subset of 229 episodes (spanning all five capability dimensions) has been executed against 5 frontier models to date; canonical benchmark size and model-evaluated N are reported separately throughout, since coverage ranges from 100% (B-Acquisition, D-physical) down to roughly 1–2% (A, E) of each family's canonical pool.

**The current sentence to retire:** *"agents acquire visual, operational, and telemetry evidence with explicit provenance and execute inspections before making downstream operational decisions"* — only PARTIALLY supported. A generation-time lineage mechanism is real and spans nearly all 4,075 episodes; the separate L1/L2/L3 evidence-source TIER system is defined in code but not persisted in any current manifest (0 matches across 4 manifests grepped), so "explicit provenance" cannot be claimed benchmark-wide as-is. Either name the lineage mechanism specifically, or scope the claim to the 50/66 B-Acquisition episodes that actually carry a `source_provenance` block.

## 5. OTHER CONFIRMED ISSUES TO FIX

- **"1,296 rendered scenes"** is a category error: 1,296 is a real, confirmed number but describes **real field photographs**, not procedurally rendered scenes (source doc explicitly says "zero procedural-render bias"). Use "1,296 real field photographs" or similar, not "rendered."
- **Figure 5** (agent-vs-environment enterprise-gate trace): no underlying trace artifact was found in the repository. Cannot be captioned/cited until the source trace data is located — flag to authors before touching this figure.
- **Human/SME validation**: no reviewer process for the 4,075-episode benchmark's own construction (worlds/scenarios/gold) was found anywhere. The only human-labeling data found concerns a *separate* 1,296-photo perception catalog, and even there the corpus-level reliability statistic was not found on disk (only a 20-scenario, explicitly-flagged-as-biased dual-label sample exists, with kappa values ranging 0.14–0.76 depending on field).
- **D-physical truncation**: DeepSeek (60% of rows) and Qwen (22.9% of rows) show material output truncation under a shared fixed 2,048-token cap. This is an infrastructure constraint, not a capability finding — disclose it alongside their D-physical numbers, but the frozen full-set result remains primary (not excluded).

## 6. RECOMMENDED REVISION ORDER (from `abstract_crosscheck_report.md`)

1. Correct the abstract/result inconsistency and freeze one versioned result artifact (write the Phase-11 corrections to disk first — everything downstream depends on this).
2. Add the main-paper scoring contract and complete the A–E protocol/result mapping.
3. Define counting units (canonical vs. model-evaluated N everywhere), report clustered/paired uncertainty where relevant, and document model/truncation conditions.
4. Validate matched interventions and disclose annotation/adjudication outcomes honestly (including the NOT FOUND items).
5. Strengthen D with class-aware metrics and separate agent behavior from safety-gate intervention (once the Figure 5 trace artifact is located).
6. Narrow unsupported conclusion/comparison-table claims, then complete the presentation edits.

---

## 7. PROMPT TO GIVE ASTRA

Copy everything between the lines below as a single prompt.

---

You are revising the InspectionBench ICLR paper draft. Do NOT invent, round, or estimate any number — every value you use must come from the verified facts below, which were produced by a direct, source-cited audit of the benchmark's manifests, scoring code, and evaluation artifacts (not from the paper itself). If a fact needed for a sentence isn't listed below, flag it as `[NEEDS SOURCE]` in your edit rather than guessing or keeping the old unsupported number.

**Task**: Revise the abstract, Section 3 (benchmark construction), Section 4 (evaluation and results), Table 1, Figure 6/7/8 captions, and the conclusion so every quantitative claim matches the verified facts below. Do not touch anything not covered by these facts (e.g. leave citations, related-work framing, and figure layout alone unless a caption states a now-wrong number).

**Four headline numbers to fix everywhere they appear** ("19 grounded worlds", "93 scenario templates", "six industrial asset types" all currently appear in the abstract, intro, and contributions list — find and fix every occurrence, not just the abstract):
- Replace "19 grounded worlds" — no current artifact supports this number. Use instead: "4 canonical grounded industrial assets, instantiated into 1,280 distinct matched-group world instances across the A/E evaluation pool" (or a narrower framing consistent with whichever capability is being discussed in that sentence).
- Replace "93 scenario templates" — this number is actually the size of the frozen historical evaluated-episode pool, not a template count. Only 14 discretely-named templates exist. Do not use "93" as a template count anywhere.
- Replace "six industrial asset types" with "4 canonical industrial assets" (name them: a chiller, two pumps, and a motor) everywhere this appears, including Table 1's asset-type framing if applicable.
- Keep "4,075 canonical episodes" but if you ever need to justify it against a per-family breakdown, use 4,009 (pre-existing) + 66 (B-Acquisition) = 4,075, and explicitly note that summing the displayed per-family counts gives 4,120 due to a documented 45-episode A/E cross-family dedup — never present 4,120 as if it were the total.

**Critical: use the corrected (Phase-11) result numbers below, not whatever is currently in the draft or in any results table you might read from the repo — those on-disk files are confirmed stale:**
- C's primary metric is `procedural_coverage`, not `ordering_satisfied`. Corrected ranking: GPT-5.2 0.576, Mistral 0.391, Qwen 0.372, Claude 0.354, DeepSeek 0.176. (This INVERTS DeepSeek from best to worst — if the current draft calls out DeepSeek as strong on C, that claim must be reversed or removed.)
- D-physical LCA must use the applicable_n=24 denominator (episodes where the oracle designates exactly one limiting constraint), not n=70. Corrected values: Claude 0.5417, GPT-5.2 0.7083, DeepSeek 0.2917, Mistral 0.375, Qwen 0.5.
- Mistral's B-Acquisition MAR/UAR: MAR=0.0606, UAR=0.0 (the current 0.0152/0.0303 values are a confirmed extraction bug).
- DeepSeek's A GSR: 0.1628 at n=43 (not 0.1458 at n=48). Qwen's A GSR: 0.4419 at n=43 (not 0.3958 at n=48).
- Claude Sonnet 4.6's A-family result must be reported explicitly, not silently excluded: TDA=0.0208 (1/48), GSR=0.0. State clearly that this is driven by Claude replying with a further tool-request instead of the required terminal verdict on 46/48 (95.8%) episodes under the frozen 2-turn protocol — a protocol-compliance finding, not evidence that Claude cannot ground its decisions. GPT-5.2 complies with the identical terminal-verdict requirement on 48/48 episodes.

**Always report canonical benchmark size and model-evaluated N together, never let one imply the other:** 4,075 canonical episodes total; only 229 have ever been evaluated by any model (93 from the frozen historical pool + 66 B-Acquisition + 70 D-physical), broken down by capability as A=48, B=78, C=9, D=76, E=18. State this explicitly wherever the abstract or intro cites the 4,075 figure in a way that could be read as "4,075 episodes were evaluated."

**Provenance claim**: the sentence "agents acquire visual, operational, and telemetry evidence with explicit provenance and execute inspections before making downstream operational decisions" is only partially supported — an L1/L2/L3 evidence-source tier is defined in code but is not persisted in any current benchmark manifest. Rewrite this to either (a) name the real generation-time lineage mechanism (world_id / matched_group_id / generator_version fields present on ~all 4,075 episodes) instead of an unqualified "explicit provenance," or (b) scope the "explicit provenance" claim specifically to the 50/66 B-Acquisition episodes that carry a `source_provenance` block.

**"1,296 rendered scenes"**: this is a category error. The 1,296 figure is real but describes real field photographs, not procedurally rendered scenes (the source documentation explicitly states "zero procedural-render bias"). Change "rendered" to "real field photographs" or equivalent wherever 1,296 appears.

**Figure 5**: do not edit its caption to add new claims — flag it back to me with `[NEEDS SOURCE: Figure 5 trace artifact not found in repository]` rather than guessing what Trace A / Trace B actually did.

**Do not add**: any human/SME validation claim about the 4,075-episode benchmark's own construction (worlds, scenarios, gold labels) — no such review process was found. If the current draft implies domain-expert review of the benchmark's construction beyond the separate 1,296-photo perception catalog's annotation process, remove or narrow that claim.

**Style constraint**: use the three abstract-sentence candidates below as your starting point for the abstract's construction sentence — pick the one that best fits the surrounding paragraph's tone, do not mix numbers from different candidates, and do not reintroduce 19/93/six anywhere else in the sentence:

[Candidate 1]: InspectionBench is an execution-grounded benchmark for agentic industrial inspection comprising 4,075 canonical episodes spanning five evaluation capabilities (evidence grounding, evidence acquisition, procedural grounding, relational physical grounding, and temporal grounding) across 4 canonical industrial assets (a chiller, two pumps, and a motor). Episodes are produced by a deterministic world-to-scenario compiler with matched-group construction (e.g. 800 A-family worlds × 3 evidence-delivery regimes, holding gold decisions invariant while varying what evidence is delivered to the agent). Of the 4,075 canonical episodes, 229 have been evaluated against a panel of 5 frontier models; the remaining episodes are scripted, generation-time constructs with no model execution against them to date.

[Candidate 2]: InspectionBench comprises 4,075 canonical, matched-generation inspection episodes over 4 grounded industrial assets, requiring agents to acquire visual, IoT/telemetry, acoustic, thermal, and maintenance-record evidence — under an explicit, generation-time lineage provenance scheme — before committing to a terminal operational decision. A frozen 2-turn protocol and a panel of 5 frontier models were used to evaluate 229 of these episodes; we find that the reported grounding gap for at least one evaluated model is substantially attributable to terminal-verdict protocol-compliance behavior rather than grounding accuracy alone, and that at least one capability's model ranking inverts under a corrected procedural-coverage metric versus the metric first used to report it.

[Candidate 3]: InspectionBench is a benchmark of 4,075 canonical inspection episodes, built by a deterministic compiler from a small, fixed set of 4 grounded industrial assets, with matched-group construction that holds gold decisions fixed while varying delivered evidence across regimes. A subset of 229 episodes (spanning all five capability dimensions) has been executed against 5 frontier models to date; canonical benchmark size and model-evaluated N are reported separately throughout, since coverage ranges from 100% (B-Acquisition, D-physical) down to roughly 1–2% (A, E) of each family's canonical pool.

After editing, produce a short changelog listing every sentence/number you changed and why, plus a list of every `[NEEDS SOURCE]` flag you inserted.

---

## 8. Section 3 reproducibility fixes (separate, newer pass)

A second audit pass went deep on Section 3.1–3.3 (grounded-world construction, scenario compilation,
episode generation/tool execution) against the current `InspectionBench-1.pdf` draft (ICLR 2027),
covering Figure 3's trace-blocking mechanism, the PerceptionGauge rendering pipeline, the "14
templates / 52 variation identifiers / 1,280 grounded worlds" claims, the annotation/adjudication
process, the tool catalog and Spot SDK provenance, and the MuJoCo/ROSClaw physical-admissibility
oracle. Full findings, replacement wording, and a `templates_inventory.csv` draft are in
**`section3_reproducibility_fixes.md`** in this directory — treat it as a second, independent
handoff document alongside this one (it duplicates none of this file's content; both should be given
to an editor together). Headline findings not already covered above:

- Figure 3's Trace A/B are real runs, but both the block reasons and Trace B's recovery framing are
  described incorrectly in the current draft (traced to `reports/ec/phase8h_figures/qa_demo/`).
- The "1,296 rendered scenes, 856/440" figure cannot be reproduced from any generator output on disk.
- "52 variation identifiers" does not reproduce under any tested counting rule (61 does).
- "Six industrial asset types" is contradicted for the actual 4,075 episodes (only 4 assets / 3 types
  appear in any canonical episode).
- No genuine human domain-expert adjudication exists anywhere in the benchmark's construction —
  `"sme_adjudicated"` means hand-authored by the generator's builder, not independently reviewed; one
  file states outright "SELF-ADJUDICATED BY CLAUDE, NOT HUMAN-REVIEWED."
- The abstract's "22.9–46.5 percentage points" does not reproduce from any artifact; the correct range
  is 16.2–40.0pp (or 4.2–40.0pp including Claude's corrected, currently-excluded result).

---
