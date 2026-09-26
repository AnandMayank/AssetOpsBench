# Section 3.2 motivation paragraph + Section 5 insights re-ordering (draft)

Written for: paper authors, to review and paste into `InspectionBench-7.pdf`'s Section 3.2
(template-family motivation) and Section 5 (evaluation insights). This is prose to be
pasted, not repo documentation — it carries no code and describes no implementation.

Two things this draft does NOT yet do, flagged so they aren't mistaken for finished work:
it does not re-verify every existing paper number against the raw data (those are taken
from `InspectionBench-7.pdf` as given), and the "further work" checklist at the bottom is
scoped, not executed — each item says what exists today vs. what would need new analysis.

---

## 1. Section 3.2 addition: why the five families are structured this way

Insert after the "Evaluation capabilities and templates" paragraph (page 6, before Table 3),
or as a short lead-in before it. This extends the sentence you drafted, keeping your wording
as the spine:

> We structure the scenario families around the progression from state understanding to
> intervention, counterfactual reasoning, and decision making used in established
> machine-understanding evaluations, extending this hierarchy to the requirements of
> executable industrial inspection. In particular, the templates test whether an agent can
> first establish the relevant physical and operational state, acquire missing evidence,
> execute the required inspection procedure, respect physical admissibility constraints, and
> account for the temporal validity of delivered evidence. FactoryBench's four levels ask
> whether a model can read a state, predict how it evolves under an intervention, reason
> about a counterfactual, and finally decide and prescribe a remediation, using data that
> is supplied to the model at every level (Merzouki et al., 2026). Our families keep that
> same state-to-decision progression but relocate each step onto an executable interface:
> a template does not ask the agent to interpret evidence it was already given, it asks the
> agent to acquire that evidence itself, under access constraints the agent must recognize
> (A, evidence grounding; B, acquisition), through a robot that can fail mid-procedure (C,
> procedural execution; D, relational physical grounding), and to notice when a previously
> acquired reading is no longer trustworthy (E, temporal grounding). This is why
> INSPECTIONBENCH's capability profile is a superset along the execution axis rather than a
> like-for-like replacement of the state/intervention/counterfactual/decision ladder: A and
> the DIGITAL ONLY / PHYSICAL ONLY regimes correspond to FactoryBench's state layer, but ask
> whether the state was actually delivered through a tool rather than handed to the model;
> D's constraint-set and limiting-constraint scores correspond to the intervention and
> counterfactual layers, but are checked against a MuJoCo-verified physical oracle rather
> than a normalized telemetry stream; and B, C, and E have no FactoryBench analogue at all,
> because they test properties -- whether to acquire more evidence, whether a multi-step
> procedure was actually completed, whether a reading is stale -- that only exist once
> evidence acquisition is itself an action the agent takes rather than a precondition it is
> handed. The five template families were seeded in part from robot-inspection scenario
> definitions authored independently in AssetOpsBenchScenarioGeneration/RobotInspection
> (the classc_fixtures.py fixture set and its groundtruth, generalized by the compiler into
> world-independent rules -- Section 3.2, Appendix C.2); the compiler's contribution is
> making those hand-authored scenarios replay across every grounded world that carries the
> matching label, at the scale Table 3 reports, rather than remaining a fixed fixture list.

**Why this framing, not a different one.** FactoryBench's related-work strength is showing
that even models strong on state and intervention collapse at the decision layer (page 8 of
the FactoryBench draft you attached: every model below 18% at L4, "the leader reaches 17.7%
... an industrial readiness gap"). The parallel claim INSPECTIONBENCH makes is not "our
decision layer also collapses" -- Figure 7's terminal-decision numbers are comparatively
high (0.73-0.89) -- it is that **decision correctness is measured on evidence the model
handed itself**, and once that constraint is added, an analogous collapse appears one layer
earlier, in the gap between the decision and its grounding (A's 22.9-40.0pp TDA-GSR gap;
Section 5.1 below). This is the connective tissue between the two papers' "readiness gap"
framings, and it is worth stating explicitly in the paragraph above or in the Section 5 lead
(see \S2).

---

## 2. Section 5 insight ordering (lead insight, then the cross-capability Gemini narrative)

Reorder Section 5's opening so the single most load-bearing claim comes first, matching your
instruction. Suggested replacement for the "Evaluation and Results" lead paragraph and the
start of "Evidence grounding (A)":

> An inspection agent must do more than choose the right operational action: it must obtain
> the evidence that justifies that action and carry out the required inspection.
> **Terminal-decision accuracy can substantially overstate evidence-grounded success, which
> is particularly important for safety-critical inspection where a correct terminal action
> is not sufficient unless the supporting evidence was actually acquired and delivered.**
> This is the central finding this section establishes and then traces across the remaining
> four capabilities: acquisition behavior (B), procedural completion (C), physical
> admissibility (D), and evidence freshness (E) each expose a further way a terminal action
> can be correct without the underlying inspection having actually happened.

Then, after presenting A's per-model TDA/GSR numbers as currently written, add the
cross-capability paragraph tying A and B together through Gemini -- this is the standout
result worth foregrounding, not burying in the B paragraph on its own:

> Two capabilities agree on which model grounds its decisions best, and for a coherent
> reason. Gemini has both the smallest TDA-GSR gap in evidence grounding (A) of any model in
> the panel and, independently, the strongest evidence-acquisition behavior (B) of any model
> apart from Qwen. Because A and B are scored from disjoint episode pools through unrelated
> predicates -- A never asks the agent to acquire anything, B never checks whether the
> terminal action matches a physical reading -- this is not one metric restating another; it
> is two independent measurements agreeing that Gemini's terminal decisions track what it
> actually observed more closely than the other models', rather than defaulting to the
> prevalent gold action the way a model with a large TDA-GSR gap does. This is consistent
> with Gemini Robotics-ER 1.5's reported strength on ASIMOV, a benchmark for whether an
> embodied agent's stated action respects the physical and safety constraints of what it can
> actually perceive and do, where Gemini Robotics-ER 1.5 was the strongest frontier model
> evaluated and no frontier model -- including GPT-5, Gemini 2.5 Pro, and Claude Opus 4.1 --
> reached a constraint-violation rate below 30% when reasoning jointly about embodiment,
> physics, and vision (Google DeepMind, 2025). ASIMOV and INSPECTIONBENCH test different
> things -- ASIMOV checks whether a proposed action violates a stated physical constraint,
> we check whether a terminal decision is grounded in evidence the agent itself acquired --
> but both point to the same underlying property: whether a model's output is coupled to
> what it can actually verify, rather than to what is plausible given the prompt. Read
> together with our result, this suggests evidence-acquisition competence (B) is not an
> independent skill orthogonal to evidence-grounding (A); a model that reliably recognizes
> when it needs more evidence before committing is, by the same disposition, less likely to
> commit on a decision its evidence does not support.

**On the extensibility motivation and RoboHarm/RoboCurve.** You asked to connect this to why
a modular, expandable evaluation matters and to bring in the recent RoboHarm result. That
benchmark is a different axis from ours -- it tests whether a robot policy *refuses* a
directly harmful instruction (RoboCurve, 2026: GPT-6-Astra attempted 97 of 100 harmful
physical tasks across five fixed scenes; Claude Fable 5.1 refused 20; MolmoAct2 refused
none) -- but it is the right citation for the argument that execution-grounded,
tool-mediated evaluation surfaces failures that prompt-only or outcome-only evaluation
cannot, which is exactly INSPECTIONBENCH's design premise (Section 1's "the robot itself can
also fail" framing). Suggested addition, placed in the Conclusion or as a closing paragraph
of Section 5:

> These results argue for evaluation that is modular across capability profiles rather than
> a single aggregate score, and for keeping that evaluation grounded in real tool execution
> rather than supplied observations. A concurrent line of work makes the same case from the
> safety side: RoboHarm, built on real bimanual manipulation hardware, finds that GPT-6-Astra
> -- the same model family evaluated here -- attempts the overwhelming majority of directly
> harmful physical instructions it is given, a failure mode invisible to any benchmark that
> does not actually execute the proposed action against a real or high-fidelity environment
> (RoboCurve, 2026). INSPECTIONBENCH's delivery firewall (Section 4) is built on the same
> premise applied to a benign rather than adversarial setting: a plausible-sounding decision
> is not evidence that the underlying inspection took place. As new frontier models and new
> robot-execution benchmarks continue to appear on a roughly quarterly cycle, a capability
> profile that can absorb a new template family (as B, C, and E were added on top of a
> FactoryBench-style state/decision core) without renormalizing the existing ones is the
> property that lets a benchmark stay current rather than becoming a snapshot of one model
> generation.

**Caution on the GPT-6-Astra "newer model, same plateau" claim.** You mentioned wanting to
show the newer model performed better at this level but "didn't show much improvement...
comparing to the earlier" -- I did not find a clean apples-to-apples pairing to support this
inside the current data: GPT-6-Astra and GPT-5.2 were run on different panels (swapped vs.
original) against overlapping but not identical pools (frozen-93 shared; A/C/E-expanded
pools differ), so a direct "GPT-6-Astra vs. GPT-5.2, same episodes" delta is not yet
computed. If this claim is meant to ship, it needs GPT-5.2 and GPT-6-Astra scored on the
exact same episode set side by side before it's stated as a finding -- happy to build that
comparison table next if you confirm you want it (it does not require new runs, both raw
files already exist for frozen-93).

---

## 3. Scoped follow-up checklist (from the pasted list)

Marked by what already exists vs. what is new analysis:

| Item | Status | Notes |
|---|---|---|
| Model/leaderboard table | **Have the data** | `reports/benchmark/figures/new_panel_main_table_v1.json` + the original panel's `main_table_v1.json` cover this; needs formatting into a combined leaderboard table, not new computation. |
| Failure-mode analysis (why some models do better/worse) | **Partially have it** | The A-selection-bias analysis (ESCALATE-only invalid episodes, universal across all 4 swapped models) and the D CC-vs-CSA/LCA gap are failure-mode findings already surfaced this session; a systematic per-model failure taxonomy (categorizing *why* each wrong answer was wrong) is new work. |
| Statistical significance / repeat runs | **Not done** | No model has been run 3x at temperature 0 to check response variance; E's pooled analysis has bootstrap CIs (done), A/C/D do not yet. This is the single most consequential gap if a reviewer pushes on it. |
| Good/bad example pairs (appendix) | **Partially have it** | Table 2's matched-regime rows and Appendix C's five compiled examples are exactly this pattern; a dedicated "one good trajectory, one bad trajectory" pair per capability, with full tool-call traces, is not yet assembled. |
| Section 3 L1-L3 connection | **This draft covers it** | See \S1 above -- the FactoryBench-hierarchy mapping paragraph. |
| Tool-related errors | **Have raw data, no rollup** | `infra_failure` records exist per model per pool (used for resume logic all session); a rollup of tool-call error rates by tool name is not yet built. |
| Cost of execution (tokens, time) | **Have raw data, no rollup** | Every raw record carries `started_at`/`finished_at` and (where the backend reports it) token usage; DeepSeek V4 Pro 0813's token-budget escalation (1024->16384) is itself a partial cost data point already in the paper's disclosed-deviation trail. A per-model $/episode and latency table is not yet built. |
| Maintainability / performance | **Not started** | Not clearly scoped yet -- would need a definition (of the benchmark harness, or of the evaluated agents?) before this can be estimated. |
| Image sequence example (Spot report video frames) | **Not started** | You mentioned an old Spot video with frame sequences; that source file wasn't in this session's context -- point me at the video/frames and I'll pull a representative sequence for the appendix. |
