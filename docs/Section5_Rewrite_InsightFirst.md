# Section 5 rewrite: insight-first structure (draft, paste-ready)

Written for: paper authors, to replace the current Section 5 (Evaluation and Results) in
`InspectionBench-7.pdf`. This carries no code; it is prose for the paper.

**Source-of-truth for every number below:** `reports/benchmark/figures/new_panel_main_table_v1.json`
(A/B/C/D/E at the frozen-93 N convention, matching Figure 5's own convention), the
per-episode raw files under `reports/benchmark/v3_full_results/` and
`reports/benchmark/b_acquisition_pilot/`, and the C-17 expansion pool's ordering/coverage
breakdown computed fresh for this rewrite (below). Every number is either already in a
committed table/figure or computed directly from raw data in this pass — none are
carried over from the original panel. Two items are explicitly flagged as **not yet
available** rather than filled with placeholder numbers: DeepSeek V4 Pro 0813's
`c_expanded_17` file (mid-rerun as of this writing) and E's final pooled precision/recall
at N=180 (E-expanded-v2 is still completing for the last model).

---

## 5 EVALUATION AND RESULTS

An inspection agent can reach the correct operational action without reliably establishing
the evidence, procedure, or physical conditions that justify that action. Robots make
physical evidence acquisition possible, but an autonomous agent must still decide what to
inspect, which channel to use, when it has enough evidence, when the procedure is actually
complete, whether the proposed physical action is admissible, and whether evidence
acquired earlier is still valid when the decision is finally made (Section 1). We therefore
use the five-capability profile introduced in Section 3.2 not as a leaderboard, but as a
diagnostic of where along this inspection-to-action path an agent fails: A asks whether the
terminal decision was actually supported by delivered evidence, B whether the agent sought
the evidence it needed, C whether it completed the required procedure, D whether it
understood the physical constraints governing execution, and E whether the evidence it
acted on was still valid.

This is a different question from the one FactoryBench asks. FactoryBench organizes
machine understanding along a state -> intervention -> counterfactual -> decision
progression and shows that frontier models collapse at the decision layer even when
telemetry is supplied directly (their Level 4 result: every model below 18%, the best at
17.7%). INSPECTIONBENCH's models are not given that telemetry -- they must acquire it
themselves, through tools, under access and modality constraints, before any decision layer
is reached at all. Our central finding is accordingly not "models also collapse at the
decision layer" -- terminal-decision accuracy in our panel is comparatively high (48.9-80.8%
on A, 72.9-88.6% on D's terminal check). It is that **once evidence acquisition is itself an
action the agent must take, high terminal correctness stops being informative on its own**:
substantial gaps reappear one layer earlier and in four further places terminal accuracy
cannot see at all -- whether the evidence behind that terminal decision was actually
delivered (A), whether the agent recognized it needed to acquire that evidence (B), whether
it completed the inspection procedure that evidence required (C), whether it understood the
physical constraints that made its proposed action admissible (D), and whether the evidence
was still valid by the time it committed (E). This is the same underlying claim -- terminal
success obscures capability gaps -- read through a different lens than FactoryBench's,
because ours is the lens created once acquisition and execution are themselves part of what
is being evaluated, not a lens that competes with or replaces theirs.

**Evaluation protocol.** We evaluate five frontier multimodal LLMs: Claude Opus 5.5,
GPT-6-Astra, DeepSeek V4 Pro 0813, Gemini 3.1 Pro Preview, and Qwen3.5-397B-A17B. [Standard
protocol paragraph carries over unchanged from the current draft -- native multimodal input
as a hard eligibility requirement, tool-execution-only observation delivery, Figure 5's
sample sizes per model-dimension cell.]

### 5.2 Overall pattern

Figure 5 (heatmap) and the capability bar chart give the full model x capability grid. The
pattern that motivates the rest of this section: no model is uniformly strong or uniformly
weak across the five capabilities. Claude Opus 5.5 leads on A (51.1%) and C (55.4%) but is
lowest on B (15.2%) and E (27.8%). Gemini 3.1 Pro Preview and Qwen3.5-397B-A17B lead on B
(69.7%, 71.2%) but are among the weakest on A and C. This is itself informative: a single
aggregate score would average these opposing strengths and weaknesses into a number that
describes none of the five underlying capabilities accurately, which is precisely the
failure mode terminal-accuracy-only evaluation exhibits at the level of a single capability.

### 5.3 Capability-level findings

**A. Evidence grounding -- "correct action does not imply grounded action."**
*Question: did the terminal decision actually follow from evidence the agent received?*

Correct terminal actions do not necessarily indicate grounded inspections. Figure 6 compares
terminal decision accuracy (TDA) against grounded success rate (GSR, which additionally
requires the cited evidence to have been delivered, attributable to the required source, and
free of a forbidden action -- Appendix A.1). Across the panel:

| Model | TDA | GSR | Gap (pp) |
|---|---|---|---|
| Claude Opus 5.5 | 80.8% | 51.1% | 29.8 |
| GPT-6-Astra | 77.1% | 47.9% | 29.2 |
| DeepSeek V4 Pro 0813 | 73.8% | 45.2% | 28.6 |
| Qwen3.5-397B-A17B | 67.4% | 44.2% | 23.2 |
| Gemini 3.1 Pro Preview | 48.9% | 42.2% | 6.7 |

Grounded success is lower than terminal accuracy for every model. The always-ESCALATE
baseline makes the risk concrete: it reaches 87.5% TDA on this pool (42/48 gold labels are
ESCALATE) while scoring zero GSR, since it acquires no evidence at all -- a high terminal
score can therefore reflect the prevalence of an action rather than an inspection having
taken place. We use GSR, not TDA, as the primary measure of A for this reason.

Gemini is the one model whose TDA is not the highest of the panel, and this is directly why
its gap is smallest: its terminal accuracy is closer to its own grounded accuracy rather than
inflated above it. This is worth stating precisely rather than as praise -- Gemini's grounded
success (42.2%) is close to the panel's grounded-success range generally (42.2-51.1%); what
distinguishes it is that its terminal layer does not overstate that grounded number the way
the other four models' terminal layers do.

*Model-development implication:* GSR identifies a grounding failure ordinary terminal
accuracy cannot see -- a model can be tuned to the prevalence of an action in training data
without that tuning reflecting evidence-based reasoning.
*Deployment implication:* this motivates an explicit evidence-delivery / provenance gate
before an operational action is trusted, independent of how confident the terminal decision
appears.

**B. Evidence acquisition -- "seeking evidence does not imply grounding the final action."**
*Question: did the agent recognize what evidence it needed and select an acquisition?*

Strong evidence-seeking behavior is measurable, but it does not guarantee a grounded final
decision. Branched acquisition-gated success (B_AGS, Appendix A.2) across the panel:

| Model | B_AGS |
|---|---|
| Qwen3.5-397B-A17B | 71.2% |
| Gemini 3.1 Pro Preview | 69.7% |
| DeepSeek V4 Pro 0813 | 37.9% |
| GPT-6-Astra | 22.7% |
| Claude Opus 5.5 | 15.2% |

Decomposing B_AGS shows the entire ranking is driven by one step. Acquisition decision
accuracy (ADA -- does the model recognize it needs more evidence) and acquisition selection
accuracy (ASA -- does it pick the right modality) both exceed 93% for every model in the
panel; every model reliably recognizes when it needs more evidence and asks for the right
thing. What separates the panel is what happens *after* that evidence arrives. Restricted to
the 33 episodes per model where acquisition was required and genuinely available, and with
ADA/ASA already correct (isolating the post-acquisition decision step):

| Model | Correct after acquiring | Over-escalated despite adequate evidence |
|---|---|---|
| GPT-6-Astra | 9% | 82% |
| Claude Opus 5.5 | 24% (+18% malformed response) | 48% |
| DeepSeek V4 Pro 0813 | 36% | 55% |
| Gemini 3.1 Pro Preview | 39% | 52% |
| Qwen3.5-397B-A17B | 55% | 36% |

This branch's gold label is COMMIT in every one of these 33 episodes by the capability
contract's own design (the paired branch, where acquisition is required but genuinely
unavailable, is scored separately via UHA and is all-ESCALATE-gold), so this specifically
and only measures whether a model over-rides adequate acquired evidence with a default
escalation -- it is a real, well-defined failure mode, not a general claim about escalation
bias in both directions. We verified this is a shared difficulty gradient rather than
unrelated per-model noise: Qwen3.5-397B-A17B's 12 failures on this branch are a strict
subset of GPT-6-Astra's 27 (every episode Qwen gets wrong, GPT-6-Astra also gets wrong, plus
15 more). Nine episodes further show two models citing the *identical* delivered observation
id and reaching opposite terminal actions -- isolating the failure to interpretation of
delivered evidence, holding acquisition behavior fixed.

Gemini and Qwen both have high acquisition-gated success and, independently, Gemini has the
smallest A-capability TDA-GSR gap. Because A and B are scored from disjoint episode pools
through unrelated predicates, this is not one metric restating another. It is consistent
with -- not proof of -- evidence-acquisition competence and evidence-grounded
decision-making sharing a common disposition: a model that reliably recognizes when it needs
more evidence before committing appears, in this panel, less likely to also override that
evidence once acquired. We state this as an alignment between two independent measurements,
not a causal claim.

*Model-development implication:* separate acquisition-selection failure (which is rare, at
&gt;93% for every model) from evidence-use/grounding failure (which drives the entire B
ranking).
*Deployment implication:* monitor specifically whether the agent, once it has the evidence
it asked for, is defaulting to a cautious action rather than using that evidence -- a
different runtime signal than whether it asked for evidence at all.

**C. Procedural execution -- "a valid order does not imply a complete inspection."**
*Question: did the inspection actually happen, rather than merely producing a plausible tool
trace?*

A plausible tool sequence can still correspond to an incomplete inspection. Ordering
satisfaction and procedural coverage are scored independently (Appendix A.3: ordering checks
only the tools that were actually called, in the required relative order; coverage is the
fraction of *distinct required tools* actually executed) -- a model that calls none of the
required tools passes the ordering check vacuously while scoring zero coverage. On the
8-episode C-expansion pool (new, independent fixtures beyond the original 9; DeepSeek V4 Pro
0813's file is still completing its rerun as of this writing and is not yet included):

| Model | Ordering satisfied | Mean procedural coverage |
|---|---|---|
| GPT-6-Astra | 100% | 18.8% |
| Gemini 3.1 Pro Preview | 100% | 2.1% |
| Claude Opus 5.5 | 88% | 35.4% |
| Qwen3.5-397B-A17B | 88% | 25.0% |

GPT-6-Astra and Gemini 3.1 Pro Preview satisfy the ordering check on every single episode in
this pool while executing, on average, well under a fifth (GPT-6-Astra) or one fiftieth
(Gemini) of the tools the procedure actually required. An agent can therefore pass the
ordering check while performing almost none of the required inspection; reporting coverage
alongside ordering is what makes that gap visible at all.

*Model-development implication:* separate sequencing competence (whether calls that do
happen are in the right order) from procedural coverage (whether the required calls happen
at all) -- a model can be perfect on one and near-zero on the other.
*Deployment implication:* check completion of required inspection steps directly, rather
than accepting a trace that merely never violates an ordering constraint.

**D. Relational physical grounding -- "correct action does not imply recovered physical
constraint."**
*Question: does the model understand the physical constraint that governs the proposed
action?*

High terminal correctness can coexist with weak recovery of the physical constraint that
makes the action admissible. Across 70 canonical episodes per model, spanning six constraint
families (reach, joint/collision feasibility, clearance, grasp/payload, stability, energy)
against a MuJoCo-verified oracle:

| Model | Terminal correctness (CC) | Constraint-set accuracy (CSA) | Limiting-constraint accuracy (LCA) |
|---|---|---|---|
| GPT-6-Astra | 88.6% | 51.4% | 46.4% |
| Claude Opus 5.5 | 87.1% | 51.4% | 51.8% |
| DeepSeek V4 Pro 0813 | 72.9% | 54.3% | 44.6% |
| Gemini 3.1 Pro Preview | 75.7% | 40.0% | 35.7% |
| Qwen3.5-397B-A17B | 72.7% | 38.6% | 41.7% |

Both GPT-6-Astra and Claude Opus 5.5 exceed this task's always-ESCALATE terminal baseline of
80% -- the first time any model evaluated on this benchmark has done so -- yet their exact
constraint-set accuracy is roughly half that, and their limiting-constraint accuracy (does
the model name the *specific* binding constraint, not just any plausible one) is lower
still. A correct-looking terminal action is therefore not evidence of correct physical
reasoning: a model can reach the right DISPATCH/ESCALATE call while still being wrong about
*why* -- misidentifying which of six constraint families (reach, collision, clearance,
payload, stability, energy) actually binds. This matters directly for physical safety, since
the specific constraint identified is what a downstream system would act on (e.g., which
parameter to correct before retrying), not the terminal label alone.

*Model-development implication:* identify whether a model is failing at action selection
(low CC) or at physical-constraint understanding (low CSA/LCA despite high CC) -- these are
different failure modes requiring different fixes.
*Deployment implication:* use physical-admissibility checks or gates whenever a decision
affects real robot execution, rather than trusting the terminal action alone as evidence the
robot's physical situation was correctly understood.

**E. Temporal grounding -- "acquired evidence does not imply permanently valid evidence."**
*Question: is the evidence still valid when the final decision is made?*

Evidence that was sufficient when acquired can become insufficient later. On the frozen-93
pool (N=18 per model, the paper's established Figure 5 convention):

| Model | Grounded temporal-consistency score |
|---|---|
| DeepSeek V4 Pro 0813 | 61.1% |
| Gemini 3.1 Pro Preview | 61.1% |
| GPT-6-Astra | 55.6% |
| Qwen3.5-397B-A17B | 47.1% (n=17/18) |
| Claude Opus 5.5 | 27.8% |

Claude Opus 5.5 is lowest here despite leading A and C, underscoring that temporal grounding
is a distinct capability from either evidence grounding or procedural completion, not a
byproduct of either. [A pooled, scaled-up precision/recall statement at N=180/model,
matching the paper's existing "5x scale-up ... no pairwise model comparison is statistically
distinguishable" sentence, is pending: E-expanded-v2 (90 additional episodes/model) has
completed for all 5 models as of this writing, but the combined v1+v2 aggregate has not yet
been recomputed -- to be added once that pass is run, rather than estimated here.]

*Model-development implication:* measure directly whether a model recognizes an observation
has gone stale, since this is invisible to any evaluation that only checks the evidence was
valid at acquisition time.
*Deployment implication:* trigger re-observation, delay, or escalation when evidence falls
outside its validity window, as a runtime policy independent of how confident the model's
terminal decision is.

### 5.4 Cross-capability implications

The five capabilities are not five unrelated benchmark tasks. They diagnose different points
of failure along one inspection-to-action loop:

- **A** -- was the decision supported by delivered evidence?
- **B** -- did the agent seek the evidence it needed, and use it once acquired?
- **C** -- did it actually complete the required inspection procedure?
- **D** -- did it understand the physical constraints governing execution?
- **E** -- did the evidence remain valid when the decision was made?

For model developers, this decomposition separates failure types that a single aggregate
score conflates: A isolates grounding failures, B isolates acquisition/evidence-use
failures (and further separates recognition from post-acquisition use, Section 5.3-B), C
isolates tool-execution/completion failures independent of sequencing competence, D isolates
physical-reasoning failures independent of action-selection competence, and E isolates
temporal/freshness failures. A model weak on one capability but strong on another (Claude
Opus 5.5: strong A and C, weak B and E; Gemini and Qwen: strong B, weak A and C) needs a
different fix depending on which capability is weak, and no single number reveals which.

For inspection-system and fleet operators, each capability motivates a distinct runtime
control rather than a single accept/reject threshold on the terminal action: A motivates an
evidence-delivery / provenance gate, B motivates monitoring whether the agent is acquiring
evidence and, separately, whether it is using what it acquires, C motivates
procedure-completion monitoring independent of whether the calls that did happen were
correctly ordered, D motivates a physical-admissibility gate before dispatching a robot
action, and E motivates a freshness / re-observation policy triggered independently of
decision confidence. We frame these as diagnostic uses and runtime policies the benchmark's
failure modes motivate, not as measured deployment outcomes -- we did not measure fleet cost,
safety-incident rates, or deployment ROI, and none of the above should be read as such.

This is the same underlying argument FactoryBench makes -- that frontier models exhibit an
industrial-readiness gap terminal-task framing alone would miss -- read through a
complementary layer. FactoryBench's models are given the state; ours must acquire it. Once
acquisition, procedure, physical admissibility, and evidence freshness are themselves part
of what is evaluated, "the model got the inspection decision right" stops being a sufficient
description of whether the inspection was actually performed. INSPECTIONBENCH's capability
profile is built to say, for any one episode or any one model, not only whether the decision
was right, but why an inspection agent failed when it did, where in the acquire-execute-decide
loop that failure occurred, and what kind of model change or runtime control would address
it.
