# Related Work — verified (P0-7)

Every citation below was fetched and read on 2026-08-11. The Rev-2 plan marked
six as **[VERIFY]** because they post-date the planner's knowledge, and
`AssetOpsBench_Ablation_Findings_Summary.md` §6 already records this repo citing
an unresolved arXiv ID once. **All six exist and match their described topic.**

Three findings below change what the paper may claim. They are listed first.

---

## Findings that affect our claims

### 1. ForesightSafety-VLA already publishes a safe/unsafe × success/failure decomposition

`arXiv:2606.27079` (Lyu, Sun, Jia, Shen, Sha, Li, Zhao, Zeng) reports
**"a four-quadrant decomposition of safe/unsafe success and failure"**, plus
cumulative safety cost (CC) and risk exposure time (RET), over 66
safety-augmented RoboTwin scenarios with a 13-category taxonomy (Safe-Core
physical / Safe-Lang instruction-side / Safe-Vis perception-side).

**Impact on E5b.** The Rev-2 plan promoted "successful but unsafe" to a headline
(Figure 3) as the cleanest separation from generic VLA success benchmarks. The
*decomposition itself is not novel* — it is published. What remains ours is the
**third axis**: their quadrants are task success × physical safety, whereas our
claim is *task succeeded, physical constraints satisfied, **and the enterprise
record is now wrong*** (FM-6 stale work order, FM-9 cascade, FM-1 skipped
verification). That is a distinct failure surface, but the paper must:

- cite ForesightSafety-VLA when introducing the quadrant view rather than
  presenting it as new;
- state the delta explicitly as the enterprise-consequence axis;
- avoid the phrase "we introduce" for anything quadrant-shaped.

### 2. RoboAbstention reports ER-1.6 abstention rising to 93.6% under defensive prompting

`arXiv:2605.20544` (Yeke, Temirel, Shreekumar, Lee, Xu, Celik), 6,069
instructions derived from five robotics datasets. Gemini 2.5 Flash abstains
39.0%; **defensive prompting lifts Gemini Robotics-ER 1.6 Preview to 93.6%.**

**Impact on A13/E2.** Our A13 found the opposite direction for the same model
family: ER's hedge-then-commit signature did *not* recover under an informed
prompt (commit-on-unreadable 0.83 → 0.83, r_pb worsening +0.33 → +0.76), which
we attribute as residual rather than instruction artifact. Both cannot be
casually true of "ER's abstention", so the paper must reconcile them rather than
cite RoboAbstention as agreeing with us. Plausible reconciliations, to test
rather than assert:

- their abstention is over *instructions* (ambiguous, infeasible, false-premise);
  ours is over *visual evidence sufficiency* on a degraded gauge;
- "defensive prompting" may be substantially stronger than our informed variant;
- the version differs (ER-1.6 Preview vs whatever snapshot A16 ends up using).

This is now a concrete, cheap experiment: run their defensive-prompt style as a
third arm of E2 on our unreadable-gauge scenarios. If ER abstains at 93% there,
our "residual" attribution was an artifact of a weak informed prompt.

### 3. SimuHome's scale was mis-stated in our source document

The Executive Summary PDF this project inherited says "600 episodes across 12
query types." The paper (`arXiv:2509.24282`, Seo, Yang, Pyo, Kim, Lee, Jo; ICLR
2026 **Oral**) says **600 episodes across four task categories** — state inquiry,
implicit intent inference, explicit device control, workflow scheduling — each
with feasible and infeasible requests, and evaluates **18 agents**. Hardest
category is workflow scheduling, with failures persisting across agent
frameworks and fine-tuning.

Use 600 episodes / 4 categories / 18 agents. Do not repeat "12 query types".

---

## Verified reference details

| Citation | ID | Status | Key facts for the related-work section |
|---|---|---|---|
| RIPL manipulation audit | 2606.04233 | ✅ verified | Jiang, Tan, Wheeler, Sun, Ayalew, Walter. Four failure modes: shortcut solvability, statistical significance, creeping overfitting, data-source dependence. Audits LIBERO, CALVIN, SimplerEnv, RoboCasa, RoboTwin 2.0. **LIBERO and CALVIN fail multiple diagnostics**; a small probe model matches reported SOTA on LIBERO and most gains lack significance; CALVIN drops when object poses are randomised within training ranges. |
| SimuHome | 2509.24282 | ✅ verified | See §3 above. Matter-protocol stateful simulation; device operations affect environmental variables over time. |
| ForesightSafety-VLA | 2606.27079 | ✅ verified | See §1 above. RoboTwin-based; varies scene structure, language command, visual observation. |
| RoboAbstention / Yes-Man | 2605.20544 | ✅ verified | See §2 above. Three-phase pipeline: visual grounding → deterministic constraint derivation → template instruction generation. Open-sourced. |
| Gemini Robotics 1.5 | 2510.03342 | ✅ verified | Gemini Robotics Team (+171). Two models: GR 1.5 (VLA) and **GR-ER 1.5** (embodied reasoning). Motion Transfer; "think before acting" natural-language reasoning. ER 1.5 targets visual/spatial understanding, task planning, progress estimation. Abstract does **not** claim safety-refusal benchmarking — do not cite it for that. |
| Production probes for Gemini | 2601.11516 | ✅ verified | Kramár, Engels, Wang, Chughtai, Shah, Nanda, Conmy. **Probes fail to generalize under production distribution shift**, especially short→long context. Deployed in production Gemini. Relevant only as a secondary direction; the plan correctly does not make interpretability a requirement. |

### Version note

Gemini Robotics 1.5 documents **ER 1.5**. Our traces use
`gemini-robotics-er-1.6-preview`, and A16's ASIMOV comparison is blocked partly
on an ER-1.6 vs ER-2 mismatch. Any claim about "Gemini Robotics-ER" must name
the exact snapshot; three distinct versions are now in play across our results
and the literature.

---

## Where this leaves the positioning

Unchanged: the **causal attribution ladder** (competence / instruction
scaffolding / shortcut solvability / evaluation framing) remains unclaimed by
any of the six. RIPL supplies the threat model but audits *manipulation*
benchmarks and makes no capability claim; ForesightSafety-VLA and SafeVLA cover
physical-rollout safety; RoboAbstention covers instruction-level abstention;
SimuHome supplies stateful-environment design without safety; the probes paper
is monitoring, not benchmark validity.

Weakened: E5b's quadrant framing (§1) and, pending investigation, the
attribution of ER's behaviour as residual rather than instruction artifact (§2).

Strengthened: V1. RIPL's finding that a *small probe model matches reported SOTA
on LIBERO* is direct precedent for our metadata-only and majority-class
conditions, and for treating a restricted-input baseline as a benchmark-validity
diagnostic rather than a weak model.
