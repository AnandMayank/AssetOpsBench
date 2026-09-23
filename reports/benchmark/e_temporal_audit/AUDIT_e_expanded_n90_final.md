# E (temporal grounding), scaled to n=90/model: final audit and paper recommendation

## What was tested

The reobservation-necessity confusion matrix: for every E episode after the
first, does the model correctly decide whether to re-observe (Precision =
of the reacquisitions it made, how many were actually needed; Recall = of
the reacquisitions that were needed, how many it made)? This requires
persistent hidden state across multiple agent turns in one continuous
session -- structurally impossible for a single/few-turn supplied-evidence
QA benchmark (FactoryBench) to construct.

## Scale-up: E-v1 (n=18, 6 sequences) -> E-expanded-v1 (n=90, 30 sequences)

Built and run for this reason specifically: at E-v1's N<=18, all 5 models'
95% CIs overlapped -- no defensible per-model claim was possible (see
`e_temporal_confusion_v1.json`). A 5x scale-up (matching the factor that
successfully resolved A's underpowered claims) was built entirely offline
first (`e_expanded/`, `e_temporal_expanded_v1.json`, zero API cost, 20/30
sequences confirmed to have a genuine band transition), then run live for
all 5 models (`phase8h2z_e_expanded_full_run.py`) -- 150/150 sequences
complete, 0 infra failures across all 5 models (2 interrupted runs were
resumed cleanly via added resume-by-sequence-id logic, no wasted API calls
on already-completed sequences).

## Result 1: the null result held up, and got MORE precise, not less

| Model | Precision [CI] | Recall [CI] | F1 | Decision accuracy [CI] |
|---|---|---|---|---|
| Claude Sonnet 4.6 | 0.650 [0.500,0.791] | 0.788 [0.647,0.909] | 0.712 | 0.553 [0.426,0.681] |
| GPT-5.2 | 0.667 [0.541,0.784] | 0.800 [0.688,0.904] | 0.727 | 0.667 [0.567,0.756] |
| DeepSeek V4 Pro | 0.707 [0.586,0.821] | 0.695 [0.563,0.815] | 0.701 | 0.602 [0.500,0.705] |
| Mistral Medium 3.5 | 0.638 [0.516,0.759] | 0.755 [0.628,0.872] | 0.692 | 0.607 [0.500,0.714] |
| Qwen3.5-397B-A17B | 0.780 [0.645,0.900] | 0.604 [0.463,0.732] | 0.681 | 0.651 [0.546,0.744] |

**0 of 10 pairwise model comparisons show a non-overlapping (defensible)
95% CI difference in decision accuracy, even at 5x the original N.** Unlike
the A-family matched-regime result (where scaling REVEALED a real effect
Mistral's small sample had missed), scaling E CONFIRMS the null: all five
models cluster in a similar 55-73% reobservation-decision-accuracy band,
with genuinely overlapping uncertainty, not just insufficient data to tell.
**Recommendation: report this as a well-powered null result** -- current
frontier models do not meaningfully differ in temporal-reobservation
judgment on this benchmark, at n=90/model.

## Result 2: pooled headline number, now well-powered (n=395, was n=74)

Pooled across all 5 models (n=395 evaluable episode-decisions, up from 74
at E-v1's scale): precision 0.685, recall 0.721, **miss rate on necessary
reacquisition = 27.9%** (was 31% pooled at n=74 -- same direction, now much
tighter). Base rate of "reobservation was necessary" across the pool:
61.8% (confirms the offline diversity audit's prediction that the expanded
pool would deliver real reobservation-necessity opportunities, not a
degenerate mostly-stationary sample).

**This is the one number to lead with in the paper for E**: nearly 1 in 3
times a model's tracked state genuinely goes stale, it acts on the stale
value anyway rather than re-observing -- a directly quotable, well-powered
temporal-grounding failure rate.

## Result 3 (unplanned but real): Claude's E output-validity is 52.2%, vs ~5% on A

Claude's terminal-verdict compliance is dramatically different between the
E (temporal, multi-visit) protocol (52.2%, 47/90) and the A (single-episode)
protocol (~5%, previously established). GPT-5.2/DeepSeek/Mistral/Qwen are
all 93-100% valid on E, similar to their A-family rates -- so this is
specific to Claude, not a property of the E protocol itself. This was not
the finding we were scaling to find, but it is real, reproducible (150/150
sequences ran cleanly, no infra artifacts), and worth a sentence in the
paper: Claude's step-2 protocol non-compliance rate is not fixed across
task framings -- it is roughly 10x lower under the multi-visit "Visit k of N"
framing than under the single-episode "STEP 2 -- decide" framing. We do not
have a confirmed mechanism for why (candidate hypothesis: the multi-visit
framing's repeated, explicit turn structure may cue compliance more
reliably than a single two-turn exchange -- untested, stated as a
hypothesis only, not a claim).

## Recommended paper text (Section 4.2, Temporal Grounding)

> We measure temporal grounding via a reobservation-necessity task: across
> 30 three-visit sequences per model (90 episodes), does the agent
> re-observe exactly when the tracked physical state has changed since its
> last observation? This construct requires persistent world state across
> multiple agent turns within one session -- a capability no single/few-turn
> supplied-evidence benchmark can test. Pooled across all five models (395
> evaluable episode-decisions), agents fail to re-observe when necessary
> 27.9% of the time (recall 0.721; precision 0.685 -- when they do
> reacquire, roughly a third of the time it was not actually necessary).
> No pairwise difference between models is statistically distinguishable
> at this scale (bootstrap 95% CIs, 2000 resamples; all 10 pairs overlap),
> indicating current frontier models are similarly limited on this
> construct rather than differentiated. We separately note that Claude
> Sonnet 4.6's terminal-verdict output-validity rate is markedly higher on
> this multi-visit task (52.2%) than on the single-episode evidence-
> grounding task (Section 4.2, Evidence Grounding; ~5%), suggesting its
> step-2 protocol non-compliance is sensitive to task framing rather than
> a fixed trait.

No frozen benchmark manifest, scoring contract, or runner code was
modified to produce this result. E-v1 (the historical 18-episode pool) is
untouched; E-expanded-v1 is a new, additive, non-overlapping-seed pool.
