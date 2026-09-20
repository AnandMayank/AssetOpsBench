# Scaling audit: was n=16 worlds enough? (superseded by A-expanded-v1, n=80)

**Question asked directly:** is the n=16-world matched-regime comparison
(`figure_matched_regime_v1`) strong enough to defend a paper claim, and if not,
scale up.

**Answer: no, n=16 was underpowered for some of the claims, and scaling to n=80
changes the conclusion for at least one model.**

## What changed

A separate, already-executed 80-world A-expanded-v1 pool exists
(`reports/benchmark/v3_full_results/a_expanded_240/`, 240 episodes/model =
80 worlds x 3 regimes, same manifest structure and scoring code as the
original 48-episode pool, seeds 4000-4079 -- no overlap with the original
3000-3015 seeds). Verified before use: 240/240 rows for GPT-5.2, DeepSeek V4
Pro, Mistral Medium 3.5, and Claude Sonnet 4.6; **Qwen3.5-397B-A17B is only
20/240 -- that run stalled (no process running, file untouched since
2026-09-18) and is excluded from this analysis pending completion.**

## n=16 vs n=80 TDA(FULL) vs TDA(DIGITAL_ONLY), bootstrap 95% CI

| Model | n=16 claim | n=80 result | CIs overlap? |
|---|---|---|---|
| GPT-5.2 | "falls from 0.69-0.75 to 0.56" | FULL 0.763 [0.663,0.850] vs DIGITAL_ONLY 0.475 [0.363,0.575] | **No -- defensible** |
| Mistral Medium 3.5 | "unchanged across all three regimes (0.69 throughout)" | FULL 0.800 [0.713,0.875] vs DIGITAL_ONLY 0.588 [0.475,0.688] | **No -- defensible, and CONTRADICTS the n=16 claim**: Mistral IS flat between FULL and PHYSICAL_ONLY (0.800 both, exactly), but drops significantly under DIGITAL_ONLY. The n=16 "unchanged across all three" claim was an artifact of insufficient power, not a correct finding. |
| DeepSeek V4 Pro | not claimed at n=16 | FULL 0.568 [0.459,0.676] vs DIGITAL_ONLY 0.538 [0.423,0.654] | Yes -- still not defensible, even at n=80 |
| Claude Sonnet 4.6 | excluded (0 evaluable pairs) | FULL n=12/240 evaluable (5.0% output validity), PHYSICAL_ONLY/DIGITAL_ONLY n=0 | Still excluded -- generalizes the 46/48 finding at 5x scale (228/240 empty verdicts here vs 46/48 in the original pool, same step-2 mechanism) |

GSR (FULL vs PHYSICAL_ONLY) CIs overlap for all 3 usable models even at n=80 --
the null result (no measurable GSR effect from withholding only digital
evidence when physical is present) is now well-powered and robust, not just
a product of small N.

## Corrected conclusion for the paper

Use the **n=80 (A-expanded-v1) numbers**, not the n=16 pool, for any claim
about evidence-regime sensitivity: they are more precise (narrower CIs) and,
for Mistral, actually reverse the smaller sample's conclusion. Recommended
paper paragraph:

> On the 80-world A-expanded-v1 pool (240 episodes/model; results for
> GPT-5.2, DeepSeek V4 Pro, and Mistral Medium 3.5, all with 90-100% output
> validity), withholding digital/IoT evidence while physical evidence
> remains available produces no statistically distinguishable change in
> grounded success (GSR bootstrap 95% CIs overlap for all three models).
> Terminal decision accuracy is more sensitive to evidence withdrawal:
> GPT-5.2's TDA falls from 0.76-0.79 (FULL/PHYSICAL_ONLY) to 0.48 when
> physical evidence -- the modality this benchmark's grounding contract
> requires -- is entirely absent (non-overlapping 95% CIs), and Mistral
> shows the same pattern (0.80 -> 0.59, non-overlapping). DeepSeek V4 Pro
> shows no such sensitivity (0.57 vs 0.54, CIs overlap) -- a genuine,
> disclosable between-model difference in how evidence-regime changes
> propagate to terminal decisions. GSR is analytically 0 whenever physical
> evidence is withheld, by construction of the evidence-grounding contract
> (Appendix, Section on scoring); we report TDA, not GSR, for the
> DIGITAL_ONLY comparison. Claude Sonnet 4.6 is excluded from this
> comparison: at n=80 worlds its output-validity rate for a terminal
> verdict is 5.0% (12/240), consistent with the step-2 protocol
> non-compliance finding at the original 48-episode scale.

## Outstanding gap

Qwen3.5-397B-A17B's expanded run is incomplete (20/240). Completing it
requires ~220 additional live model calls (real API cost) through the
existing, unmodified `scripts/phase8h2m_a_expanded_full_run.py
Qwen3.5-397B-A17B` -- not run automatically here; needs an explicit
go-ahead given the cost, consistent with this project's evaluation
constraints.

No frozen benchmark manifest, scoring contract, or runner code was changed
to produce this audit -- read-only verification and computation only,
except for using an already-completed sibling run (A-expanded-v1) that a
separate concurrent session built and executed.
