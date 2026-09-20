# Audit: always-ESCALATE baseline provenance, and IoT-agreement claim defensibility

## 1. Always-ESCALATE A baseline (87.5%) -- provenance audit

**Question audited:** is the 42/48 ESCALATE gold skew (which puts every model's TDA
below the trivial always-ESCALATE baseline) a red flag about benchmark quality, or a
legitimate, documented design choice?

**Traced to source, not inferred.**
- `src/orchestrator/scenario_gen.py:204-229` (`derive_gold`) and `:324-349`
  (`derive_gold_fm7a`): a safety-first cascade -- ESCALATE if ANY of
  (technician present, active work order, physical reading out-of-band, or, for
  contradiction worlds, physical/IoT disagreement); COMMIT only if NONE apply.
- `ACTIVE_WO_RATE = 0.25`, `TECHNICIAN_PRESENT_RATE = 0.15` (`scenario_gen.py:61-62`).

**Verification arithmetic.** Of the four evidence-agreement cells (phys_in/out x
iot_agree/disagree), three are deterministically ESCALATE by construction (any
out-of-band reading, or any contradiction). Only phys_in+iot_agree can COMMIT, with
probability `(1-0.15)(1-0.25) = 0.6375`. Expected COMMIT worlds: `4 x 0.6375 ~= 2.55`
of 16 -> `~7.6` of 48 episodes. **Observed: 6/48.** This matches the documented rule
almost exactly.

**Conclusion: legitimate, not a bug.** The skew is a direct, arithmetically-verified
consequence of a documented, safety-first decision rule (mirroring the paper's own
cited motivation -- ANYbotics' 70% false-positive verification practice, EEMUA alarm
guidance), not sampling noise or a generator defect. It should be disclosed in the
paper text next to the baseline number (one sentence, provenance above), not treated
as something to fix. It does, however, confirm that **TDA is a structurally weak,
easily-gamed metric for A** -- reinforcing, not undermining, the decision to report
GSR as the primary A metric in the main table.

## 2. IoT-agreement claim (iot_agree vs iot_disagree GSR/TDA) -- defensibility audit

**Claim under test:** does contradicting telemetry corrupt grounding even when the
required (physical) evidence is fully delivered (FULL regime only, so GSR has no
structural floor -- see `AUDIT_NOTE_digital_only_gsr.md` for why that check mattered)?

**Point estimates** (`iot_agreement_split_v1.json`) suggested models ground *slightly
better*, not worse, under disagreement (e.g. GPT-5.2 0.50 -> 0.75, Mistral 0.625 ->
0.75), with DeepSeek flat.

**Bootstrap 95% CIs** (2000 resamples, seed 42, `iot_agreement_split_with_ci_v1.json`),
computed because n=7-8 per cell is small enough that point-estimate deltas of
0.125-0.25 could easily be noise:

| Model | iot_agree GSR [CI] | iot_disagree GSR [CI] | CIs overlap? |
|---|---|---|---|
| GPT-5.2 | 0.500 [0.125, 0.875] | 0.750 [0.375, 1.000] | yes |
| DeepSeek V4 Pro | 0.286 [0.000, 0.571] | 0.286 [0.000, 0.571] | yes |
| Mistral Medium 3.5 | 0.625 [0.250, 1.000] | 0.750 [0.375, 1.000] | yes |
| Qwen3.5-397B-A17B | 0.571 [0.143, 0.857] | 0.625 [0.250, 1.000] | yes |
| Claude Sonnet 4.6 | 0.000 [n/a, n=2] | n/a (n=0 evaluable) | n/a |

**Result: 0 of 5 models show a statistically defensible (non-overlapping-CI)
directional effect.** At the current N (7-8 evaluable episodes per cell), this
comparison cannot support a claim that contradictory telemetry helps or hurts
grounding, in either direction.

**Recommendation for the paper: do not make a directional claim from this
comparison.** The honest, defensible statement is: *"On the evaluable FULL-regime
episodes (n=7-8 per condition), we find no statistically distinguishable difference
in grounded success between IoT-agreement and IoT-disagreement worlds for any
model (bootstrap 95% CIs overlap in all 5 cases); a larger evidence-agreement
sample would be needed to test this directionally."* This is a legitimate,
disclosable finding in its own right (an absence of detected confusion effect,
correctly caveated by sample size) -- it should not be dressed up as either a
capability strength or a capability gap.

No frozen benchmark manifest, scoring contract, or runner code was changed to
produce this audit -- read-only source verification and read-only computation only.
