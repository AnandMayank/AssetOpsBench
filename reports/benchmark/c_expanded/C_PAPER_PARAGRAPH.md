# Procedural execution (C) — paper paragraph and placement

## Main text (Section 6.2, insert after the D paragraph and before Section 7)

**Procedural execution (C).** Dimension C asks whether an agent executes the required tool
sequence in the order a scenario demands, and separates that from mere procedural coverage:
since ORDERING checks precedence only among calls that actually ran, it is vacuously satisfied
when no required tool ran at all -- which is why PROC (procedural coverage) is reported
separately rather than folded into it. We evaluate five models on 17 canonical C episodes (the
original 9-scenario frozen pool plus 8 additional, independently-authored, already-validated
fixtures not previously included). DeepSeek V4 Pro obtains ordering_satisfied = 1.00 while its
procedural coverage is 0.089 -- it executes almost none of the required tools, so the ordering
check is trivially true over an empty overlap. GPT-5.2 and Qwen3.5-397B-A17B obtain coverage of
0.540 and 0.315 respectively, both below their own ordering scores (0.647 each), the same
direction of gap on a smaller scale. This confirms, with a concrete example rather than only a
design justification, why grounded coverage and ordering must be reported as separate
quantities: ordering alone can overstate procedural competence for an agent that simply does
less. Claude Sonnet 4.6's output-validity rate on this dimension is 53% (9/17), consistent with
the step-2 protocol non-compliance already reported for the A and E families; scores here are
computed only over evaluable episodes, and confidence intervals (95%, bootstrap, 2,000
resamples) remain wide at this sample size -- full per-model results are in the supplementary
source.

## Why this framing, not a longer one

- Leads with the DeepSeek ordering-vs-coverage contrast, the one finding strong enough to stand
  without a caveat -- it is a real number, not a hypothesis, and it directly validates a
  methodological design choice the paper already states (Section 5.4: "ORDERING ... can be
  vacuously true when no required tool ran, which is why PROC is reported separately").
- States Claude's output-validity as one sentence, explicitly cross-referenced to the same
  mechanism already disclosed for A and E, rather than re-explaining step-2 non-compliance from
  scratch -- avoids the paper repeating the same diagnosis three times in full.
- Explicitly flags that CIs remain wide at n=17 -- this is the ceiling of already-authored,
  non-control, non-twin content (see the pool-construction audit below); going further requires
  new fixture authoring, a separate scope decision, not something claimed as done here.

## Appendix (full per-model table, methodology, and pool construction)

| Model | Output validity | Coverage [95% CI] | Ordering [95% CI] | CC [95% CI] |
|---|---|---|---|---|
| Claude Sonnet 4.6 | 53% (9/17) | 0.232 [0.083, 0.454] | 1.000 [1.000, 1.000] | 0.889 [0.667, 1.000] |
| GPT-5.2 | 100% (17/17) | 0.540 [0.366, 0.695] | 0.647 [0.412, 0.882] | 0.706 [0.471, 0.882] |
| DeepSeek V4 Pro | 94% (16/17) | 0.089 [0.000, 0.224] | 1.000 [1.000, 1.000] | 0.563 [0.313, 0.813] |
| Mistral Medium 3.5 | 88% (15/17) | 0.312 [0.161, 0.493] | 0.733 [0.533, 0.933] | 0.667 [0.400, 0.867] |
| Qwen3.5-397B-A17B | 100% (17/17) | 0.315 [0.186, 0.462] | 0.647 [0.412, 0.882] | 0.588 [0.353, 0.824] |

Bootstrap 95% CIs, 2000 resamples, seed 42. Computed only over evaluable episodes (verdict
present, not a step-2/apparatus_failure record). Claude's evaluable set: 9/17 (apparatus-failure
scenarios: R001, R006, R007, R017, R023, R024, R022, R025). DeepSeek's evaluable set: 16/17
(apparatus_failure: R024). Mistral's evaluable set: 15/17 (apparatus_failure: R021, R025).

**Pool construction.** C-expanded-v1 adds 8 scenarios (R008, R010, R011, R012, R014, R021, R022,
R025) to the original frozen-93 9-scenario C pool, drawn from 45 total hand-authored fixtures
(`classc_fixtures.py`) of which only 9 were in the frozen manifest. Each candidate was verified
against two conditions before inclusion: (1) `classc_audit.audit(sid)` parses a non-empty
`required_order` and a gold verdict from the scenario's `groundtruth.txt`
(`AssetOpsBenchScenarioGeneration/RobotInspection`), and (2) the scenario id is present in
`couchdb_executor.SCENARIO_PHYSICAL` (a real, wired-up hidden-state entry, not an orphaned
fixture definition). 11 "ordering-unconstrained control" twins (R059, R060, R064-R072) were
explicitly excluded -- they are deliberately designed to test whether ordering constraints
matter at all, not to add independent capability-measurement data, and mixing them in would
conflate a validity check with a real sample. 15 "de-leaked twin" repairs (R073-R087) were also
excluded -- patched replacements of already-used originals, not independent new scenarios.

No frozen benchmark manifest, scoring contract, or runner code was changed to produce this
result.
