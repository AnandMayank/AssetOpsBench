# Claude Sonnet 4.6 — Full A-family validation run (Phase 8H.2L)

Fresh, independent re-run of all 48 canonical A episodes through the real, unmodified
`phase8h_live_pilot.run_a_episode` (same `SYSTEM_PROMPT`, same two-turn horizon, same
terminal-verdict requirement, same `metric_contract.score_episode` scoring). Raw
responses were persisted this time via a monkey-patched capturing wrapper around
`run_l3_pilot_executed._chat` — the original `_chat` function itself was called
unchanged, so this run's scoring behavior is identical in kind to the original run, not
a re-implementation. No prompt/scoring/parser/manifest code was modified. 0 infra
failures, 0 parser failures, 3 retries never triggered (all 48 episodes completed on
the first attempt).

Raw output: `reports/benchmark/v3_full_results/frozen93/claude_A_full_validation/
raw_Claude_Sonnet_4.6_validation.jsonl` (separate file; the original
`raw_Claude_Sonnet_4.6.jsonl` was not touched).

## 1. Did the full fresh run reproduce the original Claude behavior?

Yes, closely. Both runs show the same dominant failure mode at nearly the same rate:

| | Original run | Validation run |
|---|---|---|
| Terminal verdicts produced | 2/48 (4.2%) | 4/48 (8.3%) |
| Terminal-step `tool_calls` (protocol noncompliance) | 46/48 (95.8%) | 44/48 (91.7%) |
| Infrastructure failures | 0/48 | 0/48 |
| TDA | 1/48 = 0.0208 | 2/48 = 0.0417 |
| GSR | 0/48 = 0.0000 | 0/48 = 0.0000 |

44 of 48 episodes (91.7%) show the **identical terminal-type classification** across
both runs (`claude_A_validation_episode_comparison.csv`). Of the 4 that differ: 3
flipped `tool_calls -> verdict` (the model complied on the second run where it hadn't
originally) and 1 flipped `verdict -> tool_calls` (the reverse). This is consistent
with ordinary LLM sampling variance rather than a systematic difference — the router
does not guarantee bit-for-bit determinism at `temperature=0`, and this same
stochasticity was already observed and flagged during the diagnostic phase (one
episode, `A::A-metro_pump_1-phys_out__iot_agree-3010::FULL`, complied only on a
diagnostic retry).

## 2. How many of 48 produced terminal verdicts?
**4/48 (8.3%).**

## 3. How many produced terminal-step tool_calls?
**44/48 (91.7%)** — all well-formed `{"tool_calls": [...]}` responses at the mandatory
step-2 turn, the same protocol-noncompliance pattern documented in the prior audit
(`claude_toolcall_protocol_audit.md`). None were executed (per instruction: tool_calls
at the terminal step are never dispatched, matching the original runner's behavior and
the frozen protocol).

## 4. How many were infrastructure failures?
**0/48.** All 48 episodes completed with an HTTP 200 and a parseable response on the
first attempt; no retries were needed.

## 5. What are TDA and GSR?
**TDA = 2/48 = 0.0417. GSR = 0/48 = 0.0000.**

## 6. Is N=48 defensible?
Yes. All 48 attempts are valid, scored rows with 0 exclusions and 0 infrastructure
failures. N was neither manufactured up nor trimmed down.

## 7. How many episodes changed behavior relative to the original run?
**4/48 (8.3%)** — 3 `tool_calls -> verdict`, 1 `verdict -> tool_calls`, 0 involving an
infrastructure-failure difference. See §1 for interpretation (sampling variance, not a
systematic discrepancy).

## 8. Does the fresh run support retaining the original conclusion?
Yes. Both runs agree on the central finding: under the frozen A protocol's fixed
two-turn horizon, Claude Sonnet 4.6 overwhelmingly (92–96% across the two runs) replies
at the mandatory terminal decision turn with a further tool-evidence request rather
than the required verdict schema, producing a near-zero TDA and a GSR of exactly zero
in both runs. The small (8.3%) episode-level flips are consistent with ordinary
non-determinism, not with a different underlying behavior pattern, a protocol defect,
or an infrastructure problem — no discrepancy requiring investigation was found.

## Final decision

Per the instruction's decision rule ("if they agree closely, use the fresh full run as
the primary reported Claude A evaluation and retain the original run as replication
evidence"):

- **Primary reported Claude A result (frozen, going forward): N=48, TDA=2/48=0.0417,
  GSR=0/48=0.0000**, from this validation run.
- The original run (N=48, TDA=1/48=0.0208, GSR=0/48=0.0000) is retained as replication
  evidence, not discarded and not merged (N is never combined to 96 — these are two
  replicate runs of the same 48 scenarios, not independent samples).
- No p-value was computed between the two runs, per instruction.
- No protocol, scoring, prompt, or manifest change was made.
