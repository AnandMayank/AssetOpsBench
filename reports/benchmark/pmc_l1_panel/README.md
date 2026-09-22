# PMC real-image (L1-equivalent) panel evaluation

**Status: NEW supplementary result. Does not modify, replace, or get merged into any existing
result file** (`main_table_v1.json`, `paper_results_v1.json`, `capability_profile_v1.json`, or any
frozen A–E capability number). This is a separate evidence track (real photographs, not the
synthetic A–E families) scored by a separate grader (`grade_real_pmc`, not TDA/GSR), run for the
first time against the paper's actual five-model primary panel (the only prior run used
`claude-opus-4.8`, a model not in the panel).

## Why this evaluation exists

`evidence_provenance_class` (L1/L2/L3) is real in `observation_records.json`'s 24,601-record
replayed-evidence catalog, but L1 (same-asset capture) is structurally 0 there — see
`reports/benchmark/abstract_crosscheck/section3_reproducibility_fixes.md` Item 7 and the
provenance-gated GSR diagnostic (`provenance_gated_gsr_diagnostic.json`, same directory). The
canonical four grounded assets have no real-photo evidence wired into scoring, and no ground truth
exists for the 116 facility images to legitimately construct new gold labels. The PMC/AOBv2-REAL
track (`src/orchestrator/run_pmc_benchmark.py`) is the one place in this project where real,
same-instrument photographic evidence already has real annotated ground truth and a real scorer —
just never run against this paper's actual panel. This evaluation fills that gap without inventing
any new gold labels or touching the frozen benchmark.

## Run details

- Backend: `tokenrouter`, models: `anthropic/claude-sonnet-4.6`, `openai/gpt-5.2`,
  `deepseek/deepseek-v4-pro`, `mistralai/mistral-medium-3-5`, `qwen/qwen3.5-397b-a17b` — the exact
  five-model primary panel used throughout the rest of the paper.
- N = 14 real scenarios per model (`AOBv2-REAL-002/003/004/005/006/007/008/009/010/011/013/015/016/018`),
  `min_reads=3`, `prompt_variant=baseline`, `framing=neutral`, single run per model (temp/seed fixed
  by the script), real API calls, run 2026-09-23.
- Grader: `grade_real_pmc` (`src/orchestrator/grader.py`) — pass requires the terminal action to
  match ground truth (FLAGGED with the correct recommended action, or COMMIT within 5% of the true
  gauge span), no forbidden action, no perceive-commit gap, and a valid run.
- Raw per-model output: `pmc_l1_results_<model>.json` (this directory), each a full
  `reports/pmc_benchmark_results.json`-schema file including per-scenario detail.
- Consolidated: `pmc_l1_panel_summary.json` (this directory) — includes a scenario-by-scenario
  pass/fail cross-check across all 5 models, not just the aggregate rate.

## Results

| Model | N | Pass rate | Perceive-commit gap | Forbidden-action | RSA (recovery-selection accuracy) |
|---|---|---|---|---|---|
| Claude Sonnet 4.6 | 14 | 78.6% | 0.0% | 0.0% | **64.3%** |
| GPT-5.2 | 14 | 78.6% | 0.0% | 0.0% | **57.1%** |
| DeepSeek V4 Pro | 14 | 78.6% | 0.0% | 0.0% | **57.1%** |
| Mistral Medium 3.5 | 14 | 78.6% | 0.0% | 0.0% | **71.4%** |
| Qwen3.5-397B-A17B | 14 | 78.6% | 0.0% | 0.0% | **71.4%** |

## The finding, and why pass rate should NOT be reported as a capability differentiator here

Verified scenario-by-scenario, not just at the aggregate level: **all five models pass and fail the
identical 11 of 14 scenarios.** Pass/fail on this track is structurally determined by the scripted
acquisition policy (`real_pmc_orchestrator.py`) and its fixed safety gates — readable=false
scenarios always resolve to FLAGGED regardless of which model reads them (there is no numeric value
to commit), and readable=true failures are blocked by fixed IoT-consistency gates before any model
choice matters. This is not a data artifact to hide; it is the same terminal-vs-grounded distinction
this paper already makes for TDA vs. GSR on Dimension A, now independently observed on real,
non-synthetic evidence.

**Recovery-selection accuracy (RSA) is the metric that genuinely differentiates models** on this
track, since it reflects which corrective action (`agentic_zoom` / `reposition` /
`flag_recommended_action`) a model chooses once blocked — a real per-model judgment call not gated
by the scripted policy. Range: 57.1% (GPT-5.2, DeepSeek V4 Pro) to 71.4% (Mistral Medium 3.5,
Qwen3.5-397B-A17B).

## Suggested paper text

> **Real-image evidence (AOBv2-REAL, L1-equivalent).** We additionally evaluate the primary panel
> on the benchmark's real-photograph track (N=14 real scenarios per model), the one evidence source
> in this release with genuine same-instrument physical capture. All five models achieve an
> identical 78.6% pass rate with 0% perceive-commit gap and 0% forbidden-action rate — outcome
> success on this track is dominated by the scripted acquisition policy's safety gates rather than
> model judgment, the same terminal-vs-grounded pattern already observed for Dimension A, now
> independently reproduced on real evidence. The metric that does differentiate models is
> recovery-selection accuracy (which corrective action a model chooses once blocked): 57.1%
> (GPT-5.2, DeepSeek V4 Pro) to 71.4% (Mistral Medium 3.5, Qwen3.5-397B-A17B), with Claude Sonnet
> 4.6 at 64.3%.

**Placement recommendation:** a new short subsection in Section 4 (e.g. "4.x Real-image evidence
diagnostic") or a supplementary table in the appendix, cited from the L1/L2/L3 provenance paragraph
in 3.1 as the pointer promised there ("see Section 4.x for a real-evidence evaluation"). Do not
merge these numbers into the main A–E capability table (Figure 5/6/7) — different evidence source,
different grader, different N, not comparable to TDA/GSR/B_AGS/CSA/CC_grounded on the same axis.
