# A-family output-validity appendix

Kept separate from the main table's A = GSR(evaluable) column per the reviewed
decision: evidence-grounding capability != response/parser/output validity.

| Model | N_total | N_evaluable | A1 TDA (evaluable) | A2 GSR (evaluable, = main table A) | A3 output-validity rate | Root cause of exclusions |
|---|---|---|---|---|---|---|
| GPT-5.2 | 48 | 48 | 32/48 = 66.7% | 21/48 = 43.8% | 48/48 = 100.0% | none |
| Mistral Medium 3.5 | 48 | 48 | 33/48 = 68.8% | 22/48 = 45.8% | 48/48 = 100.0% | none |
| DeepSeek V4 Pro | 48 | 43 | 27/43 = 62.8% | 7/43 = 16.3% | 43/48 = 89.6% | 5/48 empty verdict: truncation/output-budget (finish_reason=length confirmed live on 1/5); 24/48 additional episodes had a logged call_error at an earlier turn but still produced a scored verdict (TDA/GSR both computed; kept in the lenient evaluable set per reviewed decision) |
| Qwen3.5-397B-A17B | 48 | 43 | 29/43 = 67.4% | 19/43 = 44.2% | 43/48 = 89.6% | 5/48 empty verdict: 2 truncation/output-budget, 3 infrastructure (network timeout/504) |
| Claude Sonnet 4.6 | 48 | 2 | 1/2 = 50.0% | 0/2 = 0.0% | 2/48 = 4.2% | 46/48 step2_protocol_noncompliance (genuine model behavior, confirmed live via full 46-episode re-run) |

## Sensitivity: hard-zero on missing outputs (NOT primary, appendix only)

| Model | TDA (hard-zero, N=48) | GSR (hard-zero, N=48) |
|---|---|---|
| GPT-5.2 | 32/48 = 66.7% | 21/48 = 43.8% |
| Mistral Medium 3.5 | 33/48 = 68.8% | 22/48 = 45.8% |
| DeepSeek V4 Pro | 27/48 = 56.2% | 7/48 = 14.6% |
| Qwen3.5-397B-A17B | 29/48 = 60.4% | 19/48 = 39.6% |
| Claude Sonnet 4.6 | 1/48 = 2.1% | 0/48 = 0.0% |

Full per-episode diagnostic: reports/benchmark/A_evaluable_verification.json, reports/benchmark/A_evaluable_verification_full.json, reports/benchmark/claude_A_audit.csv, reports/benchmark/DeepSeek_V4_Pro_A_audit.csv, reports/benchmark/Qwen3.5-397B-A17B_A_audit.csv.
