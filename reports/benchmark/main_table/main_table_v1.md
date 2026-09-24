| Model | A GSR (evaluable N) | B B_AGS (N=66) | C coverage† (N=9) | D CSA (N=70) | E CC_grounded† (N=18/6seq) |
|---|---|---|---|---|---|
| Claude Sonnet 4.6 | 0.000 [0.000, 0.000] [1] | 0.333 [0.227, 0.455] | 0.354 [0.178, 0.561] | 0.443 [0.329, 0.557] | 0.875 [0.625, 1.000] [2] |
| GPT-5.2 | 0.438 [0.292, 0.583] | 0.348 [0.242, 0.470] | 0.576 [0.357, 0.772] | 0.571 [0.457, 0.686] | 0.235 [0.059, 0.471] [3] |
| DeepSeek V4 Pro | 0.163 [0.070, 0.279] [4] | 0.515 [0.409, 0.636] | 0.176 [0.028, 0.417] | 0.357 [0.243, 0.471] | 0.429 [0.143, 0.714] [5] |
| Mistral Medium 3.5 | 0.458 [0.312, 0.604] | 0.682 [0.561, 0.788] | 0.391 [0.228, 0.585] | 0.500 [0.386, 0.614] | 0.667 [0.444, 0.889] [6] |
| Qwen3.5-397B-A17B | 0.442 [0.302, 0.581] [7] | 0.712 [0.606, 0.818] | 0.372 [0.220, 0.561] | 0.386 [0.271, 0.500] | 0.471 [0.235, 0.706] [8] |

Footnotes:
[1] n=14/288 (pooled, see below); pooled across the frozen-93 A pool (2/48 evaluable) and A-expanded-v1 (12/240 evaluable) -- 14/288 total evaluable, all step2_protocol_noncompliance elsewhere; see claude_A_audit_summary.json
[2] n_sequences=6; CC_grounded_applicable=8/18
[3] n_sequences=6; CC_grounded_applicable=17/18
[4] n=43/48 (output-validity 43/48=89.6%, see appendix); 5/48 empty verdict excluded (truncation/output-budget); see A_evaluable_verification.json for full diagnostic incl. 24 additional partial-evidence episodes retained per reviewed decision (lenient N)
[5] n_sequences=6; CC_grounded_applicable=14/18
[6] n_sequences=6; CC_grounded_applicable=18/18
[7] n=43/48 (output-validity 43/48=89.6%, see appendix); 5/48 empty verdict excluded (2 truncation, 3 infrastructure); see A_evaluable_verification.json
[8] n_sequences=6; CC_grounded_applicable=17/18

A is GSR (Grounded Success Rate) computed on the EVALUABLE episode set only (episodes with a valid, non-empty terminal verdict) -- NOT a hard-zero over all 48. Output-validity diagnostics (why some episodes are excluded, per-model root cause, TDA/A3 validity rate, and a hard-zero sensitivity appendix) are in reports/benchmark/A_evaluable_verification.json and reports/benchmark/main_table/A_appendix_v1.md -- not folded into this column.
