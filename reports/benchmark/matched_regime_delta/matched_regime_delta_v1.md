# Matched within-world FULL vs withheld-modality paired delta (A family)

16 worlds x 3 regimes (FULL/PHYSICAL_ONLY/DIGITAL_ONLY), same world_id, same gold.
Pairs excluded when either side has an empty verdict (matches the evaluable-N
convention used for the main table's A column).

| Model | vs PHYSICAL_ONLY: n incl/excl | mean TDA delta | mean GSR delta | | vs DIGITAL_ONLY: n incl/excl | mean TDA delta | mean GSR delta |
|---|---|---|---|---|---|---|---|
| Claude Sonnet 4.6 | 0/16 | n/a | n/a | | 0/16 | n/a | n/a |
| GPT-5.2 | 16/0 | -0.062 | -0.062 | | 16/0 | +0.125 | +0.625 |
| DeepSeek V4 Pro | 13/3 | +0.308 | +0.077 | | 14/2 | +0.143 | +0.286 |
| Mistral Medium 3.5 | 16/0 | +0.000 | +0.000 | | 16/0 | +0.000 | +0.688 |
| Qwen3.5-397B-A17B | 14/2 | +0.000 | +0.000 | | 12/4 | -0.167 | +0.583 |

Positive delta = FULL scores higher than the withheld-modality regime on the same world.
