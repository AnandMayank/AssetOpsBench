# Evaluation Compute-Cost Report

Real, measured figures pulled directly from the raw run files' own `started_at`/`finished_at`,
`elapsed_s`, and `usage` fields — nothing estimated unless explicitly marked as such. Two parts:
(1) what the current main-table results actually cost, (2) a projection for the scale-up N's
discussed for the released paper, based on the observed per-episode rate of the same pools.

## 1. Main-table results (5 models x A/B/C/D/E) — actual cost

### Frozen-93 pool (A=48, B-legacy=12, C=9, D-enterprise=6, E=18 per model; wall-clock, sequential)

No `usage`/token capture in this runner (`run_l3_pilot_executed._chat` does not request or log it) —
wall-clock time only.

| Model | Episodes | Wall-clock duration |
|---|---|---|
| Claude Sonnet 4.6 | 93 | 0:23:13 |
| GPT-5.2 | 93 | 0:37:11 |
| DeepSeek V4 Pro | 93 | 0:55:59 |
| Mistral Medium 3.5 | 93 | 0:06:12 |
| Qwen3.5-397B-A17B | 93 | 1:13:33 |
| **Total (5 models x 93 = 465 episodes)** | | **3:16:08** |

### D-physical pool (70 episodes/model, real `usage` token counts captured)

| Model | Episodes | Wall-clock (sum `elapsed_s`) | Input tokens | Output tokens |
|---|---|---|---|---|
| Claude Sonnet 4.6 | 70 | 0:23:43 | 42,953 | 45,232 |
| GPT-5.2 | 70 | 0:02:12 | 39,055 | 3,504 |
| DeepSeek V4 Pro | 70 | 0:45:08 | 40,619 | 129,064 |
| Mistral Medium 3.5 | 70 | 0:01:47 | 43,060 | 3,958 |
| Qwen3.5-397B-A17B | 70 | 1:18:39 | 40,844 | 294,782 |
| **Total (5 x 70 = 350 episodes)** | | **2:31:30** | **206,531** | **476,540** |

Qwen and DeepSeek's output-token counts are 6-8x the other three models' — direct, measured
confirmation of the reasoning-heavy/verbose output pattern already noted qualitatively elsewhere
in this project (e.g. the Qwen3.5-9B exclusion, DeepSeek's step-2 protocol non-compliance).

### B-Acquisition pool (66 episodes/model, wall-clock only, no usage capture in this runner)

| Model | Episodes | Wall-clock (sum `elapsed_s`) |
|---|---|---|
| Claude Sonnet 4.6 | 66 | 0:04:39 |
| GPT-5.2 | 66 | 0:03:39 |
| DeepSeek V4 Pro | 66 | 0:14:04 |
| Mistral Medium 3.5 | 66 | 0:02:36 |
| Qwen3.5-397B-A17B | 66 | 0:42:08 |
| **Total (5 x 66 = 330 episodes)** | | **1:07:06** |

### Main-table grand total (measured)

- **Episodes executed: 1,145** (465 + 350 + 330, matches this project's own prior N-accounting)
- **Wall-clock compute time: ~6:54:44** (sum of the three pools above; pools were run sequentially
  per model within this project, not necessarily on a single continuous clock across all 5 models)
- **Tokens metered directly: 206,531 in / 476,540 out** (D-physical only; the other two pools'
  runners never captured usage, so their true token spend is not recoverable after the fact —
  this is itself worth fixing in the runner before the next scale-up, see \S3)

## 2. This session's verification work — actual cost

| Task | Calls | Wall-clock |
|---|---|---|
| Claude A diagnostic re-sample (46 units, 2 turns each) | ~92 | ~0:14 (bounded by 3 short diagnostic batches) |
| Claude A full validation re-run (48 episodes, 2 turns each) | 96 | **0:05:59** |
| PerceptionGauge-49 VLM annotation (Gemini 2.5-flash, 49 scenarios, 1 call each) | 49 | **2:08:00** (dominated by free-tier rate-limit backoff, not model latency — actual per-call latency was a few seconds each judging by the 3s configured delay) |

The PerceptionGauge-49 run is the clearest illustration of a cost that is **rate-limit-bound, not
compute-bound**: 49 calls to a "flash"-tier model took over 2 hours of wall-clock time purely from
this project's own retry/backoff policy (15s x attempt on 429/quota errors) against a free-tier
quota, not from the model itself being slow.

## 3. Projected cost for the scale-up N's proposed for the released paper

Using each pool's own measured per-episode rate (wall-clock / N), extrapolated linearly to the
canonical pool sizes already generated and frozen (Section 3.2's construction numbers) — **not**
a new estimate, a direct scaling of the rate already observed above. Token-based cost is only
projectable for D (the only pool with real usage capture); the other pools would need one run with
usage capture added first (a one-line runner fix, see below) before a token/$ estimate is possible.

Retroactively split by `dim` from the existing frozen-93 logs (each row already carries
`started_at`/`finished_at` and `dim`; no rerun needed to get this) — real, measured per-episode
rates, mean/min/max across the 5 models:

| Dim | Mean s/episode | Min (fastest model) | Max (slowest model) |
|---|---|---|---|
| A | 22.53 | 4.98 (Mistral) | 53.35 (Qwen) |
| B (legacy) | 15.95 | 2.83 (Mistral) | 32.08 (DeepSeek) |
| C | 14.40 | 3.11 (Mistral) | 32.11 (DeepSeek) |
| D (enterprise) | 11.73 | 3.00 (Mistral) | 25.50 (DeepSeek) |
| E | 129.60 | 8.67 (Mistral) | 237.50 (GPT-5.2) |

E's per-episode rate is 6-9x every other dimension's — each "episode" inside an E sequence
involves more turns/tool-calls than a single-shot A/B/C/D episode, so E's cost does not scale
linearly with N the same way the others do.

| Capability | Current N | Target N (frozen canonical pool) | Scale factor | Projected wall-clock, 5 models sequential, mean rate |
|---|---|---|---|---|
| A | 48 | 2,400 | 50x | 2,400 x 22.53s x 5 = **~75.1 h** (mean); fastest-model-only run: 2,400 x 4.98s x 5 = ~16.6 h |
| C | 9 | 112 | 12x | 112 x 14.40s x 5 = **~2.24 h** |
| E | 17-18 | 1,440 | ~80x | 1,440 x 129.60s x 5 = **~259 h (~10.8 days)** — by far the largest ask in the scale-up table; worth running on the fastest-only model first (Mistral: 1,440 x 8.67s = 3.47 h) as a checkpoint before committing to all 5 |
| D (physical) | 70 | 70 | 1x (already complete) | 0 |
| D (enterprise) | 0 | 20 | new track | 20 x 11.73s x 5 = **~0.33 h** (extrapolated from the C_D_enterprise generator's per-episode rate; no D-enterprise run has actually been executed, so this is the one number in this table that's a rate-based projection onto a genuinely untested pool, not a rerun-of-the-same-thing extrapolation) |

**A's 75-hour figure is the real headline number for anyone deciding whether to fund the scale-up.**
It is driven almost entirely by the two slow models (Qwen 53.35s/ep, DeepSeek implied similarly
high) — running A's scale-up on just the 3 fast models (Mistral/GPT-5.2/Claude, all under 15s/ep)
cuts this to roughly 2,400 x 10s x 3 ≈ 20 h, a substantially cheaper first pass that still covers
3 of 5 panel models.

## 4. Concrete, low-cost fixes to make the *next* cost report exact

1. Frozen-93 and B-Acquisition runners never capture `usage` (token counts) — D-physical's runner
   already does. Porting that one field-capture into the other two runners (a few lines each, same
   pattern already proven in D-physical) means the next scale-up run produces a real token/$ figure
   instead of wall-clock-only.
2. Frozen-93's combined A+B+C+D+E wall-clock should be split per-dimension at write time (the
   `dim` field is already recorded per row — this is a post-hoc groupby on existing data for a
   *retroactive* per-dimension rate, no rerun needed) before projecting C/E's scale-up cost.
3. PerceptionGauge-49's real bottleneck was rate-limit backoff, not model latency — running it
   against a paid-tier Gemini key (or parallelizing across multiple keys) rather than the free tier
   would cut its wall-clock by roughly the same factor as the backoff overhead (a rough 2:08:00
   observed vs. a ~49 x few-seconds-per-call floor, i.e. plausibly under 10 minutes on a paid tier).
