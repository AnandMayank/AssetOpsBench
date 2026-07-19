# Reproducing: PMC gauge-inspection data → CosmosWorld → conformal-prediction calibration

This documents the exact steps (and the two real bugs found + fixed along the
way) to go from raw PMC gauge-inspection photo bursts to a trained
`CosmosWorld` checkpoint and real conformal-prediction (CP) thresholds, using
[NVIDIA's Cosmos-Tokenizer](https://github.com/NVIDIA/Cosmos-Tokenizer) and
the [GaugeFailClassification](https://github.com/autoinspection-classification/GaugeFailClassification)
repo's `CosmosWorld`/`CosmosWorldClassifier` models.

**Current status: pipeline is fully real and runs end-to-end (training →
calibration → OOD scoring), but the checkpoint trained here (3 epochs, 678
frame pairs) shows no measurable success/failure separation yet** — see
[Known result](#known-result-no-separation-yet) below. This is included
un-sanitized so whoever continues this can pick up from a verified-correct
baseline instead of re-debugging the same issues.

**Hardware note**: NVIDIA's own Cosmos-Tokenizer model card states BF16 was
only tested on Ampere (A100) and Hopper (H100) GPUs. Everything below was run
successfully (no crashes, sane non-NaN losses) on a Quadro RTX 6000 (Turing,
24GB) — an untested-but-working configuration, not an officially validated
one. An H100/A100 is **not required** to reproduce this; it would mainly help
with (a) speed at longer training runs, and (b) matching NVIDIA's officially
tested precision path exactly.

## 0. Prerequisites

- A GaugeFailClassification checkout (`git clone git@github.com:autoinspection-classification/GaugeFailClassification.git`)
- The patches/new files in [`gauge_fail_classification_changes/`](gauge_fail_classification_changes/)
  applied on top of it (see [Bug fixes](#bug-fixes-applied) below)
- ~5GB free disk for the Cosmos checkpoint + rendered videos + frame windows
- A CUDA GPU with ≥16GB VRAM (24GB used here)

## 1. Environment setup

This repo's `pyproject.toml` requires Python ~3.10 (`cosmos_tokenizer` itself
also declares `python_requires>=3.10`), so build a dedicated venv:

```bash
python3.10 -m venv /path/to/envs/gauge_train310
source /path/to/envs/gauge_train310/bin/activate
pip install --upgrade pip

# CUDA 12.1 torch build (match to your driver; 580.x driver supports this)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# GaugeFailClassification's declared deps
pip install "lightning>=2.5.1.post0" matplotlib numpy pandas "wandb>=0.19.10" \
  "piq>=0.8.0" pyarrow scikit-learn huggingface-hub opencv-python av lpips tqdm \
  ffmpegio moviepy

# cosmos_tokenizer's own transitive deps (not pulled automatically)
pip install loguru mediapy einops imageio
```

Install the real NVIDIA Cosmos-Tokenizer package (no patching needed on
Python 3.10, unlike an earlier Python 3.8 attempt that required a
`from __future__ import annotations` patch for PEP 585 generics):

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/NVIDIA/Cosmos-Tokenizer.git /path/to/cosmos_tokenizer_src
# cosmos_tokenizer_src is used via PYTHONPATH below, not `pip install -e`,
# to avoid pulling its git-LFS test video assets.
```

Download the `Cosmos-0.1-Tokenizer-CI16x16` checkpoint (encoder.jit,
decoder.jit, autoencoder.jit — ~327MB) and symlink it into the path
`CosmosWorld`/`CosmosWorldClassifier` expect:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='nvidia/Cosmos-0.1-Tokenizer-CI16x16', local_dir='/path/to/cosmos_ckpts/Cosmos-0.1-Tokenizer-CI16x16')"

mkdir -p /path/to/GaugeFailClassification/assets/pretrained_cosmos_ckpts
ln -s /path/to/cosmos_ckpts/Cosmos-0.1-Tokenizer-CI16x16 \
  /path/to/GaugeFailClassification/assets/pretrained_cosmos_ckpts/Cosmos-0.1-Tokenizer-CI16x16
```

Always run GaugeFailClassification scripts with `cosmos_tokenizer_src` on
`PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/cosmos_tokenizer_src
```

## 2. Data pipeline (PMC photo bursts → labeled, split, rendered windows)

This repo's `src/orchestrator/data/` has the scripts (paths inside them are
hardcoded to this machine's layout — adjust before reuse):

1. **`split_windows_by_scene_full.py`** — splits raw timestamp-clustered PMC
   photo bursts into single-asset windows via grayscale histogram
   scene-change detection (`cv2.compareHist(..., HISTCMP_CORREL)`,
   threshold 0.55). Produces `pmc_windows_clean_full/manifest.json` + one
   frame directory per window. On the full archive (1,286 frames) this
   produced 180 windows.

2. **`label_windows_vlm.py`** — labels each window `success`/`failure` via:
   - ground-truth match against `perception_real.csv` where available, else
   - one whole-window VLM call (up to 4 evenly-sampled frames spanning the
     full window, first+last always included) — judges the INSPECTION
     OUTCOME across the whole sequence, not just the last frame, since PMC
     bursts sometimes zoom IN toward the gauge (last frame clearest) and
     sometimes pan AWAY (last frame worst).
   - Supports two backends: `VLM_BACKEND=gemini` (direct REST, no SDK — the
     `google-generativeai` SDK versions exposing `GenerativeModel` all need
     Python≥3.9+) or `VLM_BACKEND=tokenrouter` (OpenAI-compatible chat
     completions via `tokenrouter_backend.py`, used here to route around a
     Gemini free-tier quota wall — `generate_content_free_tier_requests,
     limit: 20`, which reset on a shorter/tighter window than expected).
   - On this run: 180/180 windows labeled (7 ground_truth + 173 vlm — 20 via
     Gemini before quota exhaustion, 153 via TokenRouter). 140 success / 40
     failure.

3. **`render_success_windows_to_video.py`** / **`render_failure_windows_to_video.py`**
   — render each labeled window's frames to a low-fps `.mp4` (2fps here,
   since PMC windows are short photo bursts, not native video — 2-18 frames
   per window) via `GaugeFailClassification/scripts/data_processing/frames_to_video.py`.
   Success-labeled windows → calibration set (140 videos, 545MB). Failure-labeled
   windows → OOD test set (40 videos, 142MB).

## 3. Bug fixes applied

See [`gauge_fail_classification_changes/`](gauge_fail_classification_changes/)
for the actual files/patch. Two are correctness bugs found while building
this pipeline, not stylistic changes:

1. **Image value range: `[0,1]` → `[-1,1]`** (`pmc_window_dataset.py`).
   `cosmos_tokenizer.image_lib.ImageTokenizer.encode()`'s docstring states
   its input contract is `Bx3xHxW, range [-1, 1]`, and this repo's own
   `SSIMLoss` (`src/models/general.py`) explicitly assumes `[-1,1]` inputs
   (`input = (input + 1.0) / 2.0`). The original `PMCWindowDataset._load_image`
   returned raw `img/255.0` (`[0,1]`) with no further rescale — feeding the
   frozen Cosmos encoder out-of-distribution inputs on every call. Fixed by
   rescaling only the tensors fed to the encoder (`img * 2.0 - 1.0`), leaving
   the `[0,1]` numpy arrays used for `build_state_obs`'s quality-feature
   proxy (`_quality_features` assumes `[0,1]`) unchanged.

2. **Mahalanobis conformal threshold: near-singular covariance → shrinkage**
   (`scripts/inference/calibrate.py`, `compute_calibration_stats()`). The
   latent space here is 4992-dim (`16 channels × 13 × 24`). With
   `ensure_length_300()` padding every short (2-18 frame) PMC video up to 300
   frames by repeating the last frame, the empirical `np.cov` over the
   calibration set is heavily rank-deficient, and `np.linalg.inv()` on it
   silently returns numerically meaningless values — observed as an
   **impossible negative Mahalanobis threshold (-5,069,454)**, when Mahalanobis
   distance (a quadratic form) can never be negative. Fixed by switching to
   `sklearn.covariance.LedoitWolf` shrinkage, which is well-conditioned and
   always invertible regardless of the sample/dimension ratio. Result went
   from -5,069,454 to a sane 3,968.09.

3. **`configure_optimizers()` return-type mismatch** (only affects a
   manual (non-`pl.Trainer`) training loop, not `calibrate.py`/`classify.py`
   themselves): `CosmosWorldClassifier.configure_optimizers()` returns a
   `{"optimizer":..., "lr_scheduler":...}` dict, but `CosmosWorld.configure_optimizers()`
   returns the bare `Adam` optimizer — different contract between the two
   sibling classes. `train_cosmos_world_pmc.py` (below) handles this.

4. Also worth knowing if you drive `training_step()`/`_generic_step()`
   directly instead of through a real `pl.Trainer`: neither `forward()`
   moves the `labels` tensor to the model's device (only `s_images`/`s_obs`/`a`
   are moved inside `forward()`) — a real `Trainer` does this via its
   `transfer_batch_to_device` hook automatically; a manual loop must do it
   itself (see `move_batch_to_device`/`move_tuple_to_device` helpers in the
   two training/test scripts here).

## 4. Training `CosmosWorld` (the unsupervised world model `calibrate.py` needs)

**Important distinction**: `calibrate.py`/`classify.py` are built around
`CosmosWorld` (unsupervised next-frame-reconstruction/anomaly model), *not*
`CosmosWorldClassifier` (a separate, supervised BCE classifier). They share
the same frozen Cosmos encoder + `LatentWorld` backbone but are different
`LightningModule`s with different heads/objectives. `CosmosWorld` needs no
labels, so it trains on all 180 real-labeled windows (678 frame pairs total)
regardless of success/failure — only the later *calibration* step restricts
to success-only videos.

`train_cosmos_world_pmc.py` (copied into
[`gauge_fail_classification_changes/`](gauge_fail_classification_changes/))
does a manual training loop (not `pl.Trainer`, for direct control/logging):

```bash
export PYTHONPATH=/path/to/cosmos_tokenizer_src
cd /path/to/GaugeFailClassification
EPOCHS=3 BATCH_SIZE=8 python -u train_cosmos_world_pmc.py
# env vars: PMC_CLEAN_DIR (default: pmc_windows_clean_full), COSMOS_WORLD_CKPT_OUT,
# EPOCHS, BATCH_SIZE
```

Result on this run (3 epochs, 84 batches/epoch, batch_size=8, Quadro RTX
6000): mean_loss **2.2446 → 2.0307 → 1.9401** — real, consistent downward
trend, no NaN/explosion, ~280s/epoch (~14 min total for all 3 epochs).
Checkpoint saved as `{"state_dict": model.state_dict()}` (compatible with
`calibrate.py`'s `torch.load(...)['state_dict']` loading).

## 5. Conformal-prediction calibration

```bash
cd /path/to/GaugeFailClassification
PYTHONPATH=/path/to/cosmos_tokenizer_src python scripts/inference/calibrate.py \
  --model_checkpoint /path/to/cosmos_world_pmc.ckpt \
  --calibration_dir /path/to/pmc_calibration_videos_success \
  --output_dir /path/to/cp_bands \
  --alpha 0.05
```

Real thresholds obtained (95th percentile, α=0.05, 140 calibration videos),
**after** the Mahalanobis fix:

| Metric | Threshold |
|---|---|
| `reconstruction_error` | 2.1388 |
| `training_loss` | 2.8209 |
| `mahalanobis` | 3968.09 |
| `l2_to_mean` | 83.9382 |
| `cosine_to_mean` | 0.8274 |
| `latent_pred_error` | 64.4365 |
| `latent_std` | 0.8538 |

## 6. OOD scoring (does it actually separate success from failure?)

```bash
cd /path/to/GaugeFailClassification
PYTHONPATH=/path/to/cosmos_tokenizer_src python scripts/inference/classify.py \
  --model_checkpoint /path/to/cosmos_world_pmc.ckpt \
  --test_dir /path/to/pmc_test_videos_failure \
  --bands_dir /path/to/cp_bands \
  --output_dir /path/to/ood_results_failure
```

(Note: despite the docstring at the top of `classify.py` calling itself
`score_ood.py`, that's the actual, current filename to run.)

### Known result: no separation yet

Running this against the 40 known-failure videos:

| Metric | OOD flagged (of 40) |
|---|---|
| `reconstruction_error` | 1 (2.5%) |
| `training_loss` | 4 (10.0%) |
| `mahalanobis` | 0 (0.0%) |
| `l2_to_mean` | 1 (2.5%) |
| `cosine_to_mean` | 1 (2.5%) |
| `latent_pred_error` | 2 (5.0%) |
| `latent_std` | 3 (7.5%) |

Since the threshold is itself the calibration set's 95th percentile, ~5% of
success videos trigger OOD by construction — a well-separated model should
flag failure videos far above that baseline (ideally near 100%). Every
metric here sits at or near 5%, and mean/median score distributions for
success vs. failure overlap almost completely per-metric (checked directly,
not just via the flag counts) — `mahalanobis` is even inverted (failure mean
666 vs. success mean 1418, the wrong direction for anomaly detection).

**This is a genuine negative result, not a pipeline bug** — every stage
(training → calibration → OOD scoring) runs correctly and produces sane,
well-formed numbers; the trained model just hasn't learned a usable
success/failure signal yet. Most likely causes, roughly in order of expected
impact:

1. **Far too little training** — 3 epochs on 678 pairs is minimal for a
   `LatentWorld` predictor trained from scratch (only the Cosmos
   encoder/decoder are pretrained; the world-model head is random-init).
   Next step: run substantially longer — 50-100+ epochs. At ~93s/epoch-per-10-batches
   observed rate this is a few hours on a single 24GB GPU, not something
   that requires bigger hardware, just more wall-clock time (or an
   A100/H100 to shorten that wait and match NVIDIA's officially-tested BF16
   path).
2. **`ensure_length_300` padding dilutes the temporal signal** — PMC windows
   are native 2-18 frames; padding to 300 by repeating the last frame means
   most of every "video" fed to the model is a frozen duplicate, likely
   washing out whatever real success/failure signal exists in the few
   genuine frame-to-frame transitions. Worth revisiting the padding strategy
   (or the video-length assumption generally — the source paper this
   pipeline follows targets ~10s/290 native videos, not 1-9s photo bursts
   padded 15-150x).
3. **Weak supervision on the *test*-split labels** — the success/failure
   labels used to decide the calibration/test split come from single-pass
   VLM judgments (whole-window, 4 sampled frames), not curated ground truth.
   `CosmosWorld` itself trains unsupervised so this doesn't affect training,
   but it does affect how trustworthy the calibration/test split boundary is.
