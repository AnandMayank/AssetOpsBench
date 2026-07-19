"""
End-to-end contract verification: PMCWindowDataset -> real Cosmos encoder ->
CosmosWorldClassifier-shaped classifier head -> real BCEWithLogitsLoss.

This proves the *data contract* is satisfiable (exact tensor shapes match what
CosmosWorldClassifier.forward()/._check_batch_shape_and_keys_classifier expect)
using the real downloaded Cosmos-0.1-Tokenizer-CI16x16 checkpoint and real
ground-truth-labeled PMC windows.

What this intentionally does NOT do: instantiate the real LatentWorld module
(needs pytorch_lightning + wandb + piq + lpips, none installed here) or train
anything. The classifier head below has the IDENTICAL architecture to
CosmosWorldClassifier.classifier, freshly initialized, run on the flattened
encoder latent directly (skipping the world-model z -> z_next prediction
step) — sufficient to prove the encoder-to-classifier shape/loss plumbing,
not sufficient to claim trained accuracy.

Run with:
  PYTHONPATH=/tmp/cosmos_tokenizer_src:~/GaugeFailClassification \
  /media/adityapachauri/second_drive/envs/gauge_reader/bin/python3 \
  verify_pmc_classifier_contract.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/media/adityapachauri/second_drive/aditya_pmc_work/cosmos_tokenizer_src")
sys.path.insert(0, os.path.expanduser("~/GaugeFailClassification"))

from cosmos_tokenizer.image_lib import ImageTokenizer  # noqa: E402
from src.data.complex_datasets.pmc_window_dataset import (  # noqa: E402
    PMCWindowDataset, collate_pmc_batch, config,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "cosmos_ckpts", "Cosmos-0.1-Tokenizer-CI16x16")
CLEAN_DIR = os.environ.get(
    "PMC_CLEAN_DIR",
    "/media/adityapachauri/second_drive/aditya_pmc_work/pmc_windows_clean_full",
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")
print(f"config: FEED_IMAGE_HEIGHT={config.FEED_IMAGE_HEIGHT} "
      f"FEED_IMAGE_WIDTH={config.FEED_IMAGE_WIDTH} "
      f"STATE_OBS_DIM={config.STATE_OBS_DIM} ACTION_DIM={config.ACTION_DIM} "
      f"NUM_FEEDS={config.NUM_FEEDS}")

# ---- 1. Real dataset, real-labeled windows only (ground_truth + vlm) ----
dataset = PMCWindowDataset(CLEAN_DIR, labeled_only=True)
print(f"\nPMCWindowDataset ({CLEAN_DIR}): {len(dataset)} labeled (frame_t, frame_t+1) pairs")
items = [dataset[i] for i in range(len(dataset))]
from collections import Counter
print(f"label_source breakdown: {Counter(it['label_source'] for it in items)}")
print(f"window breakdown: {len(Counter(it['window_name'] for it in items))} distinct windows")

feed_label = config.FEED_IMAGE_LABELS_TO_USE[0]
N = len(items)

# ---- 2. Real Cosmos encoder (same checkpoint as the round-trip demo) ----
encoder = ImageTokenizer(checkpoint_enc=os.path.join(CKPT_DIR, "encoder.jit"), device=device)
for p in encoder.parameters():
    p.requires_grad = False
encoder.eval()

cosmos_channels = 16
spatial_compression = (16, 16)
h_c = config.FEED_IMAGE_HEIGHT // spatial_compression[0]
w_c = config.FEED_IMAGE_WIDTH // spatial_compression[1]
combined_shape = (cosmos_channels * config.NUM_FEEDS, h_c, w_c)
print(f"expected combined_cosmos_latent_shape: {combined_shape}")

# ---- 3. Classifier head, architecture-identical to CosmosWorldClassifier.classifier ----
flattened_dim = int(np.prod(combined_shape))
classifier = nn.Sequential(
    nn.Linear(flattened_dim, 512), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(128, 1),
).to(device)

# ---- 4. Encode + classify + loss in mini-batches (full 678-pair dataset does
# not fit the encoder's activations in 24GB VRAM at once; this is a memory
# constraint of this GPU, not a contract issue) ----
BATCH_SIZE = 16
loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
total_loss, total_correct, all_logits, all_labels = 0.0, 0, [], []

for start in range(0, N, BATCH_SIZE):
    chunk = items[start:start + BATCH_SIZE]
    batch = collate_pmc_batch(chunk)
    x = batch["s_images"][feed_label].to(device).to(torch.bfloat16)  # (b,3,H,W)
    with torch.no_grad():
        z, = encoder.encode(x)
    z = z.float()
    b = z.shape[0]
    assert z.shape == (b, *combined_shape), f"z shape {z.shape} != expected {(b, *combined_shape)}"

    z_flat = z.view(b, -1)
    logits = classifier(z_flat).squeeze(-1)
    labels = batch["labels"].to(device)

    total_loss += loss_fn(logits, labels).item()
    predictions = (logits > 0).float()
    total_correct += (predictions == labels).sum().item()
    all_logits.extend(logits.tolist())
    all_labels.extend(labels.tolist())

    del z, z_flat, x
    torch.cuda.empty_cache()
    if (start // BATCH_SIZE) % 10 == 0:
        print(f"  encoded {min(start + BATCH_SIZE, N)}/{N} pairs...")

avg_loss = total_loss / N
accuracy = total_correct / N
print(f"encoder output z per-batch shape: (b, {combined_shape[0]}, {combined_shape[1]}, {combined_shape[2]})"
      f"  <- matches CosmosWorldClassifier's asserted shape (verified on every mini-batch)")
print(f"\nBCEWithLogitsLoss (mean over {N} examples): {avg_loss:.4f}")
print(f"accuracy (untrained head, {N} examples): {accuracy:.4f}")
print(f"label balance: {Counter(all_labels)}")
print("\nCONTRACT VERIFIED: PMCWindowDataset -> real Cosmos encoder -> "
      "classifier-head-shaped-identically-to-CosmosWorldClassifier -> real loss, "
      f"all real tensors, no shape mismatches, at full scale ({N} pairs / "
      f"{len(Counter(it['window_name'] for it in items))} windows).")
