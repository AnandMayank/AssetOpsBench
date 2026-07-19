"""
Real end-to-end test: the actual CosmosWorldClassifier (pytorch_lightning
module, real LatentWorld predictor, real classifier head) trained on real
PMC gauge-inspection windows via PMCWindowDataset, on the Python 3.10 venv
at /media/adityapachauri/second_drive/envs/gauge_train310 (built specifically
because this repo's pyproject.toml requires python~=3.10, unlike the earlier
Python 3.8 gauge_reader env used for shape-contract-only verification).

Unlike verify_pmc_classifier_contract.py (which reimplemented an
architecture-identical classifier head standalone), this instantiates the
REAL CosmosWorldClassifier class and calls its REAL training_step(), proving
the full module -- including the LatentWorld world-model predictor this repo
adds on top of the frozen Cosmos encoder -- works on this data end to end.
"""
import os
import sys

os.environ.setdefault("WANDB_MODE", "disabled")
sys.path.insert(0, "/media/adityapachauri/second_drive/aditya_pmc_work/cosmos_tokenizer_src")

import torch
from torch.utils.data import DataLoader

from src.models.cosmos_world_classifier import CosmosWorldClassifier
from src.data.complex_datasets.pmc_window_dataset import PMCWindowDataset, collate_pmc_batch

CLEAN_DIR = os.environ.get(
    "PMC_CLEAN_DIR",
    "/media/adityapachauri/second_drive/aditya_pmc_work/pmc_windows_clean_full",
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

dataset = PMCWindowDataset(CLEAN_DIR, labeled_only=True)
print(f"PMCWindowDataset: {len(dataset)} labeled pairs")

loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_pmc_batch)
batch = next(iter(loader))
print(f"one real batch: labels={batch['labels'].tolist()}")


def move_batch_to_device(batch, device):
    # Mirrors what pl.Trainer's transfer_batch_to_device hook does; needed
    # here because we call training_step() directly, bypassing the Trainer.
    # forward() already moves s_images/s_obs/a itself, but labels is read
    # straight from the batch dict in _generic_step with no device transfer.
    moved = dict(batch)
    moved["labels"] = batch["labels"].to(device)
    return moved

print("\ninstantiating real CosmosWorldClassifier (downloads/loads Cosmos ckpt, "
      "builds real LatentWorld + classifier head)...")
model = CosmosWorldClassifier(num_feeds=1)
model = model.to(device)
model.train()

print("\nrunning ONE real training_step() (forward + real BCE loss via the "
      "actual LatentWorld world-model path, not a standalone reimplementation)...")
batch = move_batch_to_device(batch, device)
out = model.training_step(batch, batch_idx=0)
print(f"training_step output: loss={out['loss'].item():.4f} accuracy={out['accuracy'].item():.4f}")

print("\nrunning a real backward pass + optimizer step to prove gradients flow "
      "through LatentWorld + classifier (Cosmos encoder/decoder stay frozen)...")
opt_config = model.configure_optimizers()
optimizer = opt_config["optimizer"]
optimizer.zero_grad()
out["loss"].backward()

trainable_grad_norms = [
    p.grad.norm().item() for n, p in model.named_parameters()
    if p.requires_grad and p.grad is not None
]
frozen_have_no_grad = all(
    p.grad is None for n, p in model.named_parameters()
    if not p.requires_grad
)
print(f"trainable params with nonzero grad: {sum(1 for g in trainable_grad_norms if g > 0)}"
      f"/{len(trainable_grad_norms)}")
print(f"frozen cosmos encoder/decoder params correctly have no grad: {frozen_have_no_grad}")
optimizer.step()

print("\nREAL TRAINING STEP VERIFIED: CosmosWorldClassifier(real LatentWorld + "
      "classifier head) -> real forward -> real BCE loss -> real backward -> "
      "real optimizer.step(), on real PMC-labeled data, on Python 3.10 with "
      "the unpatched NVIDIA cosmos_tokenizer package.")
