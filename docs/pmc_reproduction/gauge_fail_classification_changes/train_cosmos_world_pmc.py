"""
Trains the real CosmosWorld model (unsupervised next-frame world model, NOT
CosmosWorldClassifier) on the full 180-window PMC dataset, then saves a
checkpoint that scripts/inference/calibrate.py can consume to fit conformal-
prediction thresholds.

CosmosWorld needs no labels (world-model reconstruction objective), so this
trains on all real-labeled windows regardless of success/failure -- CP
calibration itself (a separate, later step) is what restricts to success-only
data, not this pretraining stage.

_check_batch_shape_and_keys() (src/models/general.py) unpacks its batch
POSITIONALLY -- batch[0..4] -- unlike CosmosWorldClassifier's dict-based
_check_batch_shape_and_keys_classifier(). PMCWindowDataset/collate_pmc_batch
return a dict, so this script wraps it into the 5-tuple CosmosWorld expects.
"""
import os
import sys
import time

os.environ.setdefault("WANDB_MODE", "disabled")
sys.path.insert(0, "/media/adityapachauri/second_drive/aditya_pmc_work/cosmos_tokenizer_src")

import torch
from torch.utils.data import DataLoader

from src.models.cosmos_world import CosmosWorld
from src.data.complex_datasets.pmc_window_dataset import PMCWindowDataset, collate_pmc_batch

CLEAN_DIR = os.environ.get(
    "PMC_CLEAN_DIR",
    "/media/adityapachauri/second_drive/aditya_pmc_work/pmc_windows_clean_full",
)
CKPT_OUT = os.environ.get(
    "COSMOS_WORLD_CKPT_OUT",
    "/media/adityapachauri/second_drive/aditya_pmc_work/cosmos_world_pmc.ckpt",
)
EPOCHS = int(os.environ.get("EPOCHS", "3"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")


def dict_batch_to_tuple(batch):
    return (
        batch["s_images"], batch["s_obs"], batch["a"],
        batch["s_images_next"], batch["s_obs_next"],
    )


def move_tuple_to_device(t, device):
    s_images, s_obs, a, s_images_next, s_obs_next = t
    return (
        {k: v.to(device) for k, v in s_images.items()},
        s_obs.to(device), a.to(device),
        {k: v.to(device) for k, v in s_images_next.items()},
        s_obs_next.to(device),
    )


dataset = PMCWindowDataset(CLEAN_DIR, labeled_only=True)
print(f"PMCWindowDataset: {len(dataset)} pairs (unsupervised training uses images only, labels ignored)")
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                    collate_fn=collate_pmc_batch, drop_last=True)
print(f"{len(loader)} batches/epoch at batch_size={BATCH_SIZE}")

print("\ninstantiating real CosmosWorld...")
model = CosmosWorld(num_feeds=1, use_unet=False, use_svd=False)
model = model.to(device)
model.train()

optimizer = model.configure_optimizers()

t0 = time.time()
for epoch in range(EPOCHS):
    epoch_losses = []
    for batch_idx, batch in enumerate(loader):
        batch_t = move_tuple_to_device(dict_batch_to_tuple(batch), device)
        optimizer.zero_grad()
        out = model._generic_step(batch_t, batch_idx, "train")
        loss = out["loss"]
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        if batch_idx % 10 == 0:
            print(f"  epoch {epoch} batch {batch_idx}/{len(loader)} loss={loss.item():.4f} "
                  f"elapsed={time.time()-t0:.0f}s")
    print(f"epoch {epoch} done: mean_loss={sum(epoch_losses)/len(epoch_losses):.4f}")

torch.save({"state_dict": model.state_dict()}, CKPT_OUT)
print(f"\nCheckpoint saved: {CKPT_OUT}")
print("REAL COSMOSWORLD TRAINING COMPLETE.")
