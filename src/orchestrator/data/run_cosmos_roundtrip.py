"""
Real Cosmos-0.1-Tokenizer-CI16x16 encode/decode round-trip on frames extracted
from the corrected, single-asset PMC window videos.

Run with the gauge_reader conda python (has torch+cuda 2.0.0+cu118, cv2, numpy):
  PYTHONPATH=/tmp/cosmos_tokenizer_src \
  /media/adityapachauri/second_drive/envs/gauge_reader/bin/python3 \
  run_cosmos_roundtrip.py
"""
import os
import sys
import numpy as np
import cv2
import torch

sys.path.insert(0, "/tmp/cosmos_tokenizer_src")
from cosmos_tokenizer.image_lib import ImageTokenizer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "cosmos_ckpts", "Cosmos-0.1-Tokenizer-CI16x16")
VIDEO = os.path.join(HERE, "pmc_videos", "AOBv2-PMC-windowA_twinneedle_dp_gauge_134114-134145.mp4")
OUT_DIR = os.path.join(HERE, "cosmos_roundtrip_out")
os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

encoder = ImageTokenizer(checkpoint_enc=os.path.join(CKPT_DIR, "encoder.jit"), device=device)
decoder = ImageTokenizer(checkpoint_dec=os.path.join(CKPT_DIR, "decoder.jit"), device=device)

cap = cv2.VideoCapture(VIDEO)
assert cap.isOpened(), f"could not open {VIDEO}"
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"video: {VIDEO} ({n_frames} frames)")

# sample a few frames spread across the window (first, middle, last)
sample_idx = sorted(set([0, n_frames // 2, n_frames - 1]))
frames = {}
for idx in sample_idx:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame_bgr = cap.read()
    assert ok, f"failed to read frame {idx}"
    frames[idx] = frame_bgr
cap.release()

results = []
for idx, frame_bgr in frames.items():
    h, w = frame_bgr.shape[:2]
    # CI16x16 needs H, W divisible by 16
    h16, w16 = (h // 16) * 16, (w // 16) * 16
    frame_bgr = frame_bgr[:h16, :w16]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    x = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    x = x.to(device).to(torch.bfloat16)

    with torch.no_grad():
        latent, = encoder.encode(x)
        recon = decoder._dec_model(latent)

    recon = recon.float().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    recon_bgr = cv2.cvtColor((recon * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    mse = float(np.mean((frame_rgb.astype(np.float32) / 255.0 - recon) ** 2))
    psnr = float(20 * np.log10(1.0 / np.sqrt(mse))) if mse > 0 else float("inf")

    out_orig = os.path.join(OUT_DIR, f"frame{idx:03d}_original.png")
    out_recon = os.path.join(OUT_DIR, f"frame{idx:03d}_reconstructed.png")
    cv2.imwrite(out_orig, frame_bgr)
    cv2.imwrite(out_recon, recon_bgr)

    results.append(dict(
        frame_idx=idx,
        input_hw=(h16, w16),
        latent_shape=tuple(latent.shape),
        mse=round(mse, 6),
        psnr_db=round(psnr, 2),
    ))
    print(f"frame {idx}: input {h16}x{w16} -> latent {tuple(latent.shape)} "
          f"(compression {h16*w16*3 / latent.numel():.1f}x) | MSE={mse:.6f} PSNR={psnr:.2f}dB")

import json
with open(os.path.join(OUT_DIR, "roundtrip_metrics.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved reconstructions and metrics to {OUT_DIR}")
