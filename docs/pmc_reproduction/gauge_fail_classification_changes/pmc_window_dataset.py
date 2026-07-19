"""
PMCWindowDataset: adapts AssetOpsBench PMC gauge-inspection photo bursts to the
CosmosWorldClassifier training contract, WITHOUT requiring the full Trajectory
format (annotations.json + feature_report.json + episode.parquet + per-feed
.mp4, at 100Hz robot state/action) that Trajectory expects.

Why this adapter exists (read before wiring this into a real training run):

Trajectory is built for robot teleop demonstrations: every timestep carries a
STATE_OBS_DIM=47 robot state vector and an ACTION_DIM=31 robot action vector
from real hardware telemetry. AssetOpsBench's PMC data is a *photographer's
walkthrough* of industrial gauges — discrete photo bursts with real timestamps
but no robot joint/command log. Forcing that data through Trajectory would mean
fabricating state/action vectors, which is not something to do silently.

This dataset instead builds the exact 6-tuple that
CosmosWorldClassifier._check_batch_shape_and_keys_classifier() expects
(s_images, s_obs, a, s_images_next, s_obs_next, labels), and is honest about
what each field means for THIS data source:

  s_images / s_images_next  -- real consecutive frames from a single-asset,
                                scene-change-filtered PMC window (see
                                split_windows_by_scene.py). This is real signal.

  s_obs / s_obs_next        -- NOT robot telemetry. A measured feature vector
                                (frame position, timing, image-quality stats,
                                perception-category one-hot when ground-truth-
                                labeled) padded to STATE_OBS_DIM. Documented as
                                a proxy, never presented as robot state.

  a                         -- NOT a real robot action log (these bursts have
                                no robot in them). A structural placeholder:
                                one-hot on a single "passive_capture" action
                                slot in the ACTION_DIM=31 vocabulary, reserved
                                so real robot-executed recovery-ladder actions
                                (navigate_to / agentic_zoom / reposition / ...)
                                can be added later without a shape change, once
                                these windows are replayed through the real
                                robot MCP tool loop instead of a static photo
                                burst.

  labels                    -- ONLY windows with label_source="ground_truth"
                                in labels.json (from label_windows.py, matched
                                against perception_real.csv) are usable for
                                supervised loss. "unlabeled" windows are valid
                                for the *unsupervised* CosmosWorld world-model
                                pretraining stage (no labels needed there) but
                                MUST be excluded from CosmosWorldClassifier
                                training/CP-calibration/test — see
                                PMCWindowDataset(labeled_only=True) below.
"""
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import src.utils.config as config
except ModuleNotFoundError as e:
    # src.utils.config imports src.utils.logging, which unconditionally
    # imports wandb + pytorch_lightning (per pyproject.toml). Those are real
    # training-time deps this repo declares, but they're not needed to prove
    # the PMCWindowDataset tensor-shape contract, so this adapter degrades to
    # a literal-matched shim rather than forcing the full install for a
    # shapes-only check. If you see this warning during a REAL training run,
    # install the missing package(s) instead of trusting this fallback --
    # the shim is a snapshot of config.py's values at adapter-write time and
    # will silently drift if config.py changes.
    import warnings
    warnings.warn(
        f"src.utils.config unavailable ({e}); using a literal-matched shim. "
        "This is only safe for shape/contract verification, not real "
        "training -- install pytorch_lightning/wandb and re-run to use the "
        "real config.", stacklevel=2,
    )

    class config:  # noqa: N801 -- mirrors the module-as-namespace usage above
        FEED_IMAGE_WIDTH = 384
        FEED_IMAGE_HEIGHT = 208
        FEED_IMAGE_CHANNELS = 3
        FEED_IMAGE_LABELS_TO_USE = ["camera_1"]
        NUM_FEEDS = len(FEED_IMAGE_LABELS_TO_USE)
        STATE_OBS_DIM = 47
        ACTION_DIM = 31

# --- PMC action vocabulary (structural placeholder; see module docstring) ---
PMC_ACTION_VOCAB = [
    "passive_capture",      # this photo burst: no robot tool was called
    "navigate_to",
    "safety_gate_check",
    "open_panel",
    "read_gauge",
    "commit_reading",
    "check_wo_similarity",
    "capture_image",
    "agentic_zoom",         # recovery ladder rung 1
    "reposition",           # recovery ladder rung 2
    "flag_recommended_action",
    "power_on", "power_off", "stand", "sit", "dock", "undock",
    "get_battery", "get_pose", "list_waypoints",
]
assert len(PMC_ACTION_VOCAB) <= config.ACTION_DIM, (
    f"PMC_ACTION_VOCAB ({len(PMC_ACTION_VOCAB)}) exceeds config.ACTION_DIM "
    f"({config.ACTION_DIM}); trim the vocabulary or the config dim changed."
)

# --- perception category one-hot slots (from perception_real.csv) ---
PMC_CATEGORY_VOCAB = [
    "gauge_degradation", "occlusion", "scale_interpretation",
    "glare_lighting", "iot_contradiction", "never_read", "unknown",
]


def _load_image(path: Path, height: int, width: int) -> np.ndarray:
    img = cv2.imread(str(path))
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0  # (H, W, 3) in [0, 1]


def _quality_features(img_rgb_float: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor((img_rgb_float * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    brightness = float(gray.mean())
    return np.array([lap_var / 200.0, contrast / 100.0, brightness / 255.0], dtype=np.float32)


def build_state_obs(frame_rgb: np.ndarray, frame_pos_ratio: float,
                     category: Optional[str]) -> np.ndarray:
    """Builds one STATE_OBS_DIM-length vector. See module docstring: this is a
    measured-feature proxy, not robot state."""
    quality = _quality_features(frame_rgb)                       # 3 dims
    cat_onehot = np.zeros(len(PMC_CATEGORY_VOCAB), dtype=np.float32)
    cat_onehot[PMC_CATEGORY_VOCAB.index(category) if category in PMC_CATEGORY_VOCAB
               else PMC_CATEGORY_VOCAB.index("unknown")] = 1.0    # 7 dims
    scalar = np.array([frame_pos_ratio], dtype=np.float32)        # 1 dim
    obs = np.concatenate([quality, cat_onehot, scalar])           # 11 dims used
    padded = np.zeros(config.STATE_OBS_DIM, dtype=np.float32)
    padded[:len(obs)] = obs
    return padded


def build_action(name: str = "passive_capture") -> np.ndarray:
    a = np.zeros(config.ACTION_DIM, dtype=np.float32)
    a[PMC_ACTION_VOCAB.index(name)] = 1.0
    return a


class PMCWindowDataset(Dataset):
    """
    One example = one (frame_t, frame_t+1) pair drawn from a single-asset PMC
    window (<clean_dir>/<window>/), with the window's label (if any) applied
    to every pair drawn from it.

    Accepts two label-file schemas (auto-detected):
      - labels.json      (label_windows.py): label_source in {"ground_truth",
        "unlabeled"}; "categories" is a list. Used by the original 13-window
        pmc_windows_clean/ demo.
      - labels_vlm.json   (label_windows_vlm.py): label_source in
        {"ground_truth", "vlm"}; "category" is a single string. Real Gemini
        VLM labels on the decision frame, used at the 180-window
        pmc_windows_clean_full/ scale. VLM labels are real model output on
        real images -- not fabricated -- but are weaker supervision than the
        curated perception_real.csv rows and should be reported as such
        (label_source is kept per-example so callers can weight/filter).

    Args:
        clean_dir: path to a pmc_windows_clean* directory containing
            manifest.json + (labels.json or labels_vlm.json).
        labeled_only: if True (required for CosmosWorldClassifier
            train/val/CP-calibration/test), drop every window whose
            label_source is not one of the "real" sources above. If False,
            includes unlabeled windows with label=-1 (caller's job to route
            those to an unsupervised objective, e.g. CosmosWorld
            reconstruction, not to BCEWithLogitsLoss).
    """

    REAL_LABEL_SOURCES = {"ground_truth", "vlm"}

    def __init__(self, clean_dir: str, labeled_only: bool = True):
        self.clean_dir = Path(clean_dir)
        manifest = json.loads((self.clean_dir / "manifest.json").read_text())

        labels_path = self.clean_dir / "labels_vlm.json"
        if not labels_path.exists():
            labels_path = self.clean_dir / "labels.json"
        labels = json.loads(labels_path.read_text())
        self.labels_path = labels_path

        self.pairs = []  # list of (window_name, frame_a, frame_b, label_int_or_None, category, pos_ratio, label_source)
        for window_name, info in manifest.items():
            lab = labels.get(window_name)
            if lab is None:
                continue
            is_real = lab["label_source"] in self.REAL_LABEL_SOURCES
            if labeled_only and not is_real:
                continue
            label_int = None
            if is_real:
                label_int = 1 if lab["label"] == "failure" else 0
            category = lab.get("category") or (lab.get("categories") or [None])[0]
            frames = info["frames"]
            for i in range(len(frames) - 1):
                self.pairs.append((
                    window_name, frames[i], frames[i + 1],
                    label_int, category,
                    i / max(1, len(frames) - 1),  # frame position ratio
                    lab["label_source"],
                ))

        if labeled_only and not self.pairs:
            raise ValueError(
                f"No real-labeled windows found under {clean_dir} (checked "
                f"{labels_path.name}). Run label_windows.py or "
                "label_windows_vlm.py first, or pass labeled_only=False for "
                "unsupervised pretraining."
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        window_name, fa, fb, label_int, category, pos_ratio, label_source = self.pairs[idx]
        wdir = self.clean_dir / window_name

        img_a = _load_image(wdir / fa, config.FEED_IMAGE_HEIGHT, config.FEED_IMAGE_WIDTH)
        img_b = _load_image(wdir / fb, config.FEED_IMAGE_HEIGHT, config.FEED_IMAGE_WIDTH)

        # cosmos_tokenizer.image_lib.ImageTokenizer.encode() documents its
        # input contract as Bx3xHxW, range [-1, 1] -- img_a/img_b themselves
        # stay [0, 1] (build_state_obs's quality features assume that range).
        feed_label = config.FEED_IMAGE_LABELS_TO_USE[0]
        s_images = {feed_label: torch.from_numpy(img_a * 2.0 - 1.0).permute(2, 0, 1)}
        s_images_next = {feed_label: torch.from_numpy(img_b * 2.0 - 1.0).permute(2, 0, 1)}

        s_obs = torch.from_numpy(build_state_obs(img_a, pos_ratio, category))
        s_obs_next = torch.from_numpy(build_state_obs(img_b, pos_ratio, category))
        a = torch.from_numpy(build_action("passive_capture"))

        label = torch.tensor(float(label_int) if label_int is not None else -1.0)

        return dict(
            s_images=s_images, s_obs=s_obs, a=a,
            s_images_next=s_images_next, s_obs_next=s_obs_next,
            labels=label,
            window_name=window_name,  # kept for debugging/inspection, not part of the model contract
            label_source=label_source,  # ditto -- "ground_truth" or "vlm"
        )


def collate_pmc_batch(batch):
    """Collates a list of __getitem__ dicts into the batched dict shape
    CosmosWorldClassifier._check_batch_shape_and_keys_classifier expects."""
    feed_label = config.FEED_IMAGE_LABELS_TO_USE[0]
    return dict(
        s_images={feed_label: torch.stack([b["s_images"][feed_label] for b in batch])},
        s_obs=torch.stack([b["s_obs"] for b in batch]),
        a=torch.stack([b["a"] for b in batch]),
        s_images_next={feed_label: torch.stack([b["s_images_next"][feed_label] for b in batch])},
        s_obs_next=torch.stack([b["s_obs_next"] for b in batch]),
        labels=torch.stack([b["labels"] for b in batch]),
    )
