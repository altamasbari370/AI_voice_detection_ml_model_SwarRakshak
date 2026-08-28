"""
train.py

Training script for the LCNN audio anti-spoofing / deepfake detector.

Pipeline per sample:
    raw waveform (AudioDataset)
        -> SpectrogramExtractor  (Log-Mel Spectrogram, [1, 1, 80, 201])
        -> LCNNClassifier        (spoof probability in [0.0, 1.0])

Targeted at a Windows + NVIDIA GPU environment (e.g. RTX 4050), with a
clean CPU fallback. Saves the best checkpoint (by validation loss) to
models/lcnn_best.pth.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split

from datasets.audio_dataset import AudioDataset
from models.lcnn_detector import LCNNClassifier
from processing.feature_extractor import SpectrogramExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Configuration -----------------------------------------------------
DATA_DIR = Path("data/train")
CHECKPOINT_DIR = Path("models")
CHECKPOINT_PATH = CHECKPOINT_DIR / "lcnn_best.pth"

TRAIN_VAL_SPLIT_RATIO = 0.8
RANDOM_SEED = 42

BATCH_SIZE = 16
NUM_WORKERS = 0  # Kept at 0 for full Windows compatibility (avoids multiprocessing issues)

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 10

LR_SCHEDULER_PATIENCE = 2
LR_SCHEDULER_FACTOR = 0.5

EER_THRESHOLD_STEPS = 1000


def setup_device() -> torch.device:
    """
    Detect and bind to a CUDA GPU if available, otherwise fall back to CPU.
    Prints device name and VRAM info when CUDA is used.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(device)
        total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        logger.info("CUDA available — using GPU: %s (%.2f GB VRAM)", gpu_name, total_vram_gb)
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available — falling back to CPU. Training will be slow.")

    return device


def build_dataloaders(
    data_dir: Path,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load the full dataset from `data_dir`, split it 80/20 into
    train/validation subsets with a fixed seed, and wrap each in a
    DataLoader.

    Raises:
        FileNotFoundError: If data_dir (or its bona_fide/spoof
            subdirectories) does not exist or contains no samples.
    """
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Training data directory not found: {data_dir.resolve()}"
        )

    full_dataset: Dataset = AudioDataset(data_dir)

    if len(full_dataset) == 0:
        raise FileNotFoundError(
            f"No .wav samples found under {data_dir.resolve()} "
            f"(expected 'bona_fide/' and/or 'spoof/' subdirectories)"
        )

    num_train = int(len(full_dataset) * TRAIN_VAL_SPLIT_RATIO)
    num_val = len(full_dataset) - num_train

    if num_train == 0 or num_val == 0:
        raise ValueError(
            f"Dataset too small to split 80/20 (total={len(full_dataset)}). "
            f"Add more training samples."
        )

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_subset, val_subset = random_split(
        full_dataset, [num_train, num_val], generator=generator
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    logger.info(
        "Dataset loaded: %d total samples -> %d train / %d val",
        len(full_dataset),
        num_train,
        num_val,
    )

    return train_loader, val_loader


def extract_batch_features(
    extractor: SpectrogramExtractor,
    waveforms: torch.Tensor,
) -> torch.Tensor:
    """
    Run SpectrogramExtractor over a batch of 1-D waveforms.

    SpectrogramExtractor.extract_features operates on a single 1-D
    waveform and returns a [1, 1, 80, time_frames] tensor, so this
    processes each waveform in the batch individually and stacks the
    results into a single [batch, 1, 80, time_frames] tensor.

    Args:
        extractor: The SpectrogramExtractor instance.
        waveforms: Float tensor of shape [batch, num_samples].

    Returns:
        Float tensor of shape [batch, 1, 80, time_frames].
    """
    per_sample_features: List[torch.Tensor] = []
    for i in range(waveforms.shape[0]):
        # extract_features returns [1, 1, 80, time_frames]; drop the
        # leading batch dim of 1 so we can re-stack across the real batch.
        features = extractor.extract_features(waveforms[i])
        per_sample_features.append(features.squeeze(0))

    return torch.stack(per_sample_features, dim=0)


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute the Equal Error Rate (EER): the point at which the False
    Acceptance Rate (FAR, i.e. false positive rate) equals the False
    Rejection Rate (FRR, i.e. false negative rate).

    Args:
        labels: 1-D array of true binary labels (0 = bona fide, 1 = spoof).
        scores: 1-D array of predicted spoof probabilities in [0, 1].

    Returns:
        EER as a fraction in [0.0, 1.0]. Returns 0.0 if either class is
        entirely absent from `labels` (EER is undefined in that case).
    """
    if labels.size == 0:
        return 0.0

    num_positive = np.sum(labels == 1)
    num_negative = np.sum(labels == 0)
    if num_positive == 0 or num_negative == 0:
        logger.warning("EER undefined: validation batch missing one class this epoch")
        return 0.0

    thresholds = np.linspace(0.0, 1.0, EER_THRESHOLD_STEPS)
    far_list = np.empty(EER_THRESHOLD_STEPS)
    frr_list = np.empty(EER_THRESHOLD_STEPS)

    for i, thresh in enumerate(thresholds):
        predicted_spoof = scores >= thresh
        false_accepts = np.sum(predicted_spoof & (labels == 0))  # bona fide misclassified as spoof
        false_rejects = np.sum(~predicted_spoof & (labels == 1))  # spoof misclassified as bona fide
        far_list[i] = false_accepts / num_negative
        frr_list[i] = false_rejects / num_positive

    diff = np.abs(far_list - frr_list)
    min_idx = int(np.argmin(diff))
    eer = float((far_list[min_idx] + frr_list[min_idx]) / 2.0)
    return eer


def train_one_epoch(
    train_loader: DataLoader,
    extractor: SpectrogramExtractor,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one full training epoch and return the average training loss."""
    model.train()
    running_loss = 0.0
    num_batches = 0

    for waveforms, labels in train_loader:
        try:
            waveforms = waveforms.to(device)
            labels = labels.to(device).view(-1, 1)

            features = extract_batch_features(extractor, waveforms)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        except Exception:
            logger.exception("Skipping a corrupted/failed training batch")
            continue

    if num_batches == 0:
        logger.warning("No successful training batches this epoch")
        return float("nan")

    return running_loss / num_batches


@torch.no_grad()
def validate_one_epoch(
    val_loader: DataLoader,
    extractor: SpectrogramExtractor,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Run one full validation epoch without gradient tracking.

    Returns:
        {
            "val_loss": float,
            "accuracy": float,   # in [0, 1]
            "eer": float,        # in [0, 1]
        }
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0
    correct = 0
    total = 0

    all_labels: List[float] = []
    all_scores: List[float] = []

    for waveforms, labels in val_loader:
        try:
            waveforms = waveforms.to(device)
            labels = labels.to(device).view(-1, 1)

            features = extract_batch_features(extractor, waveforms)
            outputs = model(features)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            num_batches += 1

            predictions = (outputs >= 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.numel()

            all_labels.extend(labels.cpu().view(-1).tolist())
            all_scores.extend(outputs.cpu().view(-1).tolist())

        except Exception:
            logger.exception("Skipping a corrupted/failed validation batch")
            continue

    if num_batches == 0 or total == 0:
        logger.warning("No successful validation batches this epoch")
        return {"val_loss": float("nan"), "accuracy": 0.0, "eer": 0.0}

    avg_loss = running_loss / num_batches
    accuracy = correct / total
    eer = compute_eer(np.array(all_labels), np.array(all_scores))

    return {"val_loss": avg_loss, "accuracy": accuracy, "eer": eer}


def save_checkpoint(model: nn.Module, path: Path, epoch: int, val_loss: float) -> None:
    """Save model weights (and minimal metadata) to `path`, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
        },
        path,
    )
    logger.info("Saved new best checkpoint to %s (epoch=%d, val_loss=%.4f)", path, epoch, val_loss)


def main() -> None:
    device = setup_device()

    try:
        train_loader, val_loader = build_dataloaders(DATA_DIR, device)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to prepare data: %s", exc)
        sys.exit(1)

    extractor = SpectrogramExtractor(device=device.type)
    model = LCNNClassifier().to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE
    )

    best_val_loss = float("inf")

    logger.info("Starting training for %d epochs on device=%s", NUM_EPOCHS, device)

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            train_loader, extractor, model, criterion, optimizer, device
        )
        val_metrics = validate_one_epoch(val_loader, extractor, model, criterion, device)

        val_loss = val_metrics["val_loss"]
        scheduler.step(val_loss)

        epoch_duration = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %2d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.2f%% | "
            "EER=%.2f%% | lr=%.2e | %.1fs",
            epoch,
            NUM_EPOCHS,
            train_loss,
            val_loss,
            val_metrics["accuracy"] * 100,
            val_metrics["eer"] * 100,
            current_lr,
            epoch_duration,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, CHECKPOINT_PATH, epoch, val_loss)

    logger.info("Training complete. Best validation loss: %.4f", best_val_loss)
    logger.info("Best checkpoint saved at: %s", CHECKPOINT_PATH.resolve())


if __name__ == "__main__":
    main()