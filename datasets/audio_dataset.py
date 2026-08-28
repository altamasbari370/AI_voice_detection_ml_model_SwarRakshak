"""
datasets/audio_dataset.py

PyTorch Dataset for audio deepfake / anti-spoofing detection.

Expects a directory layout of:

    data_dir/
        bona_fide/   # genuine audio, label = 0.0 (may contain nested subdirs)
        spoof/       # cloned/spoofed audio, label = 1.0 (may contain nested subdirs)

All .wav files under each subtree are discovered recursively via
`rglob('*.wav')`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Union

import torch
import torchaudio
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

DEFAULT_TARGET_SR = 16_000
DEFAULT_DURATION_SEC = 2.0

BONA_FIDE_LABEL = 0.0
SPOOF_LABEL = 1.0

BONA_FIDE_SUBDIR = "bona_fide"
SPOOF_SUBDIR = "spoof"


class AudioDataset(Dataset):
    """
    PyTorch Dataset that loads genuine ("bona_fide") and spoofed ("spoof")
    .wav files for binary deepfake/anti-spoofing classification.

    Each item is a fixed-length, mono, resampled 1-D waveform tensor
    paired with a float label (0.0 = genuine, 1.0 = spoofed), matching
    the input convention expected downstream by
    processing/feature_extractor.py and models/lcnn_detector.py.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        target_sr: int = DEFAULT_TARGET_SR,
        duration_sec: float = DEFAULT_DURATION_SEC,
    ) -> None:
        """
        Args:
            data_dir: Root directory containing `bona_fide/` and `spoof/`
                subdirectories of .wav files (each may be nested arbitrarily
                deep; all .wav files are discovered recursively).
            target_sr: Sample rate (Hz) all audio is resampled to.
            duration_sec: Target clip duration in seconds. Combined with
                `target_sr`, this determines `target_samples`
                (e.g. 16000 * 2.0 = 32000).

        Raises:
            FileNotFoundError: If neither the bona_fide nor spoof
                subdirectory exists under data_dir.
        """
        self.data_dir = Path(data_dir)
        self.target_sr = target_sr
        self.duration_sec = duration_sec
        self.target_samples = int(round(target_sr * duration_sec))

        bona_fide_dir = self.data_dir / BONA_FIDE_SUBDIR
        spoof_dir = self.data_dir / SPOOF_SUBDIR

        if not bona_fide_dir.is_dir() and not spoof_dir.is_dir():
            raise FileNotFoundError(
                f"Neither '{bona_fide_dir}' nor '{spoof_dir}' exists. "
                f"Expected data_dir to contain '{BONA_FIDE_SUBDIR}/' and/or "
                f"'{SPOOF_SUBDIR}/' subdirectories."
            )

        self.samples: List[Tuple[Path, float]] = []
        self._scan_directory(bona_fide_dir, BONA_FIDE_LABEL)
        self._scan_directory(spoof_dir, SPOOF_LABEL)

        if len(self.samples) == 0:
            logger.warning(
                "AudioDataset initialized with 0 samples under %s", self.data_dir
            )

        logger.info(
            "AudioDataset loaded %d samples from %s (target_sr=%d, "
            "target_samples=%d)",
            len(self.samples),
            self.data_dir,
            self.target_sr,
            self.target_samples,
        )

    def _scan_directory(self, directory: Path, label: float) -> None:
        """Recursively collect .wav file paths under `directory` with `label`."""
        if not directory.is_dir():
            logger.warning("Directory not found, skipping: %s", directory)
            return

        wav_paths = sorted(directory.rglob("*.wav"))
        for wav_path in wav_paths:
            self.samples.append((wav_path, label))

        logger.info("Found %d .wav files under %s (label=%.1f)", len(wav_paths), directory, label)

    def __len__(self) -> int:
        """Return the total number of audio samples in the dataset."""
        return len(self.samples)

    def _load_and_standardize(self, path: Path) -> torch.Tensor:
        """
        Load a .wav file and return a standardized 1-D waveform tensor:
        mono, resampled to self.target_sr, and padded/truncated to
        exactly self.target_samples.
        """
        waveform, original_sr = torchaudio.load(str(path))  # [channels, samples]

        # Convert multi-channel audio to mono by averaging across channels.
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if the file's native sample rate differs from target.
        if original_sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sr, new_freq=self.target_sr
            )
            waveform = resampler(waveform)

        # Standardize length: pad with zeros if short, truncate from the
        # start if long (i.e. keep the tail of the clip).
        num_samples = waveform.shape[-1]
        if num_samples < self.target_samples:
            pad_amount = self.target_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        elif num_samples > self.target_samples:
            waveform = waveform[..., -self.target_samples :]

        # Squeeze to a 1-D tensor of shape [target_samples].
        waveform = waveform.squeeze()
        if waveform.dim() == 0:
            # Degenerate edge case (shouldn't happen given padding above),
            # but guard against a fully-collapsed tensor.
            waveform = waveform.unsqueeze(0)

        return waveform.float()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return a single (waveform, label) pair.

        Args:
            idx: Index into the dataset.

        Returns:
            A tuple of:
                - waveform: 1-D float32 tensor of shape [target_samples].
                - label: 0-D float32 tensor (0.0 = genuine, 1.0 = spoofed).

            If the underlying audio file is missing, corrupted, or
            otherwise fails to load, a silent (all-zero) waveform of the
            correct shape is returned along with the correct label,
            rather than raising and crashing the training loop.
        """
        path, label = self.samples[idx]

        try:
            waveform = self._load_and_standardize(path)
        except Exception:
            logger.exception(
                "Failed to load audio file at index %d (%s); returning silent tensor",
                idx,
                path,
            )
            waveform = torch.zeros(self.target_samples, dtype=torch.float32)

        label_tensor = torch.tensor(label, dtype=torch.float32)
        return waveform, label_tensor


if __name__ == "__main__":
    import sys
    import tempfile

    logging.basicConfig(level=logging.INFO)

    # Build a throwaway dataset directory with a few synthetic .wav files
    # so this self-test runs standalone without requiring real audio data.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        bona_fide_dir = tmp_path / BONA_FIDE_SUBDIR / "nested"
        spoof_dir = tmp_path / SPOOF_SUBDIR
        bona_fide_dir.mkdir(parents=True)
        spoof_dir.mkdir(parents=True)

        # A "normal" 2-second clip at the target sample rate.
        normal_wave = torch.zeros(1, DEFAULT_TARGET_SR * 2)
        torchaudio.save(str(bona_fide_dir / "genuine_01.wav"), normal_wave, DEFAULT_TARGET_SR)

        # A short clip (needs padding) at a different sample rate, to
        # exercise resampling + padding.
        short_wave = torch.zeros(1, 8_000)  # 1 second at 8kHz
        torchaudio.save(str(spoof_dir / "fake_01.wav"), short_wave, 8_000)

        # A long, stereo clip (needs truncation + mono mixdown).
        long_stereo_wave = torch.zeros(2, DEFAULT_TARGET_SR * 3)
        torchaudio.save(str(spoof_dir / "fake_02.wav"), long_stereo_wave, DEFAULT_TARGET_SR)

        # A corrupted "audio" file (not actually valid audio) to exercise
        # the try/except fallback path.
        corrupted_path = spoof_dir / "corrupted.wav"
        corrupted_path.write_bytes(b"not a real wav file")

        dataset = AudioDataset(tmp_path)

        print(f"Dataset size: {len(dataset)}")
        assert len(dataset) == 4, f"Expected 4 samples, got {len(dataset)}"

        for i in range(len(dataset)):
            wf, lbl = dataset[i]
            path, expected_label = dataset.samples[i]
            print(
                f"[{i}] path={path.name:20s} shape={tuple(wf.shape)} "
                f"label={lbl.item():.1f} (expected {expected_label})"
            )
            assert wf.shape == (dataset.target_samples,), (
                f"Expected shape ({dataset.target_samples},), got {tuple(wf.shape)}"
            )
            assert wf.dtype == torch.float32
            assert lbl.item() == expected_label

        print("\nOK — all samples loaded with correct shape and labels")
        sys.exit(0)