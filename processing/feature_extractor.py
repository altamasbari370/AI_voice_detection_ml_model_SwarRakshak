"""
processing/feature_extractor.py

Converts raw speech waveform tensors into normalized Log-Mel Spectrogram
features suitable for input to a 2D CNN.
"""

from __future__ import annotations

import logging

import torch
import torchaudio

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
N_MELS = 80
N_FFT = 512
HOP_LENGTH = 160
EPS = 1e-9


class SpectrogramExtractor:
    """
    Extracts normalized Log-Mel Spectrogram features from a 1-D speech
    waveform tensor, formatted for direct input to a 2D CNN.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_mels: int = N_MELS,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            sample_rate: Expected sample rate of incoming audio.
            n_mels: Number of Mel filterbank bins.
            n_fft: FFT window size.
            hop_length: Number of samples between successive frames.
            device: torch device to run the transform on ("cpu" or "cuda").
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.device = torch.device(device)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        ).to(self.device)

        logger.info(
            "SpectrogramExtractor initialized (sample_rate=%d, n_mels=%d, "
            "n_fft=%d, hop_length=%d, device=%s)",
            self.sample_rate,
            self.n_mels,
            self.n_fft,
            self.hop_length,
            self.device,
        )

    def extract_features(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert a 1-D speech waveform into a normalized Log-Mel Spectrogram
        shaped for a 2D CNN.

        Args:
            audio_tensor: 1-D float tensor of shape [num_samples],
                representing active speech at self.sample_rate.

        Returns:
            A float32 tensor of shape [1, 1, n_mels, time_frames]
            (Batch=1, Channels=1, Mel-bins, Time), normalized to
            zero mean and unit standard deviation.

        Raises:
            ValueError: If audio_tensor is not 1-dimensional or is empty.
        """
        if audio_tensor.dim() != 1:
            raise ValueError(
                f"Expected a 1-D audio tensor, got shape {tuple(audio_tensor.shape)}"
            )
        if audio_tensor.numel() == 0:
            raise ValueError("Received empty audio tensor")

        audio_tensor = audio_tensor.to(self.device).float()

        # 1. Mel Spectrogram: [n_mels, time_frames]
        mel_spec = self.mel_transform(audio_tensor)

        # 2. Log-Mel Spectrogram (add epsilon to avoid log(0))
        log_mel_spec = torch.log(mel_spec + EPS)

        # 3. Normalize to zero mean, unit standard deviation
        mean = log_mel_spec.mean()
        std = log_mel_spec.std()
        normalized = (log_mel_spec - mean) / (std + EPS)

        # 4. Reshape to [1, 1, n_mels, time_frames] for a 2D CNN
        #    unsqueeze(0) adds the channel dim, unsqueeze(0) again adds batch dim
        features = normalized.unsqueeze(0).unsqueeze(0)

        return features


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    extractor = SpectrogramExtractor()

    # Dummy 1-D tensor representing 2 seconds of 16kHz audio
    dummy_audio = torch.randn(32_000)

    output = extractor.extract_features(dummy_audio)

    print(f"Input shape:  {tuple(dummy_audio.shape)}")
    print(f"Output shape: {tuple(output.shape)}")

    expected_shape = (1, 1, 80, 201)
    assert tuple(output.shape) == expected_shape, (
        f"Expected {expected_shape}, got {tuple(output.shape)}"
    )
    print(f"OK — output shape matches expected {expected_shape}")