"""
processing/vad_stream.py

Voice Activity Detection (VAD) for streamed audio chunks using Silero VAD.

Designed for a streaming pipeline where fixed-size (e.g. 2-second) 16kHz
mono PCM chunks arrive as raw bytes. Each chunk is scanned for speech
segments; silent frames are dropped and only the concatenated active
speech is returned.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
CHUNK_SECONDS = 2
EXPECTED_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS  # 32,000 samples for a 2s chunk


class AudioVAD:
    """
    Wraps the Silero VAD model (loaded via torch.hub) to detect and extract
    speech segments from short streaming audio chunks.

    Usage:
        vad = AudioVAD()
        speech_tensor = vad.process_2sec_chunk(raw_pcm_bytes)
        if speech_tensor is not None:
            ...  # forward to downstream model (ASR, etc.)
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            sample_rate: Expected sample rate of incoming audio (Silero VAD
                supports 8000 or 16000 Hz).
            threshold: Speech probability threshold in [0, 1]. Higher values
                make detection stricter (fewer false positives, more missed
                speech).
            device: torch device to run the model on ("cpu" or "cuda").
        """
        if sample_rate not in (8_000, 16_000):
            raise ValueError("Silero VAD only supports 8000 or 16000 Hz audio")

        self.sample_rate = sample_rate
        self.threshold = threshold
        self.device = torch.device(device)

        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        self.model.to(self.device)
        self.model.eval()

        # utils: (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks)
        (
            self._get_speech_timestamps,
            _save_audio,
            _read_audio,
            _VADIterator,
            self._collect_chunks,
        ) = utils

        logger.info(
            "AudioVAD initialized (sample_rate=%d, threshold=%.2f, device=%s)",
            self.sample_rate,
            self.threshold,
            self.device,
        )

    def _bytes_to_tensor(self, audio_bytes: bytes) -> torch.Tensor:
        """
        Convert raw little-endian 16-bit PCM bytes into a normalized
        float32 mono waveform tensor in [-1.0, 1.0].
        """
        int16_array = np.frombuffer(audio_bytes, dtype=np.int16)
        if int16_array.size == 0:
            raise ValueError("Received empty audio buffer")

        float_array = int16_array.astype(np.float32) / 32768.0
        return torch.from_numpy(float_array)

    def process_2sec_chunk(self, audio_bytes: bytes) -> Optional[torch.Tensor]:
        """
        Process a single ~2-second, 16kHz, mono, 16-bit PCM audio chunk.

        Runs Silero VAD to locate speech segments within the chunk, drops
        silent frames, and concatenates the remaining speech into one
        tensor.

        Args:
            audio_bytes: Raw little-endian int16 PCM bytes representing a
                2-second chunk at self.sample_rate.

        Returns:
            A 1-D float32 torch.Tensor containing only the active speech
            audio, or None if the chunk contains no detected speech.
        """
        try:
            waveform = self._bytes_to_tensor(audio_bytes)
        except ValueError:
            logger.warning("Skipping chunk: could not decode audio bytes")
            return None

        num_samples = waveform.shape[0]
        if num_samples != EXPECTED_SAMPLES:
            logger.debug(
                "Chunk length %d samples differs from expected %d for a %ds chunk",
                num_samples,
                EXPECTED_SAMPLES,
                CHUNK_SECONDS,
            )

        waveform = waveform.to(self.device)

        with torch.no_grad():
            speech_timestamps = self._get_speech_timestamps(
                waveform,
                self.model,
                sampling_rate=self.sample_rate,
                threshold=self.threshold,
            )

        if not speech_timestamps:
            logger.debug("Chunk is entirely silence; dropping")
            return None

        # collect_chunks concatenates only the samples within each detected
        # speech segment, discarding everything else (i.e. the silence).
        speech_audio = self._collect_chunks(speech_timestamps, waveform)

        if speech_audio is None or speech_audio.numel() == 0:
            return None

        return speech_audio.cpu()