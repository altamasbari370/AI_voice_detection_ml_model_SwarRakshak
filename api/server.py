"""
api/server.py

Production FastAPI server for audio deepfake / voice-clone detection.

Exposes a single-file inference REST API on top of the LCNN pipeline:

    uploaded audio file
        -> in-memory waveform load (torchaudio, no disk I/O)
        -> mono mixdown + resample to 16kHz + pad/truncate to 2.0s
        -> SpectrogramExtractor  (Log-Mel Spectrogram)
        -> LCNNClassifier        (spoof probability in [0.0, 1.0])

NOTE: this REST /predict design supersedes any earlier WebSocket-based
version of api/server.py from this same pipeline (which streamed live
2-second chunks over /analyze/{session_id}). This version is a
stateless, single-shot file upload endpoint instead — the two are not
meant to run side by side; choose one server per project unless the
routes are also merged into one app.
"""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from models.lcnn_detector import LCNNClassifier
from processing.feature_extractor import SpectrogramExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = Path("models/lcnn_best.pth")
TARGET_SR = 16_000
TARGET_DURATION_SEC = 2.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION_SEC)  # 32,000
SPOOF_THRESHOLD = 0.5
ALLOWED_EXTENSIONS = {".wav"}


class PredictionResponse(BaseModel):
    """Response schema for POST /predict."""

    filename: str
    spoof_probability: float
    prediction: str
    confidence: float


class HealthResponse(BaseModel):
    """Response schema for GET /health."""

    status: str
    device: str
    model_loaded: bool


class PipelineState:
    """Container for pipeline components initialized at server startup."""

    def __init__(self) -> None:
        self.device: Optional[torch.device] = None
        self.extractor: Optional[SpectrogramExtractor] = None
        self.model: Optional[LCNNClassifier] = None
        self.model_loaded: bool = False


pipeline = PipelineState()


def setup_device() -> torch.device:
    """
    Detect and bind to a CUDA GPU if available, otherwise fall back to CPU.
    Logs device name and VRAM info when CUDA is used.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(device)
        total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        logger.info("CUDA available — using GPU: %s (%.2f GB VRAM)", gpu_name, total_vram_gb)
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available — falling back to CPU")

    return device


def load_model_checkpoint(
    model: LCNNClassifier, checkpoint_path: Path, device: torch.device
) -> bool:
    """
    Load model weights from `checkpoint_path` into `model`.

    Handles the checkpoint format saved by train.py:
        {"model_state_dict": ..., "epoch": int, "val_loss": float}
    as well as a bare state_dict, for flexibility.
    """
    if not checkpoint_path.is_file():
        logger.error("Checkpoint file not found at %s", checkpoint_path.resolve())
        return False

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception:
        logger.exception("Failed to load checkpoint file at %s", checkpoint_path)
        return False

    try:
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            epoch = checkpoint.get("epoch", "unknown")
            val_loss = checkpoint.get("val_loss", "unknown")
            logger.info("Loading checkpoint (epoch=%s, val_loss=%s)", epoch, val_loss)
        else:
            state_dict = checkpoint
            logger.info("Loading checkpoint as a bare state_dict")

        model.load_state_dict(state_dict)
    except Exception:
        logger.exception("Failed to apply state_dict from checkpoint")
        return False

    return True


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    FastAPI lifespan handler: initializes the device, model, and feature
    extractor once at startup, and makes them available to request
    handlers via the module-level `pipeline` object.
    """
    logger.info("Starting up: initializing inference pipeline...")

    pipeline.device = setup_device()
    pipeline.extractor = SpectrogramExtractor(device=pipeline.device.type)
    pipeline.model = LCNNClassifier().to(pipeline.device)

    pipeline.model_loaded = load_model_checkpoint(pipeline.model, CHECKPOINT_PATH, pipeline.device)
    if not pipeline.model_loaded:
        logger.warning(
            "Server starting WITHOUT valid trained weights — "
            "/predict will use an untrained model until this is fixed"
        )

    pipeline.model.eval()

    logger.info(
        "Startup complete (device=%s, model_loaded=%s)",
        pipeline.device,
        pipeline.model_loaded,
    )

    yield

    logger.info("Shutting down inference pipeline")


app = FastAPI(
    title="🛡️ SwarRakshak AI - Voice Deepfake Detection API",
    description="""
### 📌 How to Test the Model:
1. Click on the **POST /predict** bar below.
2. Click the **"Try it out"** button on the top right.
3. Click **"Choose File"** and upload any `.wav` audio sample.
4. Click the blue **"Execute"** button to view real-time classification and confidence score.
    """,
    version="1.0.0",
    lifespan=lifespan,
)


def _validate_upload(file: UploadFile) -> None:
    """Validate the uploaded file has an acceptable audio extension."""
    if file.filename is None:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Allowed types: {sorted(ALLOWED_EXTENSIONS)}"
            ),
        )


def _load_and_standardize_waveform(audio_bytes: bytes, filename: str) -> torch.Tensor:
    """
    Load raw audio bytes into a standardized 1-D waveform tensor: mono,
    resampled to TARGET_SR, and padded/truncated to exactly TARGET_SAMPLES.
    """
    try:
        buffer = io.BytesIO(audio_bytes)
        waveform, original_sr = torchaudio.load(buffer)  # [channels, samples]
    except Exception as exc:
        logger.warning("Failed to decode audio file '%s': %s", filename, exc)
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode '{filename}' as a valid audio file",
        ) from exc

    if waveform.numel() == 0:
        raise HTTPException(status_code=400, detail=f"'{filename}' contains no audio samples")

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if original_sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=TARGET_SR)
        waveform = resampler(waveform)

    num_samples = waveform.shape[-1]
    if num_samples < TARGET_SAMPLES:
        pad_amount = TARGET_SAMPLES - num_samples
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
    elif num_samples > TARGET_SAMPLES:
        waveform = waveform[..., -TARGET_SAMPLES:]

    waveform = waveform.squeeze()
    if waveform.dim() == 0:
        waveform = waveform.unsqueeze(0)

    return waveform.float()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health-check endpoint reporting server status, device, and model load state."""
    return HealthResponse(
        status="ok",
        device=str(pipeline.device) if pipeline.device is not None else "unknown",
        model_loaded=pipeline.model_loaded,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """Run deepfake/spoof detection on an uploaded audio file."""
    _validate_upload(file)

    if pipeline.model is None or pipeline.extractor is None or pipeline.device is None:
        raise HTTPException(status_code=500, detail="Inference pipeline is not initialized")

    try:
        audio_bytes = await file.read()
    except Exception as exc:
        logger.exception("Failed to read uploaded file '%s'", file.filename)
        raise HTTPException(
            status_code=400, detail=f"Could not read uploaded file '{file.filename}'"
        ) from exc

    if not audio_bytes:
        raise HTTPException(status_code=400, detail=f"Uploaded file '{file.filename}' is empty")

    waveform = _load_and_standardize_waveform(audio_bytes, file.filename)

    try:
        waveform = waveform.to(pipeline.device)

        with torch.no_grad():
            features = pipeline.extractor.extract_features(waveform)
            output = pipeline.model(features)
            spoof_probability = float(output.item())

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Inference failed for file '%s'", file.filename)
        raise HTTPException(
            status_code=500, detail="Model inference failed unexpectedly"
        ) from exc

    prediction = "FAKE" if spoof_probability >= SPOOF_THRESHOLD else "REAL"
    confidence = (spoof_probability if prediction == "FAKE" else 1.0 - spoof_probability) * 100.0

    result = PredictionResponse(
        filename=file.filename,
        spoof_probability=round(spoof_probability, 4),
        prediction=prediction,
        confidence=round(confidence, 2),
    )

    logger.info(
        "Prediction for '%s': %s (spoof_probability=%.4f, confidence=%.2f%%)",
        file.filename,
        prediction,
        spoof_probability,
        confidence,
    )

    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
