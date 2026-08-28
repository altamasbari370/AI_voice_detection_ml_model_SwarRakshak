"""
test_client.py

Simple test client for api/server.py's WebSocket streaming endpoint.

Connects to ws://localhost:8000/analyze/{session_id}, streams 5 synthetic
2-second 16kHz PCM audio chunks (random noise -- not real speech, so the
server's VAD stage may legitimately skip most/all of them as silence),
prints each chunk's server response, then sends the finalization signal
and prints the aggregated summary.
"""

from __future__ import annotations

import asyncio
import json
import logging

import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SERVER_URI = "ws://localhost:8000/analyze/call_demo_123"
NUM_CHUNKS = 5
CHUNK_SAMPLES = 32_000  # 2 seconds at 16kHz
SLEEP_BETWEEN_CHUNKS_SECONDS = 1


# def generate_synthetic_chunk() -> bytes:
#     """
#     Generate one synthetic 2-second, 16kHz, mono, int16 PCM audio chunk
#     as raw bytes (64,000 bytes = 32,000 samples * 2 bytes/sample).
#
#     Note: this is random noise, not real speech -- it exercises the
#     wire protocol and pipeline end-to-end, but the VAD stage may
#     correctly classify some or all of these chunks as silence.
#     """
#     samples = np.random.randint(-32768, 32767, CHUNK_SAMPLES, dtype=np.int16)
#     return samples.tobytes()
# def generate_synthetic_chunk() -> bytes:
#     """
#     Generate a synthetic tone (440 Hz sine wave) that triggers the
#     Silero VAD speech detector so we can test the full pipeline.
#     """
#     sample_rate = 16_000
#     duration = 2.0
#     t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
#     # Sine wave scaled to int16 range
#     samples = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
#     return samples.tobytes()
def generate_synthetic_chunk() -> bytes:
    """
    Generate a frequency-modulated tone that mimics human vocal formants
    so the Silero VAD detector accepts it as speech.
    """
    sample_rate = 16_000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Frequency modulation to simulate vocal cord shifts (speech-like)
    freq = 200 + 50 * np.sin(2 * np.pi * 3 * t)
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate

    # Add a bit of voice-like texture
    signal = np.sin(phase) + 0.1 * np.random.randn(len(t))

    # Normalize and scale to int16 range
    signal = signal / np.max(np.abs(signal))
    samples = (signal * 32767).astype(np.int16)
    return samples.tobytes()

def print_response(chunk_number: int, response: dict) -> None:
    """Pretty-print a single chunk's server response."""
    status = response.get("status")

    if status == "skipped_silence":
        print(f"[chunk {chunk_number}] status=skipped_silence (no speech detected)")

    elif status == "ongoing":
        print(
            f"[chunk {chunk_number}] status=ongoing  "
            f"raw_score={response.get('raw_score'):.4f}  "
            f"ema={response.get('smoothed_score_ema'):.4f}  "
            f"is_anomaly={response.get('is_anomaly')}"
        )

    elif status == "error":
        print(f"[chunk {chunk_number}] status=error  detail={response.get('detail')}")

    else:
        print(f"[chunk {chunk_number}] unrecognized response: {response}")


def print_final_summary(summary: dict) -> None:
    """Pretty-print the final aggregated session summary."""
    print("\n=== Final Aggregation Summary ===")
    if "status" in summary and summary["status"] == "error":
        print(f"Server returned an error: {summary.get('detail')}")
        return

    for key in (
        "session_id",
        "global_average",
        "peak_risk",
        "fraud_chunk_ratio",
        "total_chunks_processed",
        "final_verdict",
    ):
        if key not in summary:
            continue
        value = summary[key]
        if isinstance(value, float):
            print(f"{key:25s}: {value:.4f}")
        else:
            print(f"{key:25s}: {value}")
    print("==================================\n")


async def run_client() -> None:
    """Connect to the server, stream synthetic chunks, then finalize."""
    logger.info("Connecting to %s", SERVER_URI)

    try:
        async with websockets.connect(SERVER_URI) as ws:
            logger.info("Connected. Streaming %d synthetic chunks...", NUM_CHUNKS)

            for chunk_number in range(1, NUM_CHUNKS + 1):
                chunk_bytes = generate_synthetic_chunk()
                await ws.send(chunk_bytes)

                raw_response = await ws.recv()
                response = json.loads(raw_response)
                print_response(chunk_number, response)

                if chunk_number < NUM_CHUNKS:
                    await asyncio.sleep(SLEEP_BETWEEN_CHUNKS_SECONDS)

            logger.info("Finished streaming chunks. Sending finalization signal.")
            await ws.send(json.dumps({"is_final": True}))

            raw_final_response = await ws.recv()
            final_summary = json.loads(raw_final_response)
            print_final_summary(final_summary)

    except ConnectionClosed as exc:
        logger.error("WebSocket connection closed unexpectedly: %s", exc)
    except OSError as exc:
        logger.error(
            "Could not connect to %s (%s). Is the server running?", SERVER_URI, exc
        )
    except Exception:
        logger.exception("Unexpected error during test client run")


if __name__ == "__main__":
    asyncio.run(run_client())