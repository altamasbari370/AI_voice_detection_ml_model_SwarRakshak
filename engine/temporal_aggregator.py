"""
engine/temporal_aggregator.py

Manages the temporal risk state of active streaming audio-verification
sessions. Each session accumulates per-chunk spoof/clone risk scores
(e.g. from models/lcnn_detector.py) over the life of a call, tracks a
running Exponential Moving Average (EMA) for real-time anomaly signaling,
and produces a final aggregated verdict once the session ends.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ANOMALY_EMA_THRESHOLD = 0.75
FRAUD_CHUNK_THRESHOLD = 0.75
SPOOF_GLOBAL_AVERAGE_THRESHOLD = 0.65
SPOOF_PEAK_RISK_THRESHOLD = 0.85

DEFAULT_ALPHA = 0.3


class SessionNotFoundError(KeyError):
    """Raised when an operation references a session_id with no active state."""


class _SessionState:
    """Internal per-session state container."""

    __slots__ = ("raw_scores", "ema")

    def __init__(self) -> None:
        self.raw_scores: List[float] = []
        self.ema: Optional[float] = None


class TemporalAggregator:
    """
    Tracks per-chunk risk scores for active streaming sessions and
    aggregates them into a final call-level verdict.

    Thread-safe: a single lock guards all state mutation, since chunks
    for a given session are expected to arrive sequentially over a
    streaming connection, but multiple sessions/handlers may be
    processed concurrently within the same process.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, _SessionState] = {}
        self._lock = threading.Lock()

    def add_chunk_score(
        self,
        session_id: str,
        raw_score: float,
        alpha: float = DEFAULT_ALPHA,
    ) -> Dict[str, Any]:
        """
        Record a new chunk's raw risk score for a session and update its
        running EMA.

        If this is the first chunk seen for `session_id`, session state is
        created automatically and the EMA is initialized to `raw_score`.

        Args:
            session_id: Unique identifier for the streaming session.
            raw_score: Risk score for this chunk in [0.0, 1.0], typically
                the output of LCNNClassifier.
            alpha: EMA smoothing factor in (0, 1]. Higher values weight
                recent chunks more heavily.

        Returns:
            {
                "session_id": str,
                "chunk_index": int,       # 0-based index of this chunk
                "raw_score": float,
                "smoothed_score_ema": float,
                "is_anomaly": bool,       # True if smoothed_score_ema > 0.75
            }

        Raises:
            ValueError: If raw_score or alpha are outside valid ranges.
        """
        if not 0.0 <= raw_score <= 1.0:
            raise ValueError(f"raw_score must be in [0.0, 1.0], got {raw_score}")
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0.0, 1.0], got {alpha}")

        with self._lock:
            state = self._sessions.setdefault(session_id, _SessionState())

            if state.ema is None:
                # First chunk for this session: seed the EMA with the raw score
                state.ema = raw_score
            else:
                state.ema = alpha * raw_score + (1 - alpha) * state.ema

            state.raw_scores.append(raw_score)
            chunk_index = len(state.raw_scores) - 1
            smoothed_score_ema = state.ema

        is_anomaly = smoothed_score_ema > ANOMALY_EMA_THRESHOLD

        result = {
            "session_id": session_id,
            "chunk_index": chunk_index,
            "raw_score": raw_score,
            "smoothed_score_ema": smoothed_score_ema,
            "is_anomaly": is_anomaly,
        }

        logger.debug("add_chunk_score(%s) -> %s", session_id, result)
        return result

    def finalize_session(self, session_id: str) -> Dict[str, Any]:
        """
        Compute the final aggregated risk summary for a session and
        remove its state from memory.

        Args:
            session_id: Unique identifier for the streaming session.

        Returns:
            {
                "session_id": str,
                "global_average": float,
                "peak_risk": float,
                "fraud_chunk_ratio": float,   # percentage, e.g. 40.0 for 40%
                "total_chunks_processed": int,
                "final_verdict": "SPOOF_DETECTED" | "GENUINE",
            }

        Raises:
            SessionNotFoundError: If no active state exists for session_id.
        """
        with self._lock:
            state = self._sessions.pop(session_id, None)

        if state is None:
            raise SessionNotFoundError(
                f"No active session found for session_id={session_id!r}"
            )

        scores = state.raw_scores
        total_chunks_processed = len(scores)

        if total_chunks_processed == 0:
            # Defensive fallback: a session was created but never scored.
            global_average = 0.0
            peak_risk = 0.0
            fraud_chunk_ratio = 0.0
        else:
            global_average = sum(scores) / total_chunks_processed
            peak_risk = max(scores)
            fraud_chunks = sum(1 for s in scores if s > FRAUD_CHUNK_THRESHOLD)
            fraud_chunk_ratio = (fraud_chunks / total_chunks_processed) * 100.0

        final_verdict = (
            "SPOOF_DETECTED"
            if (
                global_average > SPOOF_GLOBAL_AVERAGE_THRESHOLD
                or peak_risk > SPOOF_PEAK_RISK_THRESHOLD
            )
            else "GENUINE"
        )

        summary = {
            "session_id": session_id,
            "global_average": global_average,
            "peak_risk": peak_risk,
            "fraud_chunk_ratio": fraud_chunk_ratio,
            "total_chunks_processed": total_chunks_processed,
            "final_verdict": final_verdict,
        }

        logger.info("finalize_session(%s) -> %s", session_id, summary)
        return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    aggregator = TemporalAggregator()
    session_id = "demo-call-001"

    # Simulated 5 chunks of raw risk scores from a streaming call
    simulated_raw_scores = [0.10, 0.55, 0.82, 0.91, 0.60]

    print(f"--- Streaming session: {session_id} ---")
    for raw_score in simulated_raw_scores:
        result = aggregator.add_chunk_score(session_id, raw_score, alpha=0.3)
        print(
            f"chunk {result['chunk_index']}: "
            f"raw={result['raw_score']:.2f}  "
            f"ema={result['smoothed_score_ema']:.4f}  "
            f"anomaly={result['is_anomaly']}"
        )

    print("\n--- Finalizing session ---")
    summary = aggregator.finalize_session(session_id)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")