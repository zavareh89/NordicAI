"""Drop-in competition entry point for Dual-ASR E1.

Replace the official baseline `example.py` with this file. `api.py`, `dtos.py`,
and `utils.py` remain unchanged.
"""
from __future__ import annotations

import logging
import os
from threading import Lock

from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from utils import audio_duration_seconds, decode_audio

from e1.audio import decode_audio_bytes_16k
from e1.pipeline import DualASRE1Pipeline

logger = logging.getLogger(__name__)

_PIPELINE: DualASRE1Pipeline | None = None
_PIPELINE_LOCK = Lock()


def _build_pipeline() -> DualASRE1Pipeline:
    device = os.environ.get("E1_DEVICE", "cuda")
    config_path = os.environ.get("E1_CONFIG", "config/e1_default.json")
    logger.info("Loading Dual-ASR E1 on %s with config %s", device, config_path)
    return DualASRE1Pipeline.from_config_file(config_path, device=device)


def get_pipeline() -> DualASRE1Pipeline:
    global _PIPELINE
    if _PIPELINE is None:
        with _PIPELINE_LOCK:
            if _PIPELINE is None:
                _PIPELINE = _build_pipeline()
    return _PIPELINE


# Eager loading is the production default: uvicorn does not become ready until
# weights are resident. Set E1_LAZY_LOAD=1 only for lightweight import/tests.
if os.environ.get("E1_LAZY_LOAD", "0") != "1":
    _PIPELINE = _build_pipeline()


def predict(request: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    audio_bytes = decode_audio(request.audio_base64)
    duration = audio_duration_seconds(audio_bytes)

    logger.info(
        "%s (%.1f s, %.1f MB): %d questions",
        request.audio_filename,
        duration if duration is not None else float("nan"),
        len(audio_bytes) / 1e6,
        len(request.questions),
    )

    try:
        audio, sample_rate, decoded_duration = decode_audio_bytes_16k(audio_bytes)
        results = get_pipeline().predict_audio(
            audio,
            list(request.questions),
            sample_rate=sample_rate,
        )
        assert isinstance(results, list)
        if len(results) != len(request.questions):
            raise RuntimeError(
                f"E1 returned {len(results)} predictions for {len(request.questions)} questions"
            )

        return ASRQuestionResponseDto(
            answers=[r.answer for r in results],
            evidence_start=[r.evidence_start for r in results],
            evidence_end=[r.evidence_end for r in results],
        )

    except Exception:
        # Never let a conversation fail the HTTP contract. Catastrophic fallback
        # says YES and uses the whole clip only as a last resort. It is poor tIoU,
        # but is preferable to losing all ten questions because the endpoint died.
        logger.exception("Dual-ASR E1 failed for %s; returning safe fallback", request.audio_filename)
        fallback_duration = duration if duration is not None else 0.0
        return ASRQuestionResponseDto(
            answers=[True] * len(request.questions),
            evidence_start=[0.0 if fallback_duration > 0 else None] * len(request.questions),
            evidence_end=[fallback_duration if fallback_duration > 0 else None] * len(request.questions),
        )
