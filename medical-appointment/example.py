"""Competition entry point with switchable E1/E2 evidence experiments.

Modes:
  E2_MODE=e1              deterministic E1
  E2_MODE=e2_1            proposal-ID LLM reranker (existing E2.1)
  E2_MODE=e2_1b           direct word-index LLM selector
  E2_MODE=e2_1c           E2.1-B + calibrated boundaries
  E2_MODE=e2_1_ranker     learned proposal ranker + calibration
  E2_MODE=e2_1_ensemble   direct LLM + learned-ranker ensemble + calibration

For backward compatibility, E2_1_ENABLED=1 with no E2_MODE still selects e2_1.
"""
from __future__ import annotations

import logging
import os
from threading import Lock
from typing import Any

from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from utils import audio_duration_seconds, decode_audio

from e1.audio import decode_audio_bytes_16k
from e1.pipeline import DualASRE1Pipeline

logger = logging.getLogger(__name__)

_PIPELINE: Any | None = None
_PIPELINE_LOCK = Lock()


def _env_true(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {
        "1", "true", "yes", "on"
    }


def _mode() -> str:
    explicit = os.environ.get("E2_MODE")
    if explicit:
        return explicit.strip().lower()
    if _env_true("E2_1_ENABLED"):
        return "e2_1"
    return "e1"


def _build_pipeline() -> Any:
    device = os.environ.get("E1_DEVICE", "cuda")
    e1_config = os.environ.get("E1_CONFIG", "config/e1_evidence_tuned.json")
    mode = _mode()

    if mode == "e1":
        logger.info("Loading E1 on %s", device)
        return DualASRE1Pipeline.from_config_file(e1_config, device=device)

    if mode == "e2_1":
        from e1.e2_1_pipeline import DualASRE2_1Pipeline
        e2_config = os.environ.get("E2_1_CONFIG", "config/e2_1_default.json")
        logger.info("Loading E2.1 proposal reranker on %s", device)
        return DualASRE2_1Pipeline.from_config_files(
            e1_config, e2_config, device=device
        )

    if mode == "e2_1b":
        from e1.e2_1b_pipeline import DualASRE2_1BPipeline
        e2_config = os.environ.get("E2_1B_CONFIG", "config/e2_1b_default.json")
        logger.info("Loading E2.1-B direct word selector on %s", device)
        return DualASRE2_1BPipeline.from_config_files(
            e1_config, e2_config, device=device
        )

    if mode == "e2_1c":
        from e1.e2_1c_pipeline import DualASRE2_1CPipeline
        e2_config = os.environ.get("E2_1B_CONFIG", "config/e2_1b_default.json")
        cal_config = os.environ.get(
            "E2_1_CALIBRATION_CONFIG",
            "config/e2_1_boundary_calibration.json",
        )
        logger.info("Loading E2.1-C calibrated word selector on %s", device)
        return DualASRE2_1CPipeline.from_config_files(
            e1_config, e2_config, cal_config, device=device
        )

    if mode == "e2_1_ranker":
        from e1.e2_1_ranker_pipeline import DualASRE2_1RankerPipeline
        ranker_config = os.environ.get(
            "E2_1_RANKER_CONFIG", "config/e2_1_ranker_default.json"
        )
        logger.info("Loading E2.1 learned proposal ranker on %s", device)
        return DualASRE2_1RankerPipeline.from_config_files(
            e1_config, ranker_config, device=device
        )

    if mode == "e2_1_ensemble":
        from e1.e2_1_ensemble_pipeline import DualASRE2_1EnsemblePipeline
        word_config = os.environ.get("E2_1B_CONFIG", "config/e2_1b_default.json")
        ensemble_config = os.environ.get(
            "E2_1_ENSEMBLE_CONFIG", "config/e2_1_ensemble_default.json"
        )
        logger.info("Loading E2.1 ensemble on %s", device)
        return DualASRE2_1EnsemblePipeline.from_config_files(
            e1_config, word_config, ensemble_config, device=device
        )

    raise ValueError(f"Unknown E2_MODE={mode!r}")


def get_pipeline() -> Any:
    global _PIPELINE
    if _PIPELINE is None:
        with _PIPELINE_LOCK:
            if _PIPELINE is None:
                _PIPELINE = _build_pipeline()
    return _PIPELINE


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
        audio, sample_rate, _ = decode_audio_bytes_16k(audio_bytes)
        results = get_pipeline().predict_audio(
            audio,
            list(request.questions),
            sample_rate=sample_rate,
        )
        if not isinstance(results, list):
            raise RuntimeError("pipeline returned non-list output")
        if len(results) != len(request.questions):
            raise RuntimeError(
                f"pipeline returned {len(results)} predictions for "
                f"{len(request.questions)} questions"
            )
        return ASRQuestionResponseDto(
            answers=[item.answer for item in results],
            evidence_start=[item.evidence_start for item in results],
            evidence_end=[item.evidence_end for item in results],
        )
    except Exception:
        logger.exception("pipeline failed for %s", request.audio_filename)
        fallback_duration = duration if duration is not None else 0.0
        return ASRQuestionResponseDto(
            answers=[True] * len(request.questions),
            evidence_start=[0.0 if fallback_duration > 0 else None]
            * len(request.questions),
            evidence_end=[fallback_duration if fallback_duration > 0 else None]
            * len(request.questions),
        )
