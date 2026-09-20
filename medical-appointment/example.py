"""Competition entry point for E1 and E2.1.

Default behavior remains E1.

Enable E2.1 without changing the tuned E1 config:

    export E2_1_ENABLED=1
    export E1_CONFIG=config/e1_evidence_tuned.json
    export E2_1_CONFIG=config/e2_1_default.json

E2.1 keeps every E1 YES/NO decision unchanged and only reranks evidence spans.
"""
from __future__ import annotations

import logging
import os
from threading import Lock
from typing import Any

from dtos import (
    ASRQuestionRequestDto,
    ASRQuestionResponseDto,
)
from utils import (
    audio_duration_seconds,
    decode_audio,
)

from e1.audio import (
    decode_audio_bytes_16k,
)
from e1.pipeline import (
    DualASRE1Pipeline,
)

logger = logging.getLogger(__name__)

_PIPELINE: Any | None = None
_PIPELINE_LOCK = Lock()


def _env_true(
    name: str,
    default: str = "0",
) -> bool:
    return (
        os.environ.get(
            name,
            default,
        ).strip().lower()
        in {
            "1",
            "true",
            "yes",
            "on",
        }
    )


def _build_pipeline() -> Any:
    device = os.environ.get(
        "E1_DEVICE",
        "cuda",
    )
    e1_config_path = os.environ.get(
        "E1_CONFIG",
        "config/e1_default.json",
    )

    if _env_true(
        "E2_1_ENABLED"
    ):
        from e1.e2_1_pipeline import (
            DualASRE2_1Pipeline,
        )

        e2_config_path = os.environ.get(
            "E2_1_CONFIG",
            "config/e2_1_default.json",
        )

        logger.info(
            "Loading Dual-ASR E2.1 on %s "
            "with E1 config %s and E2.1 config %s",
            device,
            e1_config_path,
            e2_config_path,
        )

        return (
            DualASRE2_1Pipeline
            .from_config_files(
                e1_config_path,
                e2_config_path,
                device=device,
            )
        )

    logger.info(
        "Loading Dual-ASR E1 on %s with config %s",
        device,
        e1_config_path,
    )

    return (
        DualASRE1Pipeline
        .from_config_file(
            e1_config_path,
            device=device,
        )
    )


def get_pipeline() -> Any:
    global _PIPELINE

    if _PIPELINE is None:
        with _PIPELINE_LOCK:
            if _PIPELINE is None:
                _PIPELINE = (
                    _build_pipeline()
                )

    return _PIPELINE


# Eager loading is the production default. This ensures all local model weights
# are resident before uvicorn reports the service ready.
if (
    os.environ.get(
        "E1_LAZY_LOAD",
        "0",
    )
    != "1"
):
    _PIPELINE = _build_pipeline()


def predict(
    request: ASRQuestionRequestDto,
) -> ASRQuestionResponseDto:
    audio_bytes = decode_audio(
        request.audio_base64
    )
    duration = (
        audio_duration_seconds(
            audio_bytes
        )
    )

    logger.info(
        "%s (%.1f s, %.1f MB): %d questions",
        request.audio_filename,
        (
            duration
            if duration is not None
            else float("nan")
        ),
        len(audio_bytes) / 1e6,
        len(request.questions),
    )

    try:
        (
            audio,
            sample_rate,
            _decoded_duration,
        ) = decode_audio_bytes_16k(
            audio_bytes
        )

        results = (
            get_pipeline()
            .predict_audio(
                audio,
                list(
                    request.questions
                ),
                sample_rate=sample_rate,
            )
        )

        assert isinstance(
            results,
            list,
        )

        if (
            len(results)
            != len(
                request.questions
            )
        ):
            raise RuntimeError(
                "Pipeline returned "
                f"{len(results)} predictions "
                "for "
                f"{len(request.questions)} questions"
            )

        return ASRQuestionResponseDto(
            answers=[
                result.answer
                for result in results
            ],
            evidence_start=[
                result.evidence_start
                for result in results
            ],
            evidence_end=[
                result.evidence_end
                for result in results
            ],
        )

    except Exception:
        # Preserve the existing challenge HTTP contract even on a catastrophic
        # model/runtime failure.
        logger.exception(
            "Dual-ASR pipeline failed for %s; "
            "returning safe fallback",
            request.audio_filename,
        )

        fallback_duration = (
            duration
            if duration is not None
            else 0.0
        )

        return ASRQuestionResponseDto(
            answers=[
                True
            ] * len(
                request.questions
            ),
            evidence_start=[
                (
                    0.0
                    if fallback_duration > 0
                    else None
                )
            ] * len(
                request.questions
            ),
            evidence_end=[
                (
                    fallback_duration
                    if fallback_duration > 0
                    else None
                )
            ] * len(
                request.questions
            ),
        )
