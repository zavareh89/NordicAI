from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from ..config import load_config, resolve_path
from ..data.challenge_annotations import get_challenge_classes
from ..detector.factory import create_detector
from ..inference.predictor import Predictor
from ..runtime_telemetry import get_runtime_telemetry

_RUNTIME: dict[str, Any] = {}
logger = logging.getLogger(__name__)


def _resolve_default_config() -> Path:
    explicit = os.environ.get("DRONE_DETECTOR_CONFIG")
    if explicit:
        return Path(explicit).resolve()
    return (Path(__file__).resolve().parents[2] / "configs" / "e1_yolo26s.yaml").resolve()


def _get_predictor() -> Predictor:
    config_path = _resolve_default_config()
    weights_override = os.environ.get("DRONE_DETECTOR_WEIGHTS")
    cache_key = (str(config_path), str(weights_override or ""))
    if _RUNTIME.get("cache_key") == cache_key:
        return _RUNTIME["predictor"]

    cfg = load_config(config_path)
    classes = get_challenge_classes()
    detector = create_detector(cfg, classes)
    if weights_override:
        weights_path = Path(weights_override).expanduser().resolve()
    else:
        configured = cfg["model"].get("deployment_weights")
        if not configured:
            raise ValueError("model.deployment_weights must be configured for challenge serving")
        weights_path = resolve_path(cfg, configured)
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Challenge fine-tuned checkpoint not found: {weights_path}. "
            "Train the scenario first or set DRONE_DETECTOR_WEIGHTS explicitly."
        )
    detector.load(str(weights_path))
    predictor = Predictor(detector, cfg)
    _RUNTIME.clear()
    _RUNTIME.update({"cache_key": cache_key, "cfg": cfg, "predictor": predictor})
    return predictor


def initialize_runtime(*, warmup: bool = True) -> Predictor:
    """Load the static detector once before serving the first challenge frame."""
    predictor = _get_predictor()
    if warmup:
        cfg = _RUNTIME["cfg"]
        predictor.warmup(iterations=int(cfg["inference"].get("warmup_iterations", 3)))
    return predictor


def predict(request):
    """Drop-in replacement for official example.predict(), with passive telemetry."""
    predict_start = time.perf_counter()
    telemetry = get_runtime_telemetry()
    arrival = telemetry.begin_frame(
        frame=int(request.frame),
        request_id=str(request.request_id),
        arrival_monotonic=predict_start,
    )

    status = "error"
    error_text: str | None = None
    annotation_count = 0
    decode_ms = 0.0
    dto_ms = 0.0
    latency_payload: dict[str, float] = {
        "preprocess_ms": 0.0,
        "inference_ms": 0.0,
        "postprocess_ms": 0.0,
        "total_ms": 0.0,
    }

    try:
        if int(request.view.resolution_level) != 0:
            raise ValueError(
                "E1/E2 are Level-0-only baselines. A non-Level-0 request indicates the camera "
                "was moved by another solution/session. Restart the evaluator/server."
            )

        from utils import decode_view

        decode_start = time.perf_counter()
        image_bgr = decode_view(request.view)
        decode_ms = (time.perf_counter() - decode_start) * 1000.0
        if image_bgr.shape[:2] != (540, 960):
            raise ValueError(
                f"Expected official transmitted view 960x540, got "
                f"{image_bgr.shape[1]}x{image_bgr.shape[0]}"
            )

        predictor = _get_predictor()
        try:
            detections, latency = predictor.predict(image_bgr)
            latency_payload = latency.to_dict()
            status = "ok"
        except Exception as exc:
            # Keep the official frame response valid while recording the failed inference.
            logger.exception("Detector inference failed on frame %s", request.frame)
            detections = []
            status = "inference_error_empty_response"
            error_text = repr(exc)

        from .api_adapter import build_challenge_response

        dto_start = time.perf_counter()
        response = build_challenge_response(request, detections)
        dto_ms = (time.perf_counter() - dto_start) * 1000.0
        annotation_count = len(response.annotations)
        return response

    except Exception as exc:
        error_text = repr(exc)
        if status == "error":
            status = "request_error"
        raise

    finally:
        predict_total_ms = (time.perf_counter() - predict_start) * 1000.0
        timing = {
            **latency_payload,
            "decode_ms": decode_ms,
            "dto_ms": dto_ms,
            "predict_total_ms": predict_total_ms,
        }
        telemetry.finish_frame(
            frame=int(request.frame),
            request_id=str(request.request_id),
            status=status,
            timing_ms=timing,
            annotation_count=annotation_count,
            error=error_text,
        )

        newly_skipped = arrival.get("newly_skipped_frames") or []
        logger.info(
            "frame=%s status=%s skipped_now=%s interarrival_ms=%s "
            "decode_ms=%.3f preprocess_ms=%.3f inference_ms=%.3f "
            "postprocess_ms=%.3f dto_ms=%.3f predict_total_ms=%.3f detections=%d",
            request.frame,
            status,
            newly_skipped,
            (
                f"{arrival['interarrival_ms']:.3f}"
                if arrival.get("interarrival_ms") is not None
                else "n/a"
            ),
            decode_ms,
            float(latency_payload.get("preprocess_ms", 0.0)),
            float(latency_payload.get("inference_ms", 0.0)),
            float(latency_payload.get("postprocess_ms", 0.0)),
            dto_ms,
            predict_total_ms,
            annotation_count,
        )
