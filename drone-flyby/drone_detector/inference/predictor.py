from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..detector.base import BaseDetector
from ..types import Detection, LatencyBreakdown
from .postprocess import filter_detections


def _cuda_sync_if_needed() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Predictor:
    """Generic single-frame predictor. It intentionally stores no temporal state."""

    def __init__(self, detector: BaseDetector, cfg: dict[str, Any]) -> None:
        self.detector = detector
        self.cfg = cfg

    def predict(self, image_bgr: np.ndarray) -> tuple[list[Detection], LatencyBreakdown]:
        total_start = time.perf_counter()

        pre_start = time.perf_counter()
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 BGR image, got shape={image_bgr.shape}")
        image = np.ascontiguousarray(image_bgr)
        height, width = image.shape[:2]
        preprocess_ms = (time.perf_counter() - pre_start) * 1000.0

        _cuda_sync_if_needed()
        infer_start = time.perf_counter()
        raw = self.detector.predict(image)
        _cuda_sync_if_needed()
        inference_ms = (time.perf_counter() - infer_start) * 1000.0

        post_start = time.perf_counter()
        infer_cfg = self.cfg["inference"]
        # YOLO already performs NMS. RF-DETR's official decoding normally does not use
        # classical NMS, so optional extra NMS defaults off and must be enabled explicitly.
        apply_nms = bool(infer_cfg.get("apply_extra_nms", False)) and not self.detector.performs_nms
        detections = filter_detections(
            raw,
            width=width,
            height=height,
            confidence_threshold=float(infer_cfg.get("confidence_threshold", 0.08)),
            class_thresholds=infer_cfg.get("class_thresholds", {}),
            max_detections=int(infer_cfg.get("max_detections", 200)),
            apply_nms=apply_nms,
            nms_iou_threshold=float(infer_cfg.get("nms_iou_threshold", 0.60)),
        )
        postprocess_ms = (time.perf_counter() - post_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0
        return detections, LatencyBreakdown(
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
            postprocess_ms=postprocess_ms,
            total_ms=total_ms,
        )

    def warmup(self, image_shape: tuple[int, int, int] = (540, 960, 3), iterations: int = 3) -> None:
        dummy = np.zeros(image_shape, dtype=np.uint8)
        for _ in range(max(0, iterations)):
            self.predict(dummy)
