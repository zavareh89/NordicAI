from __future__ import annotations

from typing import Sequence

from ..types import Detection
from ..inference.postprocess import view_pixels_to_global_normalized


def detection_to_global_bbox(request, detection: Detection) -> tuple[float, float, float, float] | None:
    """Use the official challenge conversion helper when available."""
    width = int(request.view.width)
    height = int(request.view.height)
    x1, y1, x2, y2 = detection.bbox_xyxy
    view_normalized = (x1 / width, y1 / height, x2 / width, y2 / height)
    try:
        from utils import clip_bbox_to_frame, view_bbox_to_global

        global_bbox = view_bbox_to_global(
            view_normalized,
            request.view.source_region_xyxy,
            request.original_width,
            request.original_height,
        )
        clipped = clip_bbox_to_frame(global_bbox)
        return tuple(float(v) for v in clipped) if clipped is not None else None
    except ImportError:
        # Primarily useful for lightweight unit tests outside the official repo.
        global_bbox = view_pixels_to_global_normalized(
            detection.bbox_xyxy,
            view_width=width,
            view_height=height,
            source_region_xyxy=request.view.source_region_xyxy,
            original_width=request.original_width,
            original_height=request.original_height,
        )
        x1, y1, x2, y2 = global_bbox
        x1, x2 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))
        y1, y2 = max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))
        return None if not (x1 < x2 and y1 < y2) else (x1, y1, x2, y2)


def build_challenge_response(request, detections: Sequence[Detection]):
    from dtos import DroneFlybyPredictionDto, DroneFlybyPredictResponseDto

    annotations = []
    for detection in detections:
        global_bbox = detection_to_global_bbox(request, detection)
        if global_bbox is None:
            continue
        annotations.append(
            DroneFlybyPredictionDto(
                object_id=detection.class_name,
                bbox=list(global_bbox),
                confidence=float(detection.confidence),
            )
        )
    # Critical E1/E2 baseline behavior: null means leave camera exactly where it is.
    return DroneFlybyPredictResponseDto(
        request_id=request.request_id,
        frame=request.frame,
        annotations=annotations,
        requested_view=None,
    )
