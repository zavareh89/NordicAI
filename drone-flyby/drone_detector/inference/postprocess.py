from __future__ import annotations

from typing import Mapping, Sequence

from ..types import Detection


def clip_box_pixels(
    bbox: Sequence[float], width: int, height: int, epsilon: float = 1e-6
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    x1, x2 = max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))
    y1, y2 = max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))
    if x2 - x1 <= epsilon or y2 - y1 <= epsilon:
        return None
    return (x1, y1, x2, y2)


def _iou(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = a.bbox_xyxy
    bx1, by1, bx2, by2 = b.bbox_xyxy
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def classwise_nms(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    """Simple optional class-wise NMS. Disabled for RF-DETR by default."""
    kept: list[Detection] = []
    by_class: dict[int, list[Detection]] = {}
    for detection in detections:
        by_class.setdefault(detection.class_id, []).append(detection)
    for candidates in by_class.values():
        candidates.sort(key=lambda d: d.confidence, reverse=True)
        while candidates:
            best = candidates.pop(0)
            kept.append(best)
            candidates = [candidate for candidate in candidates if _iou(best, candidate) <= iou_threshold]
    kept.sort(key=lambda d: d.confidence, reverse=True)
    return kept


def filter_detections(
    detections: Sequence[Detection],
    *,
    width: int,
    height: int,
    confidence_threshold: float,
    class_thresholds: Mapping[str, float] | None,
    max_detections: int,
    apply_nms: bool = False,
    nms_iou_threshold: float = 0.60,
) -> list[Detection]:
    class_thresholds = class_thresholds or {}
    output: list[Detection] = []
    for detection in detections:
        threshold = float(class_thresholds.get(detection.class_name, confidence_threshold))
        if detection.confidence < threshold:
            continue
        clipped = clip_box_pixels(detection.bbox_xyxy, width, height)
        if clipped is None:
            continue
        output.append(
            Detection(
                class_id=detection.class_id,
                class_name=detection.class_name,
                confidence=min(1.0, max(0.0, float(detection.confidence))),
                bbox_xyxy=clipped,
            )
        )
    output.sort(key=lambda d: d.confidence, reverse=True)
    if apply_nms:
        output = classwise_nms(output, nms_iou_threshold)
    return output[:max_detections]


def view_pixels_to_global_normalized(
    bbox_xyxy: Sequence[float],
    *,
    view_width: int,
    view_height: int,
    source_region_xyxy: Sequence[int],
    original_width: int,
    original_height: int,
) -> tuple[float, float, float, float]:
    """Pure-math equivalent of official utils.view_bbox_to_global for tests/fallback."""
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    local = (x1 / view_width, y1 / view_height, x2 / view_width, y2 / view_height)
    sx1, sy1, sx2, sy2 = (float(v) for v in source_region_xyxy)
    sw, sh = sx2 - sx1, sy2 - sy1
    source = (
        sx1 + local[0] * sw,
        sy1 + local[1] * sh,
        sx1 + local[2] * sw,
        sy1 + local[3] * sh,
    )
    return (
        source[0] / original_width,
        source[1] / original_height,
        source[2] / original_width,
        source[3] / original_height,
    )
