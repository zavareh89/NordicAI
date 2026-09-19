from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

BBoxXYXY = tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectionAnnotation:
    frame_id: int
    class_id: int
    class_name: str
    bbox_xyxy: BBoxXYXY

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrameSample:
    frame_id: int
    image_path: Path
    width: int
    height: int
    annotations: tuple[DetectionAnnotation, ...]


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: BBoxXYXY

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LatencyBreakdown:
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_ms: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class SplitDefinition:
    train_frame_ids: tuple[int, ...]
    val_frame_ids: tuple[int, ...]
    purged_frame_ids: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, list[int]]:
        return {
            "train_frame_ids": list(self.train_frame_ids),
            "val_frame_ids": list(self.val_frame_ids),
            "purged_frame_ids": list(self.purged_frame_ids),
        }


def bbox_area_xyxy(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = (float(v) for v in box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def ensure_xyxy(box: Sequence[float]) -> BBoxXYXY:
    if len(box) != 4:
        raise ValueError(f"Expected bbox with 4 values, got {len(box)}")
    x1, y1, x2, y2 = (float(v) for v in box)
    if not (x1 < x2 and y1 < y2):
        raise ValueError(f"Invalid xyxy bbox with non-positive area: {box}")
    return (x1, y1, x2, y2)
