from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from ..types import FrameSample


def validate_samples(samples: Iterable[FrameSample], class_names: Sequence[str]) -> None:
    samples = tuple(samples)
    if not samples:
        raise ValueError("Dataset has no samples")
    counts = Counter(sample.frame_id for sample in samples)
    duplicates = [frame for frame, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate frame/image IDs: {duplicates}")
    classes = tuple(class_names)
    for sample in samples:
        if not Path(sample.image_path).is_file():
            raise FileNotFoundError(f"Missing image file: {sample.image_path}")
        if sample.width <= 0 or sample.height <= 0:
            raise ValueError(f"Invalid dimensions for frame {sample.frame_id}")
        for ann in sample.annotations:
            if not 0 <= ann.class_id < len(classes):
                raise ValueError(f"Invalid class id {ann.class_id} on frame {sample.frame_id}")
            if classes[ann.class_id] != ann.class_name:
                raise ValueError(
                    f"Class mapping mismatch on frame {sample.frame_id}: "
                    f"id={ann.class_id}, name={ann.class_name!r}"
                )
            x1, y1, x2, y2 = ann.bbox_xyxy
            if not (0.0 <= x1 < x2 <= sample.width and 0.0 <= y1 < y2 <= sample.height):
                raise ValueError(
                    f"Invalid/out-of-bounds bbox {ann.bbox_xyxy} on frame {sample.frame_id}"
                )
