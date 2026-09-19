from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import cv2

from ..types import DetectionAnnotation, FrameSample

INTERPOLATION_NAME = "INTER_AREA"


def scale_bbox_xyxy(
    bbox: tuple[float, float, float, float],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    src_w, src_h = source_size
    dst_w, dst_h = target_size
    if min(src_w, src_h, dst_w, dst_h) <= 0:
        raise ValueError("Image dimensions must be positive")
    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)
    x1, y1, x2, y2 = bbox
    scaled = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
    return _clip_nonempty_xyxy(scaled, dst_w, dst_h)


def _clip_nonempty_xyxy(
    bbox: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    x1 = min(max(float(x1), 0.0), float(width))
    x2 = min(max(float(x2), 0.0), float(width))
    y1 = min(max(float(y1), 0.0), float(height))
    y2 = min(max(float(y2), 0.0), float(height))
    if not x1 < x2 or not y1 < y2:
        raise ValueError(f"Box became empty after clipping: {bbox} -> {(x1,y1,x2,y2)}")
    return (x1, y1, x2, y2)


def transform_annotations_level0(
    sample: FrameSample,
    target_size: tuple[int, int] = (960, 540),
) -> tuple[DetectionAnnotation, ...]:
    transformed = []
    for ann in sample.annotations:
        transformed.append(
            DetectionAnnotation(
                frame_id=ann.frame_id,
                class_id=ann.class_id,
                class_name=ann.class_name,
                bbox_xyxy=scale_bbox_xyxy(
                    ann.bbox_xyxy,
                    (sample.width, sample.height),
                    target_size,
                ),
            )
        )
    return tuple(transformed)


def prepare_level0_dataset(
    samples: Iterable[FrameSample],
    output_root: str | Path,
    *,
    target_size: tuple[int, int] = (960, 540),
) -> tuple[tuple[FrameSample, ...], dict[str, Any]]:
    """Materialize exact evaluator-style Level-0 PNGs with cv2.INTER_AREA."""
    output_root = Path(output_root)
    images_out = output_root / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    prepared: list[FrameSample] = []
    manifest_frames: list[dict[str, Any]] = []
    dst_w, dst_h = target_size
    for source in sorted(samples, key=lambda s: s.frame_id):
        image = cv2.imread(str(source.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read source image {source.image_path}")
        actual_h, actual_w = image.shape[:2]
        if (actual_w, actual_h) != (source.width, source.height):
            raise ValueError(
                f"Image shape changed for {source.image_path}: sample says "
                f"{source.width}x{source.height}, read {actual_w}x{actual_h}"
            )
        resized = cv2.resize(image, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
        target_path = images_out / f"frame_{source.frame_id:06d}.png"
        if not cv2.imwrite(str(target_path), resized):
            raise RuntimeError(f"Could not write Level-0 image {target_path}")
        anns = transform_annotations_level0(source, target_size)
        sample = FrameSample(
            frame_id=source.frame_id,
            image_path=target_path.resolve(),
            width=dst_w,
            height=dst_h,
            annotations=anns,
        )
        prepared.append(sample)
        manifest_frames.append(
            {
                "frame_id": source.frame_id,
                "source_image": str(source.image_path),
                "level0_image": str(target_path.resolve()),
                "source_width": source.width,
                "source_height": source.height,
                "target_width": dst_w,
                "target_height": dst_h,
                "scale_x": dst_w / float(source.width),
                "scale_y": dst_h / float(source.height),
                "annotations": [ann.to_dict() for ann in anns],
            }
        )

    manifest = {
        "preprocessing": {
            "operation": "cv2.resize",
            "target_width": dst_w,
            "target_height": dst_h,
            "interpolation": INTERPOLATION_NAME,
        },
        "frames": manifest_frames,
    }
    with (output_root / "dataset_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return tuple(prepared), manifest
