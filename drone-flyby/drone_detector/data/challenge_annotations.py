from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import cv2

from ..types import DetectionAnnotation, FrameSample, ensure_xyxy

_FRAME_RE = re.compile(r"frame_(\d+)\.(?:png|json)$")


def get_challenge_classes() -> tuple[str, ...]:
    """Read the canonical class order from the official challenge's dtos.py."""
    try:
        from dtos import OBJECT_CLASSES
    except ImportError as exc:
        raise RuntimeError(
            "Could not import official challenge `dtos.py`. Run this project from "
            "inside (or with PYTHONPATH pointing at) the official drone-flyby directory."
        ) from exc
    return tuple(OBJECT_CLASSES)


def frame_id_from_path(path: str | Path) -> int:
    match = _FRAME_RE.match(Path(path).name)
    if not match:
        raise ValueError(f"Expected frame_XXXXXX filename, got {Path(path).name!r}")
    return int(match.group(1))


def _validate_bbox(
    bbox: Sequence[float], width: int, height: int, annotation_file: Path
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = ensure_xyxy(bbox)
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ValueError(
            f"Out-of-bounds bbox {bbox} in {annotation_file}; image={width}x{height}"
        )
    return x1, y1, x2, y2


def parse_annotation_file(
    annotation_file: str | Path,
    *,
    image_width: int,
    image_height: int,
    class_names: Sequence[str] | None = None,
) -> tuple[DetectionAnnotation, ...]:
    annotation_file = Path(annotation_file)
    classes = tuple(class_names) if class_names is not None else get_challenge_classes()
    class_to_id = {name: idx for idx, name in enumerate(classes)}
    if len(class_to_id) != len(classes):
        raise ValueError("Challenge class list contains duplicate names")

    with annotation_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("annotations"), list):
        raise ValueError(f"Malformed challenge annotation file: {annotation_file}")

    frame_id = frame_id_from_path(annotation_file)
    if "frame" in payload and int(payload["frame"]) != frame_id:
        raise ValueError(
            f"Frame mismatch in {annotation_file}: filename={frame_id}, json={payload['frame']}"
        )

    output: list[DetectionAnnotation] = []
    for index, raw in enumerate(payload["annotations"]):
        if not isinstance(raw, dict):
            raise ValueError(f"annotations[{index}] must be an object in {annotation_file}")
        class_name = raw.get("object_id")
        if class_name not in class_to_id:
            raise ValueError(
                f"Unknown object_id {class_name!r} in {annotation_file}; legal classes={classes}"
            )
        if "bbox" not in raw:
            raise ValueError(f"annotations[{index}] is missing bbox in {annotation_file}")
        bbox = _validate_bbox(raw["bbox"], image_width, image_height, annotation_file)
        output.append(
            DetectionAnnotation(
                frame_id=frame_id,
                class_id=class_to_id[class_name],
                class_name=class_name,
                bbox_xyxy=bbox,
            )
        )
    return tuple(output)


def discover_scene_samples(
    scene_dir: str | Path,
    *,
    class_names: Sequence[str] | None = None,
) -> tuple[FrameSample, ...]:
    """Parse one official scene (`images/` + `annotations/`) into canonical samples."""
    scene_dir = Path(scene_dir)
    images_dir = scene_dir / "images"
    annotations_dir = scene_dir / "annotations"
    if not images_dir.is_dir() or not annotations_dir.is_dir():
        raise FileNotFoundError(
            f"Expected official scene layout with images/ and annotations/: {scene_dir}"
        )

    image_paths = sorted(images_dir.glob("frame_*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No frame_*.png files found under {images_dir}")
    image_frame_ids = {frame_id_from_path(path) for path in image_paths}
    annotation_frame_ids = {
        frame_id_from_path(path) for path in annotations_dir.glob("frame_*.json")
    }
    orphan_annotations = sorted(annotation_frame_ids - image_frame_ids)
    if orphan_annotations:
        raise FileNotFoundError(
            "Annotation files exist without matching source images for frame IDs: "
            f"{orphan_annotations}"
        )

    samples: list[FrameSample] = []
    seen: set[int] = set()
    for image_path in image_paths:
        frame_id = frame_id_from_path(image_path)
        if frame_id in seen:
            raise ValueError(f"Duplicate frame id {frame_id} in {images_dir}")
        seen.add(frame_id)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        height, width = image.shape[:2]
        annotation_path = annotations_dir / f"frame_{frame_id:06d}.json"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing annotation file: {annotation_path}")
        annotations = parse_annotation_file(
            annotation_path,
            image_width=width,
            image_height=height,
            class_names=class_names,
        )
        samples.append(
            FrameSample(
                frame_id=frame_id,
                image_path=image_path.resolve(),
                width=width,
                height=height,
                annotations=annotations,
            )
        )
    return tuple(samples)
