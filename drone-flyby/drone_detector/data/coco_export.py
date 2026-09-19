from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable, Sequence

from ..types import FrameSample, SplitDefinition


def xyxy_to_coco(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid COCO box converted from {bbox}")
    return (x1, y1, width, height)


def _build_coco(
    samples: list[FrameSample], class_names: Sequence[str]
) -> dict:
    categories = [
        {"id": idx + 1, "name": name, "supercategory": "object"}
        for idx, name in enumerate(class_names)
    ]
    images = []
    annotations = []
    annotation_id = 1
    for image_id, sample in enumerate(samples, start=1):
        images.append(
            {
                "id": image_id,
                "file_name": sample.image_path.name,
                "width": sample.width,
                "height": sample.height,
                "frame_id": sample.frame_id,
            }
        )
        for ann in sample.annotations:
            x, y, width, height = xyxy_to_coco(ann.bbox_xyxy)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": ann.class_id + 1,
                    "bbox": [x, y, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    return {
        "info": {"description": "Nordic AI Cup Drone Flyby exact Level-0 export"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def export_coco_dataset(
    samples: Iterable[FrameSample],
    split: SplitDefinition,
    output_root: str | Path,
    class_names: Sequence[str],
) -> tuple[Path, Path]:
    """RF-DETR layout: train/_annotations.coco.json and valid/_annotations.coco.json."""
    output_root = Path(output_root)
    sample_map = {sample.frame_id: sample for sample in samples}
    outputs = []
    for part, frame_ids in (("train", split.train_frame_ids), ("valid", split.val_frame_ids)):
        part_dir = output_root / part
        part_dir.mkdir(parents=True, exist_ok=True)
        selected = []
        for frame_id in frame_ids:
            if frame_id not in sample_map:
                raise KeyError(f"Split references missing frame {frame_id}")
            sample = sample_map[frame_id]
            destination = part_dir / sample.image_path.name
            shutil.copy2(sample.image_path, destination)
            selected.append(
                FrameSample(
                    frame_id=sample.frame_id,
                    image_path=destination,
                    width=sample.width,
                    height=sample.height,
                    annotations=sample.annotations,
                )
            )
        coco = _build_coco(selected, class_names)
        annotation_path = part_dir / "_annotations.coco.json"
        with annotation_path.open("w", encoding="utf-8") as handle:
            json.dump(coco, handle, indent=2)
        outputs.append(annotation_path)
    return outputs[0], outputs[1]
