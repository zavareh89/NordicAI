from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable, Sequence

from ..types import FrameSample, SplitDefinition


def xyxy_to_coco(
    bbox: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid COCO box converted from {bbox}")
    return (x1, y1, width, height)


def _build_coco(samples: list[FrameSample], class_names: Sequence[str]) -> dict:
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


def _export_partition(
    sample_map: dict[int, FrameSample],
    frame_ids: Sequence[int],
    part_dir: Path,
    class_names: Sequence[str],
) -> Path:
    # Avoid stale images/marker files when switching split strategies.
    if part_dir.exists():
        shutil.rmtree(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)
    selected: list[FrameSample] = []
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
    annotation_path = part_dir / "_annotations.coco.json"
    with annotation_path.open("w", encoding="utf-8") as handle:
        json.dump(_build_coco(selected, class_names), handle, indent=2)
    return annotation_path


def export_coco_dataset(
    samples: Iterable[FrameSample],
    split: SplitDefinition,
    output_root: str | Path,
    class_names: Sequence[str],
) -> tuple[Path, Path]:
    """Export RF-DETR train/valid layout.

    RF-DETR 1.5.2 expects `valid/` even when we intentionally have no independent
    validation set. In all-frames mode the valid partition is therefore a mirror of
    train *for framework compatibility only*. Training disables early stopping and
    validation-based checkpoint selection, and the mirrored scores are never reported
    as held-out metrics.
    """
    output_root = Path(output_root)
    sample_map = {sample.frame_id: sample for sample in samples}

    train_ids = tuple(split.train_frame_ids)
    independent_val = bool(split.val_frame_ids)
    valid_ids = tuple(split.val_frame_ids) if independent_val else train_ids

    train_json = _export_partition(
        sample_map, train_ids, output_root / "train", class_names
    )
    valid_json = _export_partition(
        sample_map, valid_ids, output_root / "valid", class_names
    )

    if not independent_val:
        (output_root / "valid" / "VALIDATION_IS_TRAIN_MIRROR.txt").write_text(
            "This valid/ directory mirrors train/ because RF-DETR 1.5.2 requires a "
            "validation dataset. It is not an independent validation set and must not "
            "be used for model selection or reported generalization metrics.\n",
            encoding="utf-8",
        )

    return train_json, valid_json
