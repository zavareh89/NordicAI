from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from drone_detector.data.coco_export import export_coco_dataset, xyxy_to_coco
from drone_detector.data.yolo_export import export_yolo_dataset, xyxy_to_yolo
from drone_detector.types import DetectionAnnotation, FrameSample, SplitDefinition


def _sample(tmp_path: Path, frame_id: int, class_id: int, class_name: str) -> FrameSample:
    image = np.zeros((540, 960, 3), dtype=np.uint8)
    image_path = tmp_path / f"frame_{frame_id:06d}.png"
    assert cv2.imwrite(str(image_path), image)
    return FrameSample(
        frame_id=frame_id,
        image_path=image_path,
        width=960,
        height=540,
        annotations=(
            DetectionAnnotation(frame_id, class_id, class_name, (96.0, 54.0, 192.0, 108.0)),
        ),
    )


def test_yolo_box_is_normalized_to_unit_interval() -> None:
    xc, yc, width, height = xyxy_to_yolo((96, 54, 192, 108), 960, 540)
    assert (xc, yc, width, height) == (0.15, 0.15, 0.1, 0.1)
    assert all(0.0 <= value <= 1.0 for value in (xc, yc, width, height))


def test_yolo_export_uses_shared_class_order(tmp_path: Path) -> None:
    classes = ("hangar", "tank")
    samples = (_sample(tmp_path, 0, 0, "hangar"), _sample(tmp_path, 1, 1, "tank"))
    split = SplitDefinition((0,), (1,))
    data_yaml = export_yolo_dataset(samples, split, tmp_path / "yolo", classes)
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    assert payload["names"] == {0: "hangar", 1: "tank"}
    values = (tmp_path / "yolo/labels/train/frame_000000.txt").read_text().split()
    assert values[0] == "0"
    assert all(0.0 <= float(v) <= 1.0 for v in values[1:])


def test_coco_export_has_unique_ids_and_valid_xywh(tmp_path: Path) -> None:
    classes = ("hangar", "tank")
    samples = (_sample(tmp_path, 0, 0, "hangar"), _sample(tmp_path, 1, 1, "tank"))
    split = SplitDefinition((0,), (1,))
    train_json, valid_json = export_coco_dataset(samples, split, tmp_path / "rfdetr", classes)
    train = json.loads(train_json.read_text(encoding="utf-8"))
    valid = json.loads(valid_json.read_text(encoding="utf-8"))
    assert train["categories"] == [
        {"id": 1, "name": "hangar", "supercategory": "object"},
        {"id": 2, "name": "tank", "supercategory": "object"},
    ]
    assert len({row["id"] for row in train["images"]}) == len(train["images"])
    assert train["annotations"][0]["bbox"] == [96.0, 54.0, 96.0, 54.0]
    assert train["annotations"][0]["category_id"] == 1
    assert valid["annotations"][0]["category_id"] == 2
    assert xyxy_to_coco((1, 2, 5, 8)) == (1, 2, 4, 6)
