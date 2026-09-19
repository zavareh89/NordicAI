from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from drone_detector.data.challenge_annotations import discover_scene_samples, parse_annotation_file
from drone_detector.data.level0_dataset import prepare_level0_dataset, scale_bbox_xyxy
from drone_detector.types import DetectionAnnotation, FrameSample


def test_official_annotation_parser_uses_absolute_xyxy(tmp_path: Path) -> None:
    annotation = tmp_path / "frame_000007.json"
    annotation.write_text(
        json.dumps(
            {
                "frame": 7,
                "pose": {"x": 0, "y": 0},
                "annotations": [
                    {"object_id": "tank", "bbox": [400, 200, 800, 600]},
                ],
            }
        ),
        encoding="utf-8",
    )
    parsed = parse_annotation_file(
        annotation,
        image_width=3840,
        image_height=2160,
        class_names=("hangar", "tank"),
    )
    assert len(parsed) == 1
    assert parsed[0].frame_id == 7
    assert parsed[0].class_id == 1
    assert parsed[0].class_name == "tank"
    assert parsed[0].bbox_xyxy == (400.0, 200.0, 800.0, 600.0)


def test_level0_bbox_transform_is_generic_and_equals_quarter_for_4k() -> None:
    assert scale_bbox_xyxy((400, 200, 800, 600), (3840, 2160), (960, 540)) == (
        100.0,
        50.0,
        200.0,
        150.0,
    )
    # Also prove the implementation is not a hard-coded /4 transform.
    assert scale_bbox_xyxy((100, 50, 500, 250), (1000, 500), (960, 540)) == (
        96.0,
        54.0,
        480.0,
        270.0,
    )


def test_level0_image_materialization_matches_cv2_inter_area(tmp_path: Path) -> None:
    source_path = tmp_path / "frame_000000.png"
    # Small source keeps the test light while exercising the exact interpolation code path.
    source = np.arange(80 * 120 * 3, dtype=np.uint8).reshape(80, 120, 3)
    assert cv2.imwrite(str(source_path), source)
    sample = FrameSample(
        frame_id=0,
        image_path=source_path,
        width=120,
        height=80,
        annotations=(
            DetectionAnnotation(0, 0, "tank", (12.0, 8.0, 60.0, 40.0)),
        ),
    )
    prepared, manifest = prepare_level0_dataset((sample,), tmp_path / "level0", target_size=(96, 54))
    written = cv2.imread(str(prepared[0].image_path), cv2.IMREAD_COLOR)
    expected = cv2.resize(source, (96, 54), interpolation=cv2.INTER_AREA)
    assert np.array_equal(written, expected)
    assert prepared[0].annotations[0].bbox_xyxy == pytest.approx((9.6, 5.4, 48.0, 27.0))
    assert manifest["preprocessing"]["interpolation"] == "INTER_AREA"


def test_scene_discovery_rejects_annotation_without_matching_image(tmp_path: Path) -> None:
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    assert cv2.imwrite(str(images / "frame_000000.png"), image)
    (annotations / "frame_000000.json").write_text(
        json.dumps({"frame": 0, "annotations": []}), encoding="utf-8"
    )
    (annotations / "frame_000001.json").write_text(
        json.dumps({"frame": 1, "annotations": []}), encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match="without matching source images"):
        discover_scene_samples(tmp_path, class_names=("tank",))
