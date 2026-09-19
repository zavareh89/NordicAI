from __future__ import annotations

from drone_detector.inference.postprocess import filter_detections
from drone_detector.types import Detection


def test_configurable_threshold_topk_and_box_clipping() -> None:
    detections = [
        Detection(0, "hangar", 0.90, (-5.0, 10.0, 20.0, 30.0)),
        Detection(1, "tank", 0.07, (10.0, 10.0, 20.0, 20.0)),
        Detection(1, "tank", 0.60, (30.0, 30.0, 40.0, 40.0)),
    ]
    result = filter_detections(
        detections,
        width=960,
        height=540,
        confidence_threshold=0.08,
        class_thresholds={"tank": 0.10},
        max_detections=1,
    )
    assert len(result) == 1
    assert result[0].class_name == "hangar"
    assert result[0].bbox_xyxy == (0.0, 10.0, 20.0, 30.0)
