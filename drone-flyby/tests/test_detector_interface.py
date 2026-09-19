from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from drone_detector.detector.base import BaseDetector
from drone_detector.detector.rfdetr_detector import RFDETRSmallDetector
from drone_detector.detector.yolo_detector import YOLO26Detector
from drone_detector.types import Detection


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeYOLOModel:
    def predict(self, **kwargs):
        boxes = SimpleNamespace(
            xyxy=FakeTensor([[10.0, 20.0, 30.0, 40.0]]),
            conf=FakeTensor([0.25]),
            cls=FakeTensor([1]),
        )
        return [SimpleNamespace(boxes=boxes)]


class FakeRFModel:
    def predict(self, image, **kwargs):
        return SimpleNamespace(
            xyxy=np.asarray([[10.0, 20.0, 30.0, 40.0]]),
            confidence=np.asarray([0.25]),
            # Deliberately 1-based-looking numeric ID; class_name must win.
            class_id=np.asarray([2]),
            data={"class_name": np.asarray(["tank"])},
        )


def _cfg(family: str) -> dict:
    return {
        "model": {"family": family, "imgsz": 960, "resolution": 960, "rect_inference": True},
        "training": {"device": "cpu"},
        "inference": {
            "device": "cpu",
            "confidence_threshold": 0.08,
            "class_thresholds": {},
            "nms_iou_threshold": 0.60,
            "max_detections": 200,
        },
    }


def test_both_adapters_implement_same_base_and_return_detection_type() -> None:
    classes = ("hangar", "tank")
    image = np.zeros((540, 960, 3), dtype=np.uint8)

    yolo = YOLO26Detector(_cfg("yolo26"), classes)
    rf = RFDETRSmallDetector(_cfg("rfdetr"), classes)
    assert isinstance(yolo, BaseDetector)
    assert isinstance(rf, BaseDetector)

    yolo.model = FakeYOLOModel()
    rf.model = FakeRFModel()
    yolo_out = yolo.predict(image)
    rf_out = rf.predict(image)

    assert isinstance(yolo_out[0], Detection)
    assert isinstance(rf_out[0], Detection)
    assert yolo_out[0].class_name == "tank"
    assert rf_out[0].class_name == "tank"
    assert yolo_out[0].bbox_xyxy == rf_out[0].bbox_xyxy
