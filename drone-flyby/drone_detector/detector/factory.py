from __future__ import annotations

from typing import Any, Sequence

from .base import BaseDetector
from .rfdetr_detector import RFDETRSmallDetector
from .yolo_detector import YOLO26Detector


def create_detector(cfg: dict[str, Any], class_names: Sequence[str]) -> BaseDetector:
    family = str(cfg["model"]["family"]).lower()
    if family == "yolo26":
        return YOLO26Detector(cfg, class_names)
    if family == "rfdetr":
        return RFDETRSmallDetector(cfg, class_names)
    raise ValueError(f"Unsupported detector family: {family}")
