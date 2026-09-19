from __future__ import annotations

from typing import Any

from .train_rfdetr import train_rfdetr
from .train_yolo import train_yolo


def train_from_config(cfg: dict[str, Any]):
    family = str(cfg["model"]["family"]).lower()
    if family == "yolo26":
        return train_yolo(cfg)
    if family == "rfdetr":
        return train_rfdetr(cfg)
    raise ValueError(f"Unsupported detector family {family}")
