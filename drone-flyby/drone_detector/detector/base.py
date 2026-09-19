from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np

from ..types import Detection


class BaseDetector(ABC):
    """Stateless-per-frame detector contract shared by E1 and E2.

    Long-lived model weights/configuration are allowed. No previous-frame state may
    affect `predict()` output in these baselines.
    """

    def __init__(self, cfg: dict[str, Any], class_names: Sequence[str]) -> None:
        self.cfg = cfg
        self.class_names = tuple(class_names)
        self.model: Any = None

    @abstractmethod
    def load(self, weights: str | None = None) -> None:
        """Load pretrained or challenge-fine-tuned weights."""

    @abstractmethod
    def train(self, dataset_path: str, output_dir: str) -> Any:
        """Fine-tune from pretrained weights using the model-specific backend."""

    @abstractmethod
    def predict(self, image_bgr: np.ndarray) -> list[Detection]:
        """Return detections in pixel xyxy coordinates of the supplied image."""

    @property
    @abstractmethod
    def performs_nms(self) -> bool:
        """Whether the framework already performs classical NMS."""
