from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..types import Detection
from .base import BaseDetector


class YOLO26Detector(BaseDetector):
    performs_nms = True

    def __init__(self, cfg: dict[str, Any], class_names: Sequence[str]) -> None:
        super().__init__(cfg, class_names)
        self.loaded_weights: str | None = None

    def _default_weights(self) -> str:
        return str(self.cfg["model"].get("pretrained_weights", "yolo26s.pt"))

    def load(self, weights: str | None = None) -> None:
        from ultralytics import YOLO

        chosen = weights or self._default_weights()
        self.model = YOLO(chosen)
        self.loaded_weights = chosen

    def _build_custom_albumentations(self) -> list[Any] | None:
        aug = self.cfg.get("augmentation", {})
        blur_p = float(aug.get("gaussian_blur_probability", 0.0))
        noise_p = float(aug.get("gaussian_noise_probability", 0.0))
        brightness_contrast_p = float(aug.get("brightness_contrast_probability", 0.0))
        if max(blur_p, noise_p, brightness_contrast_p) <= 0:
            return None
        import albumentations as A

        transforms: list[Any] = []
        if brightness_contrast_p > 0:
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=float(aug.get("brightness_limit", 0.10)),
                    contrast_limit=float(aug.get("contrast_limit", 0.10)),
                    p=brightness_contrast_p,
                )
            )
        if blur_p > 0:
            transforms.append(A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 0.8), p=blur_p))
        if noise_p > 0:
            transforms.append(
                A.GaussNoise(
                    std_range=tuple(aug.get("gaussian_noise_std_range", [0.005, 0.015])),
                    p=noise_p,
                )
            )
        return transforms

    def train(self, dataset_path: str, output_dir: str) -> Any:
        if self.model is None:
            self.load()
        training = self.cfg["training"]
        aug = self.cfg["augmentation"]
        model_cfg = self.cfg["model"]
        custom_aug = self._build_custom_albumentations()
        kwargs: dict[str, Any] = {
            "data": dataset_path,
            "project": str(Path(output_dir).parent),
            "name": Path(output_dir).name,
            "exist_ok": True,
            "epochs": int(training["epochs"]),
            "batch": int(training["batch_size"]),
            "workers": int(training.get("workers", 4)),
            "imgsz": int(model_cfg.get("imgsz", 960)),
            "optimizer": str(training.get("optimizer", "AdamW")),
            "lr0": float(training.get("learning_rate", 1e-3)),
            "weight_decay": float(training.get("weight_decay", 5e-4)),
            "patience": int(training.get("early_stopping_patience", 30)),
            "seed": int(self.cfg["experiment"].get("seed", 42)),
            "deterministic": bool(training.get("deterministic", False)),
            "device": training.get("device", 0),
            "amp": bool(training.get("amp", True)),
            "save": True,
            "save_period": int(training.get("checkpoint_interval", 10)),
            "close_mosaic": int(aug.get("close_mosaic", 10)),
            "hsv_h": float(aug.get("hsv_h", 0.005)),
            "hsv_s": float(aug.get("hsv_s", 0.15)),
            "hsv_v": float(aug.get("hsv_v", 0.15)),
            "degrees": float(aug.get("degrees", 3.0)),
            "translate": float(aug.get("translate", 0.05)),
            "scale": _ultralytics_scale_amplitude(aug.get("scale", [0.8, 1.2])),
            "shear": float(aug.get("shear", 0.0)),
            "perspective": float(aug.get("perspective", 0.0)),
            "flipud": float(aug.get("flipud", 0.5)),
            "fliplr": float(aug.get("fliplr", 0.5)),
            "mosaic": float(aug.get("mosaic", 0.25)),
            "mixup": float(aug.get("mixup", 0.0)),
            "cutmix": float(aug.get("cutmix", 0.0)),
        }
        if custom_aug:
            kwargs["augmentations"] = custom_aug
        return self.model.train(**kwargs)

    def predict(self, image_bgr: np.ndarray) -> list[Detection]:
        if self.model is None:
            raise RuntimeError("Detector is not loaded. Call load() first.")
        infer = self.cfg["inference"]
        model_cfg = self.cfg["model"]
        minimum_threshold = min(
            [float(infer.get("confidence_threshold", 0.08))]
            + [float(v) for v in infer.get("class_thresholds", {}).values()]
        )
        results = self.model.predict(
            source=image_bgr,
            imgsz=int(model_cfg.get("imgsz", 960)),
            conf=minimum_threshold,
            iou=float(infer.get("nms_iou_threshold", 0.60)),
            max_det=int(infer.get("max_detections", 200)),
            rect=bool(model_cfg.get("rect_inference", True)),
            device=training_device_for_inference(self.cfg),
            verbose=False,
        )
        result = results[0]
        if result.boxes is None:
            return []
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        output: list[Detection] = []
        for box, confidence, class_id in zip(boxes, confidences, classes):
            if class_id < 0 or class_id >= len(self.class_names):
                raise ValueError(f"YOLO returned out-of-range class id {class_id}")
            output.append(
                Detection(
                    class_id=int(class_id),
                    class_name=self.class_names[int(class_id)],
                    confidence=float(confidence),
                    bbox_xyxy=tuple(float(v) for v in box),
                )
            )
        return output


def _ultralytics_scale_amplitude(value: Any) -> float:
    """Translate an intuitive multiplicative scale range into Ultralytics `scale`.

    Ultralytics defines `scale=s` as a random multiplicative zoom in approximately
    [1-s, 1+s]. The configs intentionally store the human-readable [min, max]
    range requested for this experiment.
    """
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError("augmentation.scale must contain [min_scale, max_scale]")
        low, high = (float(value[0]), float(value[1]))
        if not (0.0 < low <= 1.0 <= high):
            raise ValueError("augmentation.scale range must satisfy 0 < min <= 1 <= max")
        return max(1.0 - low, high - 1.0)
    scalar = float(value)
    if not 0.0 <= scalar <= 1.0:
        raise ValueError("Ultralytics scale amplitude must be within [0, 1]")
    return scalar


def training_device_for_inference(cfg: dict[str, Any]) -> Any:
    return cfg.get("inference", {}).get("device", cfg.get("training", {}).get("device", 0))
