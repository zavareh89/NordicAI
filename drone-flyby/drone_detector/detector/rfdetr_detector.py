from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import cv2
import numpy as np

from ..types import Detection
from .base import BaseDetector


@contextmanager
def _rf152_disable_builtin_random_crop(enabled: bool) -> Iterator[None]:
    """Disable RF-DETR 1.5.2's resize->crop->resize training branch.

    RF-DETR 1.5.2 hard-wires a 50/50 OneOf between direct resize and a random-crop
    branch before `aug_config`. For these 960x540 tiny-object frames that crop branch
    can discard the only instance of a class. We keep RF-DETR's official
    AlbumentationsWrapper and augmentations, but replace only the private resize-config
    builder during this training call so the model sees a direct square resize that
    matches RF-DETR 1.5.2 inference (`F.resize(..., (resolution, resolution))`).
    """
    if not enabled:
        yield
        return

    import rfdetr.datasets.coco as coco_module

    original = coco_module._build_train_resize_config

    def direct_resize_only(scales, *, square: bool, max_size=None):
        if not square:
            # E2 uses square_resize_div_64=True. Preserve upstream behavior for any
            # unexpected non-square call rather than silently changing it.
            return original(scales, square=square, max_size=max_size)
        if len(scales) == 1:
            s = int(scales[0])
            return [{"Resize": {"height": s, "width": s}}]
        return [
            {
                "OneOf": {
                    "transforms": [
                        {"Resize": {"height": int(s), "width": int(s)}} for s in scales
                    ]
                }
            }
        ]

    coco_module._build_train_resize_config = direct_resize_only
    try:
        yield
    finally:
        coco_module._build_train_resize_config = original


def _resolve_rf152_device(value: str) -> str:
    """Resolve RF-DETR 1.5.2 TrainConfig's `auto` to a torch.device-safe value."""
    value = str(value).lower()
    if value != "auto":
        if value not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"Unsupported RF-DETR device: {value!r}")
        return value

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_rf152_train_config(
    cfg: dict[str, Any], dataset_path: str, output_dir: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return RF-DETR 1.5.2 public TrainConfig kwargs + legacy-engine extras.

    `TrainConfig` in 1.5.2 forbids unknown keys. Parameters such as `seed`,
    `clip_max_norm`, and `lr_scheduler` exist in the underlying legacy engine but are
    not fields of TrainConfig, so they are deliberately passed only to
    `train_from_config()` as extras.
    """
    t = cfg["training"]
    a = cfg["augmentation"]
    infer = cfg["inference"]

    resume = t.get("resume")
    if resume:
        resume = str(Path(resume).expanduser().resolve())

    train_config_kwargs = {
        "dataset_dir": str(Path(dataset_path).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "dataset_file": "roboflow",
        "epochs": int(t.get("epochs", 150)),
        "batch_size": int(t.get("batch_size", 2)),
        "grad_accum_steps": int(t.get("grad_accum_steps", 4)),
        "lr": float(t.get("learning_rate", 5e-5)),
        "lr_encoder": float(t.get("encoder_learning_rate", 2.5e-5)),
        "weight_decay": float(t.get("weight_decay", 1e-4)),
        "device": _resolve_rf152_device(str(t.get("device", "cuda"))),
        "resume": resume,
        "ema_decay": float(t.get("ema_decay", 0.993)),
        "ema_tau": int(t.get("ema_tau", 100)),
        "lr_drop": int(t.get("lr_drop", max(1, int(t.get("epochs", 150)) - 10))),
        "checkpoint_interval": int(t.get("checkpoint_interval", 10)),
        "warmup_epochs": float(t.get("warmup_epochs", 1.0)),
        "lr_vit_layer_decay": float(t.get("lr_vit_layer_decay", 0.8)),
        "lr_component_decay": float(t.get("lr_component_decay", 0.7)),
        "drop_path": float(t.get("drop_path", 0.0)),
        "use_ema": bool(t.get("use_ema", True)),
        "num_workers": int(t.get("workers", 4)),
        "early_stopping": bool(t.get("early_stopping", True)),
        "early_stopping_patience": int(t.get("early_stopping_patience", 20)),
        "early_stopping_min_delta": float(t.get("early_stopping_min_delta", 0.001)),
        "early_stopping_use_ema": bool(t.get("early_stopping_use_ema", True)),
        "progress_bar": bool(t.get("progress_bar", True)),
        # tensorboard is an optional RF-DETR dependency. Keep it off unless explicitly requested.
        "tensorboard": bool(t.get("tensorboard", False)),
        "wandb": bool(t.get("wandb", False)),
        "mlflow": False,
        "clearml": False,
        # We export only train/valid for the blocked Helsinki experiment. In 1.5.2,
        # run_test=True tries to open dataset_dir/test before training and would fail.
        "run_test": bool(t.get("run_test", False)),
        "eval_max_dets": int(infer.get("max_detections", 200)),
        "square_resize_div_64": bool(a.get("square_resize_div_64", True)),
        "multi_scale": bool(a.get("multi_scale", False)),
        "expanded_scales": bool(a.get("expanded_scales", False)),
        "do_random_resize_via_padding": bool(a.get("do_random_resize_via_padding", False)),
        "aug_config": build_rf152_augmentation_config(cfg),
    }

    legacy_engine_extras = {
        "seed": int(cfg["experiment"].get("seed", 42)),
        "clip_max_norm": float(t.get("clip_max_norm", 0.1)),
        "lr_scheduler": str(t.get("lr_scheduler", "cosine")),
        "lr_min_factor": float(t.get("lr_min_factor", 0.05)),
    }
    return train_config_kwargs, legacy_engine_extras


def build_rf152_augmentation_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Conservative RF-DETR 1.5.2/Albumentations 1.4.x augmentation dictionary."""
    aug = cfg["augmentation"]
    config: dict[str, Any] = {
        "HorizontalFlip": {"p": float(aug.get("hflip", 0.5))},
        "VerticalFlip": {"p": float(aug.get("vflip", 0.5))},
        "RandomBrightnessContrast": {
            "brightness_limit": float(aug.get("brightness_limit", 0.10)),
            "contrast_limit": float(aug.get("contrast_limit", 0.10)),
            "p": float(aug.get("brightness_contrast_probability", 0.25)),
        },
    }

    color_p = float(aug.get("color_jitter_probability", 0.15))
    if color_p > 0:
        config["ColorJitter"] = {
            "brightness": 0.0,
            "contrast": 0.0,
            "saturation": float(aug.get("saturation_jitter", 0.10)),
            "hue": float(aug.get("hue_jitter", 0.02)),
            "p": color_p,
        }

    rotate_p = float(aug.get("rotate_probability", 0.10))
    if rotate_p > 0:
        config["Rotate"] = {
            "limit": float(aug.get("rotate_limit_degrees", 3.0)),
            "border_mode": 0,
            "p": rotate_p,
        }

    affine_p = float(aug.get("affine_probability", 0.15))
    if affine_p > 0:
        config["Affine"] = {
            "scale": tuple(float(v) for v in aug.get("scale", [0.90, 1.10])),
            "translate_percent": {
                "x": tuple(float(v) for v in aug.get("translate_x", [-0.05, 0.05])),
                "y": tuple(float(v) for v in aug.get("translate_y", [-0.05, 0.05])),
            },
            "rotate": 0.0,
            "shear": 0.0,
            "fit_output": False,
            "p": affine_p,
        }

    blur_p = float(aug.get("gaussian_blur_probability", 0.03))
    if blur_p > 0:
        config["GaussianBlur"] = {
            "blur_limit": (3, 3),
            "sigma_limit": tuple(float(v) for v in aug.get("gaussian_blur_sigma", [0.1, 0.8])),
            "p": blur_p,
        }

    noise_p = float(aug.get("gaussian_noise_probability", 0.03))
    if noise_p > 0:
        # Albumentations 1.4.24 supports normalized std_range.
        config["GaussNoise"] = {
            "std_range": tuple(float(v) for v in aug.get("gaussian_noise_std_range", [0.005, 0.015])),
            "mean_range": (0.0, 0.0),
            "p": noise_p,
        }

    return config


class RFDETRSmallDetector(BaseDetector):
    """RF-DETR Small adapter compatible with rfdetr==1.5.2."""

    performs_nms = False

    def __init__(self, cfg: dict[str, Any], class_names: Sequence[str]) -> None:
        super().__init__(cfg, class_names)
        self.loaded_weights: str | None = None

    def _default_weights(self) -> str:
        return str(self.cfg["model"].get("pretrained_weights", "rf-detr-small.pth"))

    def load(self, weights: str | None = None) -> None:
        from rfdetr import RFDETRSmall

        chosen = weights or self._default_weights()
        resolution = int(self.cfg["model"].get("resolution", 960))
        device = _resolve_rf152_device(
            str(self.cfg.get("inference", {}).get("device", self.cfg["training"].get("device", "cuda")))
        )

        # RF-DETR 1.5.2 has no from_checkpoint(). Its constructor accepts both the
        # official pretrained .pth and its own fine-tuned .pth checkpoints.
        self.model = RFDETRSmall(
            pretrain_weights=str(chosen),
            resolution=resolution,
            device=device,
        )
        self.loaded_weights = str(chosen)

    def train(self, dataset_path: str, output_dir: str) -> Any:
        if self.model is None:
            self.load()

        train_kwargs, engine_extras = build_rf152_train_config(self.cfg, dataset_path, output_dir)
        # Use the public 1.5.2 TrainConfig validator first so typos/unsupported keys fail
        # immediately. Then call train_from_config only because 1.5.2 does not expose
        # seed/clip_max_norm/lr_scheduler as TrainConfig fields even though its legacy
        # engine supports them.
        train_config = self.model.get_train_config(**train_kwargs)

        disable_crop = bool(self.cfg["augmentation"].get("disable_builtin_random_crop", True))
        with _rf152_disable_builtin_random_crop(disable_crop):
            return self.model.train_from_config(train_config, **engine_extras)

    def predict(self, image_bgr: np.ndarray) -> list[Detection]:
        if self.model is None:
            raise RuntimeError("Detector is not loaded. Call load() first.")

        infer = self.cfg["inference"]
        minimum_threshold = min(
            [float(infer.get("confidence_threshold", 0.08))]
            + [float(v) for v in infer.get("class_thresholds", {}).values()]
        )

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        detections = self.model.predict(image_rgb, threshold=minimum_threshold)

        boxes = np.asarray(detections.xyxy)
        confidences = np.asarray(detections.confidence)
        class_ids = np.asarray(detections.class_id).astype(int)

        output: list[Detection] = []
        num_classes = len(self.class_names)

        for box, confidence, class_id in zip(
            boxes,
            confidences,
            class_ids,
        ):
            class_id = int(class_id)

            # RF-DETR 1.5.2 can expose one extra background/no-object
            # output slot for custom Roboflow/COCO datasets.
            #
            # For our 16 challenge classes:
            #   foreground = 0..15
            #   extra slot = 16
            #
            # Never send this extra class to the challenge API.
            if class_id == num_classes:
                continue

            # Anything beyond the known RF-DETR background slot is genuinely
            # unexpected and should remain a hard error.
            if class_id < 0 or class_id > num_classes:
                raise ValueError(
                    f"RF-DETR returned unexpected class id {class_id}; "
                    f"expected foreground 0..{num_classes - 1} "
                    f"or background {num_classes}"
                )

            output.append(
                Detection(
                    class_id=class_id,
                    class_name=self.class_names[class_id],
                    confidence=float(confidence),
                    bbox_xyxy=tuple(float(v) for v in box),
                )
            )
        return output
