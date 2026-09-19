from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from .config import resolve_path


def pretrained_dir(cfg: dict[str, Any]) -> Path:
    path = resolve_path(cfg, cfg["paths"].get("pretrained_weights_dir", "weights/pretrained"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def challenge_weights_dir(cfg: dict[str, Any]) -> Path:
    path = resolve_path(cfg, cfg["paths"].get("challenge_weights_dir", "weights/challenge"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_yolo_weights(cfg: dict[str, Any], *, objects365: bool = False) -> Path:
    """Keep E1 behavior unchanged."""
    from ultralytics import YOLO
    from ultralytics.utils.downloads import attempt_download_asset

    key = "objects365_weights" if objects365 else "pretrained_weights"
    checkpoint = str(cfg["model"].get(key) or "")
    if not checkpoint:
        raise ValueError(f"No model.{key} configured")

    target = pretrained_dir(cfg) / Path(checkpoint).name
    if target.is_file():
        YOLO(str(target))
        return target

    resolved = Path(attempt_download_asset(checkpoint)).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Ultralytics did not produce expected checkpoint {checkpoint!r}")
    shutil.copy2(resolved, target)
    YOLO(str(target))
    return target


def download_rfdetr_weights(cfg: dict[str, Any]) -> Path:
    """Download/cache RF-DETR Small using the RF-DETR 1.5.2 asset registry.

    RF-DETR 1.5.2's ModelConfig converts a checkpoint path to an absolute path before
    model construction. Its downloader, however, recognizes hosted checkpoints by the
    *bare canonical filename*. Therefore we call the official downloader ourselves from
    inside the desired cache directory, verify the official checksum, and only then hand
    the absolute path to the detector/training adapter.
    """
    import torch
    from rfdetr.assets.model_weights import download_pretrain_weights, validate_pretrain_weights

    checkpoint = Path(str(cfg["model"].get("pretrained_weights", "rf-detr-small.pth"))).name
    if checkpoint != "rf-detr-small.pth":
        raise ValueError(
            "E2 is RF-DETR Small and expects the official checkpoint name "
            f"'rf-detr-small.pth'; got {checkpoint!r}"
        )

    target_dir = pretrained_dir(cfg)
    target = target_dir / checkpoint

    if not target.is_file():
        original_cwd = Path.cwd()
        try:
            os.chdir(target_dir)
            # Must be the bare canonical name for the 1.5.2 registry lookup.
            download_pretrain_weights(checkpoint, redownload=False, validate_md5=True)
        finally:
            os.chdir(original_cwd)

    if not target.is_file():
        raise FileNotFoundError(
            f"RF-DETR 1.5.2 did not create the expected checkpoint: {target}"
        )

    # Official asset MD5 (strict) + PyTorch deserialization sanity check.
    validate_pretrain_weights(str(target.resolve()), strict=True)
    payload = torch.load(str(target.resolve()), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Unexpected RF-DETR checkpoint structure in {target}")

    return target.resolve()


def download_for_scenario(cfg: dict[str, Any], *, initializer: str = "standard") -> Path:
    family = str(cfg["model"]["family"]).lower()
    if family == "yolo26":
        return download_yolo_weights(cfg, objects365=initializer.lower() == "objects365")
    if family == "rfdetr":
        return download_rfdetr_weights(cfg)
    raise ValueError(f"Unsupported family {family}")
