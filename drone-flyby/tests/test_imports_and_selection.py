from __future__ import annotations

import importlib
from pathlib import Path

from drone_detector.evaluation.checkpoint_selection import (
    rfdetr_checkpoint_candidates,
    yolo_checkpoint_candidates,
)


def test_all_project_modules_import_without_loading_detector_weights() -> None:
    modules = [
        "drone_detector.config",
        "drone_detector.types",
        "drone_detector.reproducibility",
        "drone_detector.experiment",
        "drone_detector.weights",
        "drone_detector.data.challenge_annotations",
        "drone_detector.data.level0_dataset",
        "drone_detector.data.split",
        "drone_detector.data.validation",
        "drone_detector.data.yolo_export",
        "drone_detector.data.coco_export",
        "drone_detector.detector.base",
        "drone_detector.detector.yolo_detector",
        "drone_detector.detector.rfdetr_detector",
        "drone_detector.detector.factory",
        "drone_detector.inference.predictor",
        "drone_detector.inference.postprocess",
        "drone_detector.evaluation.metrics",
        "drone_detector.evaluation.offline_eval",
        "drone_detector.evaluation.challenge_eval",
        "drone_detector.evaluation.checkpoint_selection",
        "drone_detector.challenge.api_adapter",
        "drone_detector.challenge.solution",
        "drone_detector.training.common",
        "drone_detector.training.train_yolo",
        "drone_detector.training.train_rfdetr",
        "drone_detector.training.train",
    ]
    for module in modules:
        importlib.import_module(module)


def test_checkpoint_candidate_discovery_is_model_specific(tmp_path: Path) -> None:
    yolo_weights = tmp_path / "yolo" / "weights"
    yolo_weights.mkdir(parents=True)
    for name in ("best.pt", "last.pt", "epoch9.pt", "ignore.txt"):
        (yolo_weights / name).touch()
    assert [p.name for p in yolo_checkpoint_candidates(tmp_path / "yolo")] == [
        "best.pt",
        "epoch9.pt",
        "last.pt",
    ]

    rf = tmp_path / "rf"
    rf.mkdir()
    for name in (
        "checkpoint_best_total.pth",
        "checkpoint_epoch=9.ckpt",
        "last.ckpt",
        "last_ema.pth",
        "ignore.txt",
    ):
        (rf / name).touch()
    assert [p.name for p in rfdetr_checkpoint_candidates(rf)] == [
        "checkpoint_best_total.pth",
        "checkpoint_epoch=9.ckpt",
        "last.ckpt",
        "last_ema.pth",
    ]
