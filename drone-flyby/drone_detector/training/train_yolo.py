from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ..config import resolve_path
from ..data.challenge_annotations import get_challenge_classes
from ..detector.yolo_detector import YOLO26Detector
from ..evaluation.offline_eval import evaluate_checkpoint
from ..evaluation.checkpoint_selection import select_best_checkpoint_by_map50, yolo_checkpoint_candidates
from ..experiment import copy_if_exists, write_json
from .common import prepare_experiment, record_training_manifest


def train_yolo(cfg: dict[str, Any]) -> Path:
    paths = prepare_experiment(cfg)
    classes = get_challenge_classes()
    pretrained = resolve_path(
        cfg,
        Path(cfg["paths"].get("pretrained_weights_dir", "weights/pretrained"))
        / Path(str(cfg["model"]["pretrained_weights"])).name,
    )
    if not pretrained.is_file():
        raise FileNotFoundError(
            f"Missing pretrained checkpoint {pretrained}. Run scripts/download_weights.py --scenario E1."
        )
    detector = YOLO26Detector(cfg, classes)
    detector.load(str(pretrained))
    dataset_yaml = resolve_path(cfg, cfg["paths"].get("yolo_dataset_yaml", "prepared/yolo/data.yaml"))
    backend_dir = paths.logs / "ultralytics"
    detector.train(str(dataset_yaml), str(backend_dir))

    framework_best = backend_dir / "weights" / "best.pt"
    framework_last = backend_dir / "weights" / "last.pt"
    if not framework_best.is_file():
        raise FileNotFoundError(f"Ultralytics training finished without expected best checkpoint: {framework_best}")
    copy_if_exists(framework_best, paths.checkpoints / "framework_best.pt")
    copy_if_exists(framework_last, paths.checkpoints / "last.pt")

    candidates = yolo_checkpoint_candidates(backend_dir)
    selected, _ = select_best_checkpoint_by_map50(
        cfg,
        candidates,
        report_path=paths.root / "checkpoint_selection.json",
    )
    selected_copy = paths.checkpoints / "selected_best.pt"
    shutil.copy2(selected, selected_copy)
    final = paths.final_model / "E1_yolo26s_best.pt"
    shutil.copy2(selected, final)
    challenge_checkpoint = resolve_path(cfg, cfg["model"]["deployment_weights"])
    challenge_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected, challenge_checkpoint)

    final_metrics, final_predictions = evaluate_checkpoint(cfg, selected)
    write_json(paths.root / "metrics.json", final_metrics)
    write_json(
        paths.predictions / "blocked_validation_predictions.json",
        final_predictions,
    )
    record_training_manifest(
        cfg,
        paths,
        {
            "pretrained_checkpoint": str(pretrained),
            "dataset": str(dataset_yaml),
            "framework_best_checkpoint": str(framework_best),
            "selection_metric": "mAP@0.50",
            "selected_source_checkpoint": str(selected),
            "best_checkpoint": str(selected_copy),
            "final_checkpoint": str(final),
            "challenge_checkpoint": str(challenge_checkpoint),
        },
    )
    return selected_copy
