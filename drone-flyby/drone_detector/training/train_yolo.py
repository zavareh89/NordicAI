from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ..config import resolve_path
from ..data.challenge_annotations import get_challenge_classes
from ..detector.yolo_detector import YOLO26Detector
from ..evaluation.checkpoint_selection import (
    select_best_checkpoint_by_map50,
    yolo_checkpoint_candidates,
)
from ..evaluation.offline_eval import evaluate_checkpoint
from ..experiment import copy_if_exists, write_json
from .common import prepare_experiment, record_training_manifest


def _load_split_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    prepared_root = resolve_path(cfg, cfg["paths"].get("prepared_root", "prepared"))
    path = prepared_root / "split.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run `python scripts/prepare_dataset.py` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def train_yolo(cfg: dict[str, Any]) -> Path:
    paths = prepare_experiment(cfg)
    classes = get_challenge_classes()
    split_meta = _load_split_metadata(cfg)
    independent_val = bool(split_meta.get("validation_is_independent", False))

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
    dataset_yaml = resolve_path(
        cfg, cfg["paths"].get("yolo_dataset_yaml", "prepared/yolo/data.yaml")
    )
    backend_dir = paths.logs / "ultralytics"
    detector.train(str(dataset_yaml), str(backend_dir))

    framework_best = backend_dir / "weights" / "best.pt"
    framework_last = backend_dir / "weights" / "last.pt"
    if not framework_last.is_file() and not framework_best.is_file():
        raise FileNotFoundError(
            f"Ultralytics training finished without last.pt/best.pt under {backend_dir / 'weights'}"
        )
    copy_if_exists(framework_best, paths.checkpoints / "framework_best.pt")
    copy_if_exists(framework_last, paths.checkpoints / "last.pt")

    if independent_val and bool(cfg["training"].get("select_best_by_map50", True)):
        selected, _ = select_best_checkpoint_by_map50(
            cfg,
            yolo_checkpoint_candidates(backend_dir),
            report_path=paths.root / "checkpoint_selection.json",
        )
        selection_metric = "mAP@0.50"
    else:
        # Final all-data training: no unbiased validation signal exists. Deploy the
        # final training checkpoint rather than a best-on-train-mirror checkpoint.
        selected = framework_last if framework_last.is_file() else framework_best
        selection_metric = "last_epoch_all_frames_no_independent_validation"
        write_json(
            paths.root / "checkpoint_selection.json",
            {
                "selection_metric": selection_metric,
                "selected_checkpoint": str(selected),
                "validation_is_independent": False,
                "note": (
                    "All labeled frames were used for training; validation-based model "
                    "selection is intentionally disabled."
                ),
            },
        )

    selected_copy = paths.checkpoints / "selected_best.pt"
    shutil.copy2(selected, selected_copy)
    final = paths.final_model / "E1_yolo26s_best.pt"
    shutil.copy2(selected, final)
    challenge_checkpoint = resolve_path(cfg, cfg["model"]["deployment_weights"])
    challenge_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected, challenge_checkpoint)

    if independent_val:
        final_metrics, final_predictions = evaluate_checkpoint(cfg, selected)
        write_json(paths.root / "metrics.json", final_metrics)
        write_json(
            paths.predictions / "blocked_validation_predictions.json",
            final_predictions,
        )
    else:
        write_json(
            paths.root / "metrics.json",
            {
                "status": "not_computed",
                "reason": "no_independent_validation_all_frames_training",
                "note": (
                    "Use the official local challenge evaluator for integration scoring, "
                    "or switch data.split.strategy to blocked_holdout for a diagnostic estimate."
                ),
            },
        )

    record_training_manifest(
        cfg,
        paths,
        {
            "pretrained_checkpoint": str(pretrained),
            "dataset": str(dataset_yaml),
            "split_strategy": split_meta.get("strategy"),
            "validation_is_independent": independent_val,
            "framework_best_checkpoint": (
                str(framework_best) if framework_best.is_file() else None
            ),
            "selection_metric": selection_metric,
            "selected_source_checkpoint": str(selected),
            "best_checkpoint": str(selected_copy),
            "final_checkpoint": str(final),
            "challenge_checkpoint": str(challenge_checkpoint),
        },
    )
    return selected_copy
