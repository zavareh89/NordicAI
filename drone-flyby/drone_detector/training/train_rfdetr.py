from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ..config import resolve_path
from ..data.challenge_annotations import get_challenge_classes
from ..detector.rfdetr_detector import RFDETRSmallDetector
from ..evaluation.checkpoint_selection import (
    rfdetr_checkpoint_candidates,
    select_best_checkpoint_by_map50,
)
from ..evaluation.offline_eval import evaluate_checkpoint
from ..experiment import copy_if_exists, write_json
from .common import prepare_experiment, record_training_manifest


def _assert_rfdetr_152() -> str:
    from importlib.metadata import version

    installed = version("rfdetr")
    if installed != "1.5.2":
        raise RuntimeError(
            "E2 is pinned to rfdetr==1.5.2 to remain compatible with "
            f"torch==2.2.1+cu121. Installed rfdetr={installed}. "
            "Run: pip install 'rfdetr==1.5.2' 'transformers==4.56.2' "
            "'albumentations==1.4.24'"
        )
    return installed


def train_rfdetr(cfg: dict[str, Any]) -> Path:
    rf_version = _assert_rfdetr_152()
    paths = prepare_experiment(cfg)
    classes = get_challenge_classes()

    pretrained = resolve_path(
        cfg,
        Path(cfg["paths"].get("pretrained_weights_dir", "weights/pretrained"))
        / Path(str(cfg["model"]["pretrained_weights"])).name,
    )
    if not pretrained.is_file():
        raise FileNotFoundError(
            f"Missing pretrained checkpoint {pretrained}. "
            "Run `python scripts/download_weights.py --scenario E2` first."
        )

    dataset_root = resolve_path(
        cfg, cfg["paths"].get("rfdetr_dataset_root", "prepared/rfdetr")
    )
    required = [
        dataset_root / "train" / "_annotations.coco.json",
        dataset_root / "valid" / "_annotations.coco.json",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "RF-DETR dataset is not prepared. Missing: " + ", ".join(missing) +
            ". Run `python scripts/prepare_dataset.py` first."
        )

    detector = RFDETRSmallDetector(cfg, classes)
    detector.load(str(pretrained))

    backend_dir = paths.logs / "rfdetr"
    backend_dir.mkdir(parents=True, exist_ok=True)
    detector.train(str(dataset_root), str(backend_dir))

    # RF-DETR 1.5.2 writes checkpoint_best_total.pth after training and also keeps
    # regular/EMA/periodic checkpoints. Do not assume Lightning last.ckpt names.
    framework_best = backend_dir / "checkpoint_best_total.pth"
    if not framework_best.is_file():
        fallbacks = [
            backend_dir / "checkpoint_best_ema.pth",
            backend_dir / "checkpoint_best_regular.pth",
            backend_dir / "checkpoint.pth",
        ]
        framework_best = next((p for p in fallbacks if p.is_file()), framework_best)
    if not framework_best.is_file():
        raise FileNotFoundError(
            f"RF-DETR 1.5.2 training finished without an expected checkpoint in {backend_dir}"
        )

    copy_if_exists(framework_best, paths.checkpoints / "framework_best.pth")
    copy_if_exists(backend_dir / "checkpoint.pth", paths.checkpoints / "last.pth")

    candidates = rfdetr_checkpoint_candidates(backend_dir)
    if bool(cfg["training"].get("select_best_by_map50", True)):
        selected, _ = select_best_checkpoint_by_map50(
            cfg,
            candidates,
            report_path=paths.root / "checkpoint_selection.json",
        )
    else:
        selected = framework_best
        write_json(
            paths.root / "checkpoint_selection.json",
            {
                "selection_metric": "framework_best_mAP50_95",
                "selected_checkpoint": str(selected),
                "note": "AP@0.50 post-selection disabled by config",
            },
        )

    selected_copy = paths.checkpoints / "selected_best.pth"
    shutil.copy2(selected, selected_copy)

    final = paths.final_model / "E2_rfdetr_small_best.pth"
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
            "rfdetr_version": rf_version,
            "pretrained_checkpoint": str(pretrained),
            "dataset": str(dataset_root),
            "framework_best_checkpoint": str(framework_best),
            "selection_metric": (
                "mAP@0.50" if bool(cfg["training"].get("select_best_by_map50", True))
                else "framework_best_mAP50_95"
            ),
            "selected_source_checkpoint": str(selected),
            "best_checkpoint": str(selected_copy),
            "final_checkpoint": str(final),
            "challenge_checkpoint": str(challenge_checkpoint),
        },
    )
    return selected_copy
