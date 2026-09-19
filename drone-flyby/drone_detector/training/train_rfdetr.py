from __future__ import annotations

import json
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


def _load_split_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    prepared_root = resolve_path(cfg, cfg["paths"].get("prepared_root", "prepared"))
    path = prepared_root / "split.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run `python scripts/prepare_dataset.py` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _last_rfdetr_checkpoint(backend_dir: Path) -> Path | None:
    # RF-DETR 1.5.2 rewrites checkpoint.pth each epoch. Prefer it for final all-frame
    # training because no independent validation metric exists to define "best".
    checkpoint = backend_dir / "checkpoint.pth"
    if checkpoint.is_file():
        return checkpoint
    periodic = sorted(backend_dir.glob("checkpoint[0-9][0-9][0-9][0-9].pth"))
    return periodic[-1] if periodic else None


def train_rfdetr(cfg: dict[str, Any]) -> Path:
    rf_version = _assert_rfdetr_152()
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
            "RF-DETR dataset is not prepared. Missing: "
            + ", ".join(missing)
            + ". Run `python scripts/prepare_dataset.py` first."
        )

    detector = RFDETRSmallDetector(cfg, classes)
    detector.load(str(pretrained))
    backend_dir = paths.logs / "rfdetr"
    backend_dir.mkdir(parents=True, exist_ok=True)
    detector.train(str(dataset_root), str(backend_dir))

    framework_best = backend_dir / "checkpoint_best_total.pth"
    if not framework_best.is_file():
        fallbacks = [
            backend_dir / "checkpoint_best_ema.pth",
            backend_dir / "checkpoint_best_regular.pth",
        ]
        framework_best = next(
            (p for p in fallbacks if p.is_file()), framework_best
        )
    last_checkpoint = _last_rfdetr_checkpoint(backend_dir)
    if last_checkpoint is None and not framework_best.is_file():
        raise FileNotFoundError(
            f"RF-DETR 1.5.2 training finished without an expected checkpoint in {backend_dir}"
        )

    if framework_best.is_file():
        copy_if_exists(framework_best, paths.checkpoints / "framework_best.pth")
    if last_checkpoint is not None:
        copy_if_exists(last_checkpoint, paths.checkpoints / "last.pth")

    if independent_val and bool(cfg["training"].get("select_best_by_map50", True)):
        selected, _ = select_best_checkpoint_by_map50(
            cfg,
            rfdetr_checkpoint_candidates(backend_dir),
            report_path=paths.root / "checkpoint_selection.json",
        )
        selection_metric = "mAP@0.50"
    else:
        selected = last_checkpoint if last_checkpoint is not None else framework_best
        selection_metric = "last_epoch_all_frames_no_independent_validation"
        write_json(
            paths.root / "checkpoint_selection.json",
            {
                "selection_metric": selection_metric,
                "selected_checkpoint": str(selected),
                "validation_is_independent": False,
                "note": (
                    "RF-DETR valid/ mirrors train/ only because 1.5.2 requires it. "
                    "Mirrored validation metrics are intentionally ignored."
                ),
            },
        )

    selected_copy = paths.checkpoints / "selected_best.pth"
    shutil.copy2(selected, selected_copy)
    final = paths.final_model / "E2_rfdetr_small_best.pth"
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
                    "The RF-DETR valid/ directory mirrors train/ for framework "
                    "compatibility; its scores are not held-out metrics."
                ),
            },
        )

    record_training_manifest(
        cfg,
        paths,
        {
            "rfdetr_version": rf_version,
            "pretrained_checkpoint": str(pretrained),
            "dataset": str(dataset_root),
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
