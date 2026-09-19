from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from ..config import resolve_path
from ..data.challenge_annotations import discover_scene_samples, get_challenge_classes
from ..data.level0_dataset import transform_annotations_level0
from ..detector.factory import create_detector
from ..experiment import ExperimentPaths, snapshot_config, write_json
from ..inference.predictor import Predictor
from ..types import FrameSample
from .metrics import coco_map50, precision_recall_at_iou50


def _load_split(path: Path) -> dict[str, list[int]]:
    with path.open("r", encoding="utf-8") as handle:
        split = json.load(handle)
    return split


def validation_samples(cfg: dict[str, Any]) -> tuple[FrameSample, ...]:
    classes = get_challenge_classes()
    scene_dir = resolve_path(cfg, cfg["paths"]["challenge_scene"])
    source_samples = discover_scene_samples(scene_dir, class_names=classes)
    prepared_root = resolve_path(cfg, cfg["paths"].get("prepared_root", "prepared"))
    split = _load_split(prepared_root / "split.json")
    val_ids = set(int(v) for v in split["val_frame_ids"])
    output = []
    for source in source_samples:
        if source.frame_id not in val_ids:
            continue
        image_path = prepared_root / "level0" / "images" / f"frame_{source.frame_id:06d}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing prepared Level-0 image {image_path}")
        output.append(
            FrameSample(
                frame_id=source.frame_id,
                image_path=image_path,
                width=960,
                height=540,
                annotations=transform_annotations_level0(source, (960, 540)),
            )
        )
    if not output:
        raise ValueError("Blocked validation split contains no available samples")
    return tuple(output)


def _default_weights(cfg: dict[str, Any], paths: ExperimentPaths) -> Path:
    family = str(cfg["model"]["family"]).lower()
    if family == "yolo26":
        candidates = [
            paths.final_model / "E1_yolo26s_best.pt",
            paths.checkpoints / "best.pt",
        ]
    else:
        candidates = [
            paths.final_model / "E2_rfdetr_small_best.pth",
            paths.checkpoints / "selected_best.pth",
            paths.checkpoints / "checkpoint_best_total.pth",
        ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    configured = cfg["model"].get("deployment_weights")
    if configured:
        resolved = resolve_path(cfg, configured)
        if resolved.is_file():
            return resolved
    raise FileNotFoundError("No challenge-fine-tuned checkpoint found; pass --weights explicitly")


def evaluate_checkpoint(
    cfg: dict[str, Any],
    weights: str | Path,
    *,
    samples: tuple[FrameSample, ...] | None = None,
    warmup_iterations: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one checkpoint without mutating experiment result files."""
    import cv2

    classes = get_challenge_classes()
    weight_path = Path(weights).resolve()
    if not weight_path.is_file():
        raise FileNotFoundError(weight_path)
    detector = create_detector(cfg, classes)
    detector.load(str(weight_path))
    predictor = Predictor(detector, cfg)
    if warmup_iterations is None:
        warmup_iterations = int(cfg["inference"].get("warmup_iterations", 3))
    predictor.warmup(iterations=warmup_iterations)

    samples = samples or validation_samples(cfg)
    predictions = {}
    latency = []
    serializable_predictions = []
    for sample in samples:
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read {sample.image_path}")
        detections, timing = predictor.predict(image)
        predictions[sample.frame_id] = detections
        latency.append(timing)
        serializable_predictions.append(
            {
                "frame_id": sample.frame_id,
                "detections": [det.to_dict() for det in detections],
                "latency": timing.to_dict(),
            }
        )

    map50, per_class = coco_map50(samples, predictions, classes)
    pr = precision_recall_at_iou50(samples, predictions)
    totals = np.asarray([item.total_ms for item in latency], dtype=float)
    metrics = {
        "weights": str(weight_path),
        "validation_frame_ids": [sample.frame_id for sample in samples],
        "mAP@0.50": map50,
        "per_class_AP@0.50": per_class,
        **pr,
        "num_predictions": sum(len(v) for v in predictions.values()),
        "latency_ms": {
            "mean_total": float(np.mean(totals)) if totals.size else 0.0,
            "p50_total": float(np.percentile(totals, 50)) if totals.size else 0.0,
            "p95_total": float(np.percentile(totals, 95)) if totals.size else 0.0,
            "mean_preprocess": mean([x.preprocess_ms for x in latency]) if latency else 0.0,
            "mean_inference": mean([x.inference_ms for x in latency]) if latency else 0.0,
            "mean_postprocess": mean([x.postprocess_ms for x in latency]) if latency else 0.0,
        },
    }
    return metrics, serializable_predictions


def evaluate_offline(cfg: dict[str, Any], *, weights: str | Path | None = None) -> dict[str, Any]:
    paths = ExperimentPaths.from_config(cfg)
    paths.create()
    snapshot_config(cfg, paths)
    weight_path = Path(weights).resolve() if weights else _default_weights(cfg, paths)
    metrics, serializable_predictions = evaluate_checkpoint(cfg, weight_path)
    write_json(paths.root / "metrics.json", metrics)
    write_json(paths.predictions / "blocked_validation_predictions.json", serializable_predictions)
    return metrics
