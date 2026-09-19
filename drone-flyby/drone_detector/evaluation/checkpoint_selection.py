from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Iterable

from ..experiment import write_json
from .offline_eval import evaluate_checkpoint


def _deduplicate_existing(paths: Iterable[str | Path]) -> list[Path]:
    output: list[Path] = []
    seen: set[Path] = set()
    for value in paths:
        path = Path(value).resolve()
        if not path.is_file() or path in seen:
            continue
        seen.add(path)
        output.append(path)
    return sorted(output, key=lambda p: p.name)


def yolo_checkpoint_candidates(backend_dir: str | Path) -> list[Path]:
    weights = Path(backend_dir) / "weights"
    return _deduplicate_existing(
        [weights / "best.pt", weights / "last.pt", *weights.glob("epoch*.pt")]
    )


def rfdetr_checkpoint_candidates(backend_dir: str | Path) -> list[Path]:
    """RF-DETR 1.5.2 emits legacy .pth files, not Lightning .ckpt files."""
    root = Path(backend_dir)
    return _deduplicate_existing(
        [
            root / "checkpoint_best_total.pth",
            root / "checkpoint_best_ema.pth",
            root / "checkpoint_best_regular.pth",
            root / "checkpoint.pth",  # overwritten each epoch: final/last state
            *root.glob("checkpoint[0-9][0-9][0-9][0-9].pth"),
        ]
    )


def _release_cuda_cache() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def select_best_checkpoint_by_map50(
    cfg: dict[str, Any],
    candidates: Iterable[str | Path],
    *,
    report_path: str | Path,
) -> tuple[Path, list[dict[str, Any]]]:
    """Select deployment checkpoint using challenge-aligned AP@IoU=0.50."""
    candidates = _deduplicate_existing(candidates)
    if not candidates:
        raise FileNotFoundError("No saved checkpoints are available for AP@0.50 selection")

    rows: list[dict[str, Any]] = []
    best_path: Path | None = None
    best_key: tuple[float, float, float] | None = None

    for candidate in candidates:
        metrics, _ = evaluate_checkpoint(cfg, candidate, warmup_iterations=1)
        row = {
            "checkpoint": str(candidate),
            "mAP@0.50": float(metrics["mAP@0.50"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "num_predictions": int(metrics["num_predictions"]),
            "mean_latency_ms": float(metrics["latency_ms"]["mean_total"]),
        }
        rows.append(row)
        key = (row["mAP@0.50"], row["recall"], row["precision"])
        if best_key is None or key > best_key:
            best_key = key
            best_path = candidate
        _release_cuda_cache()

    assert best_path is not None
    write_json(
        report_path,
        {
            "selection_metric": "mAP@0.50",
            "tie_breakers": ["recall", "precision"],
            "selected_checkpoint": str(best_path),
            "candidates": rows,
        },
    )
    return best_path, rows
