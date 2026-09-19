from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .config import resolve_path


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path
    checkpoints: Path
    predictions: Path
    logs: Path
    final_model: Path

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "ExperimentPaths":
        base = resolve_path(cfg, cfg["paths"].get("experiments_root", "experiments"))
        root = base / str(cfg["experiment"]["name"])
        return cls(
            root=root,
            checkpoints=root / "checkpoints",
            predictions=root / "predictions",
            logs=root / "logs",
            final_model=root / "final_model",
        )

    def create(self) -> None:
        for path in (self.root, self.checkpoints, self.predictions, self.logs, self.final_model):
            path.mkdir(parents=True, exist_ok=True)


def snapshot_config(cfg: dict[str, Any], paths: ExperimentPaths) -> Path:
    paths.create()
    cleaned = {k: v for k, v in cfg.items() if not k.startswith("_")}
    target = paths.root / "config.yaml"
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cleaned, handle, sort_keys=False)
    return target


def copy_if_exists(source: str | Path, target: str | Path) -> Path | None:
    source = Path(source)
    target = Path(target)
    if not source.exists():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return target


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def runtime_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import torch

        metadata.update(
            {
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": getattr(torch.version, "cuda", None),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except ImportError:
        metadata["torch_version"] = None
    return metadata
