from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import resolve_path
from ..experiment import ExperimentPaths, runtime_metadata, snapshot_config, write_json
from ..reproducibility import seed_everything


def prepare_experiment(cfg: dict[str, Any]) -> ExperimentPaths:
    paths = ExperimentPaths.from_config(cfg)
    paths.create()
    snapshot_config(cfg, paths)
    seed_state = seed_everything(
        int(cfg["experiment"].get("seed", 42)),
        deterministic=bool(cfg["training"].get("deterministic", False)),
    )
    write_json(paths.root / "runtime.json", {**runtime_metadata(), "seed_state": seed_state})

    prepared_root = resolve_path(cfg, cfg["paths"].get("prepared_root", "prepared"))
    for filename in ("dataset_manifest.json", "split.json"):
        source = prepared_root / filename
        if not source.is_file():
            raise FileNotFoundError(
                f"Missing {source}. Run `python scripts/prepare_dataset.py` before training."
            )
        (paths.root / filename).write_bytes(source.read_bytes())
    return paths


def record_training_manifest(cfg: dict[str, Any], paths: ExperimentPaths, payload: dict[str, Any]) -> Path:
    return write_json(
        paths.root / "training_manifest.json",
        {
            "experiment": cfg["experiment"],
            "model": cfg["model"],
            "training": cfg["training"],
            "augmentation": cfg["augmentation"],
            "inference": cfg["inference"],
            **payload,
        },
    )
