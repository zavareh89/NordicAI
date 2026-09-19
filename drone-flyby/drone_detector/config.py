from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ConfigError(f"Config must be a mapping: {path}")
    cfg = copy.deepcopy(cfg)
    cfg["_config_path"] = str(path)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    required_sections = (
        "experiment",
        "camera",
        "data",
        "model",
        "training",
        "augmentation",
        "inference",
        "paths",
    )
    missing = [name for name in required_sections if name not in cfg]
    if missing:
        raise ConfigError(f"Missing config sections: {', '.join(missing)}")
    if int(cfg["camera"].get("level", -1)) != 0:
        raise ConfigError("E1/E2 are detector-only Level-0 scenarios; camera.level must be 0")
    width = int(cfg["data"].get("target_width", 0))
    height = int(cfg["data"].get("target_height", 0))
    if (width, height) != (960, 540):
        raise ConfigError("Baseline Level-0 target must be exactly 960x540")
    family = str(cfg["model"].get("family", "")).lower()
    if family not in {"yolo26", "rfdetr"}:
        raise ConfigError(f"Unsupported detector family: {family}")

    if cfg["camera"].get("requested_view", None) is not None:
        raise ConfigError("E1/E2 must set camera.requested_view to null/None")
    if not bool(cfg["model"].get("pretrained", False)):
        raise ConfigError("E1/E2 require official pretrained initialization")
    if family == "rfdetr":
        resolution = int(cfg["model"].get("resolution", 0))
        if resolution <= 0 or resolution % 32 != 0:
            raise ConfigError("RF-DETR Small resolution must be a positive multiple of 32")
    conf = float(cfg["inference"].get("confidence_threshold", 0.0))
    if not 0.0 <= conf <= 1.0:
        raise ConfigError("inference.confidence_threshold must be within [0, 1]")
    max_det = int(cfg["inference"].get("max_detections", 0))
    if not 1 <= max_det <= 500:
        raise ConfigError("inference.max_detections must be between 1 and 500")


def scenario_config_path(project_root: str | Path, scenario: str) -> Path:
    scenario = scenario.strip().upper()
    names = {
        "E1": "e1_yolo26s.yaml",
        "E2": "e2_rfdetr_small.yaml",
    }
    if scenario not in names:
        raise ConfigError(f"Unknown scenario {scenario!r}; expected E1 or E2")
    return Path(project_root) / "configs" / names[scenario]


def project_root_from_config(cfg: dict[str, Any]) -> Path:
    cfg_path = Path(cfg["_config_path"])
    return cfg_path.parent.parent.resolve()


def resolve_path(cfg: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root_from_config(cfg) / path).resolve()
