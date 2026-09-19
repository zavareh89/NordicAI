from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from drone_detector.config import ConfigError, load_config, validate_config
from drone_detector.detector.yolo_detector import _ultralytics_scale_amplitude
from drone_detector.data.split import blocked_cross_validation, blocked_holdout_split

ROOT = Path(__file__).resolve().parents[1]


def test_default_25_frame_blocked_split_has_purge_gap() -> None:
    split = blocked_holdout_split(range(25), val_count=5, purge_gap=1, val_position="end")
    assert split.train_frame_ids == tuple(range(19))
    assert split.purged_frame_ids == (19,)
    assert split.val_frame_ids == (20, 21, 22, 23, 24)
    assert max(split.train_frame_ids) + 1 < min(split.val_frame_ids)


def test_blocked_cross_validation_never_mixes_validation_frames_into_training() -> None:
    folds = blocked_cross_validation(range(25), n_splits=5, purge_gap=1)
    assert len(folds) == 5
    for fold in folds:
        assert set(fold.train_frame_ids).isdisjoint(fold.val_frame_ids)
        assert set(fold.purged_frame_ids).isdisjoint(fold.val_frame_ids)


def test_both_configs_share_exact_level0_and_split() -> None:
    e1 = load_config(ROOT / "configs/e1_yolo26s.yaml")
    e2 = load_config(ROOT / "configs/e2_rfdetr_small.yaml")
    assert e1["camera"]["level"] == e2["camera"]["level"] == 0
    assert (e1["data"]["target_width"], e1["data"]["target_height"]) == (960, 540)
    assert (e2["data"]["target_width"], e2["data"]["target_height"]) == (960, 540)
    assert e1["data"]["interpolation"] == e2["data"]["interpolation"] == "INTER_AREA"
    assert e1["data"]["split"] == e2["data"]["split"]
    assert e1["camera"]["requested_view"] is None
    assert e2["camera"]["requested_view"] is None


def test_yolo_scale_range_maps_to_ultralytics_amplitude() -> None:
    assert _ultralytics_scale_amplitude([0.8, 1.2]) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        _ultralytics_scale_amplitude([1.1, 1.2])


def test_rfdetr_resolution_must_respect_32_pixel_block() -> None:
    cfg = load_config(ROOT / "configs/e2_rfdetr_small.yaml")
    broken = deepcopy(cfg)
    broken["model"]["resolution"] = 950
    with pytest.raises(ConfigError, match="multiple of 32"):
        validate_config(broken)
