from __future__ import annotations

from pathlib import Path

from drone_detector.data.class_frame_report import build_class_frame_report
from drone_detector.data.split import all_frames_split
from drone_detector.types import DetectionAnnotation, FrameSample


def _sample(frame_id: int, anns):
    return FrameSample(
        frame_id=frame_id,
        image_path=Path(f"frame_{frame_id:06d}.png"),
        width=960,
        height=540,
        annotations=tuple(anns),
    )


def test_all_frames_split_has_no_independent_validation() -> None:
    split = all_frames_split([0, 1, 2, 3])
    assert split.train_frame_ids == (0, 1, 2, 3)
    assert split.val_frame_ids == ()
    assert split.purged_frame_ids == ()


def test_class_by_frame_report_finds_singleton_and_split_coverage() -> None:
    classes = ("common", "singleton")
    samples = (
        _sample(0, [DetectionAnnotation(0, 0, "common", (1, 1, 2, 2))]),
        _sample(1, [DetectionAnnotation(1, 0, "common", (1, 1, 2, 2))]),
        _sample(2, [DetectionAnnotation(2, 1, "singleton", (1, 1, 2, 2))]),
    )
    split = all_frames_split([0, 1, 2])
    report = build_class_frame_report(
        samples, classes, split=split, split_strategy="all_frames"
    )
    assert report["singleton_classes"] == ["singleton"]
    assert report["per_class"]["singleton"]["frames"] == [2]
    assert report["split_coverage"]["classes_missing_from_train"] == []
