from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, Sequence

import yaml

from ..types import FrameSample, SplitDefinition


def xyxy_to_yolo(
    bbox: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    xc = ((x1 + x2) / 2.0) / width
    yc = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    values = (xc, yc, bw, bh)
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"YOLO normalized box outside [0,1]: {values}")
    if bw <= 0 or bh <= 0:
        raise ValueError(f"YOLO box has non-positive size: {values}")
    return values


def _write_yolo_partition(
    sample_map: dict[int, FrameSample],
    frame_ids: Sequence[int],
    output_root: Path,
    part: str,
) -> None:
    images_dir = output_root / "images" / part
    labels_dir = output_root / "labels" / part
    # Avoid stale frames/labels when switching between all_frames and blocked_holdout.
    if images_dir.exists():
        shutil.rmtree(images_dir)
    if labels_dir.exists():
        shutil.rmtree(labels_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for frame_id in frame_ids:
        if frame_id not in sample_map:
            raise KeyError(f"Split references missing frame {frame_id}")
        sample = sample_map[frame_id]
        shutil.copy2(sample.image_path, images_dir / sample.image_path.name)
        lines = []
        for ann in sample.annotations:
            xc, yc, bw, bh = xyxy_to_yolo(
                ann.bbox_xyxy, sample.width, sample.height
            )
            lines.append(
                f"{ann.class_id} {xc:.10f} {yc:.10f} {bw:.10f} {bh:.10f}"
            )
        (labels_dir / f"{sample.image_path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )


def export_yolo_dataset(
    samples: Iterable[FrameSample],
    split: SplitDefinition,
    output_root: str | Path,
    class_names: Sequence[str],
) -> Path:
    """Export YOLO data.

    In all-frames mode `split.val_frame_ids` is empty. Ultralytics still expects a
    syntactically valid `val` entry while checking the dataset, so `val` points to
    `images/train`. Training config sets `val=False`; this mirror is never treated as
    held-out evaluation.
    """
    output_root = Path(output_root)
    sample_map = {sample.frame_id: sample for sample in samples}
    _write_yolo_partition(sample_map, split.train_frame_ids, output_root, "train")

    has_independent_val = bool(split.val_frame_ids)
    if has_independent_val:
        _write_yolo_partition(sample_map, split.val_frame_ids, output_root, "val")
        val_path = "images/val"
        marker = output_root / "VALIDATION_IS_TRAIN_MIRROR.txt"
        if marker.exists():
            marker.unlink()
    else:
        # Remove a stale held-out val partition from a previous preparation run.
        for stale in (output_root / "images" / "val", output_root / "labels" / "val"):
            if stale.exists():
                shutil.rmtree(stale)
        val_path = "images/train"
        (output_root / "VALIDATION_IS_TRAIN_MIRROR.txt").write_text(
            "No independent validation split exists. data.yaml points val to train only "
            "to satisfy framework dataset checks. Do not report these metrics as held-out.\n",
            encoding="utf-8",
        )

    data_yaml = output_root / "data.yaml"
    payload = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": val_path,
        "names": {idx: name for idx, name in enumerate(class_names)},
        "nc": len(class_names),
    }
    with data_yaml.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return data_yaml
