from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..types import FrameSample, SplitDefinition


def build_class_frame_report(
    samples: Iterable[FrameSample],
    class_names: Sequence[str],
    *,
    split: SplitDefinition | None = None,
    split_strategy: str | None = None,
) -> dict[str, Any]:
    """Build a class-by-frame inventory before training.

    Counts instances, not merely presence. This makes singleton-frame classes obvious
    and reports which classes a candidate split would remove from training.
    """
    samples = tuple(sorted(samples, key=lambda s: s.frame_id))
    classes = tuple(class_names)
    class_set = set(classes)

    per_frame: list[dict[str, Any]] = []
    class_to_frames: dict[str, list[int]] = defaultdict(list)
    class_instance_counts: Counter[str] = Counter()
    matrix: dict[int, dict[str, int]] = {}

    for sample in samples:
        counts = Counter(ann.class_name for ann in sample.annotations)
        unknown = sorted(set(counts) - class_set)
        if unknown:
            raise ValueError(
                f"Frame {sample.frame_id} contains classes outside the official mapping: {unknown}"
            )
        row = {name: int(counts.get(name, 0)) for name in classes}
        matrix[sample.frame_id] = row
        present = [name for name in classes if row[name] > 0]
        per_frame.append(
            {
                "frame_id": sample.frame_id,
                "classes": present,
                "counts": {name: row[name] for name in present},
                "num_instances": int(sum(row.values())),
            }
        )
        for name in present:
            class_to_frames[name].append(sample.frame_id)
            class_instance_counts[name] += row[name]

    per_class: dict[str, Any] = {}
    for name in classes:
        frames = class_to_frames.get(name, [])
        per_class[name] = {
            "frames": frames,
            "frame_count": len(frames),
            "instance_count": int(class_instance_counts.get(name, 0)),
            "singleton_frame": frames[0] if len(frames) == 1 else None,
        }

    report: dict[str, Any] = {
        "classes": list(classes),
        "frame_ids": [sample.frame_id for sample in samples],
        "num_frames": len(samples),
        "per_frame": per_frame,
        "per_class": per_class,
        "singleton_classes": [
            name for name in classes if per_class[name]["frame_count"] == 1
        ],
        "classes_not_present_in_dataset": [
            name for name in classes if per_class[name]["frame_count"] == 0
        ],
    }

    if split is not None:
        train_ids = set(split.train_frame_ids)
        val_ids = set(split.val_frame_ids)
        purge_ids = set(split.purged_frame_ids)

        def classes_seen(frame_ids: set[int]) -> set[str]:
            seen: set[str] = set()
            for frame_id in frame_ids:
                row = matrix.get(frame_id, {})
                seen.update(name for name, count in row.items() if count > 0)
            return seen

        dataset_seen = classes_seen(set(matrix))
        train_seen = classes_seen(train_ids)
        val_seen = classes_seen(val_ids)
        purge_seen = classes_seen(purge_ids)
        report["split_coverage"] = {
            "strategy": split_strategy,
            "train_frame_ids": sorted(train_ids),
            "val_frame_ids": sorted(val_ids),
            "purged_frame_ids": sorted(purge_ids),
            "classes_seen_in_train": [name for name in classes if name in train_seen],
            "classes_seen_in_val": [name for name in classes if name in val_seen],
            "classes_seen_in_purge": [name for name in classes if name in purge_seen],
            "classes_missing_from_train": [
                name for name in classes if name in dataset_seen and name not in train_seen
            ],
            "classes_missing_from_val": [
                name for name in classes if name in dataset_seen and name not in val_seen
            ],
        }

    return report


def write_class_frame_report(
    report: dict[str, Any], output_root: str | Path
) -> tuple[Path, Path]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    json_path = output_root / "class_by_frame.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=False)

    csv_path = output_root / "class_by_frame.csv"
    classes = list(report["classes"])
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_id", *classes, "num_instances"])
        for row in report["per_frame"]:
            counts = row["counts"]
            writer.writerow(
                [
                    row["frame_id"],
                    *[int(counts.get(name, 0)) for name in classes],
                    row["num_instances"],
                ]
            )
    return json_path, csv_path


def format_class_frame_summary(report: dict[str, Any]) -> str:
    lines = [
        "Class-by-frame summary:",
        f"  frames: {report['num_frames']}",
        f"  singleton classes: {report['singleton_classes'] or 'none'}",
        f"  absent classes: {report['classes_not_present_in_dataset'] or 'none'}",
    ]
    coverage = report.get("split_coverage")
    if coverage:
        lines.extend(
            [
                f"  split strategy: {coverage.get('strategy')}",
                f"  classes missing from train: {coverage['classes_missing_from_train'] or 'none'}",
            ]
        )
    return "\n".join(lines)
