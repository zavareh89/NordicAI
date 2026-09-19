from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drone_detector.config import load_config, resolve_path
from drone_detector.data.challenge_annotations import (
    discover_scene_samples,
    get_challenge_classes,
)
from drone_detector.data.class_frame_report import (
    build_class_frame_report,
    format_class_frame_summary,
    write_class_frame_report,
)
from drone_detector.data.coco_export import export_coco_dataset
from drone_detector.data.level0_dataset import prepare_level0_dataset
from drone_detector.data.split import (
    all_frames_split,
    blocked_cross_validation,
    blocked_holdout_split,
)
from drone_detector.data.validation import validate_samples
from drone_detector.data.yolo_export import export_yolo_dataset


def _make_split(cfg: dict, frame_ids: list[int]):
    split_cfg = cfg["data"]["split"]
    strategy = str(split_cfg.get("strategy", "all_frames")).lower()
    if strategy == "all_frames":
        return strategy, all_frames_split(frame_ids)
    if strategy == "blocked_holdout":
        return strategy, blocked_holdout_split(
            frame_ids,
            val_count=int(split_cfg.get("val_count", 5)),
            purge_gap=int(split_cfg.get("purge_gap", 1)),
            val_position=str(split_cfg.get("val_position", "end")),
        )
    raise ValueError(
        f"Unsupported data.split.strategy={strategy!r}; expected all_frames or blocked_holdout"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare exact Level-0 data for E1/E2 and report class-by-frame coverage."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "e1_yolo26s.yaml"),
    )
    parser.add_argument(
        "--blocked-cv",
        action="store_true",
        help="Also save optional blocked CV folds for diagnostics.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    classes = get_challenge_classes()
    scene_dir = resolve_path(cfg, cfg["paths"]["challenge_scene"])
    prepared_root = resolve_path(cfg, cfg["paths"].get("prepared_root", "prepared"))
    level0_root = prepared_root / "level0"

    source_samples = discover_scene_samples(scene_dir, class_names=classes)
    validate_samples(source_samples, classes)
    level0_samples, manifest = prepare_level0_dataset(
        source_samples,
        level0_root,
        target_size=(
            int(cfg["data"]["target_width"]),
            int(cfg["data"]["target_height"]),
        ),
    )
    validate_samples(level0_samples, classes)

    frame_ids = [sample.frame_id for sample in level0_samples]
    strategy, split = _make_split(cfg, frame_ids)

    prepared_root.mkdir(parents=True, exist_ok=True)
    split_payload = {
        **split.to_dict(),
        "strategy": strategy,
        "validation_is_independent": bool(split.val_frame_ids),
        "framework_validation_mode": (
            "held_out" if split.val_frame_ids else "train_mirror_for_framework_only"
        ),
    }
    with (prepared_root / "split.json").open("w", encoding="utf-8") as handle:
        json.dump(split_payload, handle, indent=2)

    with (prepared_root / "dataset_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)

    report = build_class_frame_report(
        level0_samples,
        classes,
        split=split,
        split_strategy=strategy,
    )
    json_report, csv_report = write_class_frame_report(report, prepared_root)

    # A blocked holdout that removes a dataset-present class from training is unsafe
    # for this tiny challenge dataset. Fail loudly instead of silently training a model
    # that has never seen that class.
    missing_from_train = report.get("split_coverage", {}).get(
        "classes_missing_from_train", []
    )
    if strategy == "blocked_holdout" and missing_from_train:
        raise RuntimeError(
            "Blocked split removes classes from training: "
            f"{missing_from_train}. Use data.split.strategy=all_frames for final training "
            "or choose a different diagnostic split. Class report: "
            f"{json_report}"
        )

    export_yolo_dataset(level0_samples, split, prepared_root / "yolo", classes)
    export_coco_dataset(level0_samples, split, prepared_root / "rfdetr", classes)

    if args.blocked_cv:
        split_cfg = cfg["data"]["split"]
        folds = blocked_cross_validation(
            frame_ids,
            n_splits=int(split_cfg.get("blocked_cv_splits", 5)),
            purge_gap=int(split_cfg.get("purge_gap", 1)),
        )
        with (prepared_root / "blocked_cv.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump([fold.to_dict() for fold in folds], handle, indent=2)

    print(format_class_frame_summary(report))
    print(f"Class-by-frame JSON: {json_report}")
    print(f"Class-by-frame CSV:  {csv_report}")
    print(f"Prepared {len(level0_samples)} Level-0 frames under {prepared_root}")
    print(f"Training frames: {list(split.train_frame_ids)}")
    if split.val_frame_ids:
        print(f"Validation frames: {list(split.val_frame_ids)}")
    else:
        print("Independent validation: disabled (all labeled frames used for training)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
