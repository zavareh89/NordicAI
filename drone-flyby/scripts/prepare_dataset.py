from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drone_detector.config import load_config, resolve_path
from drone_detector.data.challenge_annotations import discover_scene_samples, get_challenge_classes
from drone_detector.data.coco_export import export_coco_dataset
from drone_detector.data.level0_dataset import prepare_level0_dataset
from drone_detector.data.split import blocked_cross_validation, blocked_holdout_split
from drone_detector.data.validation import validate_samples
from drone_detector.data.yolo_export import export_yolo_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare exact Level-0 data for E1 and E2.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "e1_yolo26s.yaml"))
    parser.add_argument("--blocked-cv", action="store_true", help="Also save optional blocked CV folds.")
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
        target_size=(int(cfg["data"]["target_width"]), int(cfg["data"]["target_height"])),
    )
    validate_samples(level0_samples, classes)

    split_cfg = cfg["data"]["split"]
    split = blocked_holdout_split(
        [sample.frame_id for sample in level0_samples],
        val_count=int(split_cfg.get("val_count", 5)),
        purge_gap=int(split_cfg.get("purge_gap", 1)),
        val_position=str(split_cfg.get("val_position", "end")),
    )
    prepared_root.mkdir(parents=True, exist_ok=True)
    with (prepared_root / "split.json").open("w", encoding="utf-8") as handle:
        json.dump(split.to_dict(), handle, indent=2)
    # Canonical top-level copy consumed by experiment snapshotting.
    with (prepared_root / "dataset_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    export_yolo_dataset(level0_samples, split, prepared_root / "yolo", classes)
    export_coco_dataset(level0_samples, split, prepared_root / "rfdetr", classes)

    if args.blocked_cv:
        folds = blocked_cross_validation(
            [sample.frame_id for sample in level0_samples],
            n_splits=int(split_cfg.get("blocked_cv_splits", 5)),
            purge_gap=int(split_cfg.get("purge_gap", 1)),
        )
        with (prepared_root / "blocked_cv.json").open("w", encoding="utf-8") as handle:
            json.dump([fold.to_dict() for fold in folds], handle, indent=2)

    print(f"Prepared {len(level0_samples)} Level-0 frames under {prepared_root}")
    print(f"Train frames: {list(split.train_frame_ids)}")
    print(f"Purged boundary frames: {list(split.purged_frame_ids)}")
    print(f"Validation frames: {list(split.val_frame_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
