from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from ..types import Detection, FrameSample


def _xyxy_to_xywh(box):
    x1, y1, x2, y2 = (float(v) for v in box)
    return [x1, y1, x2 - x1, y2 - y1]


def coco_map50(
    samples: Sequence[FrameSample],
    predictions: Mapping[int, Sequence[Detection]],
    class_names: Sequence[str],
) -> tuple[float, dict[str, float]]:
    from faster_coco_eval import COCO, COCOeval_faster

    frame_to_image_id = {sample.frame_id: idx for idx, sample in enumerate(samples, start=1)}
    category_ids = {name: idx for idx, name in enumerate(class_names, start=1)}
    present_classes = {
        ann.class_name for sample in samples for ann in sample.annotations
    }
    evaluated_classes = tuple(name for name in class_names if name in present_classes)
    if not evaluated_classes:
        raise ValueError("Validation set has no ground-truth objects")

    gt_annotations = []
    annotation_id = 1
    for sample in samples:
        for ann in sample.annotations:
            x, y, width, height = _xyxy_to_xywh(ann.bbox_xyxy)
            gt_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": frame_to_image_id[sample.frame_id],
                    "category_id": category_ids[ann.class_name],
                    "bbox": [x, y, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    gt = {
        "info": {"description": "Blocked Level-0 validation"},
        "licenses": [],
        "images": [
            {
                "id": frame_to_image_id[sample.frame_id],
                "file_name": sample.image_path.name,
                "width": sample.width,
                "height": sample.height,
            }
            for sample in samples
        ],
        "categories": [
            {"id": category_ids[name], "name": name, "supercategory": "object"}
            for name in class_names
        ],
        "annotations": gt_annotations,
    }
    dt = []
    for sample in samples:
        for detection in predictions.get(sample.frame_id, []):
            x, y, width, height = _xyxy_to_xywh(detection.bbox_xyxy)
            if width <= 0 or height <= 0:
                continue
            dt.append(
                {
                    "image_id": frame_to_image_id[sample.frame_id],
                    "category_id": category_ids[detection.class_name],
                    "bbox": [x, y, width, height],
                    "score": float(detection.confidence),
                }
            )
    if not dt:
        return 0.0, {name: 0.0 for name in evaluated_classes}

    coco_gt = COCO(gt)
    coco_dt = coco_gt.loadRes(dt)
    evaluator = COCOeval_faster(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = list(frame_to_image_id.values())
    evaluator.params.catIds = [category_ids[name] for name in evaluated_classes]
    evaluator.params.iouThrs = np.array([0.50])
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval["precision"]
    ap_by_class = {}
    for category_index, name in enumerate(evaluated_classes):
        values = precision[0, :, category_index, 0, -1]
        valid = values[values > -1]
        ap_by_class[name] = float(np.mean(valid)) if valid.size else 0.0
    overall = float(np.mean(list(ap_by_class.values()))) if ap_by_class else 0.0
    return max(0.0, min(1.0, overall)), {
        name: max(0.0, min(1.0, value)) for name, value in ap_by_class.items()
    }


def _iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def precision_recall_at_iou50(
    samples: Sequence[FrameSample], predictions: Mapping[int, Sequence[Detection]]
) -> dict[str, float | int]:
    gt_by_key: dict[tuple[int, str], list[tuple[float, float, float, float]]] = defaultdict(list)
    for sample in samples:
        for ann in sample.annotations:
            gt_by_key[(sample.frame_id, ann.class_name)].append(ann.bbox_xyxy)

    tp = fp = 0
    matched_total = 0
    gt_total = sum(len(v) for v in gt_by_key.values())
    for sample in samples:
        by_class: dict[str, list[Detection]] = defaultdict(list)
        for det in predictions.get(sample.frame_id, []):
            by_class[det.class_name].append(det)
        for class_name, dets in by_class.items():
            gt_boxes = gt_by_key.get((sample.frame_id, class_name), [])
            used: set[int] = set()
            for det in sorted(dets, key=lambda d: d.confidence, reverse=True):
                candidates = [
                    (idx, _iou_xyxy(det.bbox_xyxy, gt_box))
                    for idx, gt_box in enumerate(gt_boxes)
                    if idx not in used
                ]
                if candidates:
                    best_idx, best_iou = max(candidates, key=lambda pair: pair[1])
                else:
                    best_idx, best_iou = -1, 0.0
                if best_iou >= 0.50:
                    tp += 1
                    matched_total += 1
                    used.add(best_idx)
                else:
                    fp += 1
    fn = gt_total - matched_total
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}
