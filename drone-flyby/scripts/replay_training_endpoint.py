#!/usr/bin/env python3
"""Replay prepared Level-0 training frames through the official Drone Flyby API.

Run from the challenge repository. No detector imports, training or server changes.
See README_ENDPOINT_REPLAY.md for metric definitions and diagnostic limitations.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit, urlunsplit
import uuid

import cv2
import numpy as np
import requests
from faster_coco_eval import COCO, COCOeval_faster


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_protocol(root):
    sys.path.insert(0, str(Path(root).resolve()))
    import dtos
    return dtos


def finite_box(box, width, height):
    if len(box) != 4:
        raise ValueError(f"Expected xyxy box, got {box}")
    x1, y1, x2, y2 = map(float, box)
    if not all(np.isfinite([x1, y1, x2, y2])) or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"Invalid xyxy box {box} for {width}x{height}")
    return [x1, y1, x2, y2]


def load_frames(manifest_path, split_path, subset, images_dir, protocol):
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    frames = manifest["frames"]
    ids = [f["frame_id"] for f in frames]
    if not frames or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Manifest must contain unique nonnegative integer frame IDs")
    # frame_index is the position in the original sequence, not the subset index.
    positions = {fid: index for index, fid in enumerate(sorted(ids))}
    if subset != "all":
        split = read_json(split_path)
        wanted = split[f"{subset}_frame_ids"]
        if len(set(wanted)) != len(wanted) or set(wanted) - set(ids):
            raise ValueError("Split contains duplicate IDs or IDs absent from the manifest")
        frames = [f for f in frames if f["frame_id"] in set(wanted)]
    if not frames:
        raise ValueError(f"No frames in subset {subset!r}; all_frames training has no held-out val set")
    output = []
    for f in sorted(frames, key=lambda x: x["frame_id"]):
        fid = f["frame_id"]
        w, h = int(f["target_width"]), int(f["target_height"])
        sw, sh = int(f["source_width"]), int(f["source_height"])
        if (w, h) != tuple(protocol.TRANSMITTED_VIEW_SIZE) or (sw, sh) != (protocol.IMAGE_WIDTH, protocol.IMAGE_HEIGHT):
            raise ValueError(f"Frame {fid}: dimensions do not match the official Level-0 contract")
        name = Path(f["level0_image"]).name
        if images_dir:
            candidates = [Path(images_dir) / name]
        else:
            candidates = [Path(f["level0_image"]), manifest_path.parent / "level0" / "images" / name,
                          manifest_path.parent / "images" / name]
        image_path = next((p.resolve() for p in candidates if p.is_file()), None)
        if image_path is None:
            raise FileNotFoundError(f"Frame {fid}: prepared PNG not found; use --images-dir prepared/level0/images")
        encoded = image_path.read_bytes()
        if not encoded.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"Not a PNG: {image_path}")
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (h, w):
            raise ValueError(f"Invalid prepared image shape: {image_path}")
        anns = []
        for a in f["annotations"]:
            if a["class_name"] not in protocol.OBJECT_CLASSES:
                raise ValueError(f"Unknown ground-truth class: {a['class_name']}")
            if a.get("frame_id", fid) != fid:
                raise ValueError(f"Annotation frame mismatch on {fid}")
            box = finite_box(a["bbox_xyxy"], w, h)
            # Score in source pixels exactly as the official evaluator does.
            anns.append({"object_id": a["class_name"], "bbox": [box[0]*sw/w, box[1]*sh/h, box[2]*sw/w, box[3]*sh/h]})
        output.append({"frame": fid, "frame_index": positions[fid], "image_path": str(image_path),
                       "png_sha256": hashlib.sha256(encoded).hexdigest(), "width": sw, "height": sh,
                       "annotations": anns})
    if not any(f["annotations"] for f in output):
        raise ValueError("Selected subset has no ground truth; mAP is undefined")
    return output


def build_payload(frame, protocol, sequence):
    w, h = frame["width"], frame["height"]
    request_id = f"{sequence}:{frame['frame_index']}:0:{w//2}:{h//2}"
    bounds = []
    allowed = protocol.ALLOWED_RESOLUTION_LEVELS[0]
    for level in allowed:
        rw, rh = protocol.SOURCE_REGION_SIZES[level]
        bounds.append({"resolution_level": level, "width": rw, "height": rh,
                       "minimum_center_x": rw//2, "maximum_center_x": w-rw//2,
                       "minimum_center_y": rh//2, "maximum_center_y": h-rh//2})
    payload = {
        "sequence_id": sequence, "frame": frame["frame"], "frame_index": frame["frame_index"],
        "request_id": request_id, "frame_interval_ms": 333, "response_timeout_ms": 3333,
        "original_width": w, "original_height": h,
        "view": {"resolution_level": 0, "center_x": w//2, "center_y": h//2,
                 "view_id": request_id, "image": base64.b64encode(Path(frame["image_path"]).read_bytes()).decode("ascii"),
                 "image_media_type": "image/png", "width": protocol.TRANSMITTED_VIEW_SIZE[0],
                 "height": protocol.TRANSMITTED_VIEW_SIZE[1], "source_region_xyxy": [0, 0, w, h]},
        "camera_constraints": {"maximum_center_delta": protocol.MAXIMUM_CENTER_DELTA_PIXELS[0],
                               "allowed_resolution_levels": list(allowed), "center_bounds": bounds,
                               "full_view_reset_exempt_from_delta": True},
        "camera_command_feedback": None,
    }
    protocol.DroneFlybyPredictRequestDto.model_validate(payload)
    return payload


def validate_predictions(body, payload, protocol):
    response = protocol.DroneFlybyPredictResponseDto.model_validate(body)
    if response.frame != payload["frame"] or response.request_id != payload["request_id"]:
        raise ValueError("Response frame/request_id does not match request")
    # This diagnostic must keep the camera at Level 0 for every frame.
    predictions = []
    w, h = payload["original_width"], payload["original_height"]
    for a in response.annotations:
        x1, y1, x2, y2 = a.bbox
        predictions.append({"object_id": a.object_id, "confidence": float(a.confidence),
                            "bbox": [float(x1)*w, float(y1)*h, float(x2)*w, float(y2)*h]})
    return predictions, response.requested_view is not None


def coco_metrics(frames, predictions, classes, max_dets=100):
    present = [c for c in classes if any(a["object_id"] == c for f in frames for a in f["annotations"])]
    cats = {c: i+1 for i, c in enumerate(classes)}
    gt = {"info": {"description": "Endpoint replay diagnostic"}, "licenses": [], "images": [],
          "categories": [{"id": cats[c], "name": c} for c in classes], "annotations": []}
    dt = []
    for image_id, f in enumerate(frames, 1):
        gt["images"].append({"id": image_id, "width": f["width"], "height": f["height"]})
        for a in f["annotations"]:
            x1, y1, x2, y2 = a["bbox"]
            gt["annotations"].append({"id": len(gt["annotations"])+1, "image_id": image_id,
                "category_id": cats[a["object_id"]], "bbox": [x1,y1,x2-x1,y2-y1],
                "area": (x2-x1)*(y2-y1), "iscrowd": 0})
        for a in predictions.get(f["frame"], []):
            x1, y1, x2, y2 = a["bbox"]
            dt.append({"image_id": image_id, "category_id": cats[a["object_id"]],
                       "bbox": [x1,y1,x2-x1,y2-y1], "score": a["confidence"]})
    if not present:
        raise ValueError("mAP undefined: no ground-truth classes")
    cg = COCO(gt)
    if dt:
        cd = cg.loadRes(dt)
    else:
        # loadRes([]) is not supported consistently across COCO implementations.
        cd = COCO({**gt, "annotations": []})
    ev = COCOeval_faster(cg, cd, "bbox")
    ev.params.imgIds = [i["id"] for i in gt["images"]]
    ev.params.catIds = [cats[c] for c in present]
    ev.params.maxDets = [1, 10, max_dets]
    ev.evaluate()
    ev.accumulate()
    precision, recall = ev.eval["precision"], ev.eval["recall"]
    def avg(v):
        v = np.asarray(v)
        v = v[v > -1]
        return float(v.mean()) if v.size else None
    i50 = int(np.argmin(abs(ev.params.iouThrs - .50)))
    i75 = int(np.argmin(abs(ev.params.iouThrs - .75)))
    per_class = {}
    curves = []
    for k, c in enumerate(present):
        per_class[c] = {"ap50": avg(precision[i50,:,k,0,-1]), "ap75": avg(precision[i75,:,k,0,-1]),
                        "ap50_95": avg(precision[:,:,k,0,-1]), "ar50_95": avg(recall[:,k,0,-1])}
        for j, r in enumerate(ev.params.recThrs):
            curves.append({"class_name": c, "recall": float(r), "precision_iou50_interpolated": float(precision[i50,j,k,0,-1])})
    return {"map50": avg(precision[i50,:,:,0,-1]), "map75": avg(precision[i75,:,:,0,-1]),
            "map50_95": avg(precision[:,:,:,0,-1]), "mar50_95": avg(recall[:,:,0,-1]),
            "max_detections_per_image_per_class": max_dets, "per_class": per_class,
            "evaluated_classes": present, "classes_without_ground_truth": [c for c in classes if c not in present]}, curves


def iou(a, b):
    intersection = max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection/union if union else 0.0


def operating_metrics(frames, predictions, threshold, class_agnostic=False):
    tp = fp = gt_count = 0
    overlaps = []
    for f in frames:
        gt = f["annotations"]
        gt_count += len(gt)
        used = set()
        for d in sorted(predictions.get(f["frame"], []), key=lambda x: -x["confidence"]):
            if d["confidence"] < threshold:
                continue
            options = [(iou(d["bbox"], a["bbox"]), j) for j,a in enumerate(gt)
                       if j not in used and (class_agnostic or a["object_id"] == d["object_id"])]
            overlap, j = max(options, default=(0,-1))
            if overlap >= .5:
                used.add(j)
                tp += 1
                overlaps.append(overlap)
            else:
                fp += 1
    fn = gt_count-tp
    p, r = tp/(tp+fp) if tp+fp else 0.0, tp/gt_count if gt_count else 0.0
    return {"confidence_threshold": threshold, "iou_threshold": .5, "tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r, "f1": 2*p*r/(p+r) if p+r else 0.0,
            "mean_matched_iou": float(np.mean(overlaps)) if overlaps else None}


def overlay(frame, predictions, destination, threshold):
    image = cv2.imread(frame["image_path"])
    h,w = image.shape[:2]
    for annotations, color, prefix in [(frame["annotations"], (0,220,0), "GT"), (predictions, (0,140,255), "P")]:
        for a in annotations:
            if prefix == "P" and a["confidence"] < threshold:
                continue
            x1,y1,x2,y2 = a["bbox"]
            p1 = (round(x1*w/frame["width"]), round(y1*h/frame["height"]))
            p2 = (round(x2*w/frame["width"]), round(y2*h/frame["height"]))
            label = f"{prefix} {a['object_id']}" + (f" {a['confidence']:.3f}" if prefix == "P" else "")
            cv2.rectangle(image, p1, p2, color, 1)
            cv2.putText(image, label, (p1[0], max(12, p1[1]-3)), cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1, cv2.LINE_AA)
    if not cv2.imwrite(str(destination), image):
        raise OSError(f"Could not write {destination}")


def run(args):
    protocol = load_protocol(args.challenge_root)
    frames = load_frames(args.manifest, args.split, args.subset, args.images_dir, protocol)
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory is not empty: {out}; choose a new --output")
    out.mkdir(parents=True, exist_ok=True)
    (out/"responses").mkdir()
    if args.overlays:
        (out/"overlays").mkdir()
    sequence = f"diagnostic-{uuid.uuid4().hex[:12]}"
    predictions, rows = {}, []
    headers = read_json(args.headers_json) if args.headers_json else {}
    if not isinstance(headers, dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in headers.items()):
        raise ValueError("--headers-json must contain a JSON object of header names to string values")
    parsed = urlsplit(args.url)
    safe_url = urlunsplit((parsed.scheme, parsed.hostname + (f":{parsed.port}" if parsed.port else ""), parsed.path, "", ""))
    metadata = {"started_utc": datetime.now(timezone.utc).isoformat(), "run_label": args.run_label,
                "sequence_id": sequence, "endpoint": safe_url, "subset": args.subset,
                "manifest_sha256": sha256(args.manifest), "split_sha256": sha256(args.split) if args.subset != "all" else None,
                "protocol_file": str(Path(protocol.__file__).resolve()), "protocol_sha256": sha256(protocol.__file__),
                "frames": frames, "client_timeout_seconds": args.timeout,
                "sequential_no_frame_skipping": True, "server_checkpoint": "Not introspected; identify with --run-label",
                "metrics_use_all_returned_confidences": True, "operating_threshold": args.confidence}
    write_json(out/"run_metadata.json", metadata)
    with requests.Session() as session, (out/"frame_events.jsonl").open("w", encoding="utf-8") as log:
        session.headers.update(headers)
        for f in frames:
            payload = build_payload(f, protocol, sequence)
            # Encoding is outside HTTP round-trip timing. No retries or reset calls.
            body, error, status, camera_requested, http_status = None, None, "ok", False, None
            raw_text = None
            start = time.perf_counter()
            response = None
            try:
                response = session.post(args.url, json=payload, timeout=args.timeout, allow_redirects=False)
                http_status = response.status_code
                raw_text = response.text
                if not 200 <= http_status < 300:
                    raise requests.HTTPError(f"HTTP {http_status}")
                try:
                    body = response.json()
                except ValueError as exc:
                    raise ValueError("Response was not valid JSON") from exc
                http_ms = (time.perf_counter()-start)*1000
                dets, camera_requested = validate_predictions(body, payload, protocol)
            except requests.Timeout:
                status, error, dets = "timeout", "Client HTTP timeout", []
            except requests.RequestException as exc:
                status, error, dets = "http_error", type(exc).__name__, []
            except (ValueError, TypeError, KeyError) as exc:
                status, error, dets = "invalid_response", str(exc), []
            elapsed = (time.perf_counter()-start)*1000
            if status != "ok":
                http_ms = elapsed
            predictions[f["frame"]] = dets
            event = {"frame": f["frame"], "frame_index": f["frame_index"], "request_id": payload["request_id"],
                     "status": status, "http_status": http_status, "http_round_trip_ms": http_ms,
                     "over_333ms": http_ms > 333, "over_3333ms": http_ms > 3333,
                     "gt_count": len(f["annotations"]), "prediction_count": len(dets),
                     "camera_command_ignored": camera_requested, "error": error}
            log.write(json.dumps(event, allow_nan=False)+"\n")
            log.flush()
            write_json(out/"responses"/f"frame_{f['frame']:06d}.json",
                       {"request_metadata": {**payload, "view": {k:v for k,v in payload["view"].items() if k != "image"}},
                        "response": body, "raw_text_if_unparsed": raw_text if body is None else None, "event": event})
            rows.append(event)
            print(f"frame={f['frame']} {status} gt={len(f['annotations'])} predictions={len(dets)} HTTP={http_ms:.1f}ms", flush=True)
            if args.overlays:
                overlay(f, dets, out/"overlays"/f"frame_{f['frame']:06d}.png", args.confidence)
    write_json(out/"predictions_source_xyxy.json", predictions)
    write_csv(out/"per_frame.csv", rows)
    metrics, curves = coco_metrics(frames, predictions, protocol.OBJECT_CLASSES, args.max_dets)
    write_csv(out/"pr_curve_iou50.csv", curves)
    operating = operating_metrics(frames, predictions, args.confidence)
    agnostic = operating_metrics(frames, predictions, args.confidence, True)
    sweep = [operating_metrics(frames, predictions, t) for t in sorted(set([0,.01,.03,.05,.08,.1,.25,.5,.75,args.confidence]))]
    write_csv(out/"confidence_sweep.csv", sweep)
    per_class = []
    for c in protocol.OBJECT_CLASSES:
        cf = [{**f, "annotations": [a for a in f["annotations"] if a["object_id"] == c]} for f in frames]
        cp = {k:[a for a in v if a["object_id"] == c] for k,v in predictions.items()}
        per_class.append({"class_name": c, "gt_count": sum(len(f["annotations"]) for f in cf),
                          "prediction_count": sum(map(len, cp.values())),
                          **metrics["per_class"].get(c, {"ap50": None,"ap75": None,"ap50_95": None,"ar50_95": None}),
                          **operating_metrics(cf,cp,args.confidence)})
    write_csv(out/"per_class.csv", per_class)
    # This is a timeout-only approximation, not the official realtime scheduler.
    timely = {r["frame"]: predictions[r["frame"]] if r["status"] == "ok" and not r["over_3333ms"] else [] for r in rows}
    deadline_metrics = coco_metrics(frames, timely, protocol.OBJECT_CLASSES, args.max_dets)[0] if any(r["over_3333ms"] for r in rows) else metrics
    times = [r["http_round_trip_ms"] for r in rows]
    scores = [a["confidence"] for v in predictions.values() for a in v]
    summary = {"evaluation": "training replay diagnostic" if args.subset == "train" else f"{args.subset} replay diagnostic; independence depends on checkpoint provenance",
               "run_label": args.run_label, "frame_count": len(frames), "ground_truth_count": sum(len(f["annotations"]) for f in frames),
               "prediction_count": sum(map(len,predictions.values())), "statuses": dict(Counter(r["status"] for r in rows)),
               "failed_frames": [r["frame"] for r in rows if r["status"] != "ok"],
               "valid_empty_frames": [r["frame"] for r in rows if r["status"] == "ok" and not r["prediction_count"]],
               "camera_command_frames": [r["frame"] for r in rows if r["camera_command_ignored"]],
               "coco": metrics, "operating_point": operating, "class_agnostic_at_operating_point": agnostic,
               "map50_timeout_only_3333ms": deadline_metrics["map50"],
               "http_latency_ms": {"mean": float(np.mean(times)), "p50": float(np.percentile(times,50)),
                                   "p95": float(np.percentile(times,95)), "max": float(max(times))},
               "frames_over_333ms": sum(r["over_333ms"] for r in rows), "frames_over_3333ms": sum(r["over_3333ms"] for r in rows),
               "returned_confidence": {"mean": float(np.mean(scores)) if scores else None, "max": max(scores) if scores else None},
               "warnings": ["Resubstitution diagnostic, not proof of generalization.",
                            "AP uses all returned scores; the server may already filter/cap detections.",
                            "Failed/invalid responses contribute zero predictions; their ground truth remains.",
                            "HTTP-200 empty responses may hide detector exceptions; inspect server telemetry.",
                            "Sequential replay does not emulate upstream skipping, concurrency or competition realtime scoring.",
                            "A new sequence_id does not guarantee resetting server telemetry; use a separate diagnostic server/log directory."]}
    write_json(out/"metrics.json", summary)
    report = [f"Endpoint replay: {args.run_label}", "", summary["evaluation"], "",
              f"Frames: {len(frames)}; failures: {len(summary['failed_frames'])}; valid empty: {len(summary['valid_empty_frames'])}",
              f"mAP50: {metrics['map50']:.4f} ({100*metrics['map50']:.2f}%)",
              f"mAP50-95: {metrics['map50_95']:.4f}; mAP75: {metrics['map75']:.4f}",
              f"Precision / recall / F1 at confidence {args.confidence}: {operating['precision']:.4f} / {operating['recall']:.4f} / {operating['f1']:.4f}",
              f"Class-agnostic recall at same threshold: {agnostic['recall']:.4f}",
              f"HTTP latency mean / p95: {np.mean(times):.1f} / {np.percentile(times,95):.1f} ms", "", *summary["warnings"]]
    (out/"REPORT.txt").write_text("\n".join(report)+"\n", encoding="utf-8")
    print("\n"+"\n".join(report[:10])+f"\n\nSaved: {out}")
    return 2 if summary["failed_frames"] else 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:9053/predict", help="Full prediction endpoint URL")
    p.add_argument("--challenge-root", default=".", help="Directory containing official dtos.py")
    p.add_argument("--manifest", default="prepared/dataset_manifest.json")
    p.add_argument("--split", default="prepared/split.json")
    p.add_argument("--subset", choices=["train","val","all"], default="train")
    p.add_argument("--images-dir", help="Override prepared PNG directory when manifest paths came from another machine")
    p.add_argument("--output", default="endpoint_replay_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    p.add_argument("--run-label", default="endpoint_training_replay", help="Record scenario/checkpoint/epoch manually")
    p.add_argument("--timeout", type=float, default=30, help="Client timeout in seconds; diagnostic default allows slow responses")
    p.add_argument("--confidence", type=float, default=.05, help="Threshold for precision/recall/F1 and overlays ONLY; not AP or server inference")
    p.add_argument("--max-dets", type=int, default=100, help="COCO per-image per-class cap; official scorer default is 100")
    p.add_argument("--headers-json", help="Optional local JSON file of authentication headers; never copied into reports")
    p.add_argument("--overlays", action="store_true", help="Save green ground truth and orange prediction overlays")
    args = p.parse_args()
    if not np.isfinite(args.timeout) or args.timeout <= 0 or not np.isfinite(args.confidence) or not 0 <= args.confidence <= 1 or args.max_dets < 10:
        p.error("Require positive finite timeout, confidence in [0,1], and max-dets >= 10")
    if urlsplit(args.url).scheme not in ("http","https") or not urlsplit(args.url).hostname:
        p.error("--url must be an http(s) endpoint")
    try:
        return run(args)
    except (ValueError, OSError, ImportError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
