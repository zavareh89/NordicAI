# Replay training frames through the deployed endpoint

This is a detector correctness and memorization diagnostic for E1/E2 at Level 0. It sends the exact prepared PNG pixels through the real HTTP prediction endpoint and scores the returned annotations. No training, model loading, or server code changes are needed.

## Install

Extract this update in the official `drone-flyby` repository. It adds:

- `scripts/replay_training_endpoint.py`
- `tests/test_endpoint_replay.py`
- this guide and `TEST_REPORT_ENDPOINT_REPLAY.md`

Use the existing detector environment, which already declares NumPy, OpenCV, Requests, Pydantic 2 and faster-coco-eval. If missing, install only the missing packages; do not replace your detector environment or CUDA stack:

```bash
python -m pip install requests 'faster-coco-eval>=1.7.2,<2'
```

The script imports your official `dtos.py` to construct and validate the actual protocol. Run from the challenge repository root, or supply `--challenge-root /path/to/drone-flyby`.

## 1. Start the checkpoint you want to inspect

In one terminal, using your existing server command:

```bash
python scripts/run_server.py \
  --scenario E2 \
  --weights weights/challenge/E2_rfdetr_small_best.pth \
  --telemetry-dir runtime_logs_replay_E2
```

The telemetry flag requires the previously supplied telemetry update. Otherwise omit that flag. Use `--scenario E1` and the E1 checkpoint to inspect YOLO. Use a separate diagnostic process or run when no competition evaluation is active: replay adds requests and can change telemetry counters. A unique sequence ID does not necessarily reset your telemetry implementation.

## 2. Replay the actual training subset

In a second terminal, from the same repository:

```bash
python scripts/replay_training_endpoint.py \
  --url http://127.0.0.1:9053/predict \
  --manifest prepared/dataset_manifest.json \
  --split prepared/split.json \
  --subset train \
  --images-dir prepared/level0/images \
  --run-label E2_epoch150 \
  --output results/endpoint_E2_epoch150 \
  --overlays
```

Use your actual server port and checkpoint name. The default endpoint is the official local evaluator's `http://127.0.0.1:9053/predict`. Remote endpoints work too; only the client needs the prepared images and labels. Optional authentication: `--headers-json /path/to/headers.json`, whose content is a JSON object such as `{"X-API-Key":"..."}`. Headers are not copied to reports; restrict access to that local file.

The manifest contains ground truth in Level-0 pixels. The script converts it to source-frame pixels, and converts the endpoint's normalized global boxes to the same coordinate system. It sends prepared PNG bytes without another resize or lossy re-encoding. `--images-dir` handles manifests containing absolute paths from another machine.

Every selected frame is sent once, sequentially. No retries, no intentionally skipped frames, no camera movement. `frame` retains its original source ID; `frame_index` is its position in the complete manifest sequence. All images/annotations are checked before replay starts. Existing nonempty result directories are rejected to prevent mixing checkpoints.

## 3. Read the outputs

```text
results/endpoint_E2_epoch150/
  REPORT.txt                       Human-readable summary
  metrics.json                     Complete metrics and warnings
  per_class.csv                    AP and operating metrics for every legal class
  per_frame.csv                    HTTP status, counts, timing, errors
  confidence_sweep.csv             P/R/F1 at different client-side thresholds
  pr_curve_iou50.csv               COCO interpolated precision at 101 recall points
  predictions_source_xyxy.json     Accepted detections in original-frame pixels
  frame_events.jsonl               Flushed per-frame event log
  run_metadata.json                Frame list, labels, image/protocol hashes, run label
  responses/frame_000000.json      Response plus request metadata (no image Base64)
  overlays/frame_000000.png        Green ground truth; orange predictions
```

```bash
cat results/endpoint_E2_epoch150/REPORT.txt
cat results/endpoint_E2_epoch150/metrics.json
```

Metrics:

- `coco.map50`: class-averaged COCO AP at IoU 0.50, the official local scorer's metric.
- `coco.map75` and `coco.map50_95`: stricter localization metrics.
- `coco.mar50_95`: mean recall over IoU 0.50:0.95 at the specified COCO detection cap.
- `operating_point`: micro precision, recall, F1, TP, FP, FN and mean matched IoU at confidence 0.05 and IoU 0.50 by default. Matching is score-ordered and one-to-one within each frame/class.
- `class_agnostic_at_operating_point`: the same matching ignoring class labels. High recall here but low class-aware recall suggests a classification/mapping problem.
- Per-class AP is averaged only over classes with ground truth in the selected subset, matching the official scorer. Absent classes have blank AP in CSV, not zero; their false detections still count in operating-point precision.
- COCO uses its standard 101 recall points and a default cap of 100 detections per image per class. `--max-dets` changes that cap and therefore changes comparability with the official default. Operating-point counts use all returned detections at the chosen threshold.

All AP values are fractions in [0,1], so 0.90 means 90% AP. AP is not the percentage of objects detected.

### Confidence threshold

`--confidence 0.05` controls only precision/recall/F1, class-agnostic diagnostics, and overlays. AP uses every accepted detection returned by the server. The CSV sweep filters these already returned detections; it cannot recover predictions discarded inside the server.

To inspect lower-confidence predictions, edit the server's inference config and restart it. Keep this threshold, class-specific thresholds, NMS, detection caps, checkpoint/EMA choice, and AP definition consistent when comparing runs. The script cannot inspect the server checkpoint; record its identity in `--run-label`.

### Failures and timing

HTTP failures, timeouts, malformed responses, wrong frame/request IDs, and invalid annotations cost the entire response, as in the official contract. Their frames stay in the evaluation denominator with zero predictions. The script continues and writes results; exit code 2 means at least one such failure. Exit code 1 means a setup/input failure; 0 means all responses were valid, even if AP was poor.

HTTP-200 empty responses cannot be distinguished from swallowed detector exceptions by this client; cross-check server telemetry/logs. Non-null camera commands are recorded but ignored because this test holds the camera at Level 0.

Client timeout defaults to 30 seconds to collect slow predictions for accuracy diagnosis. Requests still advertise the official 333 ms interval and 3333 ms response budget. `map50_timeout_only_3333ms` additionally discards responses whose measured round trip exceeded 3333 ms. This is only a timeout diagnostic: it does not simulate the official scheduler, upstream skipping, request cancellation, or competition score. Requests' timeout is a socket connect/read timeout, not a strict total wall-clock deadline. Main AP retains slow valid responses; latency counts show how many exceeded each budget.

HTTP time includes request serialization/transport/server work/response read and JSON decoding; it excludes PNG loading/Base64 encoding and local metric calculation. It is not GPU inference time.

## 4. Compare checkpoints

Restart the diagnostic server with the next checkpoint, then use a distinct output directory and label. For example:

```bash
python scripts/replay_training_endpoint.py \
  --url http://127.0.0.1:9053/predict \
  --run-label E2_epoch50 \
  --output results/endpoint_E2_epoch50 \
  --overlays
```

Compare `map50`, per-class AP, valid-empty/failed frames, confidence distributions, and overlays. A training score around 0.9 reproduced through HTTP confirms that the deployed path retains training performance. A collapse through HTTP points to deployment/checkpoint/coordinate/response problems. Neither result alone proves why unseen performance is low.

## 5. Optional other subsets

```bash
python scripts/replay_training_endpoint.py \
  --subset val \
  --output results/endpoint_val
```

`val` requires nonempty `val_frame_ids` in `split.json`. With `all_frames`, it correctly fails because there is no holdout. Do not point it at RF-DETR's mirrored `valid/` and call that independent validation. The held-out images must never have been used to train the checkpoint being evaluated.

Use `--subset all` to score every manifest frame without a split file. This does not imply independent validation. For an independent dataset, use its prepared manifest and explicitly document how it was excluded from training.

## Tests

```bash
python -m pytest tests/test_endpoint_replay.py -q
```

Tests exercise metric edge cases, official response validation, direct agreement with `local_evaluator.score`, and complete HTTP replay against a local synthetic endpoint. They do not require weights or a GPU. See `TEST_REPORT_ENDPOINT_REPLAY.md` for the tests executed during delivery.
