# Nordic AI Cup 2026 — Drone Flyby Level-0 detector baselines

This implementation adds two **detector-only**, **stateless**, Level-0 baselines around the official Nordic AI Cup `drone-flyby` code:

- **E1 — YOLO26-S**: official Ultralytics `yolo26s.pt`, exact Level-0 data, mild aerial/small-object augmentation.
- **E2 — RF-DETR Small**: official Roboflow `rf-detr-small.pth`, the same exact Level-0 data and split, conservative aerial/small-object augmentation.

Nothing here tracks objects, remembers prior frames, requests zoom, uses Level 1/2, tiles an image, performs temporal fusion, or controls the camera. Each frame is detected independently. The detector interface and coordinate adapter are separated so those components can be added later without replacing the detector pipeline.

> **Integration model:** copy this project into the root of the official `drone-flyby/` directory. The official `api.py`, `dtos.py`, `utils.py`, `local_evaluator.py`, source data, and evaluator remain unchanged. The supplied `example.py` is the thin competition-facing adapter imported by the official API.

## 1. Verified challenge format

The implementation is based on the current official challenge code rather than generic YOLO/COCO assumptions.

- Scene layout: `src/<scene>/images/frame_XXXXXX.png` and `src/<scene>/annotations/frame_XXXXXX.json`.
- Helsinki source images are 3840×2160.
- Each annotation JSON contains an `annotations` list. Each object has `object_id` and `bbox`.
- Source annotation boxes are **absolute source-image pixel coordinates in `xyxy` order**: `[x1, y1, x2, y2]`.
- The legal 16 class names and their ordering are imported at runtime from official `dtos.OBJECT_CLASSES`; they are intentionally not duplicated in this project.
- Level 0 is the entire source image resized to **960×540** with OpenCV `INTER_AREA`.
- Participant predictions use a legal `object_id`, confidence, and a **full-frame normalized `xyxy`** box.
- Returning `requested_view=None` leaves the camera unchanged. Because these baselines never request another view, they stay at Level 0 for the entire run.
- The official local score evaluates COCO-style AP at **IoU = 0.50**. This project therefore reports `mAP@0.50` explicitly rather than treating `mAP50-95` as the primary baseline metric.

## 2. Project layout and responsibility boundaries

The common path is:

```text
official challenge source/annotations
        ↓
canonical FrameSample + DetectionAnnotation
        ↓
exact 3840×2160 → 960×540 cv2.INTER_AREA Level-0 materialization
        ↓
shared temporally blocked split
        ↓
YOLO exporter ─────────────┐
COCO/RF-DETR exporter ─────┤ model-specific training backend
                           ↓
                    BaseDetector
                           ↓
                 canonical Detection
                           ↓
              generic filtering/postprocess
                           ↓
official view_bbox_to_global / DTO adapter
                           ↓
             requested_view = None
```

`BaseDetector.predict()` returns boxes in pixels relative to the received 960×540 view. It knows nothing about FastAPI, camera movement, tracking, prior frames, or global scene memory. The challenge adapter performs the view/global-coordinate conversion separately.

## 3. Environment

A Linux + NVIDIA CUDA 12.1 environment is recommended. Python 3.10 or 3.11 is the safest common denominator for the pinned detector stacks.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` deliberately contains:

```text
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.2.1+cu121
torchvision==0.17.1+cu121
```

The detector-framework pins used by this implementation are `ultralytics==8.4.155` and `rfdetr[train,augment]==1.10.1`. Ultralytics is AGPL-3.0; the RF-DETR Small open-source package/model family is Apache-2.0. Review the competition and model licenses for your deployment context.

## 4. Exact Level-0 preprocessing

No final baseline training is performed on 4K images. The canonical dataset stage reads each official source image and performs exactly:

```python
cv2.resize(source_bgr, (960, 540), interpolation=cv2.INTER_AREA)
```

Boxes are transformed generically with:

```text
scale_x = 960 / source_width
scale_y = 540 / source_height
x1,x2 *= scale_x
y1,y2 *= scale_y
```

For a 3840×2160 source this happens to equal `/4` on both axes, but `/4` is not hard-coded. Prepared PNGs are lossless, so PNG compression settings do not alter pixel values.

Prepare both model datasets in one pass:

```bash
python scripts/prepare_dataset.py
```

Optional blocked cross-validation definitions can also be emitted without making CV the default training path:

```bash
python scripts/prepare_dataset.py --blocked-cv
```

Generated data live under `prepared/` and include:

```text
prepared/
├── dataset_manifest.json
├── split.json
├── blocked_cv.json                 # only with --blocked-cv
├── level0/images/
├── yolo/
│   ├── data.yaml
│   ├── images/{train,val}/
│   └── labels/{train,val}/
└── rfdetr/
    ├── train/_annotations.coco.json
    └── valid/_annotations.coco.json
```

The parser validates unknown classes, malformed/empty boxes, out-of-bounds boxes, missing images/annotations, duplicate frame IDs, and class-ID/name mismatches. YOLO normalization and COCO conversion are validated again by their adapters.

## 5. Temporal split and leakage prevention

The default split is deliberately **not random**. Configuration is shared by E1/E2:

```yaml
data:
  split:
    strategy: blocked_holdout
    val_count: 5
    purge_gap: 1
    val_position: end
```

For the current 25-frame Helsinki sequence numbered 0–24 this produces:

```text
train:      0 … 18
purged:     19
validation: 20 … 24
```

The purge frame is intentionally unused so the nearest training image is not the immediately preceding video-like frame. `prepared/split.json` is copied into every experiment directory so both models can be audited against the exact same frame IDs.

## 6. Official pretrained weights and cache

Large checkpoint files are ignored by Git.

```text
weights/
├── pretrained/      # generic official initialization
└── challenge/       # challenge-fine-tuned deployment checkpoints
```

### E1 — YOLO26-S

Default baseline initializer:

```bash
python scripts/download_weights.py --scenario E1
```

This asks the installed official Ultralytics library to resolve/download `yolo26s.pt`, copies it into `weights/pretrained/`, and verifies that Ultralytics can load it. The standard YOLO26 checkpoint is the E1 default.

An optional official Objects365-stage YOLO26-S initializer is also exposed, but is **not** silently used by E1:

```bash
python scripts/download_weights.py --scenario E1 --initializer objects365
```

That resolves `yolo26s-objv1-150.pt`. If you choose to use it, also change `model.pretrained_weights` in a copied experiment config so the run provenance is unambiguous.

Manual/cache-equivalent method if the helper cannot access the network:

```bash
python -c 'from ultralytics import YOLO; YOLO("yolo26s.pt")'
```

Then place the official resolved checkpoint at `weights/pretrained/yolo26s.pt` and rerun the download command; it will verify and reuse the local file instead of downloading again.

### E2 — RF-DETR Small

```bash
python scripts/download_weights.py --scenario E2
```

This uses the official `RFDETRSmall(pretrain_weights="rf-detr-small.pth")` loading mechanism. `RF_HOME` is temporarily pointed at this project's `weights/pretrained/` cache, and the resulting checkpoint is verified by loading it.

Manual/cache-equivalent method:

```bash
export RF_HOME="$PWD/weights/pretrained"
python -c 'from rfdetr import RFDETRSmall; RFDETRSmall(pretrain_weights="rf-detr-small.pth", resolution=960)'
```

The baseline never trains either detector from scratch.

## 7. E1 — YOLO26-S configuration

The baseline is in `configs/e1_yolo26s.yaml`. Important defaults are explicit:

- `yolo26s.pt` initialization.
- `imgsz=960` for train/inference.
- aspect-aware/rectangular inference enabled; Ultralytics handles its native letterboxing rather than geometrically warping the 16:9 Level-0 image.
- `fliplr=0.5`, `flipud=0.5`.
- mild HSV and brightness/contrast changes.
- ±3° rotation, 5% translation.
- scale range approximately 0.8–1.2 (stored explicitly as `[0.8, 1.2]`; the adapter converts this to Ultralytics `scale=0.2`, whose API uses an amplitude around 1.0).
- weak blur/noise with 3% probability each.
- mosaic `0.25`, configurable down to `0`; `close_mosaic=10` disables it for the final epochs.
- MixUp and CutMix disabled.
- confidence threshold `0.08`, max detections `200`, YOLO NMS IoU `0.60`.

Ultralytics already performs NMS in normal YOLO inference, so the shared postprocessor does **not** apply a second NMS pass.

Train:

```bash
python scripts/train_e1.py
```

or equivalently with an explicit config:

```bash
python scripts/train_e1.py --config configs/e1_yolo26s.yaml
```

## 8. E2 — RF-DETR Small configuration

The baseline is in `configs/e2_rfdetr_small.yaml`.

RF-DETR Small uses `patch_size=16` and `num_windows=2`, so its valid resolution block is 32 pixels. **960 is divisible by 32**, making it the closest exact requested long-dimension baseline without dropping to 640.

The source dataset remains the exact 960×540 Level-0 image. RF-DETR's model-specific preprocessing then resizes to its configured square 960 input, as expected by the official Small configuration. This RF-DETR-specific transform is downstream of the shared Level-0 materialization and is applied consistently in its training/inference backend.

Conservative defaults include:

- official `rf-detr-small.pth` initialization/backbone.
- `resolution=960`.
- `lr=5e-5`, encoder LR `2.5e-5`.
- effective batch 8 via batch 2 × grad accumulation 4.
- EMA enabled.
- early stopping enabled.
- gradient clipping `0.1`.
- custom official Albumentations integration with H/V flips, mild brightness/contrast and saturation/hue jitter, ±3° low-probability rotation, 0.9–1.1 low-probability affine scaling/translation, and weak blur/noise.
- `scale_jitter=false`, deliberately disabling RF-DETR's independent resize→crop→resize branch so tiny/border objects are not clipped.
- `multi_scale=false` for a stable small-data baseline.
- confidence threshold `0.08`, max detections `200`.
- no added classical NMS; RF-DETR remains NMS-free unless a later experiment explicitly enables shared extra NMS.

The official `AUG_AERIAL` preset was inspected but is not used blindly: it includes stronger 90° rotation behavior. The custom baseline retains aerial-friendly vertical/horizontal flips but keeps rotation and affine transforms mild because targets are tiny and the dataset has only 25 highly correlated frames.

Train:

```bash
python scripts/train_e2.py
```

## 9. Experiment outputs and reproducibility

Training creates independent experiment roots:

```text
experiments/
├── E1_yolo26s/
│   ├── config.yaml
│   ├── dataset_manifest.json
│   ├── split.json
│   ├── checkpoints/
│   ├── metrics.json
│   ├── predictions/
│   ├── logs/
│   └── final_model/
└── E2_rfdetr_small/
    └── ... same logical structure ...
```

Each run snapshots the full scenario config and shared dataset/split manifests. Python, NumPy and PyTorch seeds are initialized from the same default seed (`42`). Deterministic CUDA algorithms are configurable rather than forced because strict deterministic kernels can materially reduce throughput; the config records that choice.

After training, all available periodic plus framework-best/last checkpoints are evaluated on the shared blocked validation set and the deployment checkpoint is selected by `mAP@0.50` (recall, then precision are deterministic tie-breakers). The selected checkpoint is then evaluated once more to write the run's final `metrics.json` and `predictions/blocked_validation_predictions.json`, including per-class AP and latency statistics. This prevents an upstream framework's native mAP50-95 fitness from silently defining the competition checkpoint. The selected checkpoint is copied both into the experiment's `final_model/` and into the configured `weights/challenge/` deployment path. Generic pretrained and challenge-fine-tuned checkpoints are therefore never confused.

## 10. Offline validation

Evaluate the same blocked Level-0 validation frames:

```bash
python scripts/evaluate.py --scenario E1
python scripts/evaluate.py --scenario E2
```

or evaluate a specific checkpoint:

```bash
python scripts/evaluate.py --scenario E1 --weights /path/to/best.pt
python scripts/evaluate.py --scenario E2 --weights /path/to/best.pth
```

The shared evaluator reports:

- `mAP@0.50` using COCO evaluation with the IoU threshold fixed to 0.50;
- per-class AP@0.50 for classes present in the held-out block;
- precision and recall at IoU 0.50;
- prediction count;
- mean, p50, and p95 total detector latency;
- mean shared preprocessing, detector-backend call, and shared postprocessing latency.

GPU synchronization surrounds the detector call so asynchronous CUDA work is not falsely reported as finished. Warm-up runs occur before latency collection. Framework-internal resize/normalization lives inside the backend call; therefore `mean_inference` is deliberately an end-to-end detector-backend duration, not a claim that hidden framework preprocessing has been separately instrumented.

## 11. Competition API integration

The official `api.py` imports `predict` from `example.py`. This project supplies a minimal `example.py` that forwards to `drone_detector.challenge.solution.predict`.

The server helper loads and warms the configured model **before** Uvicorn starts accepting frames, so first-request model construction does not consume the challenge frame budget.

Run E1:

```bash
python scripts/run_server.py --scenario E1
```

Run E2:

```bash
python scripts/run_server.py --scenario E2
```

Override the challenge-fine-tuned checkpoint without editing code:

```bash
python scripts/run_server.py --scenario E1 --weights /path/to/checkpoint.pt
```

The server path performs:

```text
official request
  → official decode_view()
  → assert Level 0 + 960×540
  → BaseDetector.predict()
  → canonical Detection[]
  → configurable confidence/top-K filtering
  → official view_bbox_to_global()
  → official prediction DTOs
  → requested_view=None
```

`requested_view=None` is hard-wired in the challenge adapter for these two baselines. A request arriving at any nonzero resolution level raises immediately, which makes accidental camera movement visible instead of silently evaluating the wrong distribution.

The API path logs decode, shared preprocessing, detector-backend, shared postprocessing, DTO conversion and whole-request timing for the **current frame only**. These diagnostics are not retained as temporal state. The official local evaluator remains the authoritative end-to-end HTTP timing measurement.

## 12. Official local evaluator

Start one server in terminal 1:

```bash
python scripts/run_server.py --scenario E1
```

Then in terminal 2 invoke the unchanged official evaluator through the thin helper:

```bash
python scripts/evaluate.py --scenario E1 --official-local --challenge-root . --scene helsinki
```

For realtime/3-FPS simulation:

```bash
python scripts/evaluate.py --scenario E1 --official-local --challenge-root . --scene helsinki --realtime --verbose
```

Repeat with `--scenario E2` after starting the E2 server. The evaluator is not reimplemented or patched; the helper simply invokes official `local_evaluator.py`.

The source stream is about 3 FPS (~333 ms between source frames). A model can satisfy a larger HTTP timeout yet still skip source frames if inference is substantially slower than ~333 ms, so latency is a first-class reported result. TensorRT, quantization and other deployment optimizations are intentionally outside E1/E2.

## 13. Tests

The tests are intentionally lightweight: no checkpoint download, long training, GPU run, or official evaluator job is performed by the unit suite.

```bash
pytest -q
```

Coverage includes:

- exact challenge annotation parsing and absolute `xyxy` interpretation;
- generic 3840×2160 → 960×540 box scaling;
- exact `cv2.INTER_AREA` Level-0 image transformation;
- temporally blocked split with purge gap;
- YOLO normalization bounds;
- COCO/RF-DETR IDs/categories/boxes;
- Level-0 and generic view→global coordinate conversion;
- common output datatype from both detector adapters using model stubs (no weights/network);
- challenge response shape and `requested_view=None` using DTO-compatible test stubs;
- config invariants that prevent E1/E2 from leaving Level 0.

When the tests are run from inside the full official repository, an additional integration test uses the actual `dtos.py` when available.

## 14. Typical full workflow

```bash
# 1) Place these files in official drone-flyby/ and install dependencies
pip install -r requirements.txt

# 2) Build one canonical exact-L0 dataset + shared blocked split
python scripts/prepare_dataset.py

# 3) E1
python scripts/download_weights.py --scenario E1
python scripts/train_e1.py
python scripts/evaluate.py --scenario E1
python scripts/run_server.py --scenario E1
# In another shell:
python scripts/evaluate.py --scenario E1 --official-local --challenge-root . --scene helsinki --realtime

# 4) E2
python scripts/download_weights.py --scenario E2
python scripts/train_e2.py
python scripts/evaluate.py --scenario E2
python scripts/run_server.py --scenario E2
# In another shell:
python scripts/evaluate.py --scenario E2 --official-local --challenge-root . --scene helsinki --realtime
```

## 15. Switching E1 ↔ E2

No application code changes are needed. Training selects one config, and challenge serving uses `--scenario E1` vs `--scenario E2`. The shared parser, class mapping, Level-0 converter, split, canonical types, predictor, filtering, coordinate conversion, API adapter, metrics and experiment layout do not change.

## 16. Future extension points

A later experiment can wrap the same `BaseDetector` output with modules such as:

```text
current camera image
       ↓
BaseDetector
       ↓
Detection[]
       ↓
tracker / persistent object memory / temporal fusion
       ↓
global scene state
       ↓
camera / zoom policy
       ↓
requested next view
```

Likewise, a future tiled predictor can wrap detector inference before the canonical postprocessor without changing annotation parsing, model training, DTO conversion or camera policy. None of those future capabilities are active in E1/E2.

## 17. Scope exclusions

These baselines intentionally do **not** include tracking, prior-frame boxes, confidence smoothing, motion prediction, object identity caching, frame-number heuristics, Level-1/2 requests, SAHI/tiled inference, super-resolution, external DOTA/xView data, ensembling, hyperparameter optimization, TensorRT, quantization, or temporal fusion.

They are intended to answer one clean question first: **how strong are modern pretrained detectors after fine-tuning on the exact Level-0 distribution under a leakage-resistant validation split?**
