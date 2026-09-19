# Endpoint replay test report

Result: **14 passed** (1.93 seconds).

Command used:

```text
PYTHONPATH=sources .venv/bin/python -m pytest deliverables/endpoint_replay_update/tests/test_endpoint_replay.py -q
```

The `sources` directory contained the official repository's `dtos.py`, `utils.py`, and `local_evaluator.py`, retrieved from `amboltio/Nordic-AI-Cup-2026/main/drone-flyby` for verification. Those official files are not replaced or bundled in this update. The delivered script uses the user's own official DTOs.

Covered:

- Perfect detections produce AP50 and AP50–95 of 1.
- Empty predictions produce AP 0 and retain all false negatives.
- A missing frame remains in the denominator (one of two identical GT instances recalled gives 51/101 interpolated AP).
- A high-confidence false positive lowers AP; duplicate detections count as false positives.
- Wrong classes fail class-aware evaluation while retaining class-agnostic localization recall.
- The operating confidence threshold does not filter AP inputs.
- Wrong frame/request IDs, source-pixel response boxes, NaN scores, unknown classes, and extra response keys are rejected using official DTO validation.
- AP50 matches `local_evaluator.score` to 1e-12 on a synthetic case containing a high-confidence false positive and a missing-frame prediction.
- Two complete CLI runs against a real local HTTP server: perfect responses, and one HTTP-500 response.
- HTTP integration checks exact transmitted PNG bytes, official request validation, sparse frame IDs, relocation of prepared-image paths, output metrics/CSVs, overlay creation, and failure exit status.

Environment: Python 3.12, faster-coco-eval 1.8.0, OpenCV 5.0.0, NumPy 2.3.5, Pydantic 2.13.5, Requests 2.34.2. No training or GPU inference was performed. The user's pinned Python 3.11 / NumPy 1.26 / OpenCV 4.10 detector environment was not executed here; the script uses APIs available in those versions.

Not tested: the user's actual endpoint, trained checkpoints, real training images, or hidden challenge validation. No real accuracy result is claimed by this delivery.
