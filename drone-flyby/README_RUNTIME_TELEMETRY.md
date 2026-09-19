# Runtime frame telemetry update

Minimal runtime monitoring for deployed E1/E2 servers. Monitoring state is kept completely separate from detector state and never changes predictions.

Updated/new files:

```text
drone_detector/runtime_telemetry.py       NEW
drone_detector/challenge/solution.py      UPDATED
scripts/run_server.py                     UPDATED
tests/test_runtime_telemetry.py            NEW
```

Run Helsinki with known 25-frame length:

```bash
python scripts/run_server.py \
  --scenario E2 \
  --expected-frames 25 \
  --telemetry-dir runtime_logs
```

Generated files:

```text
runtime_logs/frame_events.jsonl
runtime_logs/frame_summary.json
runtime_logs/http_timing.jsonl
```

`frame_summary.json` reports received, successful, failed and currently skipped frame IDs; frames slower than the configured 333 ms budget; and mean/p50/p95/max processing, inference, inter-arrival and HTTP timing.

A frame is considered skipped only when its index is absent below the highest frame index that has already reached the server. If an out-of-order request later arrives, it is automatically removed from the skipped set.

`missing_expected_if_run_finished_now` is only final once the evaluator run has actually finished. `remaining_after_latest_received` represents expected future indices that simply have not been reached yet.
