from __future__ import annotations

import json
from pathlib import Path

from drone_detector.runtime_telemetry import RuntimeTelemetry


def test_tracks_processed_failed_and_skipped_frames(tmp_path: Path) -> None:
    telemetry = RuntimeTelemetry(
        log_dir=tmp_path,
        expected_frames=6,
        first_frame=0,
        slow_frame_ms=333.0,
    )

    a0 = telemetry.begin_frame(frame=0, request_id="r0", arrival_monotonic=10.0)
    assert a0["newly_skipped_frames"] == []
    telemetry.finish_frame(
        frame=0,
        request_id="r0",
        status="ok",
        timing_ms={"inference_ms": 100.0, "predict_total_ms": 120.0},
        annotation_count=1,
    )

    a2 = telemetry.begin_frame(frame=2, request_id="r2", arrival_monotonic=10.4)
    assert a2["newly_skipped_frames"] == [1]
    telemetry.finish_frame(
        frame=2,
        request_id="r2",
        status="inference_error_empty_response",
        timing_ms={"inference_ms": 350.0, "predict_total_ms": 370.0},
        annotation_count=0,
        error="boom",
    )

    # Late/out-of-order arrival removes frame 1 from the current skipped set.
    telemetry.begin_frame(frame=1, request_id="r1", arrival_monotonic=10.5)
    telemetry.finish_frame(
        frame=1,
        request_id="r1",
        status="ok",
        timing_ms={"inference_ms": 90.0, "predict_total_ms": 110.0},
        annotation_count=2,
    )

    summary = json.loads((tmp_path / "frame_summary.json").read_text())
    assert summary["received_frames"] == [0, 1, 2]
    assert summary["successful_frames"] == [0, 1]
    assert summary["failed_frames"] == [2]
    assert summary["skipped_before_latest_received"] == []
    assert summary["remaining_after_latest_received"] == [3, 4, 5]
    assert summary["frames_over_budget_ms"] == [2]
    assert summary["timing_ms"]["predict_total"]["p50"] == 120.0
