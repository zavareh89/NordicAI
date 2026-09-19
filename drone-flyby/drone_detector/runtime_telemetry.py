from __future__ import annotations

import json
import os
import statistics
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


class RuntimeTelemetry:
    """Thread-safe runtime diagnostics that never influence detector output."""

    def __init__(
        self,
        *,
        log_dir: str | Path,
        expected_frames: int | None = None,
        first_frame: int = 0,
        slow_frame_ms: float = 333.0,
        enabled: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.log_dir = Path(log_dir).resolve()
        self.expected_frames = expected_frames
        self.first_frame = int(first_frame)
        self.slow_frame_ms = float(slow_frame_ms)
        self._lock = threading.Lock()

        self._received_frames: set[int] = set()
        self._successful_frames: set[int] = set()
        self._failed_frames: set[int] = set()
        self._completed_frames: set[int] = set()
        self._duplicate_request_count = 0
        self._last_arrival_monotonic: float | None = None
        self._last_seen_frame: int | None = None
        self._frame_timings: dict[int, dict[str, float]] = {}
        self._interarrival_ms: dict[int, float] = {}
        self._http_total_ms: list[float] = []
        self._http_error_count = 0

        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._write_summary_unlocked()

    @property
    def events_path(self) -> Path:
        return self.log_dir / "frame_events.jsonl"

    @property
    def summary_path(self) -> Path:
        return self.log_dir / "frame_summary.json"

    @property
    def http_path(self) -> Path:
        return self.log_dir / "http_timing.jsonl"

    def begin_frame(
        self,
        *,
        frame: int,
        request_id: str,
        arrival_monotonic: float,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {
                "duplicate": False,
                "newly_skipped_frames": [],
                "interarrival_ms": None,
            }

        frame = int(frame)
        with self._lock:
            duplicate = frame in self._received_frames
            if duplicate:
                self._duplicate_request_count += 1

            previous_latest = (
                max(self._received_frames)
                if self._received_frames
                else self.first_frame - 1
            )
            previous_arrival = self._last_arrival_monotonic
            interarrival_ms = (
                (arrival_monotonic - previous_arrival) * 1000.0
                if previous_arrival is not None
                else None
            )

            self._received_frames.add(frame)
            self._last_arrival_monotonic = float(arrival_monotonic)
            self._last_seen_frame = frame
            if interarrival_ms is not None:
                self._interarrival_ms[frame] = interarrival_ms

            newly_skipped: list[int] = []
            if frame > previous_latest + 1:
                newly_skipped = [
                    idx
                    for idx in range(previous_latest + 1, frame)
                    if idx not in self._received_frames
                ]

            self._append_jsonl_unlocked(
                self.events_path,
                {
                    "event": "received",
                    "timestamp_utc": _utc_now(),
                    "request_id": str(request_id),
                    "frame": frame,
                    "duplicate": duplicate,
                    "interarrival_ms": interarrival_ms,
                    "newly_skipped_frames": newly_skipped,
                },
            )
            self._write_summary_unlocked()

            return {
                "duplicate": duplicate,
                "newly_skipped_frames": newly_skipped,
                "interarrival_ms": interarrival_ms,
            }

    def finish_frame(
        self,
        *,
        frame: int,
        request_id: str,
        status: str,
        timing_ms: dict[str, float | int | None],
        annotation_count: int,
        error: str | None = None,
    ) -> None:
        if not self.enabled:
            return

        frame = int(frame)
        normalized_timing = {
            key: float(value)
            for key, value in timing_ms.items()
            if value is not None
        }

        with self._lock:
            self._completed_frames.add(frame)
            if status == "ok":
                self._successful_frames.add(frame)
            else:
                self._failed_frames.add(frame)
            self._frame_timings[frame] = normalized_timing

            latest = max(self._received_frames) if self._received_frames else None
            skipped = self._skipped_before_latest_unlocked()
            self._append_jsonl_unlocked(
                self.events_path,
                {
                    "event": "completed",
                    "timestamp_utc": _utc_now(),
                    "request_id": str(request_id),
                    "frame": frame,
                    "status": status,
                    "annotation_count": int(annotation_count),
                    "error": error,
                    "timing_ms": normalized_timing,
                    "latest_received_frame": latest,
                    "skipped_before_latest_received": skipped,
                },
            )
            self._write_summary_unlocked()

    def record_http_request(
        self,
        *,
        method: str,
        path: str,
        status_code: int,
        total_ms: float,
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._http_total_ms.append(float(total_ms))
            if int(status_code) >= 400:
                self._http_error_count += 1
            self._append_jsonl_unlocked(
                self.http_path,
                {
                    "timestamp_utc": _utc_now(),
                    "method": method,
                    "path": path,
                    "status_code": int(status_code),
                    "total_ms": float(total_ms),
                },
            )
            self._write_summary_unlocked()

    def _skipped_before_latest_unlocked(self) -> list[int]:
        if not self._received_frames:
            return []
        latest = max(self._received_frames)
        return [
            idx
            for idx in range(self.first_frame, latest + 1)
            if idx not in self._received_frames
        ]

    def _summary_unlocked(self) -> dict[str, Any]:
        received = sorted(self._received_frames)
        successful = sorted(self._successful_frames)
        failed = sorted(self._failed_frames)
        completed = sorted(self._completed_frames)
        pending = sorted(self._received_frames - self._completed_frames)
        skipped = self._skipped_before_latest_unlocked()
        latest = max(received) if received else None

        remaining_after_latest: list[int] = []
        missing_if_finished_now: list[int] = []
        if self.expected_frames is not None and self.expected_frames >= 0:
            end_exclusive = self.first_frame + self.expected_frames
            expected = set(range(self.first_frame, end_exclusive))
            missing_if_finished_now = sorted(expected - self._received_frames)
            if latest is None:
                remaining_after_latest = sorted(expected)
            else:
                remaining_after_latest = sorted(
                    idx for idx in expected if idx > latest
                )

        predict_totals = [
            timing["predict_total_ms"]
            for timing in self._frame_timings.values()
            if "predict_total_ms" in timing
        ]
        inference_totals = [
            timing["inference_ms"]
            for timing in self._frame_timings.values()
            if "inference_ms" in timing
        ]
        interarrivals = list(self._interarrival_ms.values())
        frames_over_budget = sorted(
            frame
            for frame, timing in self._frame_timings.items()
            if timing.get("predict_total_ms", 0.0) > self.slow_frame_ms
        )

        def stats(values: list[float]) -> dict[str, float | None]:
            return {
                "mean": statistics.mean(values) if values else None,
                "p50": _percentile(values, 50.0),
                "p95": _percentile(values, 95.0),
                "max": max(values) if values else None,
            }

        return {
            "updated_at_utc": _utc_now(),
            "expected_frames": self.expected_frames,
            "first_expected_frame": self.first_frame,
            "slow_frame_budget_ms": self.slow_frame_ms,
            "received_count": len(received),
            "successful_count": len(successful),
            "failed_count": len(failed),
            "completed_count": len(completed),
            "duplicate_request_count": self._duplicate_request_count,
            "received_frames": received,
            "successful_frames": successful,
            "failed_frames": failed,
            "pending_frames": pending,
            "latest_received_frame": latest,
            "skipped_before_latest_received": skipped,
            "remaining_after_latest_received": remaining_after_latest,
            "missing_expected_if_run_finished_now": missing_if_finished_now,
            "frames_over_budget_ms": frames_over_budget,
            "timing_ms": {
                "predict_total": stats(predict_totals),
                "inference": stats(inference_totals),
                "interarrival": stats(interarrivals),
                "http_total": stats(self._http_total_ms),
            },
            "http_request_count": len(self._http_total_ms),
            "http_error_count": self._http_error_count,
        }

    def _write_summary_unlocked(self) -> None:
        if not self.enabled:
            return
        payload = self._summary_unlocked()
        temp = self.summary_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp.replace(self.summary_path)

    @staticmethod
    def _append_jsonl_unlocked(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")


_SINGLETON: RuntimeTelemetry | None = None
_SINGLETON_LOCK = threading.Lock()


def get_runtime_telemetry() -> RuntimeTelemetry:
    global _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is not None:
            return _SINGLETON

        enabled = os.environ.get("DRONE_TELEMETRY_ENABLED", "1").lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        expected_raw = os.environ.get("DRONE_EXPECTED_FRAMES")
        expected_frames = int(expected_raw) if expected_raw else None
        first_frame = int(os.environ.get("DRONE_FIRST_FRAME", "0"))
        slow_frame_ms = float(os.environ.get("DRONE_SLOW_FRAME_MS", "333.0"))
        log_dir = os.environ.get("DRONE_TELEMETRY_DIR", "runtime_logs")

        _SINGLETON = RuntimeTelemetry(
            log_dir=log_dir,
            expected_frames=expected_frames,
            first_frame=first_frame,
            slow_frame_ms=slow_frame_ms,
            enabled=enabled,
        )
        return _SINGLETON
