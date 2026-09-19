from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import threading
from typing import Deque, Dict, Iterable, Optional


@dataclass(frozen=True)
class TimingSample:
    wall_ms: float
    cpu_ms: float


class TimingStats:
    """Thread-safe cumulative + rolling timing statistics.

    Cumulative totals are never discarded, which is important for comparing the
    process's total work with the competition's accumulated waiting-time budget.
    A bounded rolling window is retained only for percentile estimation.
    """

    def __init__(self, *, max_samples: int = 20_000) -> None:
        self._lock = threading.Lock()
        self._wall: Deque[float] = deque(maxlen=max_samples)
        self._cpu: Deque[float] = deque(maxlen=max_samples)
        self._count = 0
        self._errors = 0
        self._wall_total_ms = 0.0
        self._cpu_total_ms = 0.0
        self._wall_max_ms = 0.0
        self._cpu_max_ms = 0.0

    def record(self, wall_ms: float, cpu_ms: float, *, error: bool = False) -> None:
        wall_ms = max(0.0, float(wall_ms))
        cpu_ms = max(0.0, float(cpu_ms))
        with self._lock:
            self._count += 1
            self._errors += int(bool(error))
            self._wall_total_ms += wall_ms
            self._cpu_total_ms += cpu_ms
            self._wall_max_ms = max(self._wall_max_ms, wall_ms)
            self._cpu_max_ms = max(self._cpu_max_ms, cpu_ms)
            self._wall.append(wall_ms)
            self._cpu.append(cpu_ms)

    def reset(self) -> None:
        with self._lock:
            self._wall.clear()
            self._cpu.clear()
            self._count = 0
            self._errors = 0
            self._wall_total_ms = 0.0
            self._cpu_total_ms = 0.0
            self._wall_max_ms = 0.0
            self._cpu_max_ms = 0.0

    @staticmethod
    def _percentile(values: list[float], q: float) -> Optional[float]:
        if not values:
            return None
        values.sort()
        index = min(len(values) - 1, max(0, int(math.ceil(q * len(values))) - 1))
        return values[index]

    @classmethod
    def _distribution(cls, values: Iterable[float]) -> Dict[str, Optional[float] | int]:
        vals = [float(v) for v in values]
        if not vals:
            return {
                "samples": 0,
                "mean_ms": None,
                "p50_ms": None,
                "p90_ms": None,
                "p95_ms": None,
                "p99_ms": None,
                "max_window_ms": None,
            }
        return {
            "samples": len(vals),
            "mean_ms": sum(vals) / len(vals),
            "p50_ms": cls._percentile(vals.copy(), 0.50),
            "p90_ms": cls._percentile(vals.copy(), 0.90),
            "p95_ms": cls._percentile(vals.copy(), 0.95),
            "p99_ms": cls._percentile(vals.copy(), 0.99),
            "max_window_ms": max(vals),
        }

    def snapshot(self) -> dict:
        with self._lock:
            wall = list(self._wall)
            cpu = list(self._cpu)
            count = self._count
            errors = self._errors
            wall_total_ms = self._wall_total_ms
            cpu_total_ms = self._cpu_total_ms
            wall_max_ms = self._wall_max_ms
            cpu_max_ms = self._cpu_max_ms
        return {
            "count": count,
            "errors": errors,
            "wall_total_ms": wall_total_ms,
            "wall_total_s": wall_total_ms / 1000.0,
            "cpu_total_ms": cpu_total_ms,
            "cpu_total_s": cpu_total_ms / 1000.0,
            "wall_mean_all_ms": wall_total_ms / count if count else None,
            "cpu_mean_all_ms": cpu_total_ms / count if count else None,
            "wall_max_all_ms": wall_max_ms if count else None,
            "cpu_max_all_ms": cpu_max_ms if count else None,
            "rolling_wall": self._distribution(wall),
            "rolling_cpu": self._distribution(cpu),
        }


class DiagnosticRegistry:
    def __init__(self, *, max_samples: int = 20_000) -> None:
        self.app_predict = TimingStats(max_samples=max_samples)
        self.controller = TimingStats(max_samples=max_samples)
        self._meta_lock = threading.Lock()
        self._reset_at = datetime.now(timezone.utc).isoformat()
        self._last_status_code: Optional[int] = None
        self._last_error: Optional[str] = None

    def reset(self) -> None:
        self.app_predict.reset()
        self.controller.reset()
        with self._meta_lock:
            self._reset_at = datetime.now(timezone.utc).isoformat()
            self._last_status_code = None
            self._last_error = None

    def set_last_status(self, status_code: int) -> None:
        with self._meta_lock:
            self._last_status_code = int(status_code)

    def set_last_error(self, error: str) -> None:
        with self._meta_lock:
            self._last_error = str(error)

    def snapshot(self) -> dict:
        app = self.app_predict.snapshot()
        controller = self.controller.snapshot()
        with self._meta_lock:
            reset_at = self._reset_at
            last_status = self._last_status_code
            last_error = self._last_error

        n = app["count"]
        app_total = float(app["wall_total_ms"])
        controller_total = float(controller["wall_total_ms"])
        non_controller_ms = max(0.0, app_total - controller_total)
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "reset_at_utc": reset_at,
            "predict_requests": n,
            "last_status_code": last_status,
            "last_error": last_error,
            "app_predict": app,
            "controller": controller,
            "derived": {
                "app_minus_controller_total_ms": non_controller_ms,
                "app_minus_controller_mean_ms": non_controller_ms / n if n else None,
                "controller_fraction_of_app_wall": (
                    controller_total / app_total if app_total > 0 else None
                ),
            },
        }

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.snapshot(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)
