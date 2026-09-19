from __future__ import annotations

"""Diagnostic wrapper around the existing production ``agent_server.py``.

Nothing in the controller policy is changed. Run this module with Uvicorn instead
of ``agent_server:app`` while diagnosing latency:

    uvicorn diagnostic_agent_server:app --host 0.0.0.0 --port 9052 \
        --workers 1 --no-access-log --log-level warning

It measures:
  * complete in-process /predict handling time;
  * SurvivalController.decide_step() wall and CPU time;
  * cumulative totals across the validation run;
  * rolling p50/p90/p95/p99/max;
  * periodic JSON snapshots for post-mortem analysis.
"""

import os
from pathlib import Path
from time import perf_counter_ns, process_time_ns
from typing import Any, Callable

from fastapi import Request
from fastapi.responses import JSONResponse

from diagnostics.perf_monitor import DiagnosticRegistry


WINDOW = int(os.getenv("SURVIVAL_DIAG_WINDOW", "20000"))
SNAPSHOT_EVERY = max(1, int(os.getenv("SURVIVAL_DIAG_SNAPSHOT_EVERY", "250")))
SNAPSHOT_FILE = Path(
    os.getenv("SURVIVAL_DIAG_SNAPSHOT_FILE", "diagnostics/perf_snapshot.json")
)
DIAG_TOKEN = os.getenv("SURVIVAL_DIAG_TOKEN", "")

registry = DiagnosticRegistry(max_samples=WINDOW)


# Patch only the timing around the real controller method. Policy logic and
# return values remain exactly the same.
from survival_policy.controller import SurvivalController  # noqa: E402

_ORIGINAL_DECIDE_STEP = SurvivalController.decide_step


def _timed_decide_step(self: SurvivalController, *args: Any, **kwargs: Any) -> Any:
    wall_start = perf_counter_ns()
    cpu_start = process_time_ns()
    error = False
    try:
        return _ORIGINAL_DECIDE_STEP(self, *args, **kwargs)
    except Exception:
        error = True
        raise
    finally:
        wall_ms = (perf_counter_ns() - wall_start) / 1e6
        cpu_ms = (process_time_ns() - cpu_start) / 1e6
        registry.controller.record(wall_ms, cpu_ms, error=error)


SurvivalController.decide_step = _timed_decide_step  # type: ignore[method-assign]


# Import the user's unchanged production server only after the class method has
# been instrumented. This preserves the exact API/controller initialization.
import agent_server as production_server  # noqa: E402

app = production_server.app


def _authorized(request: Request) -> bool:
    if not DIAG_TOKEN:
        return True
    supplied = request.headers.get("x-diagnostic-token", "")
    return supplied == DIAG_TOKEN


@app.middleware("http")
async def _diagnostic_timing_middleware(request: Request, call_next: Callable):
    if request.url.path != "/predict":
        return await call_next(request)

    wall_start = perf_counter_ns()
    cpu_start = process_time_ns()
    error = False
    status_code = 500
    try:
        response = await call_next(request)
        status_code = int(response.status_code)
        error = status_code >= 500
        return response
    except Exception as exc:
        error = True
        registry.set_last_error(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        wall_ms = (perf_counter_ns() - wall_start) / 1e6
        cpu_ms = (process_time_ns() - cpu_start) / 1e6
        registry.app_predict.record(wall_ms, cpu_ms, error=error)
        registry.set_last_status(status_code)

        count = registry.app_predict.snapshot()["count"]
        if count and count % SNAPSHOT_EVERY == 0:
            try:
                registry.save_json(SNAPSHOT_FILE)
            except Exception as exc:  # Diagnostics must never break /predict.
                registry.set_last_error(f"snapshot_write_failed: {type(exc).__name__}: {exc}")




@app.get("/diag/ping", include_in_schema=False)
def diagnostic_ping():
    return {"ok": True}


@app.get("/diag/perf", include_in_schema=False)
def diagnostic_perf(request: Request):
    if not _authorized(request):
        return JSONResponse({"detail": "unauthorized"}, status_code=403)
    return registry.snapshot()


@app.post("/diag/reset", include_in_schema=False)
def diagnostic_reset(request: Request):
    if not _authorized(request):
        return JSONResponse({"detail": "unauthorized"}, status_code=403)
    registry.reset()
    try:
        registry.save_json(SNAPSHOT_FILE)
    except Exception:
        pass
    return {"status": "reset", "snapshot_file": str(SNAPSHOT_FILE)}


@app.post("/diag/save", include_in_schema=False)
def diagnostic_save(request: Request):
    if not _authorized(request):
        return JSONResponse({"detail": "unauthorized"}, status_code=403)
    registry.save_json(SNAPSHOT_FILE)
    return {"status": "saved", "snapshot_file": str(SNAPSHOT_FILE)}
