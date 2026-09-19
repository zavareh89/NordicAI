from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from drone_detector.config import scenario_config_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the official FastAPI app with E1/E2 and passive runtime telemetry.")
    parser.add_argument("--scenario", choices=["E1", "E2", "e1", "e2"], required=True)
    parser.add_argument("--weights", default=None, help="Challenge-fine-tuned checkpoint override.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9053)
    parser.add_argument(
        "--telemetry-dir",
        default="runtime_logs",
        help="Directory for frame_events.jsonl, frame_summary.json and http_timing.jsonl.",
    )
    parser.add_argument(
        "--expected-frames",
        type=int,
        default=None,
        help="Optional total frame count, e.g. 25 for Helsinki, to report final missing IDs.",
    )
    parser.add_argument("--first-frame", type=int, default=0)
    parser.add_argument(
        "--slow-frame-ms",
        type=float,
        default=333.0,
        help="Flag frames slower than the ~3 FPS source interval.",
    )
    parser.add_argument("--disable-telemetry", action="store_true")
    args = parser.parse_args()

    config = scenario_config_path(PROJECT_ROOT, args.scenario)
    os.environ["DRONE_DETECTOR_CONFIG"] = str(config)
    if args.weights:
        os.environ["DRONE_DETECTOR_WEIGHTS"] = str(Path(args.weights).resolve())

    os.environ["DRONE_TELEMETRY_ENABLED"] = "0" if args.disable_telemetry else "1"
    os.environ["DRONE_TELEMETRY_DIR"] = str(Path(args.telemetry_dir).resolve())
    os.environ["DRONE_FIRST_FRAME"] = str(args.first_frame)
    os.environ["DRONE_SLOW_FRAME_MS"] = str(args.slow_frame_ms)
    if args.expected_frames is not None:
        os.environ["DRONE_EXPECTED_FRAMES"] = str(args.expected_frames)
    else:
        os.environ.pop("DRONE_EXPECTED_FRAMES", None)

    # Warm the model before the server accepts competition frames.
    from drone_detector.challenge.solution import initialize_runtime

    initialize_runtime(warmup=True)

    import api
    import uvicorn
    from drone_detector.runtime_telemetry import get_runtime_telemetry

    telemetry = get_runtime_telemetry()
    http_logger = logging.getLogger("drone_detector.http")

    @api.app.middleware("http")
    async def _runtime_http_timing(request, call_next):
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = int(response.status_code)
            return response
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if request.url.path == "/predict":
                telemetry.record_http_request(
                    method=request.method,
                    path=request.url.path,
                    status_code=status_code,
                    total_ms=elapsed_ms,
                )
                http_logger.info(
                    "HTTP %s %s status=%d total_ms=%.3f",
                    request.method,
                    request.url.path,
                    status_code,
                    elapsed_ms,
                )

    # Use one worker so in-memory frame accounting has one authoritative sequence.
    uvicorn.run(api.app, host=args.host, port=args.port, reload=False, workers=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
