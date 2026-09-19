from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from drone_detector.config import scenario_config_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the unchanged official FastAPI app with E1 or E2.")
    parser.add_argument("--scenario", choices=["E1", "E2", "e1", "e2"], required=True)
    parser.add_argument("--weights", default=None, help="Challenge-fine-tuned checkpoint override.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9053)
    args = parser.parse_args()

    config = scenario_config_path(PROJECT_ROOT, args.scenario)
    os.environ["DRONE_DETECTOR_CONFIG"] = str(config)
    if args.weights:
        os.environ["DRONE_DETECTOR_WEIGHTS"] = str(Path(args.weights).resolve())

    # Load and warm up before the HTTP server begins accepting evaluator frames so
    # model construction / CUDA kernel warm-up cannot consume the first frame budget.
    from drone_detector.challenge.solution import initialize_runtime

    initialize_runtime(warmup=True)
    import uvicorn

    # The official repository's api.py remains the ASGI application.
    uvicorn.run("api:app", host=args.host, port=args.port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
