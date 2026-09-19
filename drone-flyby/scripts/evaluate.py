from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from drone_detector.config import load_config, scenario_config_path
from drone_detector.evaluation.challenge_eval import run_official_local_evaluator
from drone_detector.evaluation.offline_eval import evaluate_offline


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate E1/E2 offline or invoke the official local evaluator.")
    parser.add_argument("--scenario", choices=["E1", "E2", "e1", "e2"], required=True)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--official-local", action="store_true")
    parser.add_argument("--challenge-root", default=str(PROJECT_ROOT))
    parser.add_argument("--url", default="http://localhost:9053/predict")
    parser.add_argument("--scene", default="helsinki")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.official_local:
        return run_official_local_evaluator(
            args.challenge_root,
            url=args.url,
            scene=args.scene,
            realtime=args.realtime,
            verbose=args.verbose,
        )
    cfg = load_config(scenario_config_path(PROJECT_ROOT, args.scenario))
    metrics = evaluate_offline(cfg, weights=args.weights)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
