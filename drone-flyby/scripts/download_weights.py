from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drone_detector.config import load_config, scenario_config_path
from drone_detector.weights import download_for_scenario


def main() -> int:
    parser = argparse.ArgumentParser(description="Download/cache official pretrained detector weights.")
    parser.add_argument("--scenario", choices=["E1", "E2", "e1", "e2"], required=True)
    parser.add_argument(
        "--initializer",
        choices=["standard", "objects365"],
        default="standard",
        help="E1 only. Standard yolo26s.pt is the baseline default; Objects365 is optional.",
    )
    args = parser.parse_args()
    cfg = load_config(scenario_config_path(PROJECT_ROOT, args.scenario))
    if args.scenario.upper() == "E2" and args.initializer != "standard":
        parser.error("--initializer objects365 is only valid for E1")
    path = download_for_scenario(cfg, initializer=args.initializer)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
