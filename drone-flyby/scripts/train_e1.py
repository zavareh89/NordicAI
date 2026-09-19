from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from drone_detector.config import load_config
from drone_detector.training.train_yolo import train_yolo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "e1_yolo26s.yaml"))
    args = parser.parse_args()
    checkpoint = train_yolo(load_config(args.config))
    print(f"Best checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
