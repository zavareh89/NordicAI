from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drone_detector.config import load_config
from drone_detector.training.train_rfdetr import train_rfdetr


def main() -> int:
    parser = argparse.ArgumentParser(description="Train E2 RF-DETR Small with RF-DETR 1.5.2.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "e2_rfdetr_small.yaml"),
    )
    parser.add_argument("--resume", default=None, help="Resume from RF-DETR 1.5.2 checkpoint.pth")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument(
        "--skip-map50-selection",
        action="store_true",
        help="Use RF-DETR's native best checkpoint instead of post-selecting saved checkpoints by AP@0.50.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = copy.deepcopy(cfg)
    if args.resume is not None:
        cfg["training"]["resume"] = str(Path(args.resume).expanduser().resolve())
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.grad_accum_steps is not None:
        cfg["training"]["grad_accum_steps"] = args.grad_accum_steps
    if args.device is not None:
        cfg["training"]["device"] = args.device
        cfg["inference"]["device"] = args.device
    if args.skip_map50_selection:
        cfg["training"]["select_best_by_map50"] = False

    checkpoint = train_rfdetr(cfg)
    print(f"Selected E2 checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
