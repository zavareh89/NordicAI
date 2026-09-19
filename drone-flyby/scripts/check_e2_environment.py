from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drone_detector.config import load_config, resolve_path
from drone_detector.detector.rfdetr_detector import build_rf152_train_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight E2 without starting training.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "e2_rfdetr_small.yaml"),
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    import torch
    import albumentations
    import transformers
    import rfdetr
    from rfdetr.config import TrainConfig

    expected = {
        "torch": "2.2.1+cu121",
        "rfdetr": "1.5.2",
        "transformers": "4.56.2",
        "albumentations": "1.4.24",
    }
    actual = {
        "torch": torch.__version__,
        "rfdetr": version("rfdetr"),
        "transformers": transformers.__version__,
        "albumentations": albumentations.__version__,
    }
    print("Versions:")
    print(json.dumps(actual, indent=2))
    problems = [f"{k}: expected {v}, got {actual[k]}" for k, v in expected.items() if actual[k] != v]
    if problems:
        raise RuntimeError("E2 dependency mismatch:\n  " + "\n  ".join(problems))

    weights = resolve_path(
        cfg,
        Path(cfg["paths"].get("pretrained_weights_dir", "weights/pretrained"))
        / Path(cfg["model"]["pretrained_weights"]).name,
    )
    dataset = resolve_path(cfg, cfg["paths"].get("rfdetr_dataset_root", "prepared/rfdetr"))
    required = [
        weights,
        dataset / "train" / "_annotations.coco.json",
        dataset / "valid" / "_annotations.coco.json",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing E2 prerequisites:\n  " + "\n  ".join(missing))

    kwargs, extras = build_rf152_train_config(cfg, str(dataset), "/tmp/e2-rfdetr-preflight")
    # This is the exact 1.5.2 validator used by RFDETRSmall.get_train_config().
    validated = TrainConfig(**kwargs)
    print("RF-DETR 1.5.2 TrainConfig: OK")
    print("dataset_dir:", validated.dataset_dir)
    print("epochs:", validated.epochs)
    print("batch_size:", validated.batch_size)
    print("grad_accum_steps:", validated.grad_accum_steps)
    print("resolution (model config):", cfg["model"]["resolution"])
    print("legacy engine extras:", extras)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("E2 preflight: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
