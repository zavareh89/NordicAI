#!/usr/bin/env python3
"""Pre-download/check the three ASR repositories."""
from __future__ import annotations

import argparse
import os

from asr_backends import MODEL_SPECS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS),
        default=["parakeet_v3", "medasr", "whisper_large_v3"],
    )
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or None
    failures = 0

    for name in args.models:
        spec = MODEL_SPECS[name]
        print(f"\n{name}: {spec['model_id']}")

        try:
            path = snapshot_download(
                repo_id=spec["model_id"],
                revision=args.revision,
                token=token,
            )
            print(f"  ready: {path}")

        except Exception as exc:
            failures += 1
            print(f"  FAILED: {exc}")
            print(f"  model page: {spec['page']}")

            if name == "medasr":
                print("  MedASR is gated. Accept Google's terms, then run")
                print("  `hf auth login` or export HF_TOKEN=hf_... and retry.")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
