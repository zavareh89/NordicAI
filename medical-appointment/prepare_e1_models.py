#!/usr/bin/env python3
"""Download/check the three models required by Dual-ASR E1."""
from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download

MODELS = {
    "medasr": "google/medasr",
    "parakeet_v3": "nvidia/parakeet-tdt-0.6b-v3",
    "nli": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), default=list(MODELS))
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or None
    failures = 0

    for name in args.models:
        repo = MODELS[name]
        print(f"\n{name}: {repo}")
        try:
            path = snapshot_download(repo_id=repo, revision=args.revision, token=token)
            print(f"  ready: {path}")
        except Exception as exc:
            failures += 1
            print(f"  FAILED: {exc}")
            if name == "medasr":
                print("  Accept the MedASR terms at https://huggingface.co/google/medasr")
                print("  and authenticate with `hf auth login` or HF_TOKEN.")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
