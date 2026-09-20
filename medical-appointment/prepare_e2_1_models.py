#!/usr/bin/env python3
"""Download/check the additional local model required by E2.1.

E2.1 reuses the existing E1 MedASR, Parakeet and DeBERTa weights.
This script downloads only the evidence-reranking LLM by default.
"""
from __future__ import annotations

import argparse
import os

from huggingface_hub import (
    snapshot_download,
)

DEFAULT_MODEL = (
    "Qwen/Qwen3-4B-Instruct-2507"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--revision",
        default=None,
    )
    args = parser.parse_args()

    token = (
        os.environ.get(
            "HF_TOKEN"
        )
        or None
    )

    print(
        f"E2.1 evidence LLM: {args.model}"
    )

    try:
        path = snapshot_download(
            repo_id=args.model,
            revision=args.revision,
            token=token,
        )
        print(
            f"ready: {path}"
        )
        return 0

    except Exception as exc:
        print(
            f"FAILED: {exc}"
        )
        print(
            "Model page: "
            "https://huggingface.co/"
            f"{args.model}"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
