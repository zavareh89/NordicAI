#!/usr/bin/env python3
"""Compare E1 and E2.1 evidence spans on one audio file.

This script is for manual inspection. It intentionally runs both pipelines
separately so the user can verify that YES/NO answers are invariant while
evidence boundaries may change.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from e1.audio import (
    decode_audio_path_16k,
)
from e1.e2_1_pipeline import (
    DualASRE2_1Pipeline,
)
from e1.pipeline import (
    DualASRE1Pipeline,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument(
        "questions_json",
        help=(
            "JSON file containing a list "
            "of question strings"
        ),
    )
    parser.add_argument(
        "--e1-config",
        default=(
            "config/e1_evidence_tuned.json"
        ),
    )
    parser.add_argument(
        "--e2-config",
        default=(
            "config/e2_1_default.json"
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    args = parser.parse_args()

    questions = json.loads(
        Path(
            args.questions_json
        ).read_text(
            encoding="utf-8"
        )
    )

    if (
        not isinstance(
            questions,
            list,
        )
        or not all(
            isinstance(
                question,
                str,
            )
            for question
            in questions
        )
    ):
        raise ValueError(
            "questions_json must contain "
            "a JSON list of strings"
        )

    (
        audio,
        sample_rate,
        _duration,
    ) = decode_audio_path_16k(
        args.audio
    )

    e1 = (
        DualASRE1Pipeline
        .from_config_file(
            args.e1_config,
            device=args.device,
        )
    )
    e1_result = e1.predict_audio(
        audio,
        questions,
        sample_rate,
    )

    del e1

    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    e2 = (
        DualASRE2_1Pipeline
        .from_config_files(
            args.e1_config,
            args.e2_config,
            device=args.device,
        )
    )
    e2_result = e2.predict_audio(
        audio,
        questions,
        sample_rate,
    )

    answer_mismatches = 0

    for index, (
        question,
        old,
        new,
    ) in enumerate(
        zip(
            questions,
            e1_result,
            e2_result,
        )
    ):
        if old.answer != new.answer:
            answer_mismatches += 1

        print(
            f"\n[{index}] {question}"
        )
        print(
            "  answer: "
            f"E1={old.answer} "
            f"E2.1={new.answer}"
        )
        print(
            "  E1 evidence:   "
            f"{old.evidence_start} .. "
            f"{old.evidence_end}"
        )
        print(
            "  E2.1 evidence: "
            f"{new.evidence_start} .. "
            f"{new.evidence_end}"
        )

    print(
        "\nAnswer mismatches:",
        answer_mismatches,
    )

    if answer_mismatches:
        raise RuntimeError(
            "E2.1 changed at least one E1 "
            "classification, which violates "
            "the E2.1 design invariant."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
