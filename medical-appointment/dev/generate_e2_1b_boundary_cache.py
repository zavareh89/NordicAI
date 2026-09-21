#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import json
from pathlib import Path
import time

from e1.audio import decode_audio_path_16k
from e1.config import E1Config
from e1.e2_1b_config import E2_1BConfig
from e1.e2_1b_pipeline import DualASRE2_1BPipeline
from utils import AUDIO_DIRECTORY, group_questions_by_conversation


def write_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1-config", default="config/e1_evidence_tuned.json")
    parser.add_argument("--e2-config", default="config/e2_1b_default.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output", default="dev_cache/e2_1b_boundary_cache.json.gz"
    )
    parser.add_argument("--limit-conversations", type=int, default=None)
    args = parser.parse_args()

    e2_config = E2_1BConfig.from_json(args.e2_config)
    # Cache raw word-boundary times. Calibration is learned afterwards.
    e2_config = replace(e2_config, padding_before_s=0.0, padding_after_s=0.0)
    pipeline = DualASRE2_1BPipeline(
        config=E1Config.from_json(args.e1_config),
        e2_config=e2_config,
        device=args.device,
    )

    conversations = list(group_questions_by_conversation())
    if args.limit_conversations:
        conversations = conversations[: args.limit_conversations]

    records = []
    started = time.perf_counter()
    for index, (audio_filename, rows) in enumerate(conversations, start=1):
        audio, sample_rate, _ = decode_audio_path_16k(AUDIO_DIRECTORY / audio_filename)
        questions = [row["question"] for row in rows]
        predictions = pipeline.predict_audio(audio, questions, sample_rate=sample_rate)

        for row, pred in zip(rows, predictions):
            records.append(
                {
                    "question_id": row["question_id"],
                    "transcript_id": row["transcript_id"],
                    "question": row["question"],
                    "question_type": row["question_type"],
                    "label": int(row["label"]),
                    "answer": bool(pred.answer),
                    "source": pred.source,
                    "raw_start": pred.evidence_start,
                    "raw_end": pred.evidence_end,
                    "gold_start": (
                        float(row["evidence_start"])
                        if row.get("evidence_start") else None
                    ),
                    "gold_end": (
                        float(row["evidence_end"])
                        if row.get("evidence_end") else None
                    ),
                }
            )

        print(f"[{index}/{len(conversations)}] {audio_filename}", flush=True)

    write_json(
        args.output,
        {
            "schema_version": 1,
            "e1_config": args.e1_config,
            "e2_config": args.e2_config,
            "generation_seconds": time.perf_counter() - started,
            "records": records,
        },
    )
    print(f"Wrote {len(records)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
