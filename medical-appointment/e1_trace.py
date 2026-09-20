#!/usr/bin/env python3
"""Run E1 on one local audio file and print per-question diagnostics.

This is a manual debugging tool; it is not used by the competition server.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from e1.audio import decode_audio_path_16k
from e1.pipeline import DualASRE1Pipeline


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("questions_json", help="JSON file containing a list of questions")
    p.add_argument("--config", default="config/e1_default.json")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    questions = json.loads(Path(args.questions_json).read_text(encoding="utf-8"))
    if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
        raise ValueError("questions_json must contain a JSON list of strings")

    audio, sr, _ = decode_audio_path_16k(args.audio)
    pipeline = DualASRE1Pipeline.from_config_file(args.config, device=args.device)
    predictions, traces = pipeline.predict_audio(audio, questions, sr, return_traces=True)

    for i, (prediction, trace) in enumerate(zip(predictions, traces)):
        print(f"\n[{i}] {trace.question}")
        print(f"  answer={prediction.answer} confidence={prediction.confidence:.3f} reason={prediction.reason}")
        print(f"  evidence={prediction.evidence_start}..{prediction.evidence_end} source={prediction.source}")
        for source, candidates in trace.candidates.items():
            print(f"  {source}:")
            for c in candidates:
                print(f"    rank={c.rank} retr={c.retrieval_score:.3f} {c.start:.2f}-{c.end:.2f}: {c.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
