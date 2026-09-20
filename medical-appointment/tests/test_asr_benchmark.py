import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from asr_backends import (
    ctc_collapse_runs,
    parakeet_tokens_to_words,
    sentencepiece_run_groups,
)
from benchmark_asr import (
    bm25_rank,
    build_windows,
    cache_signature,
    environment_snapshot,
    evidence_detail,
    temporal_iou,
    validate_transcript_payload,
)


class BenchmarkTests(unittest.TestCase):
    def test_temporal_iou(self):
        self.assertAlmostEqual(
            temporal_iou(21.62, 26.24, 21.62, 26.24),
            1.0,
        )
        self.assertEqual(
            temporal_iou(40.0, 45.0, 21.62, 26.24),
            0.0,
        )

    def test_parakeet_alignment(self):
        text = "blood pressure is 135 over 88"
        stamps = [
            {"token": "blood", "start": 1.0, "end": 1.2},
            {"token": " pressure", "start": 1.3, "end": 1.6},
            {"token": " is", "start": 1.7, "end": 1.8},
            {"token": " 13", "start": 1.9, "end": 2.0},
            {"token": "5", "start": 2.0, "end": 2.1},
            {"token": " over", "start": 2.2, "end": 2.3},
            {"token": " 88", "start": 2.4, "end": 2.6},
        ]

        words = parakeet_tokens_to_words(text, stamps)

        self.assertEqual(
            [w.text for w in words],
            ["blood", "pressure", "is", "135", "over", "88"],
        )
        self.assertEqual(
            (words[3].start, words[3].end),
            (1.9, 2.1),
        )

    def test_ctc_collapse_runs(self):
        # blank=0. Token 4 occurs twice separated by a blank and must stay twice.
        frames = [0, 4, 4, 0, 4, 5, 5, 0]
        runs = ctc_collapse_runs(frames, blank_id=0)
        self.assertEqual(
            runs,
            [
                (4, 1, 3),
                (4, 4, 5),
                (5, 5, 7),
            ],
        )

    def test_sentencepiece_groups(self):
        tokens = ["▁blood", "pressure", "▁is", "▁135", "▁over", "▁88"]
        groups = sentencepiece_run_groups(tokens)
        self.assertEqual(
            groups,
            [
                (0, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
            ],
        )

    def test_validation_rejects_negative_interval(self):
        payload = {
            "transcript": {
                "words": [
                    {"text": "x", "start": 2.0, "end": 1.0}
                ]
            }
        }
        with self.assertRaises(ValueError):
            validate_transcript_payload(payload)

    def test_cache_signature_changes(self):
        a = cache_signature("whisper_large_v3", {"beam": 5})
        b = cache_signature("whisper_large_v3", {"beam": 1})
        self.assertNotEqual(a, b)

    def test_retrieval_detail(self):
        token_text = [
            "hello", "the", "blood", "pressure", "was", "135/88",
            "today", "and", "the", "patient", "is", "well",
        ]

        words = [
            {
                "text": text,
                "start": float(i),
                "end": float(i) + 0.8,
                "confidence": None,
            }
            for i, text in enumerate(token_text)
        ]

        transcript = {
            "transcript": {
                "text": " ".join(token_text),
                "words": words,
                "segments": [],
            }
        }

        row = {
            "question_id": "q1",
            "transcript_id": "sample_1",
            "question": "Was the blood pressure measured at 135/88?",
            "evidence_start": 2.0,
            "evidence_end": 6.0,
            "label": 1,
        }

        detail = evidence_detail(
            "whisper_large_v3",
            row,
            transcript,
            window_words=7,
            window_stride=3,
            context_s=1.0,
        )

        self.assertGreater(detail["timestamp_ceiling_tiou"], 0.7)
        self.assertEqual(detail["numeric_token_recall"], 1.0)
        self.assertIn("135/88", detail["asr_evidence_text"])

    def test_bm25(self):
        words = [
            {"text": "cat", "start": 0.0, "end": 0.5},
            {"text": "garden", "start": 0.6, "end": 1.0},
            {"text": "metformin", "start": 1.1, "end": 1.5},
            {"text": "500", "start": 1.6, "end": 1.8},
            {"text": "mg", "start": 1.9, "end": 2.1},
        ]

        windows = build_windows(words, size=3, stride=2)
        ranked = bm25_rank("metformin 500 mg", windows)
        self.assertIn("metformin", windows[ranked[0][0]]["text"])

    def test_environment_snapshot(self):
        env = environment_snapshot()
        self.assertIn("python", env)
        self.assertIn("platform", env)


if __name__ == "__main__":
    unittest.main()
