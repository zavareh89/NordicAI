import unittest

from asr_backends import TranscriptResult, WordTimestamp
from e1.config import E1Config
from e1.e2_1b_config import E2_1BConfig
from e1.e2_1b_pipeline import DualASRE2_1BPipeline
from e1.llm_word_evidence import WordSelection
from e1.schemas import NLIResult


class FakeASR:
    def __init__(self, text):
        self.text = text

    def transcribe(self, audio, sample_rate=16000):
        tokens = self.text.split()
        return TranscriptResult(
            self.text,
            [WordTimestamp(t, i * 0.3, i * 0.3 + 0.24) for i, t in enumerate(tokens)],
            [],
        )


class FakeNLI:
    def predict(self, premises, questions):
        out = []
        for p, q in zip(premises, questions):
            if "metformin" in p.lower() and "500" in p and "500" in q:
                out.append(NLIResult(0.86, 0.11, 0.03))
            else:
                out.append(NLIResult(0.10, 0.80, 0.10))
        return out


class FakeSelector:
    def select(self, tasks, transcripts):
        result = {}
        for task in tasks:
            allowed = task.ranges[0]
            # Deliberately choose a valid inner span.
            result[task.task_id] = WordSelection(
                allowed.source,
                min(allowed.left + 1, allowed.right - 1),
                min(allowed.left + 6, allowed.right - 1),
            )
        return result


class E21BPipelineTests(unittest.TestCase):
    def test_classification_is_frozen(self):
        text = "today we will start metformin 500 mg twice daily with meals"
        pipeline = DualASRE2_1BPipeline(
            config=E1Config(
                retrieval_window_words=16,
                retrieval_secondary_window_words=8,
                retrieval_stride_words=4,
                retrieval_secondary_stride_words=2,
            ),
            e2_config=E2_1BConfig(),
            device="cpu",
            medasr=FakeASR(text),
            parakeet=FakeASR(text),
            nli=FakeNLI(),
            word_selector=FakeSelector(),
        )
        result = pipeline.predict_audio(
            [0.0],
            ["Was metformin prescribed at 500 mg twice daily?"],
        )
        self.assertTrue(result[0].answer)
        self.assertIsNotNone(result[0].evidence_start)
        self.assertIsNotNone(result[0].evidence_end)


if __name__ == "__main__":
    unittest.main()
