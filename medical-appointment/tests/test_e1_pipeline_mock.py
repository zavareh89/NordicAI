import unittest

from asr_backends import TranscriptResult, WordTimestamp
from e1.config import E1Config
from e1.pipeline import DualASRE1Pipeline
from e1.schemas import NLIResult


class FakeASR:
    def __init__(self, text: str):
        self.text = text

    def transcribe(self, audio, sample_rate=16000):
        tokens = self.text.split()
        words = [
            WordTimestamp(token, i * 0.3, i * 0.3 + 0.24)
            for i, token in enumerate(tokens)
        ]
        return TranscriptResult(self.text, words, [])


class FakeNLI:
    def predict(self, premises, questions):
        results = []
        for premise, question in zip(premises, questions):
            p = premise.lower()
            q = question.lower()
            if "200 mg" in q and "100 mg" in p:
                results.append(NLIResult(0.80, 0.10, 0.10))  # fact guard must fix this
            elif "500 mg" in q and ("500 mg" in p or "five hundred" in p):
                results.append(NLIResult(0.92, 0.06, 0.02))
            elif "500 mg" in q and ("50 mg" in p or "fifty" in p):
                results.append(NLIResult(0.04, 0.08, 0.88))
            else:
                results.append(NLIResult(0.10, 0.80, 0.10))
        return results


class PipelineMockTests(unittest.TestCase):
    def test_positive_dual_asr(self):
        pipeline = DualASRE1Pipeline(
            config=E1Config(retrieval_window_words=16, retrieval_stride_words=4),
            device="cpu",
            medasr=FakeASR("the doctor prescribed metformin 500 mg twice daily with meals"),
            parakeet=FakeASR("the doctor prescribed metformin 500 mg twice daily with meals"),
            nli=FakeNLI(),
        )
        out = pipeline.predict_audio([0.0], ["Was metformin prescribed at 500 mg twice daily?"])
        self.assertTrue(out[0].answer)
        self.assertIsNotNone(out[0].evidence_start)
        self.assertIsNotNone(out[0].evidence_end)

    def test_hard_negative_even_when_nli_is_fooled(self):
        pipeline = DualASRE1Pipeline(
            config=E1Config(retrieval_window_words=16, retrieval_stride_words=4),
            device="cpu",
            medasr=FakeASR("the dose was 100 mg once daily"),
            parakeet=FakeASR("the dose was 100 mg once daily"),
            nli=FakeNLI(),
        )
        out = pipeline.predict_audio([0.0], ["Was the dose 200 mg once daily?"])
        self.assertFalse(out[0].answer)
        self.assertIsNone(out[0].evidence_start)

    def test_asr_disagreement_keeps_exact_fact_yes(self):
        pipeline = DualASRE1Pipeline(
            config=E1Config(retrieval_window_words=16, retrieval_stride_words=4),
            device="cpu",
            medasr=FakeASR("start metformin 50 mg twice daily"),
            parakeet=FakeASR("start metformin 500 mg twice daily"),
            nli=FakeNLI(),
        )
        out = pipeline.predict_audio([0.0], ["Was metformin prescribed at 500 mg twice daily?"])
        self.assertTrue(out[0].answer)
        self.assertEqual(out[0].source, "parakeet_v3")


if __name__ == "__main__":
    unittest.main()
