import unittest

from asr_backends import (
    TranscriptResult,
    WordTimestamp,
)
from e1.config import E1Config
from e1.e2_1_config import E2_1Config
from e1.e2_1_pipeline import (
    DualASRE2_1Pipeline,
)
from e1.pipeline import (
    DualASRE1Pipeline,
)
from e1.schemas import NLIResult


class FakeASR:
    def __init__(
        self,
        text: str,
    ):
        self.text = text

    def transcribe(
        self,
        audio,
        sample_rate=16000,
    ):
        tokens = self.text.split()

        return TranscriptResult(
            text=self.text,
            words=[
                WordTimestamp(
                    token,
                    i * 0.30,
                    i * 0.30 + 0.24,
                )
                for i, token
                in enumerate(tokens)
            ],
            segments=[],
        )


class FakeNLI:
    def predict(
        self,
        premises,
        questions,
    ):
        results = []

        for premise, question in zip(
            premises,
            questions,
        ):
            p = premise.lower()
            q = question.lower()

            if (
                "metformin" in p
                and "500" in p
                and "metformin" in q
                and "500" in q
            ):
                results.append(
                    NLIResult(
                        0.82,
                        0.15,
                        0.03,
                    )
                )
            elif (
                "200 mg" in q
                and "500 mg" in p
            ):
                results.append(
                    NLIResult(
                        0.05,
                        0.05,
                        0.90,
                    )
                )
            else:
                results.append(
                    NLIResult(
                        0.12,
                        0.78,
                        0.10,
                    )
                )

        return results


class FakeReranker:
    def __init__(self):
        self.calls = 0
        self.tasks_seen = 0

    def rank(self, tasks):
        self.calls += 1
        self.tasks_seen += len(tasks)

        selections = {}

        for task in tasks:
            # Prefer the most complete non-retrieval clause-like candidate.
            chosen = max(
                task.candidates,
                key=lambda candidate: (
                    "clause"
                    in candidate.proposal.kind,
                    candidate.proposal.end_word
                    - candidate.proposal.start_word,
                ),
            )
            selections[
                task.task_id
            ] = chosen.candidate_id

        return selections


class FailingReranker:
    def rank(self, tasks):
        raise RuntimeError(
            "simulated LLM failure"
        )


class E21PipelineMockTests(unittest.TestCase):
    def _config(self):
        return E1Config(
            retrieval_window_words=16,
            retrieval_secondary_window_words=8,
            retrieval_stride_words=4,
            retrieval_secondary_stride_words=2,
            evidence_max_words=32,
            evidence_fixed_widths=[
                5,
                8,
                12,
                16,
            ],
            evidence_pause_thresholds_s=[
                0.30,
                0.45,
            ],
            evidence_max_proposals_per_source=30,
        )

    def _asrs(self):
        text = (
            "today we reviewed diabetes and I would like "
            "you to start metformin 500 mg twice daily "
            "with meals and follow up next month"
        )
        return (
            FakeASR(text),
            FakeASR(text),
        )

    def test_e2_1_preserves_classification(self):
        med1, par1 = self._asrs()
        med2, par2 = self._asrs()

        e1 = DualASRE1Pipeline(
            config=self._config(),
            device="cpu",
            medasr=med1,
            parakeet=par1,
            nli=FakeNLI(),
        )

        reranker = FakeReranker()

        e2 = DualASRE2_1Pipeline(
            config=self._config(),
            e2_config=E2_1Config(),
            device="cpu",
            medasr=med2,
            parakeet=par2,
            nli=FakeNLI(),
            llm_reranker=reranker,
        )

        questions = [
            "Was metformin prescribed at 500 mg twice daily?",
            "Was metformin prescribed at 200 mg twice daily?",
        ]

        old = e1.predict_audio(
            [0.0],
            questions,
        )
        new = e2.predict_audio(
            [0.0],
            questions,
        )

        self.assertEqual(
            [item.answer for item in old],
            [item.answer for item in new],
        )
        self.assertEqual(
            old[1].evidence_start,
            new[1].evidence_start,
        )
        self.assertFalse(
            new[1].answer
        )
        self.assertGreaterEqual(
            reranker.tasks_seen,
            1,
        )

    def test_llm_failure_falls_back_to_e1_behavior(self):
        med, par = self._asrs()

        e2 = DualASRE2_1Pipeline(
            config=self._config(),
            e2_config=E2_1Config(),
            device="cpu",
            medasr=med,
            parakeet=par,
            nli=FakeNLI(),
            llm_reranker=FailingReranker(),
        )

        out = e2.predict_audio(
            [0.0],
            [
                "Was metformin prescribed at 500 mg twice daily?"
            ],
        )

        self.assertTrue(
            out[0].answer
        )
        self.assertIsNotNone(
            out[0].evidence_start
        )
        self.assertIsNotNone(
            out[0].evidence_end
        )


if __name__ == "__main__":
    unittest.main()
