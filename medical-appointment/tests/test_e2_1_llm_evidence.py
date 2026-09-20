import unittest

from e1.config import E1Config
from e1.e2_1_config import E2_1Config
from e1.llm_evidence import (
    RerankTask,
    build_batch_prompt,
    build_rerank_candidates,
    parse_selection_output,
)
from e1.schemas import (
    CandidateAssessment,
    CandidateWindow,
    EvidenceProposal,
    EvidenceSpan,
    FactComparison,
    NLIResult,
)


def make_proposal(
    start,
    end,
    text,
    kind,
    entailment,
    strict=False,
):
    candidate = CandidateWindow(
        source="medasr",
        left=0,
        right=20,
        start=0.0,
        end=5.0,
        text=text,
        normalized_text=text.lower(),
        retrieval_score=1.0,
        topic_overlap=1.0,
    )
    assessment = CandidateAssessment(
        candidate=candidate,
        nli=NLIResult(
            entailment=entailment,
            neutral=1.0 - entailment,
            contradiction=0.0,
        ),
        facts=FactComparison(),
        yes_score=entailment,
        no_score=0.0,
    )
    return EvidenceProposal(
        source="medasr",
        start_word=start,
        end_word=end,
        text=text,
        topic_coverage=1.0,
        exact_fact_fraction=1.0,
        has_strict_contradiction=strict,
        heuristic_score=1.0,
        parent=assessment,
        kind=kind,
    )


class E21LLMEvidenceTests(unittest.TestCase):
    def test_candidate_builder_keeps_e1_choice(self):
        p1 = make_proposal(
            1,
            3,
            "500 mg twice daily",
            "anchor_cluster",
            0.90,
        )
        p2 = make_proposal(
            0,
            6,
            "start metformin 500 mg twice daily",
            "clause",
            0.82,
        )
        p3 = make_proposal(
            0,
            10,
            "today we will start metformin 500 mg twice daily with meals",
            "pause_0.45",
            0.78,
        )

        baseline = EvidenceSpan(
            source="medasr",
            start_word=1,
            end_word=3,
            start=1.0,
            end=2.0,
        )

        candidates = build_rerank_candidates(
            task_id=0,
            proposals=[p1, p2, p3],
            nli_results=[
                NLIResult(0.90, 0.08, 0.02),
                NLIResult(0.82, 0.15, 0.03),
                NLIResult(0.78, 0.19, 0.03),
            ],
            baseline_span=baseline,
            e1_config=E1Config(),
            e2_config=E2_1Config(),
        )

        self.assertGreaterEqual(
            len(candidates),
            2,
        )
        self.assertTrue(
            any(
                c.proposal.start_word == 1
                and c.proposal.end_word == 3
                for c in candidates
            )
        )

    def test_strict_fact_contradiction_is_not_exposed(self):
        good = make_proposal(
            0,
            5,
            "start metformin 500 mg daily",
            "clause",
            0.75,
        )
        bad = make_proposal(
            0,
            5,
            "start metformin 50 mg daily",
            "clause",
            0.95,
            strict=True,
        )

        candidates = build_rerank_candidates(
            task_id=2,
            proposals=[good, bad],
            nli_results=[
                NLIResult(0.75, 0.20, 0.05),
                NLIResult(0.95, 0.03, 0.02),
            ],
            baseline_span=None,
            e1_config=E1Config(),
            e2_config=E2_1Config(),
        )

        self.assertTrue(candidates)
        self.assertFalse(
            any(
                c.proposal.has_strict_contradiction
                for c in candidates
            )
        )

    def test_parser_accepts_only_valid_candidate_ids(self):
        p1 = make_proposal(
            0,
            3,
            "candidate one",
            "clause",
            0.8,
        )
        p2 = make_proposal(
            0,
            5,
            "candidate two",
            "pause_0.45",
            0.8,
        )

        candidates = build_rerank_candidates(
            task_id=4,
            proposals=[p1, p2],
            nli_results=[
                NLIResult(0.8, 0.15, 0.05),
                NLIResult(0.8, 0.15, 0.05),
            ],
            baseline_span=None,
            e1_config=E1Config(),
            e2_config=E2_1Config(),
        )

        task = RerankTask(
            task_id=4,
            question="Was treatment started?",
            candidates=candidates,
        )

        valid = candidates[0].candidate_id

        parsed = parse_selection_output(
            '{"4": "' + valid + '"}',
            [task],
        )
        self.assertEqual(
            parsed[4],
            valid,
        )

        invalid = parse_selection_output(
            '{"4": "INVENTED"}',
            [task],
        )
        self.assertEqual(
            invalid,
            {},
        )

    def test_prompt_forbids_answer_changes_and_timestamps(self):
        p1 = make_proposal(
            0,
            4,
            "start medication",
            "clause",
            0.8,
        )
        p2 = make_proposal(
            0,
            8,
            "we will start medication today",
            "pause_0.45",
            0.8,
        )

        candidates = build_rerank_candidates(
            task_id=0,
            proposals=[p1, p2],
            nli_results=[
                NLIResult(0.8, 0.15, 0.05),
                NLIResult(0.8, 0.15, 0.05),
            ],
            baseline_span=None,
            e1_config=E1Config(),
            e2_config=E2_1Config(),
        )

        prompt = build_batch_prompt(
            [
                RerankTask(
                    task_id=0,
                    question="Was medication started?",
                    candidates=candidates,
                )
            ]
        )

        self.assertIn(
            "DO NOT reconsider or change",
            prompt,
        )
        self.assertIn(
            "Never invent text, word indices, or timestamps",
            prompt,
        )


if __name__ == "__main__":
    unittest.main()
