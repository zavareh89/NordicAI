import unittest

from e1.e2_1_ensemble_config import E2_1EnsembleConfig
from e1.e2_1_ensemble_pipeline import should_use_llm
from e1.schemas import EvidenceSpan, NLIResult


class EnsembleTests(unittest.TestCase):
    def test_agreement_prefers_direct_llm_boundary(self):
        config = E2_1EnsembleConfig(agreement_tiou_threshold=0.30)
        llm = EvidenceSpan("medasr", 1, 5, 1.0, 3.0)
        ranker = EvidenceSpan("medasr", 2, 6, 1.4, 3.4)
        self.assertTrue(
            should_use_llm(
                llm,
                NLIResult(0.70, 0.25, 0.05),
                ranker,
                NLIResult(0.72, 0.23, 0.05),
                config,
            )
        )

    def test_large_disagreement_needs_semantic_advantage(self):
        config = E2_1EnsembleConfig(
            agreement_tiou_threshold=0.50,
            llm_entailment_advantage=0.10,
        )
        llm = EvidenceSpan("medasr", 1, 3, 1.0, 2.0)
        ranker = EvidenceSpan("medasr", 10, 15, 5.0, 7.0)
        self.assertFalse(
            should_use_llm(
                llm,
                NLIResult(0.70, 0.25, 0.05),
                ranker,
                NLIResult(0.68, 0.27, 0.05),
                config,
            )
        )
        self.assertTrue(
            should_use_llm(
                llm,
                NLIResult(0.85, 0.10, 0.05),
                ranker,
                NLIResult(0.68, 0.27, 0.05),
                config,
            )
        )


if __name__ == "__main__":
    unittest.main()
