import unittest

from dev.evidence_cv_tune import evidence_metrics


class EvidenceCvTuneTests(unittest.TestCase):
    def test_exact_boundary_scores_one(self):
        config = {
            "event_member_yes_floor": 0.48,
            "dual_event_yes_threshold": 0.56,
            "single_source_yes_threshold": 0.62,
            "exact_fact_yes_threshold": 0.58,
            "asr_fact_disagreement_yes_threshold": 0.64,
            "generic_disagreement_yes_threshold": 0.68,
            "strong_entailment_threshold": 0.66,
            "event_yes_no_margin": 0.03,
            "dual_agreement_bonus": 0.05,
            "paired_neutral_bonus": 0.02,
            "minimum_relevance_for_yes": 0.10,
            "evidence_min_words": 3,
            "evidence_max_words": 32,
            "evidence_min_entailment": 0.48,
            "evidence_entailment_weight": 0.60,
            "evidence_fact_weight": 0.20,
            "evidence_topic_weight": 0.12,
            "evidence_compactness_weight": 0.03,
            "evidence_medasr_tie_bonus": 0.0,
            "evidence_padding_before_s": 0.0,
            "evidence_padding_after_s": 0.0,
            "evidence_kind_bias": {},
        }
        record = {
            "label": 1,
            "transcript_id": "sample",
            "gold_start": 10.0,
            "gold_end": 12.0,
            "events": [{
                "members": [{
                    "source": "medasr",
                    "yes_score": 0.8,
                    "no_score": 0.05,
                    "exact_match": False,
                    "strict_contradiction": False,
                    "relevance": 1.0,
                    "nli": {"entailment": 0.8, "neutral": 0.15, "contradiction": 0.05},
                }],
                "proposals": [{
                    "source": "medasr",
                    "kind": "pause_0.45",
                    "word_count": 8,
                    "raw_start": 10.0,
                    "raw_end": 12.0,
                    "topic_coverage": 1.0,
                    "exact_fact_fraction": 0.0,
                    "has_strict_contradiction": False,
                    "nli": {"entailment": 0.9, "neutral": 0.08, "contradiction": 0.02},
                }],
            }],
        }
        metrics = evidence_metrics([record], config)
        self.assertAlmostEqual(metrics["mean_tiou_all_positives"], 1.0)


if __name__ == "__main__":
    unittest.main()
