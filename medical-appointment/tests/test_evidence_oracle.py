import unittest

from dev.evidence_oracle import analyze_record


class EvidenceOracleTests(unittest.TestCase):
    def test_oracle_is_at_least_selected_for_same_event(self):
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
            "evidence_max_words": 14,
            "evidence_min_entailment": 0.48,
            "evidence_entailment_weight": 0.60,
            "evidence_fact_weight": 0.20,
            "evidence_topic_weight": 0.12,
            "evidence_compactness_weight": 0.08,
            "evidence_medasr_tie_bonus": 0.015,
            "evidence_padding_before_s": 0.0,
            "evidence_padding_after_s": 0.0,
        }
        member = {
            "source": "medasr",
            "yes_score": 0.8,
            "no_score": 0.05,
            "exact_match": False,
            "strict_contradiction": False,
            "relevance": 1.0,
            "nli": {"entailment": 0.8, "neutral": 0.15, "contradiction": 0.05},
            "candidate": {"start": 9.0, "end": 15.0},
        }
        record = {
            "question_id": "q",
            "transcript_id": "s",
            "question": "q?",
            "label": 1,
            "gold_start": 10.0,
            "gold_end": 12.0,
            "events": [{
                "members": [member],
                "proposals": [
                    {
                        "source": "medasr",
                        "kind": "compact",
                        "word_count": 5,
                        "raw_start": 10.0,
                        "raw_end": 12.0,
                        "topic_coverage": 0.5,
                        "exact_fact_fraction": 0.0,
                        "has_strict_contradiction": False,
                        "nli": {"entailment": 0.9, "neutral": 0.08, "contradiction": 0.02},
                    },
                    {
                        "source": "medasr",
                        "kind": "compact",
                        "word_count": 5,
                        "raw_start": 10.5,
                        "raw_end": 12.5,
                        "topic_coverage": 1.0,
                        "exact_fact_fraction": 0.0,
                        "has_strict_contradiction": False,
                        "nli": {"entailment": 0.95, "neutral": 0.04, "contradiction": 0.01},
                    },
                ],
            }],
        }
        row = analyze_record(record, config)
        self.assertGreaterEqual(
            row["selected_event_proposal_oracle_tiou"],
            row["selected_tiou"],
        )


if __name__ == "__main__":
    unittest.main()
