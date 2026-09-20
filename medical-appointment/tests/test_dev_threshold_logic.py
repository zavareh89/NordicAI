import unittest

from dev.common import score_records, select_cached_evidence
from dev.threshold_logic import decide_cached_record


BASE = {
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
    "medasr_evidence_preference_tolerance": 0.05,
    "evidence_min_words": 3,
    "evidence_max_words": 14,
    "evidence_min_entailment": 0.48,
    "evidence_entailment_weight": 0.60,
    "evidence_fact_weight": 0.20,
    "evidence_topic_weight": 0.12,
    "evidence_compactness_weight": 0.08,
    "evidence_medasr_tie_bonus": 0.015,
    "evidence_padding_before_s": 0.03,
    "evidence_padding_after_s": 0.06,
}


def member(source, yes, no, exact=False, strict=False, neutral=0.2):
    return {
        "source": source,
        "yes_score": yes,
        "no_score": no,
        "exact_match": exact,
        "strict_contradiction": strict,
        "relevance": 1.0,
        "nli": {
            "entailment": yes,
            "neutral": neutral,
            "contradiction": no,
        },
        "candidate": {"start": 10.0, "end": 14.0},
    }


class DevThresholdLogicTests(unittest.TestCase):
    def test_dual_yes(self):
        record = {
            "events": [{
                "members": [
                    member("medasr", 0.70, 0.05),
                    member("parakeet_v3", 0.72, 0.04),
                ],
                "proposals": [],
            }]
        }
        decision = decide_cached_record(record, BASE)
        self.assertTrue(decision["answer"])
        self.assertEqual(decision["reason"], "paired_dual_yes")

    def test_exact_fact_disagreement_can_still_be_yes(self):
        record = {
            "events": [{
                "members": [
                    member("medasr", 0.05, 0.98, strict=True),
                    member("parakeet_v3", 0.80, 0.05, exact=True),
                ],
                "proposals": [],
            }]
        }
        decision = decide_cached_record(record, BASE)
        self.assertTrue(decision["answer"])
        self.assertEqual(decision["source"], "parakeet_v3")

    def test_cached_evidence_uses_nli_and_timestamps(self):
        event = {
            "proposals": [{
                "source": "medasr",
                "kind": "compact",
                "word_count": 6,
                "raw_start": 10.0,
                "raw_end": 12.0,
                "start_word": 5,
                "end_word": 10,
                "topic_coverage": 0.8,
                "exact_fact_fraction": 1.0,
                "has_strict_contradiction": False,
                "nli": {"entailment": 0.9, "neutral": 0.08, "contradiction": 0.02},
            }]
        }
        evidence = select_cached_evidence(event, BASE)
        self.assertIsNotNone(evidence)
        self.assertLess(evidence["start"], 10.0)
        self.assertGreater(evidence["end"], 12.0)


if __name__ == "__main__":
    unittest.main()
