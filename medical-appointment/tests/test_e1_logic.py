import unittest

from e1.config import E1Config
from e1.consensus import assess_candidate, decide
from e1.evidence import select_evidence
from e1.facts import compare_facts, extract_facts
from e1.pairing import candidates_align
from e1.retrieval import retrieve
from e1.schemas import CandidateWindow, NLIResult, NormalizedWord, TranscriptView
from e1.textnorm import normalize_text


class E1LogicTests(unittest.TestCase):
    def setUp(self):
        self.config = E1Config()

    def test_number_and_unit_normalization(self):
        self.assertIn("500 mg", normalize_text("five hundred milligrams"))
        self.assertIn("14 day", normalize_text("fourteen days"))

    def test_mass_dose_conversion(self):
        a = extract_facts("0.5 grams")
        b = extract_facts("500 mg")
        comparison = compare_facts(a, b)
        self.assertIn("dose_mg", comparison.matched)

    def test_hard_negative_fact_guard(self):
        question = extract_facts("Was the dose 200 mg?")
        candidate = extract_facts("The dose was 100 mg.")
        comparison = compare_facts(question, candidate)
        self.assertTrue(comparison.has_strict_contradiction)
        self.assertIn("dose_mg", comparison.mismatched)

    def test_retrieval_does_not_require_wrong_number_to_match(self):
        words = [
            NormalizedWord(i, t, normalize_text(t), float(i), float(i) + 0.4)
            for i, t in enumerate(
                "we discussed exercise then prescribed metformin five hundred milligrams twice daily for diabetes follow up".split()
            )
        ]
        transcript = TranscriptView("medasr", " ".join(w.text for w in words), words)
        results = retrieve("Was metformin prescribed at 200 mg twice daily?", transcript, self.config)
        self.assertTrue(results)
        self.assertIn("metformin", results[0].text.lower())

    def test_temporal_alignment(self):
        a = CandidateWindow("medasr", 0, 5, 10.0, 14.0, "a", "a")
        b = CandidateWindow("parakeet_v3", 0, 5, 10.5, 14.5, "b", "b")
        self.assertTrue(candidates_align(a, b, self.config))

    def test_fact_guard_overrides_false_entailment(self):
        candidate = CandidateWindow(
            "medasr", 0, 5, 1.0, 3.0,
            "metformin 100 mg daily", "metformin 100 mg daily",
            facts=extract_facts("metformin 100 mg daily"),
            retrieval_score=1.0,
        )
        assessment = assess_candidate(
            "Was metformin prescribed at 200 mg daily?",
            candidate,
            NLIResult(0.90, 0.05, 0.05),
            self.config,
        )
        self.assertGreater(assessment.no_score, assessment.yes_score)
        self.assertTrue(assessment.facts.has_strict_contradiction)

    def test_dual_yes_consensus(self):
        med = CandidateWindow(
            "medasr", 0, 6, 10.0, 12.0,
            "metformin 500 mg twice daily", "metformin 500 mg twice daily",
            facts=extract_facts("metformin 500 mg twice daily"), retrieval_score=1.0,
        )
        par = CandidateWindow(
            "parakeet_v3", 0, 6, 10.1, 12.1,
            "metformin 500 mg twice daily", "metformin 500 mg twice daily",
            facts=extract_facts("metformin 500 mg twice daily"), retrieval_score=1.0,
        )
        q = "Was metformin prescribed at 500 mg twice daily?"
        assessments = [
            assess_candidate(q, med, NLIResult(0.90, 0.08, 0.02), self.config),
            assess_candidate(q, par, NLIResult(0.92, 0.06, 0.02), self.config),
        ]
        decision = decide(q, assessments, self.config)
        self.assertTrue(decision.answer)
        self.assertEqual(decision.reason, "dual_yes_consensus")

    def test_evidence_is_tighter_than_retrieval_window(self):
        text = "today we discussed diet and then start metformin 500 mg twice daily with meals and review in four weeks"
        pieces = text.split()
        words = [
            NormalizedWord(i, token, normalize_text(token), i * 0.25, i * 0.25 + 0.2)
            for i, token in enumerate(pieces)
        ]
        transcript = TranscriptView("medasr", text, words)
        candidate = CandidateWindow(
            "medasr", 0, len(words), words[0].start, words[-1].end,
            text, normalize_text(text), facts=extract_facts(text), retrieval_score=1.0,
        )
        assessment = assess_candidate(
            "Was metformin prescribed at 500 mg twice daily?",
            candidate,
            NLIResult(0.95, 0.04, 0.01),
            self.config,
        )
        span = select_evidence(
            "Was metformin prescribed at 500 mg twice daily?",
            assessment,
            transcript,
            self.config,
        )
        self.assertLessEqual(span.end_word - span.start_word + 1, self.config.evidence_max_words)
        self.assertGreater(span.end, span.start)


if __name__ == "__main__":
    unittest.main()
