import unittest

from e1.config import E1Config
from e1.consensus import assess_candidate
from e1.evidence import build_evidence_proposals
from e1.facts import extract_facts
from e1.schemas import CandidateWindow, NLIResult, NormalizedWord, TranscriptView
from e1.textnorm import normalize_text


class EvidenceCandidateTests(unittest.TestCase):
    def _fixture(self):
        tokens = [
            ("today", 0.00, 0.20),
            ("we", 0.23, 0.35),
            ("reviewed", 0.38, 0.65),
            ("things.", 0.67, 0.90),
            ("start", 1.55, 1.75),
            ("metformin", 1.78, 2.10),
            ("500", 2.12, 2.28),
            ("mg", 2.30, 2.42),
            ("twice", 2.45, 2.66),
            ("daily.", 2.68, 2.92),
            ("okay", 3.60, 3.80),
        ]
        words = [
            NormalizedWord(i, text, normalize_text(text), start, end)
            for i, (text, start, end) in enumerate(tokens)
        ]
        transcript = TranscriptView("medasr", " ".join(t[0] for t in tokens), words)
        text = transcript.text
        candidate = CandidateWindow(
            "medasr", 0, len(words), 0.0, 3.8, text, normalize_text(text),
            facts=extract_facts(text), retrieval_score=1.0, topic_overlap=1.0,
        )
        assessment = assess_candidate(
            "Was metformin prescribed at 500 mg twice daily?",
            candidate,
            NLIResult(0.90, 0.08, 0.02),
            E1Config(),
        )
        return transcript, assessment

    def test_pause_clause_and_multiscale_proposals_exist(self):
        transcript, assessment = self._fixture()
        proposals = build_evidence_proposals(
            "Was metformin prescribed at 500 mg twice daily?",
            assessment,
            transcript,
            E1Config(),
        )
        kinds = {proposal.kind for proposal in proposals}
        self.assertTrue(any(kind.startswith("pause_") for kind in kinds))
        self.assertIn("clause", kinds)
        self.assertTrue(any(kind.startswith("fixed_") for kind in kinds))
        self.assertTrue(any(kind.startswith("retrieval_") for kind in kinds))

    def test_pause_span_stops_at_large_gaps(self):
        transcript, assessment = self._fixture()
        proposals = build_evidence_proposals(
            "Was metformin prescribed at 500 mg twice daily?",
            assessment,
            transcript,
            E1Config(),
        )
        pause = [p for p in proposals if p.kind == "pause_0.45"]
        self.assertTrue(pause)
        self.assertTrue(any(p.start_word >= 4 and p.end_word <= 9 for p in pause))


if __name__ == "__main__":
    unittest.main()
