import unittest

from e1.e2_1b_config import E2_1BConfig
from e1.llm_word_evidence import (
    AllowedWordRange,
    WordSelection,
    WordSelectionTask,
    parse_word_selections,
    validate_word_selection,
)
from e1.schemas import NormalizedWord, TranscriptView
from e1.textnorm import normalize_text


class WordEvidenceTests(unittest.TestCase):
    def _transcript(self):
        tokens = "we will start metformin 500 mg twice daily with meals".split()
        return TranscriptView(
            "medasr",
            " ".join(tokens),
            [
                NormalizedWord(i, token, normalize_text(token), i * 0.2, i * 0.2 + 0.15)
                for i, token in enumerate(tokens)
            ],
        )

    def test_parser(self):
        task = WordSelectionTask(
            0,
            "Was metformin prescribed at 500 mg twice daily?",
            (AllowedWordRange("medasr", 0, 9),),
            None,
            (),
        )
        parsed = parse_word_selections(
            '{"0":{"source":"medasr","start_word":2,"end_word":7}}',
            [task],
        )
        self.assertEqual(parsed[0].start_word, 2)
        self.assertEqual(parsed[0].end_word, 7)

    def test_validation_rejects_out_of_range(self):
        transcript = self._transcript()
        task = WordSelectionTask(
            0,
            "Was metformin prescribed at 500 mg twice daily?",
            (AllowedWordRange("medasr", 1, 8),),
            None,
            (),
        )
        invalid = validate_word_selection(
            task,
            WordSelection("medasr", 0, 7),
            {"medasr": transcript},
            E2_1BConfig(),
        )
        self.assertIsNone(invalid)

    def test_validation_materializes_word_timestamps(self):
        transcript = self._transcript()
        task = WordSelectionTask(
            0,
            "Was metformin prescribed at 500 mg twice daily?",
            (AllowedWordRange("medasr", 0, len(transcript.words)),),
            None,
            (),
        )
        span = validate_word_selection(
            task,
            WordSelection("medasr", 2, 7),
            {"medasr": transcript},
            E2_1BConfig(padding_before_s=0.0, padding_after_s=0.0),
        )
        self.assertIsNotNone(span)
        self.assertAlmostEqual(span.start, transcript.words[2].start)
        self.assertAlmostEqual(span.end, transcript.words[7].end)


if __name__ == "__main__":
    unittest.main()
