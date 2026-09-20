from __future__ import annotations

from difflib import SequenceMatcher

from .config import E1Config
from .facts import compare_facts, extract_facts
from .schemas import CandidateAssessment, EvidenceSpan, TranscriptView
from .textnorm import normalize_text, topic_tokens


def _token_match(question_tokens: set[str], word_text: str) -> bool:
    norm = normalize_text(word_text)
    if not norm:
        return False
    pieces = norm.split()
    for piece in pieces:
        if piece in question_tokens:
            return True
        if len(piece) >= 4 and any(
            SequenceMatcher(None, piece, q).ratio() >= 0.86
            for q in question_tokens
            if len(q) >= 4
        ):
            return True
    return False


def _fact_anchor_indices(question: str, transcript: TranscriptView, left: int, right: int) -> set[int]:
    q_facts = extract_facts(question)
    if not q_facts:
        return set()

    anchors: set[int] = set()
    # Small local n-grams are enough to capture "500 milligrams", "two weeks",
    # "135 over 88", etc. Only exact fact matches become evidence anchors.
    for i in range(left, right):
        for width in range(1, 6):
            j = min(right, i + width)
            if j <= i:
                continue
            text = " ".join(w.text for w in transcript.words[i:j])
            facts = extract_facts(text)
            if compare_facts(q_facts, facts).has_exact_match:
                anchors.update(range(i, j))
    return anchors


def select_evidence(
    question: str,
    assessment: CandidateAssessment,
    transcript: TranscriptView,
    config: E1Config,
) -> EvidenceSpan:
    candidate = assessment.candidate
    left, right = candidate.left, candidate.right
    q_tokens = set(topic_tokens(question))

    anchors = {
        i
        for i in range(left, right)
        if _token_match(q_tokens, transcript.words[i].text)
    }
    anchors |= _fact_anchor_indices(question, transcript, left, right)

    if not anchors:
        # Retrieval/NLI found the window semantically but lexical anchoring found
        # nothing. Use a small center span rather than returning all 28 words.
        center = (left + right - 1) // 2
        half = max(2, config.evidence_max_words // 4)
        start_i = max(left, center - half)
        end_i = min(right - 1, center + half)
    else:
        start_i = min(anchors)
        end_i = max(anchors)

        # If anchors are far apart, keep the densest max-length region rather
        # than returning a huge low-tIoU span.
        if end_i - start_i + 1 > config.evidence_max_words:
            best: tuple[int, int, int] | None = None
            for s in range(left, max(left + 1, right - config.evidence_max_words + 1)):
                e = min(right - 1, s + config.evidence_max_words - 1)
                count = sum(s <= a <= e for a in anchors)
                item = (count, -abs((s + e) / 2.0 - sum(anchors) / len(anchors)), s)
                if best is None or item > best:
                    best = item
            assert best is not None
            start_i = best[2]
            end_i = min(right - 1, start_i + config.evidence_max_words - 1)

    start_i = max(left, start_i - config.evidence_context_words)
    end_i = min(right - 1, end_i + config.evidence_context_words)

    start = max(0.0, transcript.words[start_i].start - config.evidence_padding_before_s)
    end = transcript.words[end_i].end + config.evidence_padding_after_s

    return EvidenceSpan(
        source=transcript.source,
        start_word=start_i,
        end_word=end_i,
        start=start,
        end=end,
    )
