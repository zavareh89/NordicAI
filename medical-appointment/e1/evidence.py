from __future__ import annotations

from difflib import SequenceMatcher

from .config import E1Config
from .facts import compare_facts, extract_facts
from .schemas import (
    CandidateAssessment,
    EvidenceProposal,
    EvidenceSpan,
    NLIResult,
    TranscriptView,
)
from .textnorm import normalize_text, topic_tokens


def _token_match(
    question_tokens: set[str],
    word_text: str,
) -> bool:
    norm = normalize_text(word_text)
    if not norm:
        return False

    for piece in norm.split():
        if piece in question_tokens:
            return True
        if len(piece) >= 4 and any(
            SequenceMatcher(None, piece, q).ratio() >= 0.86
            for q in question_tokens
            if len(q) >= 4
        ):
            return True

    return False


def _fact_anchor_indices(
    question: str,
    transcript: TranscriptView,
    left: int,
    right: int,
) -> set[int]:
    q_facts = extract_facts(question)
    if not q_facts:
        return set()

    anchors: set[int] = set()

    # Local n-grams capture "500 milligrams", "two weeks", "135 over 88", etc.
    for i in range(left, right):
        for width in range(1, 6):
            j = min(right, i + width)
            if j <= i:
                continue

            text = " ".join(
                w.text
                for w in transcript.words[i:j]
            )
            facts = extract_facts(text)

            if compare_facts(q_facts, facts).has_exact_match:
                anchors.update(range(i, j))

    return anchors


def _anchor_clusters(
    anchors: set[int],
) -> list[tuple[int, int]]:
    if not anchors:
        return []

    ordered = sorted(anchors)
    clusters: list[tuple[int, int]] = []
    start = prev = ordered[0]

    for index in ordered[1:]:
        if index - prev <= 2:
            prev = index
            continue
        clusters.append((start, prev))
        start = prev = index

    clusters.append((start, prev))
    return clusters


def _proposal_metrics(
    question: str,
    text: str,
) -> tuple[float, float, bool]:
    q_topic = set(topic_tokens(question))
    p_topic = set(topic_tokens(text))
    topic_coverage = (
        len(q_topic & p_topic) / len(q_topic)
        if q_topic
        else 0.0
    )

    q_facts = extract_facts(question)
    p_facts = extract_facts(text)
    comparison = compare_facts(q_facts, p_facts)

    q_fact_types = {fact.kind for fact in q_facts}
    exact_fraction = (
        len(comparison.matched) / len(q_fact_types)
        if q_fact_types
        else 0.0
    )

    return (
        topic_coverage,
        exact_fraction,
        comparison.has_strict_contradiction,
    )


def _make_proposal(
    question: str,
    assessment: CandidateAssessment,
    transcript: TranscriptView,
    start_i: int,
    end_i: int,
    config: E1Config,
) -> EvidenceProposal:
    text = " ".join(
        word.text
        for word in transcript.words[start_i : end_i + 1]
    )

    topic_coverage, exact_fraction, strict = _proposal_metrics(
        question,
        text,
    )

    length = end_i - start_i + 1
    compactness = 1.0 - (
        (length - config.evidence_min_words)
        / max(
            1,
            config.evidence_max_words - config.evidence_min_words,
        )
    )
    compactness = max(0.0, min(1.0, compactness))

    heuristic = (
        0.50 * topic_coverage
        + 0.35 * exact_fraction
        + 0.15 * compactness
    )
    if strict:
        heuristic -= 1.0

    return EvidenceProposal(
        source=assessment.candidate.source,
        start_word=start_i,
        end_word=end_i,
        text=text,
        topic_coverage=topic_coverage,
        exact_fact_fraction=exact_fraction,
        has_strict_contradiction=strict,
        heuristic_score=heuristic,
        parent=assessment,
    )


def build_evidence_proposals(
    question: str,
    assessment: CandidateAssessment,
    transcript: TranscriptView,
    config: E1Config,
) -> list[EvidenceProposal]:
    """Create compact local spans inside the supporting retrieval window."""

    candidate = assessment.candidate
    left, right = candidate.left, candidate.right
    if right <= left:
        return []

    q_tokens = set(topic_tokens(question))

    lexical_anchors = {
        i
        for i in range(left, right)
        if _token_match(
            q_tokens,
            transcript.words[i].text,
        )
    }
    fact_anchors = _fact_anchor_indices(
        question,
        transcript,
        left,
        right,
    )
    anchors = lexical_anchors | fact_anchors

    intervals: set[tuple[int, int]] = set()

    def add_interval(start_i: int, end_i: int) -> None:
        start_i = max(left, start_i)
        end_i = min(right - 1, end_i)
        if end_i < start_i:
            return

        length = end_i - start_i + 1
        if length < config.evidence_min_words:
            need = config.evidence_min_words - length
            start_i = max(left, start_i - need // 2)
            end_i = min(
                right - 1,
                start_i + config.evidence_min_words - 1,
            )
            start_i = max(
                left,
                end_i - config.evidence_min_words + 1,
            )

        if end_i - start_i + 1 > config.evidence_max_words:
            end_i = start_i + config.evidence_max_words - 1

        intervals.add((start_i, end_i))

    if anchors:
        # Tight spans around dense anchor clusters.
        for cluster_start, cluster_end in _anchor_clusters(anchors):
            add_interval(
                cluster_start - config.evidence_context_words,
                cluster_end + config.evidence_context_words,
            )

        # Wider semantic context around each anchor region gives NLI a choice
        # between the exact fact phrase and the complete local proposition.
        widths = sorted({
            config.evidence_min_words,
            min(6, config.evidence_max_words),
            min(9, config.evidence_max_words),
            config.evidence_max_words,
        })

        for anchor in sorted(anchors):
            for width in widths:
                half = width // 2
                start_i = anchor - half
                add_interval(
                    start_i,
                    start_i + width - 1,
                )

        # If all relevant anchors already fit into a compact span, include it.
        span_start = min(anchors) - config.evidence_context_words
        span_end = max(anchors) + config.evidence_context_words
        if span_end - span_start + 1 <= config.evidence_max_words:
            add_interval(span_start, span_end)

    else:
        # Semantic retrieval can succeed without lexical overlap. In that case,
        # scan a small number of compact subspans rather than returning the
        # entire 16/32-word retrieval window.
        widths = sorted({
            min(7, config.evidence_max_words),
            min(10, config.evidence_max_words),
            config.evidence_max_words,
        })
        for width in widths:
            stride = max(2, width // 2)
            for start_i in range(left, right, stride):
                add_interval(
                    start_i,
                    start_i + width - 1,
                )
                if start_i + width >= right:
                    break

    proposals = [
        _make_proposal(
            question,
            assessment,
            transcript,
            start_i,
            end_i,
            config,
        )
        for start_i, end_i in intervals
    ]

    proposals.sort(
        key=lambda proposal: (
            proposal.has_strict_contradiction,
            -proposal.heuristic_score,
            proposal.end_word - proposal.start_word,
        )
    )

    return proposals[: config.evidence_max_proposals_per_source]


def proposal_final_score(
    proposal: EvidenceProposal,
    nli: NLIResult,
    config: E1Config,
) -> float:
    length = proposal.end_word - proposal.start_word + 1

    compactness = 1.0 - (
        (length - config.evidence_min_words)
        / max(
            1,
            config.evidence_max_words - config.evidence_min_words,
        )
    )
    compactness = max(0.0, min(1.0, compactness))

    score = (
        config.evidence_entailment_weight * nli.entailment
        + config.evidence_fact_weight * proposal.exact_fact_fraction
        + config.evidence_topic_weight * proposal.topic_coverage
        + config.evidence_compactness_weight * compactness
    )

    if proposal.source == "medasr":
        score += config.evidence_medasr_tie_bonus

    if proposal.has_strict_contradiction:
        score -= 2.0

    return score


def select_refined_evidence(
    proposals: list[EvidenceProposal],
    nli_results: list[NLIResult],
    transcripts: dict[str, TranscriptView],
    config: E1Config,
) -> EvidenceSpan:
    if not proposals:
        raise ValueError("No evidence proposals")
    if len(proposals) != len(nli_results):
        raise ValueError("proposals and nli_results must have the same length")

    scored = [
        (
            proposal_final_score(proposal, nli, config),
            nli.entailment,
            -(proposal.end_word - proposal.start_word + 1),
            proposal,
        )
        for proposal, nli in zip(proposals, nli_results)
    ]

    eligible = [
        item
        for item in scored
        if (
            item[1] >= config.evidence_min_entailment
            and not item[3].has_strict_contradiction
        )
    ]

    pool = eligible if eligible else scored
    _, _, _, best = max(
        pool,
        key=lambda item: (item[0], item[1], item[2]),
    )

    transcript = transcripts[best.source]
    start = max(
        0.0,
        transcript.words[best.start_word].start
        - config.evidence_padding_before_s,
    )
    end = (
        transcript.words[best.end_word].end
        + config.evidence_padding_after_s
    )

    return EvidenceSpan(
        source=best.source,
        start_word=best.start_word,
        end_word=best.end_word,
        start=start,
        end=end,
    )


def fallback_evidence(
    assessment: CandidateAssessment,
    transcript: TranscriptView,
    config: E1Config,
) -> EvidenceSpan:
    """Safe deterministic fallback if compact-span NLI refinement fails."""

    candidate = assessment.candidate
    width = min(
        config.evidence_max_words,
        max(
            config.evidence_min_words,
            candidate.right - candidate.left,
        ),
    )
    center = (candidate.left + candidate.right - 1) // 2
    start_i = max(candidate.left, center - width // 2)
    end_i = min(
        candidate.right - 1,
        start_i + width - 1,
    )
    start_i = max(
        candidate.left,
        end_i - width + 1,
    )

    return EvidenceSpan(
        source=transcript.source,
        start_word=start_i,
        end_word=end_i,
        start=max(
            0.0,
            transcript.words[start_i].start
            - config.evidence_padding_before_s,
        ),
        end=(
            transcript.words[end_i].end
            + config.evidence_padding_after_s
        ),
    )
