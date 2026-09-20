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


CLAUSE_END = (".", "?", "!", ";", ":")
CLAUSE_START_WORDS = {
    "but", "however", "although", "though", "then", "so", "therefore",
    "because", "while", "whereas", "instead", "otherwise",
}


def _token_match(question_tokens: set[str], word_text: str) -> bool:
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
    """Return only minimal local spans that contain exact question facts.

    The previous Step-4 implementation marked *every* word in every n-gram
    containing a matching fact. For example, if "500 mg" was the true fact,
    a five-word n-gram such as "reviewed things start metformin 500" could also
    become an anchor. That polluted the anchor cluster and caused pause-bounded
    proposals to start before an actual large speech pause.

    We first collect all matching n-grams, then discard any span that strictly
    contains a smaller matching span. The remaining spans are the local fact
    anchors we actually want.
    """

    q_facts = extract_facts(question)
    if not q_facts:
        return set()

    matching_spans: list[tuple[int, int]] = []

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
            comparison = compare_facts(q_facts, facts)

            if comparison.has_exact_match:
                matching_spans.append((i, j - 1))

    if not matching_spans:
        return set()

    # Shorter exact-fact spans are preferred. A wider span survives only when
    # there is no contained narrower span representing the same local fact.
    matching_spans = sorted(
        set(matching_spans),
        key=lambda span: (
            span[1] - span[0] + 1,
            span[0],
            span[1],
        ),
    )

    minimal: list[tuple[int, int]] = []

    for start_i, end_i in matching_spans:
        contains_existing = any(
            start_i <= kept_start
            and end_i >= kept_end
            and (start_i, end_i) != (kept_start, kept_end)
            for kept_start, kept_end in minimal
        )

        if contains_existing:
            continue

        minimal.append((start_i, end_i))

    anchors: set[int] = set()
    for start_i, end_i in minimal:
        anchors.update(range(start_i, end_i + 1))

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

    q_fact_types = {
        fact.kind
        for fact in q_facts
    }

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


def _trim_around_anchor(
    start_i: int,
    end_i: int,
    anchor_start: int,
    anchor_end: int,
    max_words: int,
) -> tuple[int, int]:
    if end_i - start_i + 1 <= max_words:
        return start_i, end_i

    center = (anchor_start + anchor_end) // 2

    start_i = max(
        start_i,
        center - max_words // 2,
    )
    end_i = start_i + max_words - 1

    if end_i < anchor_end:
        end_i = anchor_end
        start_i = end_i - max_words + 1

    return start_i, end_i


def _pause_bounds(
    transcript: TranscriptView,
    left: int,
    right: int,
    anchor_start: int,
    anchor_end: int,
    threshold: float,
    max_words: int,
) -> tuple[int, int]:
    """Expand an anchor until a sufficiently large inter-word pause.

    `left` is inclusive and `right` is exclusive.
    """

    start_i = anchor_start

    while start_i > left:
        gap = (
            transcript.words[start_i].start
            - transcript.words[start_i - 1].end
        )

        if gap >= threshold:
            break

        start_i -= 1

    end_i = anchor_end

    while end_i + 1 < right:
        gap = (
            transcript.words[end_i + 1].start
            - transcript.words[end_i].end
        )

        if gap >= threshold:
            break

        end_i += 1

    # This can only shrink the detected pause-bounded span; it never crosses
    # the detected pause boundary.
    return _trim_around_anchor(
        start_i,
        end_i,
        anchor_start,
        anchor_end,
        max_words,
    )


def _is_clause_boundary_before(
    word_text: str,
) -> bool:
    stripped = word_text.strip()
    return stripped.endswith(CLAUSE_END)


def _is_clause_boundary_at(
    word_text: str,
) -> bool:
    token = normalize_text(word_text).split()
    return bool(
        token
        and token[0] in CLAUSE_START_WORDS
    )


def _clause_bounds(
    transcript: TranscriptView,
    left: int,
    right: int,
    anchor_start: int,
    anchor_end: int,
    max_words: int,
) -> tuple[int, int]:
    start_i = anchor_start

    while start_i > left:
        if _is_clause_boundary_before(
            transcript.words[start_i - 1].text
        ):
            break

        if _is_clause_boundary_at(
            transcript.words[start_i].text
        ):
            break

        start_i -= 1

    end_i = anchor_end

    while end_i + 1 < right:
        if _is_clause_boundary_before(
            transcript.words[end_i].text
        ):
            break

        if _is_clause_boundary_at(
            transcript.words[end_i + 1].text
        ):
            break

        end_i += 1

    return _trim_around_anchor(
        start_i,
        end_i,
        anchor_start,
        anchor_end,
        max_words,
    )


def _make_proposal(
    question: str,
    assessment: CandidateAssessment,
    transcript: TranscriptView,
    start_i: int,
    end_i: int,
    config: E1Config,
    kind: str,
    pause_threshold_s: float | None = None,
) -> EvidenceProposal:
    text = " ".join(
        word.text
        for word in transcript.words[start_i : end_i + 1]
    )

    (
        topic_coverage,
        exact_fraction,
        strict,
    ) = _proposal_metrics(
        question,
        text,
    )

    length = end_i - start_i + 1

    compactness = 1.0 - (
        (length - config.evidence_min_words)
        / max(
            1,
            config.evidence_max_words
            - config.evidence_min_words,
        )
    )
    compactness = max(
        0.0,
        min(1.0, compactness),
    )

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
        kind=kind,
        pause_threshold_s=pause_threshold_s,
    )


def build_evidence_proposals(
    question: str,
    assessment: CandidateAssessment,
    transcript: TranscriptView,
    config: E1Config,
) -> list[EvidenceProposal]:
    """Generate annotation-like alternatives around the supporting event.

    Proposal families:
    - anchor_cluster
    - fixed_N
    - pause_X
    - clause
    - retrieval_window
    - scan_N
    """

    candidate = assessment.candidate
    left = candidate.left
    right = candidate.right

    if right <= left:
        return []

    q_tokens = set(
        topic_tokens(question)
    )

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

    anchors = (
        lexical_anchors
        | fact_anchors
    )

    specs: dict[
        tuple[int, int, str],
        float | None,
    ] = {}

    def add_interval(
        start_i: int,
        end_i: int,
        kind: str,
        anchor_start: int | None = None,
        anchor_end: int | None = None,
        pause_threshold_s: float | None = None,
        hard_boundaries: bool = False,
    ) -> None:
        """Register a proposal interval.

        `hard_boundaries=True` is used for pause/clause proposals. These
        intervals are allowed to be shorter than evidence_min_words rather than
        being expanded across the very pause/clause boundary that defined them.
        """

        start_i = max(
            left,
            start_i,
        )
        end_i = min(
            right - 1,
            end_i,
        )

        if end_i < start_i:
            return

        length = (
            end_i - start_i + 1
        )

        if (
            length
            < config.evidence_min_words
            and not hard_boundaries
        ):
            need = (
                config.evidence_min_words
                - length
            )

            start_i = max(
                left,
                start_i - need // 2,
            )

            end_i = min(
                right - 1,
                start_i
                + config.evidence_min_words
                - 1,
            )

            start_i = max(
                left,
                end_i
                - config.evidence_min_words
                + 1,
            )

        if (
            end_i - start_i + 1
            > config.evidence_max_words
        ):
            a0 = (
                anchor_start
                if anchor_start is not None
                else start_i
            )
            a1 = (
                anchor_end
                if anchor_end is not None
                else a0
            )

            start_i, end_i = _trim_around_anchor(
                start_i,
                end_i,
                a0,
                a1,
                config.evidence_max_words,
            )

            start_i = max(
                left,
                start_i,
            )
            end_i = min(
                right - 1,
                end_i,
            )

        specs[
            (
                start_i,
                end_i,
                kind,
            )
        ] = pause_threshold_s

    clusters = _anchor_clusters(
        anchors
    )

    if clusters:
        for (
            cluster_start,
            cluster_end,
        ) in clusters:
            add_interval(
                cluster_start
                - config.evidence_context_words,
                cluster_end
                + config.evidence_context_words,
                "anchor_cluster",
                cluster_start,
                cluster_end,
            )

            center = (
                cluster_start
                + cluster_end
            ) // 2

            for width in sorted(
                set(
                    config.evidence_fixed_widths
                )
            ):
                width = min(
                    int(width),
                    config.evidence_max_words,
                )

                if (
                    width
                    < config.evidence_min_words
                ):
                    continue

                start_i = (
                    center
                    - width // 2
                )

                add_interval(
                    start_i,
                    start_i + width - 1,
                    f"fixed_{width}",
                    cluster_start,
                    cluster_end,
                )

            # IMPORTANT:
            # Each local cluster gets its own pause-bounded proposal.
            # We never use min(all_anchors)/max(all_anchors) here.
            for threshold in sorted(
                set(
                    config.evidence_pause_thresholds_s
                )
            ):
                p_start, p_end = _pause_bounds(
                    transcript,
                    left,
                    right,
                    cluster_start,
                    cluster_end,
                    float(threshold),
                    config.evidence_max_words,
                )

                add_interval(
                    p_start,
                    p_end,
                    f"pause_{float(threshold):.2f}",
                    cluster_start,
                    cluster_end,
                    float(threshold),
                    hard_boundaries=True,
                )

            c_start, c_end = _clause_bounds(
                transcript,
                left,
                right,
                cluster_start,
                cluster_end,
                min(
                    config.evidence_clause_max_words,
                    config.evidence_max_words,
                ),
            )

            add_interval(
                c_start,
                c_end,
                "clause",
                cluster_start,
                cluster_end,
                hard_boundaries=True,
            )

        all_start = (
            min(anchors)
            - config.evidence_context_words
        )
        all_end = (
            max(anchors)
            + config.evidence_context_words
        )

        add_interval(
            all_start,
            all_end,
            "anchor_all",
            min(anchors),
            max(anchors),
        )

    else:
        candidate_length = (
            right - left
        )

        centers = {
            (left + right - 1) // 2,
            left + candidate_length // 3,
            left
            + (2 * candidate_length) // 3,
        }

        for width in sorted(
            set(
                config.evidence_fixed_widths
            )
        ):
            width = min(
                int(width),
                config.evidence_max_words,
                candidate_length,
            )

            if (
                width
                < config.evidence_min_words
            ):
                continue

            for center in centers:
                start_i = (
                    center
                    - width // 2
                )

                add_interval(
                    start_i,
                    start_i + width - 1,
                    f"scan_{width}",
                    center,
                    center,
                )

    if (
        config.evidence_include_retrieval_window
    ):
        fallback_center = (
            (left + right - 1) // 2
        )

        add_interval(
            left,
            right - 1,
            f"retrieval_{right - left}",
            (
                min(anchors)
                if anchors
                else fallback_center
            ),
            (
                max(anchors)
                if anchors
                else fallback_center
            ),
        )

    proposals = [
        _make_proposal(
            question,
            assessment,
            transcript,
            start_i,
            end_i,
            config,
            kind,
            pause_threshold_s,
        )
        for (
            start_i,
            end_i,
            kind,
        ), pause_threshold_s
        in specs.items()
    ]

    proposals.sort(
        key=lambda proposal: (
            proposal.has_strict_contradiction,
            -proposal.heuristic_score,
            proposal.end_word
            - proposal.start_word,
        )
    )

    return proposals[
        : config.evidence_max_proposals_per_source
    ]


def _kind_group(
    kind: str,
) -> str:
    if kind.startswith("pause_"):
        return "pause"

    if kind.startswith("clause"):
        return "clause"

    if kind.startswith("retrieval"):
        return "retrieval"

    if kind.startswith("anchor"):
        return "anchor"

    if kind.startswith("fixed_"):
        try:
            width = int(
                kind.split(
                    "_",
                    1,
                )[1]
            )
        except Exception:
            width = 0

        return (
            "fixed_long"
            if width >= 16
            else "fixed_short"
        )

    if kind.startswith("scan_"):
        return "scan"

    return kind


def proposal_final_score(
    proposal: EvidenceProposal,
    nli: NLIResult,
    config: E1Config,
) -> float:
    length = (
        proposal.end_word
        - proposal.start_word
        + 1
    )

    compactness = 1.0 - (
        (length - config.evidence_min_words)
        / max(
            1,
            config.evidence_max_words
            - config.evidence_min_words,
        )
    )

    compactness = max(
        0.0,
        min(1.0, compactness),
    )

    score = (
        config.evidence_entailment_weight
        * nli.entailment
        + config.evidence_fact_weight
        * proposal.exact_fact_fraction
        + config.evidence_topic_weight
        * proposal.topic_coverage
        + config.evidence_compactness_weight
        * compactness
    )

    if (
        proposal.source
        == "medasr"
    ):
        score += (
            config.evidence_medasr_tie_bonus
        )

    score += float(
        config.evidence_kind_bias.get(
            _kind_group(
                proposal.kind
            ),
            0.0,
        )
    )

    if (
        proposal.has_strict_contradiction
    ):
        score -= 2.0

    return score


def select_refined_evidence(
    proposals: list[EvidenceProposal],
    nli_results: list[NLIResult],
    transcripts: dict[
        str,
        TranscriptView,
    ],
    config: E1Config,
) -> EvidenceSpan:
    if not proposals:
        raise ValueError(
            "No evidence proposals"
        )

    if (
        len(proposals)
        != len(nli_results)
    ):
        raise ValueError(
            "proposals and nli_results "
            "must have the same length"
        )

    scored = [
        (
            proposal_final_score(
                proposal,
                nli,
                config,
            ),
            nli.entailment,
            -(
                proposal.end_word
                - proposal.start_word
                + 1
            ),
            proposal,
        )
        for proposal, nli
        in zip(
            proposals,
            nli_results,
        )
    ]

    eligible = [
        item
        for item in scored
        if (
            item[1]
            >= config.evidence_min_entailment
            and not item[
                3
            ].has_strict_contradiction
        )
    ]

    pool = (
        eligible
        if eligible
        else scored
    )

    _, _, _, best = max(
        pool,
        key=lambda item: (
            item[0],
            item[1],
            item[2],
        ),
    )

    transcript = transcripts[
        best.source
    ]

    start = max(
        0.0,
        transcript.words[
            best.start_word
        ].start
        - config.evidence_padding_before_s,
    )

    end = (
        transcript.words[
            best.end_word
        ].end
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
    candidate = (
        assessment.candidate
    )

    width = min(
        config.evidence_max_words,
        max(
            config.evidence_min_words,
            candidate.right
            - candidate.left,
        ),
    )

    center = (
        candidate.left
        + candidate.right
        - 1
    ) // 2

    start_i = max(
        candidate.left,
        center - width // 2,
    )

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
            transcript.words[
                start_i
            ].start
            - config.evidence_padding_before_s,
        ),
        end=(
            transcript.words[
                end_i
            ].end
            + config.evidence_padding_after_s
        ),
    )
