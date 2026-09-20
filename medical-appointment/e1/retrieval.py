from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
import math

from .config import E1Config
from .facts import compare_facts, extract_facts, fact_type_overlap
from .schemas import CandidateWindow, TranscriptView
from .textnorm import normalize_text, normalize_tokens, topic_tokens


def _bm25_scores(query_tokens: list[str], docs: list[list[str]]) -> list[float]:
    if not docs:
        return []
    n = len(docs)
    avgdl = sum(len(doc) for doc in docs) / max(1, n)
    dfs: Counter[str] = Counter()
    for doc in docs:
        dfs.update(set(doc))

    k1, b = 1.5, 0.75
    scores: list[float] = []
    for doc in docs:
        tf = Counter(doc)
        score = 0.0
        for term in query_tokens:
            if tf[term] == 0:
                continue
            df = dfs[term]
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            freq = tf[term]
            denom = freq + k1 * (1.0 - b + b * len(doc) / max(avgdl, 1e-9))
            score += idf * (freq * (k1 + 1.0)) / denom
        scores.append(score)
    return scores


def _fuzzy_overlap(query: list[str], doc: list[str]) -> float:
    q = [t for t in query if len(t) >= 4]
    d = [t for t in doc if len(t) >= 4]
    if not q:
        return 0.0
    hits = 0
    for qt in q:
        if qt in d:
            hits += 1
            continue
        if any(SequenceMatcher(None, qt, dt).ratio() >= 0.84 for dt in d):
            hits += 1
    return hits / len(q)


def _temporal_iou(a: CandidateWindow, b: CandidateWindow) -> float:
    intersection = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return intersection / union if union > 0 else 0.0


def build_windows(
    transcript: TranscriptView,
    size: int,
    stride: int,
) -> list[CandidateWindow]:
    words = transcript.words
    if not words:
        return []

    starts = list(range(0, len(words), stride))
    if starts and starts[-1] + size < len(words):
        starts.append(max(0, len(words) - size))

    windows: list[CandidateWindow] = []
    seen: set[tuple[int, int]] = set()

    for left in starts:
        right = min(len(words), left + size)
        if right <= left or (left, right) in seen:
            continue
        seen.add((left, right))
        chunk = words[left:right]
        text = " ".join(w.text for w in chunk)
        windows.append(
            CandidateWindow(
                source=transcript.source,
                left=left,
                right=right,
                start=chunk[0].start,
                end=chunk[-1].end,
                text=text,
                normalized_text=normalize_text(text),
                facts=extract_facts(text),
            )
        )
        if right == len(words):
            break

    return windows


def _score_windows(
    question: str,
    windows: list[CandidateWindow],
    config: E1Config,
) -> None:
    if not windows:
        return

    full_q = normalize_tokens(question)
    topic_q = topic_tokens(question)
    q_facts = extract_facts(question)

    full_docs = [normalize_tokens(w.text) for w in windows]
    topic_docs = [topic_tokens(w.text) for w in windows]
    bm25_full = _bm25_scores(full_q, full_docs)
    bm25_topic = _bm25_scores(topic_q, topic_docs)

    topic_set = set(topic_q)

    for i, window in enumerate(windows):
        doc_topic = set(topic_docs[i])
        topic_overlap = len(topic_set & doc_topic) / max(1, len(topic_set))
        fuzzy = _fuzzy_overlap(topic_q, topic_docs[i])
        type_overlap = fact_type_overlap(q_facts, window.facts)
        comparison = compare_facts(q_facts, window.facts)
        exact_fact = min(
            1.0,
            len(comparison.matched) / max(1, len({f.kind for f in q_facts})),
        )

        window.topic_overlap = topic_overlap
        window.fuzzy_overlap = fuzzy
        window.fact_type_overlap = type_overlap

        # Topic-only BM25 prevents a deliberately wrong question value from
        # suppressing the true passage. Full BM25 still rewards exact values.
        window.raw_score = (
            config.retrieval_bm25_weight
            * (0.55 * bm25_topic[i] + 0.45 * bm25_full[i])
            + config.retrieval_topic_overlap_weight * topic_overlap
            + config.retrieval_fuzzy_overlap_weight * fuzzy
            + config.retrieval_fact_type_weight * type_overlap
            + config.retrieval_exact_fact_weight * exact_fact
        )


def _deduplicate_ranked(
    ranked: list[CandidateWindow],
    top_k: int,
    tiou_threshold: float,
) -> list[CandidateWindow]:
    """Prefer diverse spoken events instead of six overlapping copies of one hit."""

    selected: list[CandidateWindow] = []
    deferred: list[CandidateWindow] = []

    for candidate in ranked:
        duplicate = any(
            _temporal_iou(candidate, existing) >= tiou_threshold
            for existing in selected
        )
        if duplicate:
            deferred.append(candidate)
            continue
        selected.append(candidate)
        if len(selected) >= top_k:
            break

    # If the conversation has very few distinct events, fill remaining slots
    # with the best deferred windows so NLI still sees alternate context sizes.
    if len(selected) < top_k:
        seen_keys = {c.key for c in selected}
        for candidate in deferred:
            if candidate.key in seen_keys:
                continue
            selected.append(candidate)
            seen_keys.add(candidate.key)
            if len(selected) >= top_k:
                break

    return selected


def retrieve(
    question: str,
    transcript: TranscriptView,
    config: E1Config,
) -> list[CandidateWindow]:
    primary = build_windows(
        transcript,
        config.retrieval_window_words,
        config.retrieval_stride_words,
    )

    secondary: list[CandidateWindow] = []
    if (
        config.retrieval_secondary_window_words > 0
        and config.retrieval_secondary_window_words != config.retrieval_window_words
    ):
        secondary = build_windows(
            transcript,
            config.retrieval_secondary_window_words,
            config.retrieval_secondary_stride_words,
        )

    # Keep both scales. Exact duplicate word ranges can occur near transcript
    # ends, so remove only exact duplicates before scoring.
    windows: list[CandidateWindow] = []
    seen: set[tuple[int, int]] = set()
    for window in primary + secondary:
        key = (window.left, window.right)
        if key in seen:
            continue
        seen.add(key)
        windows.append(window)

    if not windows:
        return []

    _score_windows(question, windows, config)
    ranked = sorted(windows, key=lambda w: (-w.raw_score, w.start, w.right - w.left))

    top = _deduplicate_ranked(
        ranked,
        config.retrieval_top_k_per_source,
        config.retrieval_dedup_tiou,
    )

    best = max((w.raw_score for w in top), default=0.0)
    for rank, window in enumerate(top, start=1):
        window.rank = rank
        if best > 0:
            window.retrieval_score = max(
                0.0,
                min(1.0, window.raw_score / best),
            )
        else:
            window.retrieval_score = max(0.20, 1.0 - 0.15 * (rank - 1))

    return top
