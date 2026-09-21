from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .schemas import EvidenceProposal, NLIResult, TranscriptView


FEATURE_NAMES = [
    "nli_entailment",
    "nli_neutral",
    "nli_contradiction",
    "topic_coverage",
    "exact_fact_fraction",
    "heuristic_score",
    "word_count",
    "duration_s",
    "parent_yes_score",
    "parent_no_score",
    "parent_retrieval_score",
    "parent_relevance",
    "parent_topic_overlap",
    "parent_fuzzy_overlap",
    "parent_fact_type_overlap",
    "source_medasr",
    "source_parakeet",
    "kind_pause",
    "kind_clause",
    "kind_retrieval",
    "kind_anchor",
    "kind_fixed_short",
    "kind_fixed_long",
    "kind_scan",
    "pause_threshold_s",
]


def kind_group(kind: str) -> str:
    kind = (kind or "").lower()
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
            width = int(kind.split("_", 1)[1])
        except Exception:
            width = 0
        return "fixed_long" if width >= 16 else "fixed_short"
    if kind.startswith("scan_"):
        return "scan"
    return "other"


def feature_dict_from_values(
    *,
    nli_entailment: float,
    nli_neutral: float,
    nli_contradiction: float,
    topic_coverage: float,
    exact_fact_fraction: float,
    heuristic_score: float,
    word_count: int,
    duration_s: float,
    parent_yes_score: float,
    parent_no_score: float,
    parent_retrieval_score: float,
    parent_relevance: float,
    parent_topic_overlap: float,
    parent_fuzzy_overlap: float,
    parent_fact_type_overlap: float,
    source: str,
    kind: str,
    pause_threshold_s: float | None,
) -> dict[str, float]:
    group = kind_group(kind)
    return {
        "nli_entailment": float(nli_entailment),
        "nli_neutral": float(nli_neutral),
        "nli_contradiction": float(nli_contradiction),
        "topic_coverage": float(topic_coverage),
        "exact_fact_fraction": float(exact_fact_fraction),
        "heuristic_score": float(heuristic_score),
        "word_count": float(word_count),
        "duration_s": float(duration_s),
        "parent_yes_score": float(parent_yes_score),
        "parent_no_score": float(parent_no_score),
        "parent_retrieval_score": float(parent_retrieval_score),
        "parent_relevance": float(parent_relevance),
        "parent_topic_overlap": float(parent_topic_overlap),
        "parent_fuzzy_overlap": float(parent_fuzzy_overlap),
        "parent_fact_type_overlap": float(parent_fact_type_overlap),
        "source_medasr": float(source == "medasr"),
        "source_parakeet": float(source == "parakeet_v3"),
        "kind_pause": float(group == "pause"),
        "kind_clause": float(group == "clause"),
        "kind_retrieval": float(group == "retrieval"),
        "kind_anchor": float(group == "anchor"),
        "kind_fixed_short": float(group == "fixed_short"),
        "kind_fixed_long": float(group == "fixed_long"),
        "kind_scan": float(group == "scan"),
        "pause_threshold_s": float(pause_threshold_s or 0.0),
    }


def runtime_feature_dict(
    proposal: EvidenceProposal,
    nli: NLIResult,
    transcript: TranscriptView,
) -> dict[str, float]:
    parent = proposal.parent
    candidate = parent.candidate
    start = transcript.words[proposal.start_word].start
    end = transcript.words[proposal.end_word].end
    return feature_dict_from_values(
        nli_entailment=nli.entailment,
        nli_neutral=nli.neutral,
        nli_contradiction=nli.contradiction,
        topic_coverage=proposal.topic_coverage,
        exact_fact_fraction=proposal.exact_fact_fraction,
        heuristic_score=proposal.heuristic_score,
        word_count=proposal.end_word - proposal.start_word + 1,
        duration_s=max(0.0, end - start),
        parent_yes_score=parent.yes_score,
        parent_no_score=parent.no_score,
        parent_retrieval_score=candidate.retrieval_score,
        parent_relevance=candidate.relevance,
        parent_topic_overlap=candidate.topic_overlap,
        parent_fuzzy_overlap=candidate.fuzzy_overlap,
        parent_fact_type_overlap=candidate.fact_type_overlap,
        source=proposal.source,
        kind=proposal.kind,
        pause_threshold_s=proposal.pause_threshold_s,
    )


def vector_from_dict(values: dict[str, float]) -> np.ndarray:
    return np.asarray([float(values.get(name, 0.0)) for name in FEATURE_NAMES], dtype=np.float64)


class EvidenceLinearRanker:
    """Tiny standardized ridge regressor trained to predict proposal tIoU."""

    def __init__(
        self,
        means: Sequence[float],
        scales: Sequence[float],
        weights: Sequence[float],
        intercept: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.means = np.asarray(means, dtype=np.float64)
        self.scales = np.asarray(scales, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.intercept = float(intercept)
        self.metadata = dict(metadata or {})
        if not (
            len(self.means)
            == len(self.scales)
            == len(self.weights)
            == len(FEATURE_NAMES)
        ):
            raise ValueError("Ranker feature dimensions do not match FEATURE_NAMES")
        self.scales = np.where(self.scales < 1e-9, 1.0, self.scales)

    @classmethod
    def from_json(cls, path: str | Path) -> "EvidenceLinearRanker":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("feature_names") != FEATURE_NAMES:
            raise ValueError("Ranker feature schema mismatch")
        return cls(
            means=payload["means"],
            scales=payload["scales"],
            weights=payload["weights"],
            intercept=payload["intercept"],
            metadata=payload.get("metadata"),
        )

    def score_features(self, values: dict[str, float]) -> float:
        x = vector_from_dict(values)
        z = (x - self.means) / self.scales
        return float(self.intercept + z @ self.weights)

    def score_runtime(
        self,
        proposal: EvidenceProposal,
        nli: NLIResult,
        transcript: TranscriptView,
    ) -> float:
        return self.score_features(runtime_feature_dict(proposal, nli, transcript))

    def select_best(
        self,
        proposals: Sequence[EvidenceProposal],
        nlies: Sequence[NLIResult],
        transcripts: dict[str, TranscriptView],
        min_entailment: float = 0.0,
        require_no_strict_contradiction: bool = True,
    ) -> tuple[EvidenceProposal, NLIResult, float] | None:
        candidates = []
        for proposal, nli in zip(proposals, nlies):
            if require_no_strict_contradiction and proposal.has_strict_contradiction:
                continue
            if nli.entailment < min_entailment:
                continue
            transcript = transcripts.get(proposal.source)
            if transcript is None:
                continue
            score = self.score_runtime(proposal, nli, transcript)
            candidates.append((score, nli.entailment, proposal, nli))
        if not candidates:
            return None
        score, _, proposal, nli = max(candidates, key=lambda item: (item[0], item[1]))
        return proposal, nli, float(score)
