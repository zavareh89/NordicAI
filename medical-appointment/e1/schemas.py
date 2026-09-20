from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

SourceName = Literal["medasr", "parakeet_v3"]


@dataclass(frozen=True)
class NormalizedWord:
    index: int
    text: str
    normalized: str
    start: float
    end: float
    confidence: float | None = None


@dataclass
class TranscriptView:
    source: SourceName
    text: str
    words: list[NormalizedWord]


@dataclass(frozen=True)
class Fact:
    kind: str
    value: Any
    unit: str | None = None
    raw: str = ""


@dataclass(frozen=True)
class FactComparison:
    matched: tuple[str, ...] = ()
    mismatched: tuple[str, ...] = ()

    @property
    def has_exact_match(self) -> bool:
        return bool(self.matched)

    @property
    def has_strict_contradiction(self) -> bool:
        return bool(self.mismatched)


@dataclass
class CandidateWindow:
    source: SourceName
    left: int
    right: int  # exclusive global word index
    start: float
    end: float
    text: str
    normalized_text: str
    facts: list[Fact] = field(default_factory=list)
    raw_score: float = 0.0
    retrieval_score: float = 0.0
    rank: int = 0
    topic_overlap: float = 0.0
    fuzzy_overlap: float = 0.0
    fact_type_overlap: float = 0.0

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.source, self.left, self.right)

    @property
    def center(self) -> float:
        return 0.5 * (self.start + self.end)

    @property
    def relevance(self) -> float:
        return max(self.topic_overlap, self.fuzzy_overlap, self.fact_type_overlap)


@dataclass(frozen=True)
class CandidatePair:
    medasr: CandidateWindow | None
    parakeet: CandidateWindow | None
    temporal_iou: float
    center_distance_s: float


@dataclass(frozen=True)
class NLIResult:
    entailment: float
    neutral: float
    contradiction: float

    @property
    def label(self) -> str:
        values = {
            "entailment": self.entailment,
            "neutral": self.neutral,
            "contradiction": self.contradiction,
        }
        return max(values, key=values.get)


@dataclass
class CandidateAssessment:
    candidate: CandidateWindow
    nli: NLIResult
    facts: FactComparison
    yes_score: float
    no_score: float


@dataclass(frozen=True)
class EvidenceEvent:
    """One spoken event represented by zero/one candidate from each ASR."""

    medasr: CandidateAssessment | None
    parakeet: CandidateAssessment | None
    temporal_iou: float
    center_distance_s: float

    @property
    def members(self) -> tuple[CandidateAssessment, ...]:
        return tuple(
            item
            for item in (self.medasr, self.parakeet)
            if item is not None
        )

    @property
    def paired(self) -> bool:
        return self.medasr is not None and self.parakeet is not None


@dataclass
class SourceVote:
    # Retained for backward-compatible traces. E1 v2 makes the final decision
    # from EvidenceEvent objects, not from global source votes.
    source: SourceName
    label: Literal["yes", "no", "neutral"]
    yes_score: float
    no_score: float
    yes_candidate: CandidateAssessment | None
    no_candidate: CandidateAssessment | None


@dataclass
class QuestionDecision:
    answer: bool
    confidence: float
    candidate: CandidateAssessment | None
    reason: str
    event: EvidenceEvent | None = None
    med_vote: SourceVote | None = None
    par_vote: SourceVote | None = None


@dataclass(frozen=True)
class EvidenceProposal:
    source: SourceName
    start_word: int
    end_word: int  # inclusive
    text: str
    topic_coverage: float
    exact_fact_fraction: float
    has_strict_contradiction: bool
    heuristic_score: float
    parent: CandidateAssessment


@dataclass(frozen=True)
class EvidenceSpan:
    source: SourceName
    start_word: int
    end_word: int  # inclusive
    start: float
    end: float


@dataclass
class E1Prediction:
    answer: bool
    evidence_start: float | None
    evidence_end: float | None
    confidence: float = 0.0
    reason: str = ""
    source: str | None = None


@dataclass
class QuestionTrace:
    question: str
    candidates: dict[str, list[CandidateWindow]]
    assessments: list[CandidateAssessment]
    events: list[EvidenceEvent]
    decision: QuestionDecision
    evidence: EvidenceSpan | None
