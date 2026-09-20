from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Any


@dataclass
class E1Config:
    # Retrieval
    retrieval_window_words: int = 28
    retrieval_stride_words: int = 8
    retrieval_top_k_per_source: int = 4
    retrieval_bm25_weight: float = 1.00
    retrieval_topic_overlap_weight: float = 1.35
    retrieval_fuzzy_overlap_weight: float = 0.55
    retrieval_fact_type_weight: float = 0.85
    retrieval_exact_fact_weight: float = 0.45

    # Cross-ASR temporal pairing
    pair_tiou_min: float = 0.15
    pair_center_distance_s: float = 2.5

    # NLI
    nli_model_id: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    nli_batch_size: int = 32
    nli_max_length: int = 384
    nli_hypothesis_template: str = "The correct answer is yes to this question: {question}"

    # Source voting / consensus
    source_yes_threshold: float = 0.50
    source_no_threshold: float = 0.55
    source_vote_margin: float = 0.04
    strong_entailment_threshold: float = 0.72
    single_source_yes_threshold: float = 0.74
    yes_vs_no_exact_fact_threshold: float = 0.70
    guard_contradiction_probability: float = 0.98
    fact_match_entailment_bonus: float = 0.08
    medasr_evidence_preference_tolerance: float = 0.08

    # Evidence tightening
    evidence_context_words: int = 2
    evidence_max_words: int = 18
    evidence_padding_before_s: float = 0.05
    evidence_padding_after_s: float = 0.10

    # Robustness
    catastrophic_fallback_answer: bool = True

    @classmethod
    def from_json(cls, path: str | Path | None) -> "E1Config":
        if path is None:
            return cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        valid = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - valid)
        if unknown:
            raise ValueError(f"Unknown E1 config keys: {unknown}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}
