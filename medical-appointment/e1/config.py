from __future__ import annotations

from dataclasses import dataclass, field, fields
import json
from pathlib import Path
from typing import Any


@dataclass
class E1Config:
    # Multi-scale retrieval.
    retrieval_window_words: int = 32
    retrieval_secondary_window_words: int = 16
    retrieval_stride_words: int = 8
    retrieval_secondary_stride_words: int = 4
    retrieval_top_k_per_source: int = 6
    retrieval_dedup_tiou: float = 0.72
    retrieval_bm25_weight: float = 1.00
    retrieval_topic_overlap_weight: float = 1.35
    retrieval_fuzzy_overlap_weight: float = 0.55
    retrieval_fact_type_weight: float = 0.85
    retrieval_exact_fact_weight: float = 0.45

    # Cross-ASR temporal event pairing.
    pair_tiou_min: float = 0.15
    pair_center_distance_s: float = 2.5

    # NLI.
    nli_model_id: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    nli_batch_size: int = 32
    nli_max_length: int = 384

    # Candidate assessment.
    guard_contradiction_probability: float = 0.98
    fact_match_entailment_bonus: float = 0.08
    assessment_retrieval_floor: float = 0.78

    # Event-level consensus.
    event_member_yes_floor: float = 0.48
    dual_event_yes_threshold: float = 0.56
    single_source_yes_threshold: float = 0.62
    strong_entailment_threshold: float = 0.66
    exact_fact_yes_threshold: float = 0.58
    asr_fact_disagreement_yes_threshold: float = 0.64
    generic_disagreement_yes_threshold: float = 0.68
    event_yes_no_margin: float = 0.03
    dual_agreement_bonus: float = 0.05
    paired_neutral_bonus: float = 0.02
    minimum_relevance_for_yes: float = 0.10
    medasr_evidence_preference_tolerance: float = 0.05

    # Evidence proposal generation. E1 v3 no longer assumes the shortest
    # entailing phrase is annotation-like. It proposes compact, clause,
    # pause-bounded and larger fixed-width alternatives and lets ranking choose.
    evidence_min_words: int = 3
    evidence_max_words: int = 32
    evidence_context_words: int = 1
    evidence_fixed_widths: list[int] = field(
        default_factory=lambda: [5, 8, 12, 16, 24, 32]
    )
    evidence_pause_thresholds_s: list[float] = field(
        default_factory=lambda: [0.30, 0.45, 0.60]
    )
    evidence_clause_max_words: int = 32
    evidence_include_retrieval_window: bool = True
    evidence_max_proposals_per_source: int = 40

    # Evidence ranking.
    evidence_min_entailment: float = 0.48
    evidence_entailment_weight: float = 0.60
    evidence_fact_weight: float = 0.20
    evidence_topic_weight: float = 0.12
    evidence_compactness_weight: float = 0.03
    evidence_medasr_tie_bonus: float = 0.015
    evidence_padding_before_s: float = 0.03
    evidence_padding_after_s: float = 0.06

    # Step 5 adds optional kind biases. Keeping it here already makes configs
    # forward-compatible; an empty mapping changes nothing.
    evidence_kind_bias: dict[str, float] = field(default_factory=dict)

    # Robustness.
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
