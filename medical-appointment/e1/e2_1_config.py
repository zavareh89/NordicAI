from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Any


@dataclass
class E2_1Config:
    """Evidence-only LLM reranker configuration.

    E2.1 does not change retrieval, NLI, consensus, or the final YES/NO label.
    It only chooses among evidence proposals for questions that E1 already
    classified as YES.
    """

    model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_candidates_per_question: int = 8
    max_tasks_per_generation: int = 10
    max_input_tokens: int = 8192
    max_new_tokens: int = 128

    # Safety filters for spans exposed to the LLM.
    min_nli_entailment: float = 0.30
    min_topic_coverage: float = 0.0
    max_baseline_score_drop: float = 0.45

    require_no_strict_fact_contradiction: bool = True
    always_include_e1_choice: bool = True
    prefer_candidate_family_diversity: bool = True

    # Kept separate from E1 so E2.1 can be tuned independently later.
    padding_before_s: float = 0.03
    padding_after_s: float = 0.06

    @classmethod
    def from_json(cls, path: str | Path | None) -> "E2_1Config":
        if path is None:
            return cls()

        data = json.loads(
            Path(path).read_text(encoding="utf-8")
        )
        valid = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - valid)

        if unknown:
            raise ValueError(
                f"Unknown E2.1 config keys: {unknown}"
            )

        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
        }
