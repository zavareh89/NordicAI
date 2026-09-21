from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Any


@dataclass
class E2_1BConfig:
    """Direct word-index evidence selection.

    E2.1-B keeps the E1 classification decision frozen. The LLM only chooses
    one contiguous word range inside the already-selected temporal evidence
    event. Timestamps are always reconstructed deterministically from ASR words.
    """

    model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_tasks_per_generation: int = 10
    max_input_tokens: int = 8192
    max_new_tokens: int = 192

    min_span_words: int = 2
    max_span_words: int = 32
    event_context_words: int = 0
    max_proposal_hints: int = 5

    require_no_strict_fact_contradiction: bool = True
    require_topic_or_exact_fact: bool = False
    min_topic_coverage: float = 0.05

    padding_before_s: float = 0.03
    padding_after_s: float = 0.06

    @classmethod
    def from_json(cls, path: str | Path | None) -> "E2_1BConfig":
        if path is None:
            return cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        valid = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - valid)
        if unknown:
            raise ValueError(f"Unknown E2.1-B config keys: {unknown}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}
