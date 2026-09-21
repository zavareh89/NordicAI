from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path


@dataclass
class E2_1EnsembleConfig:
    ranker_model_path: str = "config/e2_1_evidence_ranker.json"
    calibration_path: str = "config/e2_1_boundary_calibration.json"
    ranker_min_nli_entailment: float = 0.25
    agreement_tiou_threshold: float = 0.30
    minimum_llm_entailment: float = 0.45
    llm_entailment_advantage: float = 0.08
    require_no_strict_fact_contradiction: bool = True

    @classmethod
    def from_json(cls, path: str | Path | None) -> "E2_1EnsembleConfig":
        if path is None:
            return cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        valid = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - valid)
        if unknown:
            raise ValueError(f"Unknown E2.1 ensemble config keys: {unknown}")
        return cls(**data)
