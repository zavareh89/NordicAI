from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path


@dataclass
class E2_1RankerConfig:
    model_path: str = "config/e2_1_evidence_ranker.json"
    calibration_path: str = "config/e2_1_boundary_calibration.json"
    min_nli_entailment: float = 0.25
    require_no_strict_fact_contradiction: bool = True

    @classmethod
    def from_json(cls, path: str | Path | None) -> "E2_1RankerConfig":
        if path is None:
            return cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        valid = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - valid)
        if unknown:
            raise ValueError(f"Unknown E2.1 ranker config keys: {unknown}")
        return cls(**data)
