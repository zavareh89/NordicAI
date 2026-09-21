from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .schemas import EvidenceSpan


@dataclass(frozen=True)
class BoundaryShift:
    start_shift_s: float = 0.0
    end_shift_s: float = 0.0


class BoundaryCalibrator:
    def __init__(
        self,
        global_shift: BoundaryShift | None = None,
        by_source: dict[str, BoundaryShift] | None = None,
    ) -> None:
        self.global_shift = global_shift or BoundaryShift()
        self.by_source = dict(by_source or {})

    @classmethod
    def from_json(cls, path: str | Path | None) -> "BoundaryCalibrator":
        if path is None:
            return cls()
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        global_data = payload.get("global") or {}
        by_source_data = payload.get("by_source") or {}
        return cls(
            global_shift=BoundaryShift(
                float(global_data.get("start_shift_s", 0.0)),
                float(global_data.get("end_shift_s", 0.0)),
            ),
            by_source={
                str(source): BoundaryShift(
                    float(values.get("start_shift_s", 0.0)),
                    float(values.get("end_shift_s", 0.0)),
                )
                for source, values in by_source_data.items()
            },
        )

    def shift_for(self, source: str) -> BoundaryShift:
        return self.by_source.get(source, self.global_shift)

    def apply(self, span: EvidenceSpan) -> EvidenceSpan:
        shift = self.shift_for(span.source)
        start = max(0.0, span.start + shift.start_shift_s)
        end = span.end + shift.end_shift_s
        if end <= start:
            return span
        return EvidenceSpan(
            source=span.source,
            start_word=span.start_word,
            end_word=span.end_word,
            start=start,
            end=end,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "global": {
                "start_shift_s": self.global_shift.start_shift_s,
                "end_shift_s": self.global_shift.end_shift_s,
            },
            "by_source": {
                source: {
                    "start_shift_s": shift.start_shift_s,
                    "end_shift_s": shift.end_shift_s,
                }
                for source, shift in self.by_source.items()
            },
        }
