from __future__ import annotations

import hashlib
from typing import Iterable


def temporal_iou(gs, ge, ps, pe) -> float:
    if None in (gs, ge, ps, pe):
        return 0.0
    gs, ge, ps, pe = map(float, (gs, ge, ps, pe))
    if ge <= gs or pe <= ps:
        return 0.0
    inter = max(0.0, min(ge, pe) - max(gs, ps))
    union = max(ge, pe) - min(gs, ps)
    return inter / union if union > 0 else 0.0


def shifted_tiou(record, start_shift: float, end_shift: float) -> float:
    start = max(0.0, float(record["raw_start"]) + start_shift)
    end = float(record["raw_end"]) + end_shift
    if end <= start:
        return 0.0
    return temporal_iou(
        record["gold_start"], record["gold_end"], start, end
    )


def mean_tiou(records: Iterable[dict], start_shift: float, end_shift: float) -> float:
    values = [shifted_tiou(r, start_shift, end_shift) for r in records]
    return sum(values) / len(values) if values else 0.0


def best_shift(records: list[dict], grid: list[float]) -> tuple[float, float, float]:
    if not records:
        return 0.0, 0.0, 0.0
    best = (-1.0, 0.0, 0.0)
    for start_shift in grid:
        for end_shift in grid:
            score = mean_tiou(records, start_shift, end_shift)
            if score > best[0]:
                best = (score, start_shift, end_shift)
    return best[1], best[2], best[0]


def fold_for(transcript_id: str, folds: int) -> int:
    digest = hashlib.sha1(str(transcript_id).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds
