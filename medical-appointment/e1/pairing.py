from __future__ import annotations

from .config import E1Config
from .schemas import CandidatePair, CandidateWindow


def temporal_iou(a: CandidateWindow, b: CandidateWindow) -> float:
    intersection = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return intersection / union if union > 0 else 0.0


def candidates_align(a: CandidateWindow | None, b: CandidateWindow | None, config: E1Config) -> bool:
    if a is None or b is None:
        return False
    return (
        temporal_iou(a, b) >= config.pair_tiou_min
        or abs(a.center - b.center) <= config.pair_center_distance_s
    )


def pair_candidates(
    medasr: list[CandidateWindow],
    parakeet: list[CandidateWindow],
    config: E1Config,
) -> list[CandidatePair]:
    pairs: list[CandidatePair] = []
    used_par: set[int] = set()

    for med in medasr:
        choices: list[tuple[float, float, int, CandidateWindow]] = []
        for j, par in enumerate(parakeet):
            if j in used_par:
                continue
            tiou = temporal_iou(med, par)
            dist = abs(med.center - par.center)
            if tiou >= config.pair_tiou_min or dist <= config.pair_center_distance_s:
                # Higher tIoU and smaller center distance are better.
                choices.append((tiou, -dist, j, par))
        if choices:
            tiou, neg_dist, j, par = max(choices, key=lambda x: (x[0], x[1]))
            used_par.add(j)
            pairs.append(CandidatePair(med, par, tiou, -neg_dist))
        else:
            pairs.append(CandidatePair(med, None, 0.0, float("inf")))

    for j, par in enumerate(parakeet):
        if j not in used_par:
            pairs.append(CandidatePair(None, par, 0.0, float("inf")))

    return pairs
