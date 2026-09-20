from __future__ import annotations

from .config import E1Config
from .schemas import (
    CandidateAssessment,
    CandidatePair,
    CandidateWindow,
    EvidenceEvent,
)


def temporal_iou(a: CandidateWindow, b: CandidateWindow) -> float:
    intersection = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return intersection / union if union > 0 else 0.0


def candidates_align(
    a: CandidateWindow | None,
    b: CandidateWindow | None,
    config: E1Config,
) -> bool:
    if a is None or b is None:
        return False
    return (
        temporal_iou(a, b) >= config.pair_tiou_min
        or abs(a.center - b.center) <= config.pair_center_distance_s
    )


def _alignment_edges(
    medasr: list[CandidateWindow],
    parakeet: list[CandidateWindow],
    config: E1Config,
) -> list[tuple[float, float, int, int]]:
    edges: list[tuple[float, float, int, int]] = []

    for i, med in enumerate(medasr):
        for j, par in enumerate(parakeet):
            tiou = temporal_iou(med, par)
            distance = abs(med.center - par.center)

            if (
                tiou < config.pair_tiou_min
                and distance > config.pair_center_distance_s
            ):
                continue

            # Prefer temporal overlap first, then close centers and good
            # retrieval candidates. This avoids depending on source list order.
            retrieval_bonus = 0.10 * min(
                med.retrieval_score,
                par.retrieval_score,
            )
            distance_score = max(
                0.0,
                1.0 - distance / max(config.pair_center_distance_s, 1e-6),
            )
            alignment_score = tiou + 0.15 * distance_score + retrieval_bonus
            edges.append((alignment_score, tiou, i, j))

    return sorted(edges, reverse=True)


def pair_candidates(
    medasr: list[CandidateWindow],
    parakeet: list[CandidateWindow],
    config: E1Config,
) -> list[CandidatePair]:
    """One-to-one temporal pairing retained for diagnostics/tests."""

    pairs: list[CandidatePair] = []
    used_med: set[int] = set()
    used_par: set[int] = set()

    for _, tiou, i, j in _alignment_edges(medasr, parakeet, config):
        if i in used_med or j in used_par:
            continue
        used_med.add(i)
        used_par.add(j)
        med = medasr[i]
        par = parakeet[j]
        pairs.append(
            CandidatePair(
                medasr=med,
                parakeet=par,
                temporal_iou=tiou,
                center_distance_s=abs(med.center - par.center),
            )
        )

    for i, med in enumerate(medasr):
        if i not in used_med:
            pairs.append(CandidatePair(med, None, 0.0, float("inf")))

    for j, par in enumerate(parakeet):
        if j not in used_par:
            pairs.append(CandidatePair(None, par, 0.0, float("inf")))

    return pairs


def pair_assessments(
    medasr: list[CandidateAssessment],
    parakeet: list[CandidateAssessment],
    config: E1Config,
) -> list[EvidenceEvent]:
    """Build the event objects that E1 v2 actually reasons over."""

    med_windows = [item.candidate for item in medasr]
    par_windows = [item.candidate for item in parakeet]

    used_med: set[int] = set()
    used_par: set[int] = set()
    events: list[EvidenceEvent] = []

    for _, tiou, i, j in _alignment_edges(
        med_windows,
        par_windows,
        config,
    ):
        if i in used_med or j in used_par:
            continue
        used_med.add(i)
        used_par.add(j)
        med = medasr[i]
        par = parakeet[j]
        events.append(
            EvidenceEvent(
                medasr=med,
                parakeet=par,
                temporal_iou=tiou,
                center_distance_s=abs(
                    med.candidate.center - par.candidate.center
                ),
            )
        )

    for i, med in enumerate(medasr):
        if i not in used_med:
            events.append(
                EvidenceEvent(
                    medasr=med,
                    parakeet=None,
                    temporal_iou=0.0,
                    center_distance_s=float("inf"),
                )
            )

    for j, par in enumerate(parakeet):
        if j not in used_par:
            events.append(
                EvidenceEvent(
                    medasr=None,
                    parakeet=par,
                    temporal_iou=0.0,
                    center_distance_s=float("inf"),
                )
            )

    return events
