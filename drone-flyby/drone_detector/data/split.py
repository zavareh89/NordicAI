from __future__ import annotations

from typing import Iterable

from ..types import SplitDefinition


def all_frames_split(frame_ids: Iterable[int]) -> SplitDefinition:
    """Use every labeled frame for final training.

    No statistically independent validation set exists in this mode. Exporters may
    create a *framework-only* validation mirror when a backend requires a valid/
    directory, but those mirrored metrics must never be reported as held-out results.
    """
    ids = tuple(sorted(set(int(v) for v in frame_ids)))
    if not ids:
        raise ValueError("At least one frame is required")
    return SplitDefinition(train_frame_ids=ids, val_frame_ids=(), purged_frame_ids=())


def blocked_holdout_split(
    frame_ids: Iterable[int],
    *,
    val_count: int = 5,
    purge_gap: int = 1,
    val_position: str = "end",
) -> SplitDefinition:
    ids = tuple(sorted(set(int(v) for v in frame_ids)))
    if len(ids) < 2:
        raise ValueError("At least two frames are required for a train/validation split")
    if not 1 <= val_count < len(ids):
        raise ValueError(f"val_count must be in [1, {len(ids)-1}], got {val_count}")
    if purge_gap < 0:
        raise ValueError("purge_gap must be >= 0")

    if val_position == "end":
        val_start = len(ids) - val_count
    elif val_position == "start":
        val_start = 0
    elif val_position == "middle":
        val_start = (len(ids) - val_count) // 2
    else:
        raise ValueError("val_position must be one of: start, middle, end")
    val_end = val_start + val_count
    val_ids = ids[val_start:val_end]

    purge_start = max(0, val_start - purge_gap)
    purge_end = min(len(ids), val_end + purge_gap)
    purged = tuple(
        ids[i]
        for i in range(purge_start, purge_end)
        if not (val_start <= i < val_end)
    )
    excluded = set(val_ids) | set(purged)
    train_ids = tuple(frame for frame in ids if frame not in excluded)
    if not train_ids:
        raise ValueError("Split settings leave no training frames")
    return SplitDefinition(
        train_frame_ids=train_ids,
        val_frame_ids=val_ids,
        purged_frame_ids=purged,
    )


def blocked_cross_validation(
    frame_ids: Iterable[int], *, n_splits: int = 5, purge_gap: int = 1
) -> tuple[SplitDefinition, ...]:
    ids = tuple(sorted(set(int(v) for v in frame_ids)))
    if not 2 <= n_splits <= len(ids):
        raise ValueError("n_splits must be between 2 and number of frames")
    base, remainder = divmod(len(ids), n_splits)
    folds: list[SplitDefinition] = []
    start = 0
    for fold in range(n_splits):
        size = base + (1 if fold < remainder else 0)
        end = start + size
        val_ids = ids[start:end]
        purge_start = max(0, start - purge_gap)
        purge_end = min(len(ids), end + purge_gap)
        purged = tuple(
            ids[i]
            for i in range(purge_start, purge_end)
            if not (start <= i < end)
        )
        excluded = set(val_ids) | set(purged)
        train_ids = tuple(frame for frame in ids if frame not in excluded)
        if train_ids:
            folds.append(SplitDefinition(train_ids, val_ids, purged))
        start = end
    return tuple(folds)
