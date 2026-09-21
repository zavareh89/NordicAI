#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from dev.boundary_tuning import best_shift, fold_for, mean_tiou, shifted_tiou


def read_json(path: str | Path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def grid_values(minimum: float, maximum: float, step: float) -> list[float]:
    values = []
    x = minimum
    while x <= maximum + 1e-9:
        values.append(round(x, 6))
        x += step
    return values


def usable(records):
    return [
        r for r in records
        if int(r["label"]) == 1
        and bool(r["answer"])
        and r.get("raw_start") is not None
        and r.get("raw_end") is not None
        and r.get("gold_start") is not None
        and r.get("gold_end") is not None
        and r.get("source")
    ]


def fit_model(records, grid, min_source_samples):
    gs, ge, _ = best_shift(records, grid)
    model = {
        "global": {"start_shift_s": gs, "end_shift_s": ge},
        "by_source": {},
    }
    for source in sorted({r["source"] for r in records}):
        subset = [r for r in records if r["source"] == source]
        if len(subset) < min_source_samples:
            continue
        ss, se, _ = best_shift(subset, grid)
        model["by_source"][source] = {
            "start_shift_s": ss,
            "end_shift_s": se,
        }
    return model


def score_model(records, model):
    values = []
    for r in records:
        shift = model["by_source"].get(r["source"], model["global"])
        values.append(
            shifted_tiou(r, shift["start_shift_s"], shift["end_shift_s"])
        )
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="dev_cache/e2_1b_boundary_cache.json.gz")
    parser.add_argument(
        "--output", default="config/e2_1_boundary_calibration.json"
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--grid-min", type=float, default=-0.30)
    parser.add_argument("--grid-max", type=float, default=0.30)
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument("--min-source-samples", type=int, default=25)
    args = parser.parse_args()

    payload = read_json(args.cache)
    records = usable(payload["records"])
    if not records:
        raise RuntimeError("No detected positive evidence records in cache")

    grid = grid_values(args.grid_min, args.grid_max, args.grid_step)
    raw_mean = mean_tiou(records, 0.0, 0.0)

    fold_scores = []
    for fold in range(args.folds):
        train = [r for r in records if fold_for(r["transcript_id"], args.folds) != fold]
        test = [r for r in records if fold_for(r["transcript_id"], args.folds) == fold]
        if not train or not test:
            continue
        model = fit_model(train, grid, args.min_source_samples)
        fold_scores.append(score_model(test, model))

    final_model = fit_model(records, grid, args.min_source_samples)
    final_score = score_model(records, final_model)
    output = {
        "schema_version": 1,
        **final_model,
        "samples": len(records),
        "raw_mean_tiou": raw_mean,
        "cv_mean_tiou": (
            sum(fold_scores) / len(fold_scores) if fold_scores else None
        ),
        "fit_mean_tiou": final_score,
        "fold_scores": fold_scores,
    }
    write_json(args.output, output)
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
