from __future__ import annotations

import math

import pytest

from evaluate import planned_seeds, summarize_runs


def test_planned_seeds_one_experiment_per_core():
    assert planned_seeds(base_seed=100, n_cores=4) == [100, 101, 102, 103]


def test_planned_seeds_rejects_invalid_core_count():
    with pytest.raises(ValueError):
        planned_seeds(base_seed=1, n_cores=0)


def test_parallel_summary_is_seed_sorted_and_aggregated():
    runs = [
        {
            "seed": 12,
            "status": "success",
            "score": 30.0,
            "survival_time": 20.0,
            "final_population": 1,
            "maximum_population": 5,
            "predator_deaths": 1,
            "starvation_deaths": 0,
            "total_herbivore_deaths": 1,
            "reproduction_count": 1,
            "runtime_seconds": 2.0,
        },
        {
            "seed": 11,
            "status": "success",
            "score": 10.0,
            "survival_time": 10.0,
            "final_population": 0,
            "maximum_population": 4,
            "predator_deaths": 0,
            "starvation_deaths": 2,
            "total_herbivore_deaths": 2,
            "reproduction_count": 0,
            "runtime_seconds": 1.0,
        },
    ]
    summary = summarize_runs(
        runs,
        base_seed=11,
        n_cores=2,
        config_hash="abc",
        wall_runtime_seconds=2.5,
    )

    assert summary["status"] == "success"
    assert summary["seeds"] == [11, 12]
    assert summary["successful_runs"] == 2
    assert summary["failed_runs"] == 0
    assert math.isclose(summary["aggregate"]["score"]["mean"], 20.0)
    assert math.isclose(summary["aggregate"]["survival_time"]["median"], 15.0)


def test_parallel_summary_preserves_failure_without_polluting_aggregates():
    runs = [
        {
            "seed": 4,
            "status": "success",
            "score": 50.0,
            "survival_time": 40.0,
            "final_population": 1,
            "maximum_population": 5,
            "predator_deaths": 0,
            "starvation_deaths": 1,
            "total_herbivore_deaths": 1,
            "reproduction_count": 0,
            "runtime_seconds": 1.0,
        },
        {"seed": 5, "status": "failure", "error": "boom"},
    ]
    summary = summarize_runs(
        runs,
        base_seed=4,
        n_cores=2,
        config_hash="abc",
        wall_runtime_seconds=1.5,
    )

    assert summary["status"] == "partial_failure"
    assert summary["successful_runs"] == 1
    assert summary["failed_runs"] == 1
    assert summary["aggregate"]["score"]["mean"] == 50.0
