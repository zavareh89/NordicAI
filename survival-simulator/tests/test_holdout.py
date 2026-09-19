from __future__ import annotations

import json

from validate_holdout import compare_results


def _summary(mean, p10, seeds):
    return {
        "n_runs": len(seeds),
        "successful_runs": len(seeds),
        "failed_runs": 0,
        "seeds": seeds,
        "controller_config_hash": "abc",
        "aggregate": {
            "score": {
                "mean": mean,
                "median": mean - 1,
                "p10": p10,
                "min": p10 - 10,
                "max": mean + 50,
            },
            "survival_time": {"mean": mean},
        },
    }


def test_holdout_compare_requires_common_seeds_and_ranks_robustly(tmp_path):
    seeds = list(range(50000, 50030))
    for cid, mean, p10 in [("C119", 600.0, 350.0), ("C167", 570.0, 450.0)]:
        d = tmp_path / cid
        d.mkdir()
        (d / "summary.json").write_text(json.dumps(_summary(mean, p10, seeds)), encoding="utf-8")
    rows = compare_results(tmp_path, expected_runs=30)
    # robust: C119=550, C167=546 -> C119 stays first.
    assert rows[0]["case_id"] == "C119"
    assert (tmp_path / "holdout_leaderboard.csv").exists()
    assert (tmp_path / "holdout_comparison.json").exists()
