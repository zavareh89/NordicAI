from __future__ import annotations

import math

import optuna
import pytest

from hpo import (
    SEARCH_SPACE,
    HPO_FIXED_FIELDS,
    case_id_for_trial_number,
    config_from_trial_params,
    objective_from_summary,
    suggest_config,
)
from survival_policy.config import PolicyConfig


def test_case_names_reserve_c001_for_default():
    assert case_id_for_trial_number(0) == "C002"
    assert case_id_for_trial_number(99) == "C101"


def test_search_space_covers_original_tunable_fields_and_fixes_architecture_v2():
    config_fields = set(PolicyConfig.__dataclass_fields__)
    assert set(SEARCH_SPACE) == config_fields - HPO_FIXED_FIELDS


def test_fixed_trial_builds_valid_config_and_preserves_master_seed():
    base = PolicyConfig()
    values = {}
    for name, spec in SEARCH_SPACE.items():
        if spec.kind == "int":
            values[name] = int((int(spec.low) + int(spec.high)) // 2)
        else:
            values[name] = (float(spec.low) + float(spec.high)) / 2.0

    trial = optuna.trial.FixedTrial(values)
    config = suggest_config(trial, base)
    config.validate()
    assert config.master_seed == base.master_seed

    reconstructed = config_from_trial_params(values, base)
    assert reconstructed.to_dict() == config.to_dict()


def test_robust_objective_uses_mean_and_p10():
    summary = {
        "n_runs": 30,
        "successful_runs": 30,
        "failed_runs": 0,
        "aggregate": {
            "score": {
                "mean": 100.0,
                "median": 95.0,
                "min": 20.0,
                "max": 180.0,
                "p10": 60.0,
            }
        },
    }
    assert math.isclose(objective_from_summary(summary, "robust"), 92.0)
    assert math.isclose(objective_from_summary(summary, "mean"), 100.0)


def test_partial_batch_is_not_scored():
    summary = {
        "n_runs": 30,
        "successful_runs": 29,
        "failed_runs": 1,
        "aggregate": {"score": {"mean": 100.0, "p10": 60.0}},
    }
    with pytest.raises(ValueError):
        objective_from_summary(summary)


def test_hpo_orchestration_creates_sequential_case_directories(monkeypatch, tmp_path):
    import json
    import hpo

    def fake_run_candidate(*, base_seed, config, output_dir, n_cores):
        output_dir.mkdir(parents=True, exist_ok=True)
        config.to_json(output_dir / "params.json")
        mean = 50.0 + config.predator_repulsion_weight
        summary = {
            "status": "success",
            "base_seed": base_seed,
            "n_cores": n_cores,
            "n_runs": n_cores,
            "successful_runs": n_cores,
            "failed_runs": 0,
            "aggregate": {
                "score": {
                    "mean": mean,
                    "median": mean - 1.0,
                    "min": mean - 10.0,
                    "max": mean + 10.0,
                    "p10": mean - 5.0,
                }
            },
            "runs": [],
        }
        (output_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(hpo, "_run_candidate", fake_run_candidate)
    study = hpo.run_hpo(
        results_root=tmp_path,
        base_config=PolicyConfig(),
        base_seed=10,
        n_cores=2,
        n_trials=3,
        sampler_seed=123,
        startup_trials=1,
        objective_mode="robust",
        study_name="test_hpo",
    )

    assert len(study.trials) == 3
    assert (tmp_path / "C002" / "hpo_trial.json").exists()
    assert (tmp_path / "C003" / "hpo_trial.json").exists()
    assert (tmp_path / "C004" / "hpo_trial.json").exists()
    assert (tmp_path / "hpo_trials.csv").exists()
    assert (tmp_path / "hpo_summary.json").exists()
    assert (tmp_path / "best_params.json").exists()


def test_default_configuration_lies_inside_every_search_interval():
    base = PolicyConfig()
    for name, spec in SEARCH_SPACE.items():
        value = getattr(base, name)
        assert spec.low <= value <= spec.high, name
