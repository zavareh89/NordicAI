from __future__ import annotations

import json
import math
from pathlib import Path

import optuna
import pytest

from hpo import (
    DEFAULT_BASE_CONFIG_PATH,
    HPO_FIXED_FIELDS,
    SEARCH_RATIONALE,
    SEARCH_SPACE,
    case_id_for_trial_number,
    config_from_trial_params,
    objective_from_summary,
    suggest_config,
    validate_focused_base_config,
)
from survival_policy.config import PolicyConfig


def _base() -> PolicyConfig:
    return PolicyConfig.from_json(DEFAULT_BASE_CONFIG_PATH)


def test_case_names_continue_after_historical_v1_study():
    assert case_id_for_trial_number(0) == "C192"
    assert case_id_for_trial_number(99) == "C291"


def test_focused_search_only_tunes_declared_subset():
    expected = {
        "emergency_predator_distance",
        "predator_danger_distance",
        "predator_prediction_weight",
        "escape_persistence_ticks",
        "evasion_sprint_fraction",
        "exploration_change_interval",
        "fruit_distance_penalty",
        "spawn_min_energy",
        "population_soft_cap",
        "planner_horizon_steps",
        "planner_angle_spread",
        "planner_predator_weight",
        "planner_food_weight",
        "planner_energy_weight",
        "planner_wall_clearance",
        "ttc_danger_ticks",
        "dynamic_population_bonus",
    }
    config_fields = set(PolicyConfig.__dataclass_fields__)
    assert set(SEARCH_SPACE) == expected
    assert HPO_FIXED_FIELDS == config_fields - expected
    assert set(SEARCH_RATIONALE) == expected


def test_c167_v2_base_is_valid_and_inside_every_search_interval():
    base = _base()
    validate_focused_base_config(base)
    assert base.architecture_v2_enabled is True
    assert base.planner_enabled is True
    for name, spec in SEARCH_SPACE.items():
        value = getattr(base, name)
        assert spec.low <= value <= spec.high, name


def test_v1_base_is_rejected_for_focused_hpo():
    values = _base().to_dict()
    values["architecture_v2_enabled"] = False
    with pytest.raises(ValueError, match="architecture_v2_enabled"):
        validate_focused_base_config(PolicyConfig.from_dict(values))


def test_fixed_trial_builds_valid_config_and_preserves_frozen_fields():
    base = _base()
    sampled = {}
    for name, spec in SEARCH_SPACE.items():
        if spec.kind == "int":
            sampled[name] = int((int(spec.low) + int(spec.high)) // 2)
        else:
            sampled[name] = (float(spec.low) + float(spec.high)) / 2.0

    trial = optuna.trial.FixedTrial(sampled)
    config = suggest_config(trial, base)
    config.validate()
    assert config.master_seed == base.master_seed

    for name in HPO_FIXED_FIELDS:
        assert getattr(config, name) == getattr(base, name), name

    reconstructed = config_from_trial_params(sampled, base)
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


def test_hpo_orchestration_enqueues_exact_base_then_samples(monkeypatch, tmp_path):
    import hpo

    base = _base()

    def fake_run_candidate(*, base_seed, config, output_dir, n_cores):
        output_dir.mkdir(parents=True, exist_ok=True)
        config.to_json(output_dir / "params.json")
        mean = 50.0 + config.planner_predator_weight
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
        base_config=base,
        base_seed=70000,
        n_cores=2,
        n_trials=3,
        sampler_seed=123,
        startup_trials=1,
        objective_mode="robust",
        study_name="test_focused_hpo",
    )

    assert len(study.trials) == 3
    assert (tmp_path / "C192" / "hpo_trial.json").exists()
    assert (tmp_path / "C193" / "hpo_trial.json").exists()
    assert (tmp_path / "C194" / "hpo_trial.json").exists()
    assert (tmp_path / "hpo_v2_focused_study.db").exists()
    assert (tmp_path / "hpo_base_config.json").exists()
    assert (tmp_path / "hpo_trials.csv").exists()
    assert (tmp_path / "hpo_summary.json").exists()
    assert (tmp_path / "best_params.json").exists()

    c192 = PolicyConfig.from_json(tmp_path / "C192" / "params.json")
    assert c192.to_dict() == base.to_dict()

    meta = json.loads((tmp_path / "C192" / "hpo_trial.json").read_text())
    assert meta["is_baseline"] is True


def test_resume_rejects_changed_seed_batch(monkeypatch, tmp_path):
    import hpo

    base = _base()

    def fake_run_candidate(*, base_seed, config, output_dir, n_cores):
        output_dir.mkdir(parents=True, exist_ok=True)
        config.to_json(output_dir / "params.json")
        summary = {
            "n_runs": n_cores,
            "successful_runs": n_cores,
            "failed_runs": 0,
            "aggregate": {
                "score": {
                    "mean": 100.0,
                    "median": 100.0,
                    "min": 100.0,
                    "max": 100.0,
                    "p10": 100.0,
                }
            },
        }
        (output_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(hpo, "_run_candidate", fake_run_candidate)
    hpo.run_hpo(
        results_root=tmp_path,
        base_config=base,
        base_seed=70000,
        n_cores=2,
        n_trials=1,
        sampler_seed=123,
        startup_trials=0,
        objective_mode="robust",
        study_name="resume_guard",
    )

    with pytest.raises(RuntimeError, match="base_seed"):
        hpo.run_hpo(
            results_root=tmp_path,
            base_config=base,
            base_seed=71000,
            n_cores=2,
            n_trials=1,
            sampler_seed=123,
            startup_trials=0,
            objective_mode="robust",
            study_name="resume_guard",
        )
