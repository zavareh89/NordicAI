from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import optuna
from optuna.trial import TrialState

from survival_policy.config import PolicyConfig, load_default_config


TOTAL_HPO_CASES = 100
FIRST_CASE_NUMBER = 2
DEFAULT_N_CORES = 30
DEFAULT_BASE_SEED = 1000
DEFAULT_SAMPLER_SEED = 20260918
DEFAULT_STARTUP_TRIALS = 20
DEFAULT_STUDY_NAME = "survival_tpe_100"

# Architecture-v2 parameters are intentionally held fixed during the original
# 28-D HPO study. This preserves comparability with C002+ and prevents the
# search space from exploding before the new planner has been holdout-tested.
HPO_FIXED_FIELDS = {
    "master_seed",
    "architecture_v2_enabled",
    "planner_enabled",
    "planner_horizon_steps",
    "planner_angle_spread",
    "planner_predator_weight",
    "planner_food_weight",
    "planner_energy_weight",
    "planner_wall_clearance",
    "planner_wall_soft_clearance",
    "ttc_danger_ticks",
    "fruit_track_ttl",
    "fruit_track_match_angle",
    "fruit_track_match_distance",
    "dynamic_population_bonus",
}


@dataclass(frozen=True)
class SearchRange:
    kind: Literal["float", "int"]
    low: float | int
    high: float | int
    log: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "low": self.low,
            "high": self.high,
            "log": self.log,
        }


# This is deliberately a local search around the hand-engineered C001 defaults.
# With only 100 candidate configurations and the original 28 tunable dimensions, very wide
# ranges would waste most of the budget on obviously poor behavior. The static
# ranges also guarantee PolicyConfig constraints such as critical < low energy
# and emergency predator distance < danger distance without conditional spaces.
SEARCH_SPACE: dict[str, SearchRange] = {
    # Predator handling
    "emergency_predator_distance": SearchRange("float", 35.0, 60.0),
    "predator_danger_distance": SearchRange("float", 120.0, 190.0),
    "predator_repulsion_weight": SearchRange("float", 3.0, 7.0),
    "predator_prediction_weight": SearchRange("float", 0.25, 1.00),
    "escape_persistence_ticks": SearchRange("int", 4, 10),

    # Walls / navigation / spacing
    "wall_danger_distance": SearchRange("float", 40.0, 70.0),
    "wall_repulsion_weight": SearchRange("float", 2.0, 5.0),
    "herbivore_repulsion_radius": SearchRange("float", 30.0, 60.0),
    "herbivore_repulsion_weight": SearchRange("float", 0.30, 1.00),
    "max_turn_angle": SearchRange("float", 0.25, 0.45),

    # Food / targeting
    "critical_energy_ratio": SearchRange("float", 0.18, 0.28),
    "low_energy_ratio": SearchRange("float", 0.31, 0.43),
    "fruit_attraction_weight": SearchRange("float", 1.70, 3.50),
    "fruit_distance_penalty": SearchRange("float", 0.006, 0.016, log=True),
    "fruit_competition_penalty": SearchRange("float", 0.70, 1.80),
    "target_persistence_bonus": SearchRange("float", 0.40, 1.10),
    "target_timeout_ticks": SearchRange("int", 16, 36),

    # Movement / exploration
    "forage_move_fraction": SearchRange("float", 0.75, 0.95),
    "explore_move_fraction": SearchRange("float", 0.45, 0.68),
    "conserve_move_fraction": SearchRange("float", 0.20, 0.40),
    "evasion_sprint_fraction": SearchRange("float", 0.68, 0.90),
    "exploration_change_interval": SearchRange("int", 30, 70),

    # Reproduction
    "spawn_min_energy": SearchRange("float", 190.0, 260.0),
    "spawn_old_age": SearchRange("float", 42.0, 70.0),
    "population_soft_cap": SearchRange("int", 9, 16),
    "reproduction_cooldown_ticks": SearchRange("int", 65, 130),

    # Stuck / recovery
    "stuck_tick_threshold": SearchRange("int", 5, 10),
    "recovery_ticks": SearchRange("int", 4, 9),
}


OBJECTIVE_DESCRIPTION = "0.8 * mean(score) + 0.2 * p10(score)"


def suggest_config(trial: optuna.Trial, base_config: PolicyConfig) -> PolicyConfig:
    """Sample one complete candidate while keeping non-HPO fields unchanged."""
    values = base_config.to_dict()
    for name, spec in SEARCH_SPACE.items():
        if spec.kind == "float":
            values[name] = trial.suggest_float(
                name,
                float(spec.low),
                float(spec.high),
                log=spec.log,
            )
        else:
            values[name] = trial.suggest_int(name, int(spec.low), int(spec.high))

    # master_seed is intentionally fixed. Optimizing a random seed would select
    # lucky exploration noise rather than a transferable controller behavior.
    values["master_seed"] = base_config.master_seed
    return PolicyConfig.from_dict(values)


def config_from_trial_params(params: dict[str, Any], base_config: PolicyConfig) -> PolicyConfig:
    """Reconstruct a candidate exactly from a persisted Optuna trial."""
    values = base_config.to_dict()
    values.update(params)
    values["master_seed"] = base_config.master_seed
    return PolicyConfig.from_dict(values)


def objective_from_summary(summary: dict[str, Any], mode: str = "robust") -> float:
    """Convert one fixed-seed batch summary into the scalar TPE objective."""
    if summary.get("failed_runs", 0):
        raise ValueError("Candidate contains failed simulator runs; refusing to score a partial batch")
    if summary.get("successful_runs", 0) != summary.get("n_runs", 0):
        raise ValueError("Candidate batch did not complete all simulator runs")

    score = summary.get("aggregate", {}).get("score")
    if not score:
        raise ValueError("Candidate summary has no aggregate score")

    mean_score = float(score["mean"])
    p10_score = float(score["p10"])
    if mode == "mean":
        return mean_score
    if mode == "robust":
        return 0.8 * mean_score + 0.2 * p10_score
    raise ValueError(f"Unknown objective mode: {mode}")


def case_id_for_trial_number(trial_number: int) -> str:
    """Optuna trial 0 is C002 because C001 is reserved for defaults."""
    return f"C{trial_number + FIRST_CASE_NUMBER:03d}"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_candidate(
    *,
    base_seed: int,
    config: PolicyConfig,
    output_dir: Path,
    n_cores: int,
) -> dict[str, Any]:
    # Lazy import keeps the search-space helpers importable/testable without the
    # official simulator package on sys.path. In the challenge repository this
    # resolves to the existing parallel evaluator unchanged.
    from evaluate import run_parallel

    return run_parallel(
        base_seed=base_seed,
        config=config,
        output_dir=output_dir,
        n_cores=n_cores,
    )


def _score_stats(summary: dict[str, Any]) -> dict[str, float]:
    score = summary["aggregate"]["score"]
    return {
        "mean_score": float(score["mean"]),
        "median_score": float(score["median"]),
        "p10_score": float(score["p10"]),
        "min_score": float(score["min"]),
        "max_score": float(score["max"]),
    }


def _write_case_metadata(
    case_dir: Path,
    *,
    case_id: str,
    trial_number: int,
    objective_value: float | None,
    objective_mode: str,
    config: PolicyConfig,
    summary: dict[str, Any] | None,
    state: str,
    error: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "case_id": case_id,
        "optuna_trial_number": trial_number,
        "trial_state": state,
        "objective_mode": objective_mode,
        "objective_description": OBJECTIVE_DESCRIPTION if objective_mode == "robust" else "mean(score)",
        "objective_value": objective_value,
        "controller_config_hash": config.stable_hash,
        "master_seed_fixed": config.master_seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "error": error,
    }
    if summary and summary.get("aggregate", {}).get("score"):
        payload.update(_score_stats(summary))
    (case_dir / "hpo_trial.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _trial_row(trial: optuna.trial.FrozenTrial, results_root: Path) -> dict[str, Any]:
    case_id = trial.user_attrs.get("case_id", case_id_for_trial_number(trial.number))
    metadata_path = results_root / case_id / "hpo_trial.json"
    metadata = _read_json(metadata_path) if metadata_path.exists() else {}
    row: dict[str, Any] = {
        "case_id": case_id,
        "trial_number": trial.number,
        "state": trial.state.name,
        "objective_value": trial.value,
        "mean_score": trial.user_attrs.get("mean_score", metadata.get("mean_score")),
        "median_score": trial.user_attrs.get("median_score", metadata.get("median_score")),
        "p10_score": trial.user_attrs.get("p10_score", metadata.get("p10_score")),
        "min_score": trial.user_attrs.get("min_score", metadata.get("min_score")),
        "max_score": trial.user_attrs.get("max_score", metadata.get("max_score")),
        "config_hash": trial.user_attrs.get("config_hash", metadata.get("controller_config_hash")),
        "error": trial.user_attrs.get("error", metadata.get("error")),
    }
    row.update(trial.params)
    return row


def write_global_reports(
    study: optuna.Study,
    *,
    results_root: Path,
    base_config: PolicyConfig,
    base_seed: int,
    n_cores: int,
    objective_mode: str,
    target_trials: int,
    sampler_seed: int,
    startup_trials: int,
) -> None:
    """Persist a compact HPO ledger after every candidate for crash resilience."""
    results_root.mkdir(parents=True, exist_ok=True)
    trials = sorted(study.trials, key=lambda t: t.number)
    rows = [_trial_row(t, results_root) for t in trials if t.number < target_trials]

    fieldnames = [
        "case_id", "trial_number", "state", "objective_value",
        "mean_score", "median_score", "p10_score", "min_score", "max_score",
        "config_hash", "error",
        *SEARCH_SPACE.keys(),
    ]
    with (results_root / "hpo_trials.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    completed = [t for t in trials if t.state == TrialState.COMPLETE and t.number < target_trials]
    best_payload: dict[str, Any] | None = None
    if completed:
        best = max(completed, key=lambda t: float(t.value))
        best_case = best.user_attrs.get("case_id", case_id_for_trial_number(best.number))
        best_config = config_from_trial_params(best.params, base_config)
        best_config.to_json(results_root / "best_params.json")
        best_metadata_path = results_root / best_case / "hpo_trial.json"
        best_metadata = _read_json(best_metadata_path) if best_metadata_path.exists() else {}
        best_payload = {
            "case_id": best_case,
            "trial_number": best.number,
            "objective_value": float(best.value),
            "mean_score": best.user_attrs.get("mean_score", best_metadata.get("mean_score")),
            "median_score": best.user_attrs.get("median_score", best_metadata.get("median_score")),
            "p10_score": best.user_attrs.get("p10_score", best_metadata.get("p10_score")),
            "config_hash": best_config.stable_hash,
        }

    summary_payload = {
        "study_name": study.study_name,
        "sampler": "Optuna TPESampler(multivariate=True)",
        "sampler_seed": sampler_seed,
        "startup_trials": startup_trials,
        "target_hpo_cases": target_trials,
        "case_range": f"C{FIRST_CASE_NUMBER:03d}-C{FIRST_CASE_NUMBER + target_trials - 1:03d}",
        "completed_trials": sum(t.state == TrialState.COMPLETE for t in trials if t.number < target_trials),
        "failed_trials": sum(t.state == TrialState.FAIL for t in trials if t.number < target_trials),
        "running_trials": sum(t.state == TrialState.RUNNING for t in trials if t.number < target_trials),
        "base_seed": base_seed,
        "n_cores": n_cores,
        "seeds": [base_seed + i for i in range(n_cores)],
        "objective_mode": objective_mode,
        "objective_description": OBJECTIVE_DESCRIPTION if objective_mode == "robust" else "mean(score)",
        "master_seed_fixed": base_config.master_seed,
        "best_hpo_case": best_payload,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (results_root / "hpo_summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _record_success(
    trial: optuna.Trial,
    summary: dict[str, Any],
    config: PolicyConfig,
    objective_value: float,
) -> None:
    stats = _score_stats(summary)
    trial.set_user_attr("config_hash", config.stable_hash)
    trial.set_user_attr("objective_value", objective_value)
    for name, value in stats.items():
        trial.set_user_attr(name, value)


def _repair_running_trials(
    study: optuna.Study,
    *,
    results_root: Path,
    base_config: PolicyConfig,
    base_seed: int,
    n_cores: int,
    objective_mode: str,
    target_trials: int,
) -> None:
    """Finish trials left RUNNING by an interrupted HPO process.

    If a complete summary already exists it is reused. Otherwise the exact saved
    Optuna parameters are rerun on the same deterministic seed batch.
    """
    running = [
        t for t in study.trials
        if t.state == TrialState.RUNNING and t.number < target_trials
    ]
    for frozen in running:
        case_id = frozen.user_attrs.get("case_id", case_id_for_trial_number(frozen.number))
        case_dir = results_root / case_id
        config = config_from_trial_params(frozen.params, base_config)
        summary_path = case_dir / "summary.json"
        try:
            if summary_path.exists():
                summary = _read_json(summary_path)
                # A partial/failure summary is not reusable as a valid objective.
                if summary.get("failed_runs", 0) or summary.get("successful_runs") != n_cores:
                    summary = _run_candidate(
                        base_seed=base_seed,
                        config=config,
                        output_dir=case_dir,
                        n_cores=n_cores,
                    )
            else:
                summary = _run_candidate(
                    base_seed=base_seed,
                    config=config,
                    output_dir=case_dir,
                    n_cores=n_cores,
                )

            value = objective_from_summary(summary, objective_mode)
            # Study.tell() is public and is enough to repair the persisted state.
            # Detailed score/config metadata is kept in Cxxx/hpo_trial.json.
            study.tell(frozen.number, value)
            _write_case_metadata(
                case_dir,
                case_id=case_id,
                trial_number=frozen.number,
                objective_value=value,
                objective_mode=objective_mode,
                config=config,
                summary=summary,
                state="COMPLETE",
            )
            print(f"recovered {case_id}: objective={value:.3f}")
        except Exception as exc:
            study.tell(frozen.number, state=TrialState.FAIL)
            case_dir.mkdir(parents=True, exist_ok=True)
            _write_case_metadata(
                case_dir,
                case_id=case_id,
                trial_number=frozen.number,
                objective_value=None,
                objective_mode=objective_mode,
                config=config,
                summary=None,
                state="FAIL",
                error=f"{type(exc).__name__}: {exc}",
            )
            print(f"failed to recover {case_id}: {type(exc).__name__}: {exc}")


def _storage_url(path: Path) -> str:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.as_posix()}"


def _write_search_metadata(
    *,
    results_root: Path,
    base_config: PolicyConfig,
    base_seed: int,
    n_cores: int,
    objective_mode: str,
    target_trials: int,
    sampler_seed: int,
    startup_trials: int,
) -> None:
    search_payload = {
        "fixed_parameters": {name: getattr(base_config, name) for name in sorted(HPO_FIXED_FIELDS)},
        "search_space": {name: spec.to_dict() for name, spec in SEARCH_SPACE.items()},
    }
    (results_root / "hpo_search_space.json").write_text(
        json.dumps(search_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    run_payload = {
        "algorithm": "Optuna TPE",
        "multivariate": True,
        "startup_trials": startup_trials,
        "sampler_seed": sampler_seed,
        "n_hpo_cases": target_trials,
        "case_range": f"C{FIRST_CASE_NUMBER:03d}-C{FIRST_CASE_NUMBER + target_trials - 1:03d}",
        "n_cores_per_case": n_cores,
        "base_seed": base_seed,
        "seeds_per_case": [base_seed + i for i in range(n_cores)],
        "objective_mode": objective_mode,
        "objective_description": OBJECTIVE_DESCRIPTION if objective_mode == "robust" else "mean(score)",
        "C001_reserved_for_default": True,
        "architecture_v2_enabled": base_config.architecture_v2_enabled,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (results_root / "hpo_config.json").write_text(
        json.dumps(run_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_hpo(
    *,
    results_root: Path,
    base_config: PolicyConfig,
    base_seed: int = DEFAULT_BASE_SEED,
    n_cores: int = DEFAULT_N_CORES,
    n_trials: int = TOTAL_HPO_CASES,
    sampler_seed: int = DEFAULT_SAMPLER_SEED,
    startup_trials: int = DEFAULT_STARTUP_TRIALS,
    objective_mode: str = "robust",
    study_name: str = DEFAULT_STUDY_NAME,
    reset_study: bool = False,
) -> optuna.Study:
    if n_cores < 1:
        raise ValueError("n_cores must be >= 1")
    if n_trials < 1 or n_trials > TOTAL_HPO_CASES:
        raise ValueError(f"n_trials must be in [1, {TOTAL_HPO_CASES}]")
    if startup_trials < 0:
        raise ValueError("startup_trials must be >= 0")
    if objective_mode not in {"robust", "mean"}:
        raise ValueError("objective_mode must be 'robust' or 'mean'")

    results_root.mkdir(parents=True, exist_ok=True)
    existing_hpo_config = results_root / "hpo_config.json"
    if existing_hpo_config.exists():
        previous = json.loads(existing_hpo_config.read_text(encoding="utf-8"))
        # Historical HPO runs predate architecture-v2 metadata, so missing means v1.
        previous_v2 = bool(previous.get("architecture_v2_enabled", False))
        if previous_v2 != base_config.architecture_v2_enabled:
            raise RuntimeError(
                "Refusing to mix v1 and architecture-v2 trials in the same HPO results root. "
                "Use the matching base config to resume the old study or choose a fresh results directory."
            )
    storage_path = results_root / "hpo_study.db"
    if reset_study and storage_path.exists():
        storage_path.unlink()

    _write_search_metadata(
        results_root=results_root,
        base_config=base_config,
        base_seed=base_seed,
        n_cores=n_cores,
        objective_mode=objective_mode,
        target_trials=n_trials,
        sampler_seed=sampler_seed,
        startup_trials=startup_trials,
    )

    sampler = optuna.samplers.TPESampler(
        seed=sampler_seed,
        n_startup_trials=startup_trials,
        multivariate=True,
        warn_independent_sampling=False,
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=_storage_url(storage_path),
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )

    _repair_running_trials(
        study,
        results_root=results_root,
        base_config=base_config,
        base_seed=base_seed,
        n_cores=n_cores,
        objective_mode=objective_mode,
        target_trials=n_trials,
    )

    if len(study.trials) > n_trials:
        raise RuntimeError(
            f"Study already contains {len(study.trials)} trials, more than requested {n_trials}. "
            "Use a fresh results directory or --reset-study."
        )

    write_global_reports(
        study,
        results_root=results_root,
        base_config=base_config,
        base_seed=base_seed,
        n_cores=n_cores,
        objective_mode=objective_mode,
        target_trials=n_trials,
        sampler_seed=sampler_seed,
        startup_trials=startup_trials,
    )

    while len(study.trials) < n_trials:
        trial = study.ask()
        if trial.number >= n_trials:
            # Defensive guard if the persistent study was externally modified.
            study.tell(trial, state=TrialState.FAIL)
            break

        case_id = case_id_for_trial_number(trial.number)
        case_dir = results_root / case_id
        trial.set_user_attr("case_id", case_id)

        if case_dir.exists() and any(case_dir.iterdir()):
            # A valid resume is handled through the persistent Optuna study above.
            # Refuse silent overwrites if directories and DB no longer correspond.
            study.tell(trial, state=TrialState.FAIL)
            raise FileExistsError(
                f"{case_dir} already exists but is not represented by a finished study trial. "
                "Use the matching hpo_study.db, move the old directory, or use a fresh results root."
            )

        config = suggest_config(trial, base_config)
        trial.set_user_attr("config_hash", config.stable_hash)
        case_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"{case_id} | trial {trial.number + 1}/{n_trials} | "
            f"{n_cores}-seed batch on {n_cores} worker(s)"
        )

        try:
            summary = _run_candidate(
                base_seed=base_seed,
                config=config,
                output_dir=case_dir,
                n_cores=n_cores,
            )
            objective_value = objective_from_summary(summary, objective_mode)
            _record_success(trial, summary, config, objective_value)
            _write_case_metadata(
                case_dir,
                case_id=case_id,
                trial_number=trial.number,
                objective_value=objective_value,
                objective_mode=objective_mode,
                config=config,
                summary=summary,
                state="COMPLETE",
            )
            study.tell(trial, objective_value)

            stats = _score_stats(summary)
            print(
                f"{case_id} complete | objective={objective_value:.3f} | "
                f"mean={stats['mean_score']:.3f} | p10={stats['p10_score']:.3f} | "
                f"median={stats['median_score']:.3f}"
            )
        except KeyboardInterrupt:
            # Leave the trial RUNNING so the next invocation can recover/rerun it
            # with exactly the same sampled parameters and seeds.
            print(f"Interrupted during {case_id}; it will be recovered on resume.")
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            trial.set_user_attr("error", error)
            _write_case_metadata(
                case_dir,
                case_id=case_id,
                trial_number=trial.number,
                objective_value=None,
                objective_mode=objective_mode,
                config=config,
                summary=None,
                state="FAIL",
                error=error,
            )
            study.tell(trial, state=TrialState.FAIL)
            print(f"{case_id} failed | {error}")

        write_global_reports(
            study,
            results_root=results_root,
            base_config=base_config,
            base_seed=base_seed,
            n_cores=n_cores,
            objective_mode=objective_mode,
            target_trials=n_trials,
            sampler_seed=sampler_seed,
            startup_trials=startup_trials,
        )

    return study


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run sequential Optuna/TPE HPO over controller parameter sets. Each candidate "
            "is evaluated by evaluate.run_parallel() on the same deterministic seed batch."
        )
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--base-config", type=Path, default=None)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--n-cores", type=int, default=DEFAULT_N_CORES)
    parser.add_argument("--n-trials", type=int, default=TOTAL_HPO_CASES)
    parser.add_argument("--sampler-seed", type=int, default=DEFAULT_SAMPLER_SEED)
    parser.add_argument("--startup-trials", type=int, default=DEFAULT_STARTUP_TRIALS)
    parser.add_argument("--objective", choices=("robust", "mean"), default="robust")
    parser.add_argument("--study-name", default=DEFAULT_STUDY_NAME)
    parser.add_argument(
        "--reset-study",
        action="store_true",
        help="Delete only results/hpo_study.db before starting. Existing Cxxx directories are never silently deleted.",
    )
    args = parser.parse_args()

    base_config = (
        PolicyConfig.from_json(args.base_config)
        if args.base_config is not None
        else load_default_config()
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = run_hpo(
        results_root=args.results_root,
        base_config=base_config,
        base_seed=args.base_seed,
        n_cores=args.n_cores,
        n_trials=args.n_trials,
        sampler_seed=args.sampler_seed,
        startup_trials=args.startup_trials,
        objective_mode=args.objective,
        study_name=args.study_name,
        reset_study=args.reset_study,
    )

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if completed:
        best = max(completed, key=lambda t: float(t.value))
        best_case = best.user_attrs.get("case_id", case_id_for_trial_number(best.number))
        print(
            f"HPO finished: completed={len(completed)}/{args.n_trials}, "
            f"best={best_case}, objective={float(best.value):.3f}"
        )
    else:
        print("HPO finished with no completed trials.")


if __name__ == "__main__":
    main()
