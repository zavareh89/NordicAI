from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import optuna
from optuna.trial import TrialState

from survival_policy.config import PolicyConfig


TOTAL_HPO_CASES = 100
# The second focused v2 study produced C202-C401. Continue numbering without
# overwriting historical candidates; C402 is the exact C376-v3 baseline.
FIRST_CASE_NUMBER = 402
DEFAULT_N_CORES = 30
# Keep all earlier tuning/holdout seed ranges isolated. V3 HPO uses a fresh
# common-random-number batch so every candidate sees identical simulations.
DEFAULT_BASE_SEED = 90000
DEFAULT_SAMPLER_SEED = 20260920
DEFAULT_STARTUP_TRIALS = 15
DEFAULT_STUDY_NAME = "survival_v3_focused_tpe_100"
DEFAULT_RESULTS_ROOT = Path("results_v3_hpo")
DEFAULT_BASE_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "C376_v3_hpo_base.json"


@dataclass(frozen=True)
class SearchRange:
    kind: Literal["float", "int"]
    low: float | int
    high: float | int
    log: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "low": self.low, "high": self.high, "log": self.log}


# V3 is a structural planner upgrade, so most well-converged v1/v2 values are
# frozen to C376. Search only parameters whose semantics changed in v3 or whose
# previous optimum sat on a meaningful boundary. Beam width and horizon are fixed
# for the first v3 HPO because they directly control latency.
SEARCH_SPACE: dict[str, SearchRange] = {
    # Predator timing still interacts strongly with CPA/TTC rollout scoring.
    "predator_danger_distance": SearchRange("float", 205.0, 245.0),
    "escape_persistence_ticks": SearchRange("int", 14, 22),
    "evasion_sprint_fraction": SearchRange("float", 0.88, 0.97),
    "ttc_danger_ticks": SearchRange("float", 24.0, 38.0),

    # Keep C376's successful coarse action envelope, but retune how future
    # rollout steps branch and how safety/reward terms trade off.
    "planner_angle_spread": SearchRange("float", 0.34, 0.49),
    "planner_predator_weight": SearchRange("float", 4.4, 6.3),
    "planner_food_weight": SearchRange("float", 1.8, 2.55),
    "planner_followup_angle_spread": SearchRange("float", 0.12, 0.30),
    "planner_rollout_discount": SearchRange("float", 0.76, 0.96),
    "planner_cpa_weight": SearchRange("float", 0.80, 1.60),
    "planner_energy_score_scale": SearchRange("float", 0.55, 1.15),
    "planner_wall_clearance": SearchRange("float", 10.0, 16.0),
    "planner_path_wall_soft_clearance": SearchRange("float", 22.0, 36.0),

    # V3 population/exploration behavior is new enough to warrant a compact
    # retune. The stable base population remains near C376's high-cap regime.
    "population_soft_cap": SearchRange("int", 19, 24),
    "population_capacity_smoothing": SearchRange("float", 0.10, 0.38),
    "exploration_no_food_ticks": SearchRange("int", 60, 125),
    "reproduction_slots_per_tick": SearchRange("int", 1, 2),
}

HPO_FIXED_FIELDS = set(PolicyConfig.__dataclass_fields__) - set(SEARCH_SPACE)

SEARCH_RATIONALE: dict[str, str] = {
    "predator_danger_distance": "C376/v2 favored very early danger awareness; CPA rollouts change the best radius.",
    "escape_persistence_ticks": "C376 sat at the high end and v3 replans sequentially after an escape begins.",
    "evasion_sprint_fraction": "High v2 values worked well; exact rollout energy accounting may shift the best sprint fraction.",
    "ttc_danger_ticks": "Still gates high-level threat state while v3 uses candidate-specific CPA in the final action.",
    "planner_angle_spread": "State-adaptive templates preserve the C376 envelope but may need a slightly different first-step spread.",
    "planner_predator_weight": "Main high-level safety scale feeding the new CPA risk term.",
    "planner_food_weight": "Balances capture progress against safety in cumulative rollout scoring.",
    "planner_followup_angle_spread": "New v3 parameter controlling branching of future beam steps.",
    "planner_rollout_discount": "New v3 temporal tradeoff between immediate and future action quality.",
    "planner_cpa_weight": "New v3 closest-point-of-approach risk multiplier.",
    "planner_energy_score_scale": "V3 uses a simulator-aligned cumulative energy estimate rather than the old free energy weight.",
    "planner_wall_clearance": "V3 evaluates clearance along the actual path, so C376's endpoint threshold needs retuning.",
    "planner_path_wall_soft_clearance": "New v3 soft path-clearance margin.",
    "population_soft_cap": "C376 remained at the old upper boundary; v3 uses a smoother stable-capacity model.",
    "population_capacity_smoothing": "New v3 parameter controlling how quickly carrying capacity reacts to noisy observations.",
    "exploration_no_food_ticks": "V3 exploration is event-driven; this controls the no-food event rather than periodic timer rotation.",
    "reproduction_slots_per_tick": "New v3 population-level coordination control preventing synchronized spawn bursts.",
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
    """Map focused-v3 Optuna trial 0 to C402, after historical C202-C401."""
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
        "is_baseline": trial_number == 0,
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
        "is_baseline": trial.number == 0,
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
        "case_id", "trial_number", "is_baseline", "state", "objective_value",
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
        "sampler": "Optuna TPESampler(multivariate=True), focused v3 search",
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
        "base_config_hash": base_config.stable_hash,
        "base_config_source": "selected C376 baseline with architecture_v3_enabled=true",
        "fixed_parameters": {name: getattr(base_config, name) for name in sorted(HPO_FIXED_FIELDS)},
        "search_space": {name: spec.to_dict() for name, spec in SEARCH_SPACE.items()},
        "search_rationale": SEARCH_RATIONALE,
    }
    base_config.to_json(results_root / "hpo_base_config.json")
    (results_root / "hpo_search_space.json").write_text(
        json.dumps(search_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    run_payload = {
        "algorithm": "Optuna TPE focused architecture-v3 search",
        "multivariate": True,
        "baseline_case": f"C{FIRST_CASE_NUMBER:03d}",
        "baseline_is_exact_base_config": True,
        "searched_dimensions": len(SEARCH_SPACE),
        "fixed_dimensions": len(HPO_FIXED_FIELDS),
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
        "architecture_v3_enabled": base_config.architecture_v3_enabled,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (results_root / "hpo_config.json").write_text(
        json.dumps(run_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_focused_base_config(base_config: PolicyConfig) -> None:
    """Fail fast if the v3 study is accidentally started from v1/v2/default."""
    base_config.validate()
    if not base_config.architecture_v2_enabled or not base_config.architecture_v3_enabled:
        raise ValueError(
            "Focused HPO requires architecture_v2_enabled=true and architecture_v3_enabled=true. "
            "Use config/C376_v3_hpo_base.json or an equivalent validated v3 config."
        )
    if not base_config.planner_enabled:
        raise ValueError("Focused HPO requires planner_enabled=true")
    # Latency-sensitive structural parameters are deliberately fixed for this study.
    if base_config.planner_beam_width != 1 or base_config.planner_horizon_steps != 3:
        raise ValueError("First v3 HPO expects planner_beam_width=1 and planner_horizon_steps=3")
    for name, spec in SEARCH_SPACE.items():
        value = getattr(base_config, name)
        if not (spec.low <= value <= spec.high):
            raise ValueError(
                f"Base value {name}={value} is outside focused search range "
                f"[{spec.low}, {spec.high}]"
            )

def _enqueue_base_candidate_if_needed(study: optuna.Study, base_config: PolicyConfig) -> None:
    """Make C402 the exact C376-v3 baseline on the new common seed set."""
    if study.trials:
        return
    params = {name: getattr(base_config, name) for name in SEARCH_SPACE}
    study.enqueue_trial(params)


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
    validate_focused_base_config(base_config)
    if n_cores < 1:
        raise ValueError("n_cores must be >= 1")
    if n_trials < 1 or n_trials > TOTAL_HPO_CASES:
        raise ValueError(f"n_trials must be in [1, {TOTAL_HPO_CASES}]")
    if startup_trials < 0:
        raise ValueError("startup_trials must be >= 0")
    if startup_trials >= n_trials and n_trials > 1:
        raise ValueError("startup_trials should be smaller than n_trials so TPE gets adaptive trials")
    if objective_mode not in {"robust", "mean"}:
        raise ValueError("objective_mode must be 'robust' or 'mean'")

    results_root.mkdir(parents=True, exist_ok=True)
    existing_hpo_config = results_root / "hpo_config.json"
    existing_search_config = results_root / "hpo_search_space.json"
    existing_base_config = results_root / "hpo_base_config.json"
    if existing_hpo_config.exists():
        previous = json.loads(existing_hpo_config.read_text(encoding="utf-8"))
        checks = {
            "architecture_v2_enabled": base_config.architecture_v2_enabled,
        "architecture_v3_enabled": base_config.architecture_v3_enabled,
            "base_seed": base_seed,
            "n_cores_per_case": n_cores,
            "objective_mode": objective_mode,
        }
        for key, expected in checks.items():
            if previous.get(key) != expected:
                raise RuntimeError(
                    f"Refusing to resume focused HPO with changed {key}: "
                    f"stored={previous.get(key)!r}, requested={expected!r}. "
                    "Use the original settings or a fresh results directory."
                )
    if existing_search_config.exists():
        previous_search = json.loads(existing_search_config.read_text(encoding="utf-8"))
        current_search = {name: spec.to_dict() for name, spec in SEARCH_SPACE.items()}
        if previous_search.get("search_space") != current_search:
            raise RuntimeError(
                "Focused HPO search space differs from the persisted study. "
                "Do not resume across code/search-space changes; use a fresh results directory."
            )
    if existing_base_config.exists():
        previous_base = PolicyConfig.from_json(existing_base_config)
        if previous_base.stable_hash != base_config.stable_hash:
            raise RuntimeError(
                "Focused HPO base config differs from the persisted C376-v3 baseline. "
                "Use the original base config or a fresh results directory."
            )
    storage_path = results_root / "hpo_v3_focused_study.db"
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
    _enqueue_base_candidate_if_needed(study, base_config)

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

    while True:
        relevant = [t for t in study.trials if t.number < n_trials]
        finished = sum(t.state in {TrialState.COMPLETE, TrialState.FAIL} for t in relevant)
        if finished >= n_trials:
            break

        trial = study.ask()
        if trial.number >= n_trials:
            # Defensive guard if the persistent study was externally modified.
            study.tell(trial, state=TrialState.FAIL)
            break

        case_id = case_id_for_trial_number(trial.number)
        case_dir = results_root / case_id
        trial.set_user_attr("case_id", case_id)
        trial.set_user_attr("is_baseline", trial.number == 0)

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
            "Run focused architecture-v3 Optuna/TPE HPO around the selected C376 baseline. Each candidate "
            "is evaluated by evaluate.run_parallel() on the same deterministic seed batch."
        )
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=DEFAULT_BASE_CONFIG_PATH,
        help="Base/fixed configuration. Defaults to selected C376-v3.",
    )
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
        help="Delete only the focused study SQLite DB before starting. Existing Cxxx directories are never silently deleted.",
    )
    args = parser.parse_args()

    if not args.base_config.exists():
        raise FileNotFoundError(
            f"Base config not found: {args.base_config}. "
            "Provide --base-config pointing to config/C376_v3_hpo_base.json or another validated v3 config."
        )
    base_config = PolicyConfig.from_json(args.base_config)

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
