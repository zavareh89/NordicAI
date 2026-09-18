from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import platform
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.core import SimulationCore
from src.utils.DTOs import ObservationResponse, StepResponse
from survival_policy import PolicyConfig, SurvivalController, load_default_config
from survival_policy.metrics import PopulationSample, RunMetrics


def _step_response_from_state(sim: SimulationCore, state: dict[str, Any]) -> StepResponse:
    status = [ObservationResponse(**obs) for obs in state["observations"] if obs is not None]
    game_over = state["num_agents"] == 0 or sim.env.time > 3000
    return StepResponse(
        game_status="game_over" if game_over else "ok",
        score=state["score"],
        sim_time=state["sim_time"],
        n_agents=state["num_agents"],
        agent_status=status,
    )


def run_one(seed: int, config: PolicyConfig, output_dir: str | Path) -> RunMetrics:
    """Run one isolated simulator episode and persist its complete result."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config.to_json(out / "params.json")
    run_config = {
        "simulator_seed": seed,
        "controller_config_hash": config.stable_hash,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    (out / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    metrics = RunMetrics(seed=seed)
    started = time.perf_counter()
    try:
        sim = SimulationCore(seed=seed)
        controller = SurvivalController(config)

        # Instrument the official kill hook without changing simulator source.
        original_kill = sim.env.kill_agent

        def tracked_kill(agent):
            if getattr(agent, "energy", 0.0) <= 0.0:
                metrics.starvation_deaths += 1
            else:
                metrics.predator_deaths += 1
            metrics.total_herbivore_deaths += 1
            return original_kill(agent)

        sim.env.kill_agent = tracked_kill

        state = sim.step([])
        step = _step_response_from_state(sim, state)
        known_ids = {a.agent_id for a in sim.env.agents}
        metrics.maximum_population = len(known_ids)
        next_sample_time = 0.0

        while step.game_status != "game_over":
            actions_raw = controller.decide_step(step)

            # Validate through the official DTO before simulator integration.
            from src.utils.DTOs import ActionRequest

            actions = [(int(a["agent_id"]), ActionRequest(**a)) for a in actions_raw]
            state = sim.step(actions)
            current_ids = {a.agent_id for a in sim.env.agents}
            newborns = current_ids - known_ids
            metrics.reproduction_count += len(newborns)
            known_ids |= current_ids
            metrics.maximum_population = max(metrics.maximum_population, len(current_ids))
            step = _step_response_from_state(sim, state)

            if sim.env.time + 1e-9 >= next_sample_time:
                metrics.population_trajectory.append(
                    PopulationSample(
                        sim_time=float(sim.env.time),
                        population=len(current_ids),
                        score=float(state["score"]),
                    )
                )
                next_sample_time += 1.0

        metrics.score = float(step.score)
        metrics.survival_time = float(step.sim_time)
        metrics.reached_3000_seconds = step.sim_time >= 3000.0
        metrics.final_population = int(step.n_agents)
        metrics.status = "success"
    except Exception as exc:
        metrics.status = "failure"
        metrics.error = f"{type(exc).__name__}: {exc}"
        metrics.runtime_seconds = time.perf_counter() - started
        metrics.save(out)
        raise

    metrics.runtime_seconds = time.perf_counter() - started
    metrics.save(out)
    return metrics


def planned_seeds(base_seed: int, n_cores: int) -> list[int]:
    """Return deterministic seeds: exactly one experiment per requested core."""
    if n_cores < 1:
        raise ValueError("n_cores must be >= 1")
    return [base_seed + i for i in range(n_cores)]


def _run_one_worker(seed: int, config_dict: dict[str, Any], output_dir: str) -> dict[str, Any]:
    """Top-level multiprocessing worker; all mutable simulator/controller state stays local."""
    config = PolicyConfig.from_dict(config_dict)
    try:
        metrics = run_one(seed=seed, config=config, output_dir=output_dir)
        return metrics.to_summary_dict()
    except Exception as exc:
        # run_one already persisted the failure details before re-raising.
        return {
            "seed": seed,
            "status": "failure",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _numeric_summary(values: Iterable[float]) -> dict[str, float] | None:
    vals = [float(v) for v in values]
    if not vals:
        return None
    vals.sort()
    # Nearest-rank p10 avoids optional numerical dependencies.
    p10_index = max(0, int(0.10 * (len(vals) - 1)))
    return {
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "min": vals[0],
        "max": vals[-1],
        "p10": vals[p10_index],
    }


def summarize_runs(
    run_summaries: list[dict[str, Any]],
    *,
    base_seed: int,
    n_cores: int,
    config_hash: str,
    wall_runtime_seconds: float,
) -> dict[str, Any]:
    """Build a deterministic batch summary from worker result dictionaries."""
    runs = sorted(run_summaries, key=lambda r: int(r["seed"]))
    successes = [r for r in runs if r.get("status") == "success"]
    failures = [r for r in runs if r.get("status") != "success"]

    aggregate: dict[str, Any] = {}
    numeric_fields = (
        "score",
        "survival_time",
        "final_population",
        "maximum_population",
        "predator_deaths",
        "starvation_deaths",
        "total_herbivore_deaths",
        "reproduction_count",
        "runtime_seconds",
    )
    for field in numeric_fields:
        summary = _numeric_summary(r[field] for r in successes if field in r)
        if summary is not None:
            aggregate[field] = summary

    return {
        "status": "success" if not failures else ("failure" if not successes else "partial_failure"),
        "base_seed": base_seed,
        "seeds": [int(r["seed"]) for r in runs],
        "n_cores": n_cores,
        "n_runs": len(runs),
        "successful_runs": len(successes),
        "failed_runs": len(failures),
        "controller_config_hash": config_hash,
        "wall_runtime_seconds": wall_runtime_seconds,
        "aggregate": aggregate,
        "runs": runs,
    }


def _write_batch_results(output_dir: Path, summary: dict[str, Any]) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fields = [
        "seed",
        "status",
        "score",
        "survival_time",
        "reached_3000_seconds",
        "final_population",
        "maximum_population",
        "predator_deaths",
        "starvation_deaths",
        "total_herbivore_deaths",
        "reproduction_count",
        "runtime_seconds",
        "error",
    ]
    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for run in summary["runs"]:
            writer.writerow(run)


def run_parallel(
    base_seed: int,
    config: PolicyConfig,
    output_dir: str | Path,
    n_cores: int,
) -> dict[str, Any]:
    """Run exactly ``n_cores`` independent episodes using ``n_cores`` worker processes.

    The mapping is deterministic: worker/run i receives ``base_seed + i``. Each
    process owns its own SimulationCore and SurvivalController, so no per-agent
    memory, RNG state, simulator state, or result files are shared across runs.
    """
    seeds = planned_seeds(base_seed, n_cores)
    out = Path(output_dir)
    runs_dir = out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    config.to_json(out / "params.json")

    batch_config = {
        "base_seed": base_seed,
        "seeds": seeds,
        "n_cores": n_cores,
        "n_runs": len(seeds),
        "controller_config_hash": config.stable_hash,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "multiprocessing_start_method": "spawn",
    }
    (out / "batch_config.json").write_text(
        json.dumps(batch_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    started = time.perf_counter()
    config_dict = config.to_dict()
    run_summaries: list[dict[str, Any]] = []

    # spawn gives every worker a clean interpreter, avoiding accidental inherited
    # simulator/controller globals and making behavior consistent across platforms.
    mp_context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_cores, mp_context=mp_context) as pool:
        future_to_seed = {}
        for run_index, seed in enumerate(seeds):
            run_dir = runs_dir / f"run_{run_index:03d}_seed_{seed}"
            future = pool.submit(_run_one_worker, seed, config_dict, str(run_dir))
            future_to_seed[future] = seed

        for future in as_completed(future_to_seed):
            seed = future_to_seed[future]
            try:
                run_summaries.append(future.result())
            except Exception as exc:  # defensive: worker normally catches its own errors
                run_summaries.append(
                    {
                        "seed": seed,
                        "status": "failure",
                        "error": f"worker_process_error: {type(exc).__name__}: {exc}",
                    }
                )

    summary = summarize_runs(
        run_summaries,
        base_seed=base_seed,
        n_cores=n_cores,
        config_hash=config.stable_hash,
        wall_runtime_seconds=time.perf_counter() - started,
    )
    _write_batch_results(out, summary)
    return summary


def _print_single_summary(metrics: RunMetrics) -> None:
    print(
        f"score={metrics.score:.3f}, survival_time={metrics.survival_time:.1f}, "
        f"final_population={metrics.final_population}, predator_deaths={metrics.predator_deaths}, "
        f"starvation_deaths={metrics.starvation_deaths}, runtime={metrics.runtime_seconds:.2f}s"
    )


def _print_batch_summary(summary: dict[str, Any]) -> None:
    score_stats = summary.get("aggregate", {}).get("score") or {}
    survival_stats = summary.get("aggregate", {}).get("survival_time") or {}
    score_text = f"{score_stats.get('mean', float('nan')):.3f}" if score_stats else "n/a"
    survival_text = f"{survival_stats.get('mean', float('nan')):.1f}" if survival_stats else "n/a"
    print(
        f"runs={summary['n_runs']}, successful={summary['successful_runs']}, "
        f"failed={summary['failed_runs']}, mean_score={score_text}, "
        f"mean_survival_time={survival_text}, wall_runtime={summary['wall_runtime_seconds']:.2f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the Survival Simulator controller. --n-cores N runs exactly N "
            "independent seeds in parallel; no hyperparameter search is performed."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Single-run seed, or base seed for a parallel batch (seeds are seed..seed+n_cores-1).",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/evaluation"))
    parser.add_argument(
        "--n-cores",
        type=int,
        default=1,
        help="Number of worker processes. For N>1, exactly N experiments are launched concurrently.",
    )
    args = parser.parse_args()

    if args.n_cores < 1:
        parser.error("--n-cores must be >= 1")

    config = PolicyConfig.from_json(args.config) if args.config else load_default_config()

    if args.n_cores == 1:
        metrics = run_one(args.seed, config, args.output_dir)
        _print_single_summary(metrics)
        return

    summary = run_parallel(
        base_seed=args.seed,
        config=config,
        output_dir=args.output_dir,
        n_cores=args.n_cores,
    )
    _print_batch_summary(summary)
    if summary["failed_runs"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
