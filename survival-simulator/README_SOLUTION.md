# Survival Simulator controller overlay

This directory is an overlay for `survival-simulator/` in the Nordic AI Cup 2026 repository.
Copy these files into that directory. The only existing challenge file intentionally replaced is `agent_server.py`.
All official `src/` simulator files should remain unchanged.

The controller itself is lightweight and deterministic. Evaluation can now run either a single episode or a parallel
batch of independent episodes. Parallel evaluation does **not** perform hyperparameter optimization: every worker uses
the exact same `PolicyConfig`; only the simulator seed differs.

## Production server

```bash
python agent_server.py
```

Optional environment variables:

- `SURVIVAL_POLICY_CONFIG=/path/to/params.json`
- `SURVIVAL_AGENT_HOST=0.0.0.0`
- `SURVIVAL_AGENT_PORT=9052`
- `LOG_LEVEL=INFO`

The production API remains single-controller inference code. The multiprocessing implementation exists only in
`evaluate.py`; no process pool or evaluation logic is added to the serving path.

## Unit tests

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

The tests include deterministic seed planning and batch-summary aggregation in addition to the controller/API tests.

## Single local evaluation

`--n-cores 1` preserves the single-run behavior:

```bash
python evaluate.py \
  --seed 1 \
  --n-cores 1 \
  --output-dir results/single_run
```

The single-run directory contains:

```text
results/single_run/
├── params.json
├── run_config.json
├── metrics.json
└── population_trajectory.csv
```

## Parallel evaluation

For `N > 1`, `--n-cores N` starts **N independent worker processes and exactly N simulator experiments**.
For example:

```bash
python evaluate.py \
  --seed 1000 \
  --n-cores 30 \
  --output-dir results/batch_30
```

This launches 30 experiments concurrently with deterministic seeds:

```text
1000, 1001, 1002, ..., 1029
```

The rule is always:

```text
seed_i = base_seed + i,  i = 0 .. n_cores-1
```

Therefore the same configuration, base seed, and `n_cores` produce the same seed set every time. Each worker creates a
fresh `SimulationCore` and a fresh `SurvivalController`, so simulator state, controller memory, RNG state, and result
files are isolated between experiments.

The multiprocessing start method is explicitly `spawn`. This avoids inheriting mutable simulator/controller globals
and also keeps the implementation portable across Linux, macOS, and Windows.

A parallel result directory looks like:

```text
results/batch_30/
├── params.json
├── batch_config.json
├── summary.json
├── runs.csv
└── runs/
    ├── run_000_seed_1000/
    │   ├── params.json
    │   ├── run_config.json
    │   ├── metrics.json
    │   └── population_trajectory.csv
    ├── run_001_seed_1001/
    │   └── ...
    └── run_029_seed_1029/
        └── ...
```

`summary.json` records the exact seeds, configuration hash, worker count, success/failure count, total wall-clock runtime,
per-run scalar metrics, and aggregate statistics over successful runs. `runs.csv` provides one compact row per run.
Each run directory still preserves its complete population trajectory and exact parameter/config metadata.

If a worker encounters an exception, its run directory stores failure metrics and the batch continues collecting results
from the other workers. The CLI exits non-zero if any worker failed, while still preserving `summary.json` and `runs.csv`.

### Choosing `n_cores`

`n_cores` is deliberately an evaluation setting, not a policy parameter. A reasonable value is the number of CPU cores
you actually want to dedicate to simulation. For example, on a 32-core machine you might use 30 and leave two cores for
the OS/other tasks:

```bash
python evaluate.py --seed 1 --n-cores 30 --output-dir results/batch_30
```

One execution then means 30 experiments. There is no nested multiprocessing inside an experiment, so CPU allocation is
straightforward and there is no shared controller state between runs.

Be aware that each worker is a separate Python process with its own simulator memory. Do not set `n_cores` higher than
the machine can comfortably support merely because more logical CPUs are reported.

## Custom configuration

Defaults live in:

```text
config/default_params.json
```

Run either single or parallel evaluation with another configuration using:

```bash
python evaluate.py \
  --config path/to/candidate_params.json \
  --seed 1000 \
  --n-cores 30 \
  --output-dir results/candidate_001
```

`PolicyConfig.from_json(...)` validates a replacement JSON. `PolicyConfig.to_json(...)` saves a deterministic
representation, and `stable_hash` fingerprints the exact controller parameter vector.

## Optimization architecture

The controller and evaluator remain optimizer-independent:

```text
parameter dictionary / JSON
        ↓
PolicyConfig
        ↓
SurvivalController
        ↓
evaluate.py
        ↓
run_one(...) or run_parallel(...)
```

`hpo.py` now sits above this unchanged interface. It proposes a candidate `PolicyConfig`, calls `run_parallel(...)`,
records the resulting statistics, and feeds one scalar objective back to Optuna/TPE. No optimizer code is present in the
production controller or API path.

## Reproducibility

For a parallel batch, reproducibility is controlled by:

1. the exact `params.json` / configuration hash;
2. the base `--seed`;
3. `--n-cores`, which determines the deterministic seed set;
4. the simulator/repository version.

The order in which worker processes finish does not affect `summary.json`: results are sorted by seed before they are
written. Worker completion timing therefore cannot change aggregation ordering.

## Sandbox smoke-test note

The bundled historical `results/smoke_test` records the earlier correctness smoke run performed in the ChatGPT sandbox.
That sandbox could not clone GitHub or install pygame, so the run used a headless compatibility copy reconstructed from
the verified public simulator mechanics. Those compatibility files are deliberately **not** included in this overlay.

For authoritative pre-submission verification, run the evaluator inside a fresh official repository checkout. Parallel
execution changes only evaluation orchestration; it does not alter the production controller or API behavior.

## Hyperparameter optimization: 200 TPE candidates × 30 seeds

The repository now includes `hpo.py`, a thin optimization layer on top of the existing `run_parallel(...)` evaluator.
The controller, API, simulator integration, per-agent memory, and evaluation path are unchanged.

### Why TPE

For this controller, Optuna's Tree-structured Parzen Estimator (TPE) is a better fit than a Gaussian-process Bayesian
optimizer or a large evolutionary population because the budget is only 200 candidate configurations while the search
space contains a mixture of continuous and integer parameters. TPE is inexpensive, handles mixed bounded spaces directly,
and its multivariate mode can learn interactions such as predator distance/weight or reproduction threshold/cooldown.

The schedule is:

```text
C001 = your existing default configuration (never overwritten by hpo.py)
C002..C041 = 40 TPE startup/random candidates
C022..C201 = 80 adaptive TPE candidates
```

Candidate configurations are evaluated **sequentially at the HPO level** because TPE needs the preceding results to
choose the next configuration. Inside every candidate, the existing evaluator launches 30 simulations concurrently:

```text
1 candidate parameter set
        ↓
30 fixed simulator seeds
        ↓
30 worker processes / cores
        ↓
summary.json
        ↓
TPE observes the candidate objective and proposes the next candidate
```

Therefore 200 HPO candidates correspond to 3,000 simulator episodes, but at most 30 simulator workers are active at once.
Do not set Optuna itself to run multiple candidates concurrently on the same 30-core machine; the parallelism belongs
inside each candidate batch.

### Fair comparison across candidates

All candidates use the same seeds by default:

```text
1000, 1001, ..., 1029
```

This common-random-number design substantially reduces noise when comparing two parameter vectors. Otherwise TPE could
reward a candidate simply because it happened to receive easier simulator seeds.

The default optimization objective is a mildly downside-aware score:

```text
objective = 0.8 * mean_score + 0.2 * p10_score
```

The mean remains dominant, while the p10 term discourages parameter sets that have a good average but repeatedly collapse
on a minority of seeds. Every raw run and the full mean/median/min/max/p10 summary are still stored. If you want to optimize
only expected score, pass `--objective mean`.

### Search ranges

The ranges are intentionally local around the hand-engineered C001 defaults. With only 200 samples in a 28-dimensional
space, very broad intervals would spend most of the budget in implausible regions.

| Parameter | C001 default | HPO range |
|---|---:|---:|
| `master_seed` | 20260918 | **fixed** (not optimized) |
| `emergency_predator_distance` | 45 | 35–60 |
| `predator_danger_distance` | 150 | 120–190 |
| `predator_repulsion_weight` | 4.5 | 3.0–7.0 |
| `predator_prediction_weight` | 0.65 | 0.25–1.00 |
| `escape_persistence_ticks` | 6 | 4–10 integer |
| `wall_danger_distance` | 52 | 40–70 |
| `wall_repulsion_weight` | 3.2 | 2.0–5.0 |
| `herbivore_repulsion_radius` | 42 | 30–60 |
| `herbivore_repulsion_weight` | 0.65 | 0.30–1.00 |
| `max_turn_angle` | 0.35 | 0.25–0.45 |
| `critical_energy_ratio` | 0.23 | 0.18–0.28 |
| `low_energy_ratio` | 0.36 | 0.31–0.43 |
| `fruit_attraction_weight` | 2.4 | 1.70–3.50 |
| `fruit_distance_penalty` | 0.010 | 0.006–0.016, log sampled |
| `fruit_competition_penalty` | 1.25 | 0.70–1.80 |
| `target_persistence_bonus` | 0.70 | 0.40–1.10 |
| `target_timeout_ticks` | 24 | 16–36 integer |
| `forage_move_fraction` | 0.86 | 0.75–0.95 |
| `explore_move_fraction` | 0.58 | 0.45–0.68 |
| `conserve_move_fraction` | 0.32 | 0.20–0.40 |
| `evasion_sprint_fraction` | 0.78 | 0.68–0.90 |
| `exploration_change_interval` | 45 | 30–70 integer |
| `spawn_min_energy` | 225 | 190–260 |
| `spawn_old_age` | 55 | 42–70 |
| `population_soft_cap` | 12 | 9–16 integer |
| `reproduction_cooldown_ticks` | 90 | 65–130 integer |
| `stuck_tick_threshold` | 7 | 5–10 integer |
| `recovery_ticks` | 6 | 4–9 integer |

`master_seed` is intentionally not optimized. Selecting a lucky controller RNG seed would be optimization of randomness,
not optimization of behavior, and is unlikely to transfer to hidden evaluation conditions. It remains fixed in every
candidate. The static intervals also guarantee the existing config constraints: the emergency predator range is always
below the danger range, and the critical-energy range is always below the low-energy range.

### Install HPO dependency

```bash
python -m pip install -r requirements-hpo.txt
```

### Run all 200 candidates

From the official `survival-simulator/` directory:

```bash
python hpo.py \
  --results-root results \
  --n-trials 200 \
  --n-cores 30 \
  --base-seed 1000
```

This leaves `results/C001` untouched and creates exactly:

```text
results/
├── C001/                         # existing default, untouched
├── C002/
│   ├── params.json
│   ├── batch_config.json
│   ├── summary.json
│   ├── runs.csv
│   ├── hpo_trial.json
│   └── runs/
├── C003/
│   └── ...
├── ...
├── C201/
│   └── ...
├── hpo_study.db                 # persistent Optuna study
├── hpo_config.json              # HPO-level reproducibility settings
├── hpo_search_space.json        # exact parameter ranges
├── hpo_trials.csv               # one row per Cxxx candidate
├── hpo_summary.json             # current progress + best HPO case
└── best_params.json             # current best HPO parameter vector
```

Each `Cxxx/summary.json` is produced by the unchanged parallel evaluator and therefore contains the full batch statistics
and per-seed scalar results. `hpo_trials.csv` gives a compact cross-candidate table including objective value, score
statistics, config hash, and all sampled parameters.

### Resume after interruption

The Optuna study is persisted in `results/hpo_study.db`. Running the same command again resumes it instead of starting
another study. If the process was interrupted while one case was RUNNING, `hpo.py` checks that case first: if a complete
`summary.json` exists it reuses it; otherwise it reruns exactly that saved parameter vector on the same fixed seed batch.
It then continues with the next `Cxxx` identifier.

Do not delete `hpo_study.db` while keeping partially populated `C002..C201` directories. The database is what keeps
Optuna trial numbers synchronized with those directory names. For a completely fresh optimization, use a fresh results
root. `--reset-study` deletes only the SQLite study and intentionally does not silently delete existing candidate results.

### Short functionality test

Before committing thousands of simulator episodes, the HPO orchestration can be smoke-tested with a small temporary
results directory:

```bash
python hpo.py \
  --results-root results/hpo_smoke \
  --n-trials 2 \
  --n-cores 2 \
  --base-seed 1
```

This tests TPE sampling, parameter validation, the nested 2-process evaluator, result persistence, Optuna ask/tell state,
and global HPO summaries. It is only a functionality test; do not use its scores for parameter selection.
