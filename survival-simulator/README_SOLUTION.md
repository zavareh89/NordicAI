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

## Future hyperparameter optimization

No optimizer is included or invoked by this implementation. The separation is intentionally:

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

A future Optuna/TPE/CMA-ES/etc. script can therefore create a candidate `PolicyConfig` and call the evaluator without
changing controller internals. A candidate can be evaluated over `n_cores` deterministic seeds in one invocation, but
this repository itself does not search for candidate parameter values.

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
