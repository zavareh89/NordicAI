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

## Hyperparameter optimization: 100 TPE candidates × 30 seeds

The repository now includes `hpo.py`, a thin optimization layer on top of the existing `run_parallel(...)` evaluator.
The controller, API, simulator integration, per-agent memory, and evaluation path are unchanged.

### Why TPE

For this controller, Optuna's Tree-structured Parzen Estimator (TPE) is a better fit than a Gaussian-process Bayesian
optimizer or a large evolutionary population because the budget is only 100 candidate configurations while the search
space contains a mixture of continuous and integer parameters. TPE is inexpensive, handles mixed bounded spaces directly,
and its multivariate mode can learn interactions such as predator distance/weight or reproduction threshold/cooldown.

The schedule is:

```text
C001 = your existing default configuration (never overwritten by hpo.py)
C002..C021 = 20 TPE startup/random candidates
C022..C101 = 80 adaptive TPE candidates
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

Therefore 100 HPO candidates correspond to 3,000 simulator episodes, but at most 30 simulator workers are active at once.
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

The ranges are intentionally local around the hand-engineered C001 defaults. With only 100 samples in a 28-dimensional
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

### Run all 100 candidates

From the official `survival-simulator/` directory:

```bash
python hpo.py \
  --results-root results \
  --n-trials 100 \
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
├── C101/
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

Do not delete `hpo_study.db` while keeping partially populated `C002..C101` directories. The database is what keeps
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

## Holdout selection of the best original HPO candidate

Do not select the final v1 configuration using the same `1000..1029` seeds that drove TPE. The repository now includes
`validate_holdout.py` so shortlisted candidates can be evaluated apples-to-apples on a completely new common seed set.
The default holdout set is:

```text
50000, 50001, ..., 50029
```

Use the **same base seed for every candidate**. The five shortlisted configurations from the supplied HPO table are
C119, C181, C173, C156, and C167. Ready-to-run copies are included under `config/holdout_candidates/`.

Because architecture v2 changes the controller, original HPO candidates must first be selected with the old algorithm.
Pass `--legacy-v1` to force `architecture_v2_enabled=false` even if the source JSON was created before that field existed:

```bash
python validate_holdout.py run \
  --case-id C119 \
  --config results/C119/params.json \
  --results-root results_holdout \
  --base-seed 50000 \
  --n-seeds 30 \
  --n-cores 30 \
  --legacy-v1
```

Repeat only the case/config arguments while keeping the seed arguments identical:

```bash
python validate_holdout.py run --case-id C181 --config results/C181/params.json --results-root results_holdout --base-seed 50000 --n-seeds 30 --n-cores 30 --legacy-v1
python validate_holdout.py run --case-id C173 --config results/C173/params.json --results-root results_holdout --base-seed 50000 --n-seeds 30 --n-cores 30 --legacy-v1
python validate_holdout.py run --case-id C156 --config results/C156/params.json --results-root results_holdout --base-seed 50000 --n-seeds 30 --n-cores 30 --legacy-v1
python validate_holdout.py run --case-id C167 --config results/C167/params.json --results-root results_holdout --base-seed 50000 --n-seeds 30 --n-cores 30 --legacy-v1
```

If the original `results/Cxxx` folders are unavailable, substitute the bundled files such as
`config/holdout_candidates/C119.json`.

After all candidates finish:

```bash
python validate_holdout.py compare \
  --results-root results_holdout \
  --expected-runs 30
```

The comparison refuses to rank candidates if their seed lists differ. It writes `holdout_leaderboard.csv` and
`holdout_comparison.json`, ranking by the same robust criterion used during HPO:

```text
0.8 * mean_score + 0.2 * p10_score
```

The mean, median, p10, minimum, maximum, and mean survival time are also retained. This holdout set should remain a
selection set: do not feed its results back into another long TPE run if you want it to remain an unbiased check.

## Architecture v2: incremental controller upgrade

The original hierarchy, per-agent memory, potential-field proposal, API, parallel evaluator, and HPO plumbing remain in
place. Architecture v2 changes only the decision layer around them. It can be disabled with:

```json
"architecture_v2_enabled": false
```

This gate was verified against the previous overlay on a synthetic multi-tick sequence: the v1-gated controller produced
identical actions. This is also what `validate_holdout.py --legacy-v1` uses.

### 1. Short-horizon micro-planner

New file: `survival_policy/planner.py`.

The hierarchy/potential field still proposes the preferred direction. `MicroPlanner` then evaluates 20 nearby actions
(5 direction offsets × 4 movement magnitudes) over a two-step geometric horizon. Candidate utility contains predator
separation/TTC, fruit progress, approximate movement/turning energy, herbivore spacing, the v1 preferred-direction prior,
and wall feasibility. No neural inference or global optimization is introduced.

Integration block: `SurvivalController._decide_agent()` after `_desired_vector()`, `_turn_angle()`, and
`_movement_distance()` compute the v1 proposal.

### 2. Predictive predator handling and time-to-collision

Modified files: `survival_policy/memory.py` and `survival_policy/controller.py`.

`AgentMemory` now keeps a smoothed closing rate, bearing-rate estimate, and `predator_ttc_ticks`. The controller derives a
dynamic danger distance from TTC, energy reserve, and wall pressure. Imminent closing threats therefore trigger earlier
than equally distant predators that are not closing. The micro-planner also scores predicted predator separation and
preserves the useful v1 behavior of keeping the body partially oriented toward a nearby predator while moving tangentially
or away.

### 3. State-dependent movement and thresholds

Modified block: `SurvivalController._movement_distance()` and the new `_dynamic_predator_distance()` / `_dynamic_spawn_threshold()` helpers.

Foraging slows near a fruit to reduce overshoot, conservation becomes more aggressive as energy falls, critical foraging
can spend slightly more movement energy, and evasion sprint fraction increases when TTC is urgent. The v1 constants remain
the baseline values rather than being discarded.

### 4. Dynamic reproduction and carrying capacity

Modified files: `survival_policy/coordination.py` and `survival_policy/controller.py`.

`PopulationContext` aggregates only observable signals: population count/trend, average energy ratio, fruit sightings per
agent, and predator sightings per agent. `estimate_carrying_capacity()` adjusts the original soft cap within a bounded
bonus. The spawn threshold falls in safe/food-rich/high-energy or collapsing-population conditions and rises with nearby
danger, crowding, or capacity pressure. The simulator's hard spawn mechanics are not changed.

### 5. Temporal pseudo-fruit identities and stronger assignment

Modified files: `survival_policy/memory.py`, `survival_policy/coordination.py`, and the fruit-selection block in
`survival_policy/controller.py`.

`FruitTrack` provides short-lived per-agent pseudo-IDs using angle/distance matching in that agent's internal heading
frame. Target persistence can therefore refer to a tracked fruit rather than only a loose geometric match.

A truly global fruit reservation table is intentionally **not** implemented because the challenge API exposes neither
fruit IDs nor globally aligned herbivore positions. Instead `local_fruit_owner()` deterministically gives a candidate to
the mutually visible herbivore with the stronger geometric claim, while the existing soft competition cost remains as a
secondary signal. This is the strongest assignment justified by the observable state without inventing coordinates.

### 6. Coordinated exploration sectors

Modified block: `SurvivalController._ensure_exploration_heading()`.

Architecture v2 sorts active IDs and distributes exploration headings across equal angular sectors, then applies a small
deterministic jitter and slow epoch rotation. This replaces per-agent random-like phase offsets with population-aware
coverage while keeping exploration deterministic and persistent.

### 7. Wall handling as feasibility rather than a dominant force

Modified files: `survival_policy/geometry.py`, `survival_policy/planner.py`, and `_desired_vector()`.

The v1 wall field remains as a weak proposal prior and as an emergency wall-avoidance state. The final action is instead
penalized/rejected when its projected endpoint violates hard/soft segment-clearance margins. This prevents wall repulsion
from fighting food/predator objectives when the wall is not actually constraining the next action.

### Architecture-v2 configuration and HPO compatibility

The new fields are appended to `PolicyConfig`, so historical Cxxx JSON files still load: omitted fields take conservative
v2 defaults. The original 28 HPO dimensions are unchanged; architecture-v2 controls are currently fixed rather than
silently expanding an already large search space.

`hpo.py` now records these fixed fields and refuses to mix v1 and v2 trials in the same results root. An old HPO directory
whose metadata predates the v2 flag is interpreted as v1. To optimize v2 later, use a **fresh results root**; do not append
v2 trials to the database that produced C002..C191.
