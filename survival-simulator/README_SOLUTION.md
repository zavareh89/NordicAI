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

## Architecture v3: C376 baseline + latency-bounded rollout planner

C376 is now the selected v2 baseline. V3 is implemented incrementally on top of
that controller; the hierarchy, memory, pseudo-fruit tracking, population
coordination, API and evaluator remain intact.

The main v3 changes are:

- true sequential horizon-3 rollout instead of multiplying one action to a terminal point;
- candidate-specific predator closest-point-of-approach / TTC scoring;
- cumulative simulator-aligned energy scoring across rollout steps;
- full movement-segment wall intersection/clearance checks;
- stable carrying-capacity smoothing plus population-level reproduction slots;
- event-driven exploration-sector switching;
- state-adaptive candidate action templates;
- ETA/time-to-capture fruit ownership and travel cost.

See `ARCHITECTURE_V3_CHANGES.md` for the exact file/block map.

### Compatibility

The old planner is preserved. To reproduce C376-v2 exactly, use:

```text
config/C376_v2.json
```

or set:

```json
"architecture_v3_enabled": false
```

The production `config/default_params.json` now contains C376 with v3 enabled.

### Controller latency

The v3 planner is intentionally bounded to horizon 3 and beam width 1. The
framework supports wider beams, but the competition default prioritizes the API
time budget. On the included 20-agent synthetic stress benchmark, controller-only
latency was:

```text
v2_us_per_agent=139.61
v3_us_per_agent=204.30
overhead_percent=46.33
```

Run the benchmark on the deployment machine:

```bash
python benchmark_v3_latency.py \
  --config config/C376_v3_hpo_base.json \
  --ticks 500 --agents 20 --repeats 5
```

The script exits non-zero if measured overhead exceeds 60%.

### Focused v3 HPO

Do not reopen all historical controller parameters. The v1/v2 studies already
converged many of them, and v3 changes only a subset of semantics. The new
`hpo.py` therefore freezes the C376 values for converged parameters and searches
17 parameters related to CPA/TTC, rollout scoring, wall-path clearance,
population smoothing/reproduction, and event-driven exploration.

The latency-sensitive structural values are fixed during this first v3 HPO:

```text
planner_horizon_steps = 3
planner_beam_width = 1
```

The new candidate numbering continues after the previous C202-C401 study:

```text
C402        exact C376-v3 baseline on the new HPO seed batch
C403-C501   focused v3 TPE candidates
```

The default common seed set is fresh:

```text
90000, 90001, ..., 90029
```

All candidates see exactly the same 30 seeds.

Install HPO dependencies if needed:

```bash
python -m pip install -r requirements-hpo.txt
```

Run the focused v3 study on 30 cores:

```bash
python hpo.py \
  --base-config config/C376_v3_hpo_base.json \
  --results-root results_v3_hpo \
  --n-trials 100 \
  --n-cores 30 \
  --base-seed 90000
```

The robust objective remains:

```text
0.8 * mean(score) + 0.2 * p10(score)
```

The study writes:

```text
results_v3_hpo/
├── C402/
├── C403/
├── ...
├── C501/
├── hpo_v3_focused_study.db
├── hpo_base_config.json
├── hpo_config.json
├── hpo_search_space.json
├── hpo_trials.csv
├── hpo_summary.json
└── best_params.json
```

Do not choose the final competition model solely on `90000..90029`. Shortlist a
few candidates and validate them on another untouched common seed batch with
`validate_holdout.py`.

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

The new fields are appended to `PolicyConfig`, so historical Cxxx JSON files still load. The original C002-C191 study is
kept as historical v1 evidence. The current `hpo.py` is a separate, focused architecture-v2 study built around the
holdout-selected C167 values; it does not append trials to the old study or reopen every legacy parameter.
