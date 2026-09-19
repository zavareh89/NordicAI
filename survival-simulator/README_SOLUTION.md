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

## Focused architecture-v2 HPO: C167-v2 baseline + 99 TPE candidates

The original 190-trial v1 study is now treated as **historical evidence**, not restarted. C167 was selected on unseen
seeds, and the architecture-v2 comparison showed a large improvement over C167-v1. The new `hpo.py` therefore performs
a focused search around **C167-v2** instead of reopening all 28 legacy parameters.

This is intentional: with a 100-candidate budget, searching 28 old dimensions plus every new v2 control would make TPE
spend most of its budget relearning parameter regions that the first study already established.

### Search schedule and seed hygiene

The focused study uses a fresh common seed set:

```text
70000, 70001, ..., 70029
```

These seeds are distinct from the original HPO (`1000..1029`), v1 candidate holdout (`50000..50029`), and v1-v2
architecture comparison (`60000..60029`). Every focused-HPO candidate is evaluated on the same 30 seeds.

Case numbering continues after the historical C002-C191 study:

```text
C192 = exact C167-v2 base configuration, reevaluated on 70000..70029
C193..C291 = 99 focused TPE candidates
```

C192 is deliberately enqueued before sampling so the new study always contains an apples-to-apples baseline on its own
training seeds.

The scalar objective remains:

```text
objective = 0.8 * mean_score + 0.2 * p10_score
```

This keeps the optimization criterion consistent with the original model-selection procedure while still penalizing
catastrophic tail failures.

### Parameters searched

Only 17 parameters are reopened.

| Parameter | C167-v2 base | Focused range | Why it remains tunable |
|---|---:|---:|---|
| `emergency_predator_distance` | 39.11 | 33–45 | v2 TTC changes the close-range emergency boundary |
| `predator_danger_distance` | 187.45 | 175–225 | old HPO pushed upward; it is the base v2 danger radius |
| `predator_prediction_weight` | 0.960 | 0.80–1.15 | old upper-bound signal + predictive planner use |
| `escape_persistence_ticks` | 10 | 9–18 | old HPO hit its upper bound; important for tail survival |
| `evasion_sprint_fraction` | 0.805 | 0.75–0.95 | directly defines nominal v2 evasion speed |
| `exploration_change_interval` | 68 | 55–100 integer | old upper-bound signal; v2 sector exploration makes persistence directly relevant |
| `fruit_distance_penalty` | 0.01523 | 0.013–0.022 | strong old directional signal and still used in target selection |
| `spawn_min_energy` | 201.89 | 180–225 | base of the new dynamic reproduction threshold |
| `population_soft_cap` | 15 | 14–20 | base of the v2 adaptive carrying capacity |
| `planner_horizon_steps` | 2 | 1–3 integer | controls candidate projection horizon |
| `planner_angle_spread` | 0.42 | 0.28–0.60 | controls directional alternatives around the v1 proposal |
| `planner_predator_weight` | 4.0 | 2.8–6.5 | main final-action safety weight |
| `planner_food_weight` | 1.7 | 1.1–2.6 | safety/food tradeoff in final action selection |
| `planner_energy_weight` | 0.55 | 0.30–0.90 | movement/turn-energy tradeoff |
| `planner_wall_clearance` | 18 | 13–24 | hard predicted wall-clearance threshold |
| `ttc_danger_ticks` | 22 | 14–34 | predictive danger urgency window |
| `dynamic_population_bonus` | 6 | 3–10 integer | amount the adaptive carrying capacity may move from the base cap |

Everything else is **fixed to the holdout-selected C167-v2 values**. Important examples deliberately fixed include old
wall-force parameters, `max_turn_angle`, energy thresholds, general foraging/exploration speeds, fruit-attraction and
persistence terms, herbivore-repulsion terms, reproduction cooldown/old-age logic, stuck recovery, fruit-track matching
thresholds, and `planner_wall_soft_clearance`.

Those fields are fixed for specific reasons: the first HPO already converged them sufficiently, their v2 role is mostly
superseded by another mechanism, or they are lower-level implementation tolerances that should not consume a 100-trial
search budget without evidence that they are limiting performance.

The complete searched/fixed values and a rationale for every searched field are written to `hpo_search_space.json` when
the study starts.

### Base configuration

The package includes:

```text
config/C167_v2_hpo_base.json
```

This is the exact holdout-selected C167 parameter vector with `architecture_v2_enabled=true`. The focused HPO refuses to
start from a v1 configuration or with the planner disabled.

### Install HPO dependency

```bash
python -m pip install -r requirements-hpo.txt
```

### Run the 100-case focused search on 30 cores

From the official `survival-simulator/` directory:

```bash
python hpo.py \
  --base-config config/C167_v2_hpo_base.json \
  --results-root results_v2_hpo \
  --n-trials 100 \
  --n-cores 30 \
  --base-seed 70000
```

The default arguments already match this command, so `python hpo.py` is sufficient if the bundled paths are unchanged.

The output is:

```text
results_v2_hpo/
├── C192/                         # exact C167-v2 baseline on new HPO seeds
├── C193/
├── ...
├── C291/
├── hpo_v2_focused_study.db       # persistent Optuna study
├── hpo_base_config.json          # exact frozen baseline
├── hpo_config.json               # seed/objective/study settings
├── hpo_search_space.json         # searched + fixed parameters and rationale
├── hpo_trials.csv                # one row per candidate
├── hpo_summary.json              # progress + current best
└── best_params.json              # current best complete PolicyConfig
```

At the HPO level candidates are still evaluated sequentially. Inside each candidate, `evaluate.run_parallel()` launches
30 independent simulator episodes concurrently, one per core. Thus 100 cases correspond to 3,000 simulator episodes but
never to 100 simultaneous Optuna candidates.

### Resume safety

The study remains resumable after interruption. In addition to the existing Optuna SQLite state, focused HPO now refuses
to resume if the base seed, number of per-candidate seeds/cores, objective, base C167-v2 config, or search-space definition
has changed. This prevents silently mixing incomparable trials in one study.

### After focused HPO

Do **not** choose the final competition configuration only from `70000..70029`; those seeds are now optimization data.
Shortlist the best few C192-C291 candidates and evaluate them on another untouched common seed set, e.g.:

```text
80000..80029
```

Use `validate_holdout.py` for that final comparison, exactly as for the earlier C167 selection.

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
