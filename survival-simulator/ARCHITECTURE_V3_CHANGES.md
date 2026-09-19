# Architecture V3 change map

V3 is an incremental upgrade on top of the existing C376-v2 controller. The API,
parallel evaluator, memory model, HPO orchestration and v1/v2 compatibility are
kept. `architecture_v3_enabled=false` returns to the v2 planner path.

## Files changed

### `survival_policy/config.py`
**Block: `PolicyConfig` v3 section**

Adds the v3 gate and latency-bounded rollout controls:

- `architecture_v3_enabled`
- `planner_beam_width`
- `planner_rollout_discount`
- `planner_followup_angle_spread`
- `planner_cpa_weight`
- `planner_energy_score_scale`
- `planner_path_wall_soft_clearance`
- `reproduction_slots_per_tick`
- `population_capacity_smoothing`
- `exploration_no_food_ticks`
- `exploration_stale_fallback_ticks`

Validation was extended for these fields. The default production configuration is
now the selected C376 parameter vector with v3 enabled.

### `survival_policy/geometry.py`
**Block: segment / predictive geometry helpers**

Adds:

- `dot()` / `cross()`
- `segments_intersect()`
- `segment_segment_distance()`
- `closest_approach()`

These are used by the v3 planner for movement-path wall checks and
candidate-specific predator closest-point-of-approach (CPA) scoring.

### `survival_policy/memory.py`
**Block: `AgentMemory` exploration fields**

Adds event-driven exploration state:

- `exploration_last_change_tick`
- `exploration_epoch`
- `no_food_ticks`
- `was_in_predator_state`

**Block: `PopulationMemory`**

Adds a smoothed carrying-capacity estimate so reproduction does not react to
single-tick fruit/predator observation noise.

### `survival_policy/coordination.py`
**Block: fruit ownership**

Adds `local_fruit_owner_eta()`. V3 deconfliction estimates time-to-capture rather
than using only raw Euclidean distance. Because other herbivore speed is not
exposed by the public API, their direction is used to adjust a common local speed
scale; no unavailable global state is invented.

**Block: reproduction coordination**

Adds:

- `reproduction_priority()`
- `select_reproduction_slots()`

Only the best one (configurable) current reproduction candidate is allowed to
enter `REPRODUCTION`, preventing synchronized population bursts.

### `survival_policy/planner.py`
**Block: v2 compatibility path**

The old endpoint planner remains in `_plan_v2()` and is selected whenever
`architecture_v3_enabled=false`.

**Block: state-adaptive candidate generation**

`_state_template()` creates different low-cost candidate sets for evasion,
foraging, conservation, wall avoidance, recovery and exploration.

**Block: true sequential rollout**

`_plan_v3()` replaces the v2 terminal-point multiplier with a real sequence of
future state transitions. The default is deliberately bounded to:

- horizon: 3
- beam width: 1
- roughly 10 first-step candidates + 2 follow-up candidates per future step

That is about 14 scored action-steps per decision versus 20 independent v2
candidates. The beam-search framework supports wider beams, but they are not the
competition default because of latency.

**Block: predator CPA/TTC**

`_cpa_predator_score()` scores each candidate using predicted closest approach
rather than only current distance. Multiple predators use a worst-threat-dominant
aggregate.

**Block: path-based wall geometry**

`_path_wall_score()` checks the full movement segment against observed edge
segments. A crossing receives a hard penalty; near-path clearance receives a
soft penalty. This replaces v2's endpoint-only wall clearance in the final action
selection.

**Block: cumulative energy scoring**

`_simulator_aligned_energy_score()` accumulates energy cost at each rollout step.
The normal-movement component uses the verified challenge coefficient
`0.05 * distance`. Sprint and turning are kept as deterministic convex/turning
surrogates because those exact simulator constants are not present in this
standalone overlay. If the official `environment.py` is supplied in the working
tree, this is the one block to align byte-for-byte with its exact formulas.

### `survival_policy/controller.py`
**Block: `decide_step()` population context**

Adds carrying-capacity smoothing and population-level reproduction-slot
allocation before per-agent decisions.

**Block: `_ensure_exploration_heading()`**

V3 exploration is event-driven. A sector changes after meaningful events such as
prolonged no-food exploration, escape/recovery transitions or the long safety
fallback, rather than simply every old exploration interval.

**Block: `_fruit_utility()` / `_select_fruit()`**

V3 converts fruit travel cost from raw distance to a local time-to-capture proxy
and uses ETA-based local ownership. Existing pseudo-fruit tracks and target
persistence remain intact.

**Block: `_state_for()`**

Reproduction now requires both normal eligibility and a population-level spawn
slot.

**Block: `_decide_agent()`**

Adds event bookkeeping and passes `max_energy` into the v3 rollout planner. The
existing hierarchy and potential field still generate the preferred direction;
v3 only replaces the final action-selection layer.

### `hpo.py`
**Block: focused v3 search definition**

The HPO baseline is now C376-v3. Historical candidate numbering is preserved:

- C202-C401: previous focused v2 study
- C402: exact C376-v3 baseline on the new v3 HPO seed batch
- C403-C501: v3 TPE candidates

The default v3 tuning seeds are `90000..90029`.

The first v3 HPO intentionally fixes latency-sensitive structural values:

- `planner_horizon_steps = 3`
- `planner_beam_width = 1`

Only 17 parameters whose semantics changed or remain materially uncertain are
searched. The previous C376 values for converged v1/v2 parameters stay frozen.

### `benchmark_v3_latency.py`
New deterministic synthetic controller-only benchmark. It compares C376-v2 and
C376-v3 on the same preconstructed observations and fails if v3 exceeds 60%
overhead.

### `config/C376_v3_hpo_base.json`
New C376-v3 base and HPO baseline.

### `config/C376_v2.json`
Same C376 parameter vector with `architecture_v3_enabled=false` for direct v2-v3
comparison.

### `config/default_params.json`
Updated to C376-v3 so the production API loads v3 by default.

### `tests/test_architecture_v3.py`
Adds tests for:

- path-wall intersection/distance
- CPA geometry
- ETA fruit ownership
- reproduction-slot coordination
- finite v3 planner actions
- event-driven exploration persistence

### `tests/test_hpo.py`
Updated for C376-v3, C402-C501 numbering, the new focused search dimensions,
and v3 resume protection.

## Files intentionally unchanged

- `agent_server.py` API contract
- `evaluate.py` parallel simulation infrastructure
- `validate_holdout.py` common-seed validation workflow
- official simulator files under `src/`

## Latency result

Synthetic stress benchmark, 20 agents, fruits + predator + two walls + another
agent per observation, 120 ticks, 7 repetitions:

```text
v2_us_per_agent=139.61
v3_us_per_agent=204.30
overhead_percent=46.33
```

This is inside the requested 50-60% maximum overhead. Run the same benchmark on
the deployment machine before final submission:

```bash
python benchmark_v3_latency.py \
  --config config/C376_v3_hpo_base.json \
  --ticks 500 --agents 20 --repeats 5
```
