# Architecture v2 change map

This upgrade is incremental. The existing hierarchy, memory ownership, API contract, parallel evaluator, and original 28-D HPO search are retained.

## New files

- `survival_policy/planner.py`
  - `MicroPlanner`: 20-action short-horizon candidate generator/scorer.
  - `PlannedAction`: selected movement/turn result.
- `validate_holdout.py`
  - `run` subcommand: one shortlisted v1 config × 30 unseen seeds × 30 cores.
  - `compare` subcommand: verifies common seeds and writes/ranks holdout summaries.
- `config/holdout_candidates/C119.json`, `C181.json`, `C173.json`, `C156.json`, `C167.json`
  - Ready-to-run v1-gated copies reconstructed from the supplied HPO table.
- `tests/test_architecture_v2.py`
  - Pseudo-fruit IDs, local ownership, dynamic capacity, TTC, planner constraints, v1 gate.
- `tests/test_holdout.py`
  - Common-seed holdout ranking/persistence.

## Modified `survival_policy/config.py`

### `PolicyConfig`
Appended architecture-v2 fields only; all original HPO fields are unchanged.

Key additions:
- `architecture_v2_enabled`
- micro-planner geometry/weights
- TTC horizon
- pseudo-fruit tracking tolerances/TTL
- dynamic population bonus

Old Cxxx JSON files remain loadable because omitted new fields take dataclass defaults.

## Modified `survival_policy/memory.py`

### New `FruitTrack`
Short-lived, per-agent pseudo-ID for fruit observations.

### `AgentMemory`
Added:
- `target_track_id`
- local `fruit_tracks`
- predator closing-rate/bearing-rate estimates
- `predator_ttc_ticks`

### `PopulationMemory`
Added:
- sorted active IDs
- bounded population history
- population trend

## Modified `survival_policy/coordination.py`

Added:
- `PopulationContext`
- `local_fruit_owner()`
- `estimate_carrying_capacity()`
- `build_population_context()`

The old `local_competition_cost()` and `local_density()` remain.

## Modified `survival_policy/geometry.py`

Added general point-to-segment geometry:
- `closest_point_on_segment()`
- `segment_distance_to_point()`
- `subtract()`

Existing origin-distance helpers now delegate to the general implementation.

## Modified `survival_policy/controller.py`

### `__init__`
Creates one `MicroPlanner` per controller instance.

### `decide_step()`
Now updates bounded population history/context before individual decisions.

### `_ensure_exploration_heading()`
V2: equal population sectors + deterministic jitter/rotation.
V1 gate: original ID/golden-angle exploration is preserved exactly.

### `_fruit_utility()` / `_select_fruit()`
V2:
- deterministic local fruit owner
- persistent pseudo-track target bonus
- existing soft competition retained

V1 gate preserves original behavior.

### `_dynamic_predator_distance()`
New TTC/energy/wall-pressure dependent danger distance.

### `_dynamic_spawn_threshold()`
New food/energy/population/danger dependent reproduction threshold.

### `_state_for()`
Uses dynamic predator range, dynamic carrying capacity, and dynamic spawn threshold in v2.

### `_movement_distance()`
V2 uses state/energy/target/TTC dependent movement. V1 branch reproduces original movement logic.

### `_desired_vector()`
Potential fields remain the proposal generator. Wall force is weakened in v2 because wall feasibility is checked by the planner.

### `_update_predator_memory()`
Adds smoothed closing-rate and TTC estimation.

### `_decide_agent()`
Main integration point:
1. update fruit pseudo-tracks;
2. build target/threat state;
3. compute dynamic thresholds;
4. run existing hierarchy/potential-field proposal;
5. run `MicroPlanner` on 20 nearby actions;
6. return the selected action through the unchanged API schema.

## Modified `hpo.py`

- Original 28-dimensional search space is unchanged.
- New architecture controls are listed in `HPO_FIXED_FIELDS`.
- HPO metadata records `architecture_v2_enabled` and all fixed fields.
- Refuses to mix v1 and v2 trials in the same results root/database.

Use a fresh results root for future v2 optimization.

## Modified `tests/conftest.py`

Adds test-only conditional DTO/core import stubs when the overlay is tested outside the official repository. They are inactive when official `src/` is available.

## Modified `README_SOLUTION.md`

Added:
- unseen-seed holdout procedure;
- top-five commands;
- v1/v2 compatibility explanation;
- architecture-v2 implementation details;
- HPO study-separation warning.
