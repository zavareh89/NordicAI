from __future__ import annotations

import math

from survival_policy import PolicyConfig, SurvivalController
from survival_policy.coordination import estimate_carrying_capacity, local_fruit_owner
from survival_policy.memory import ActionMemory, AgentMemory
from survival_policy.planner import MicroPlanner
from tests.conftest import make_status, make_step


def test_fruit_pseudo_id_persists_across_small_observation_change():
    cfg = PolicyConfig()
    mem = AgentMemory(agent_id=1)
    mem.begin_tick(300.0, 1.0)
    first = mem.update_fruit_tracks(
        [{"type": "Fruit", "distance": 100.0, "angle": 0.10}],
        match_angle=cfg.fruit_track_match_angle,
        match_distance=cfg.fruit_track_match_distance,
        ttl=cfg.fruit_track_ttl,
    )
    tid = first[0]["_track_id"]
    mem.finalize_tick(energy=300.0, age=1.0, action=ActionMemory())
    mem.begin_tick(299.0, 1.1)
    second = mem.update_fruit_tracks(
        [{"type": "Fruit", "distance": 95.0, "angle": 0.13}],
        match_angle=cfg.fruit_track_match_angle,
        match_distance=cfg.fruit_track_match_distance,
        ttl=cfg.fruit_track_ttl,
    )
    assert second[0]["_track_id"] == tid


def test_local_owner_prefers_agent_closer_to_fruit():
    owner = local_fruit_owner(
        self_id=2,
        fruit_distance=100.0,
        fruit_angle=0.0,
        visible_agents=[{"id": 1, "distance": 92.0, "angle": 0.0}],
    )
    assert owner == 1


def test_carrying_capacity_rises_with_food_and_falls_with_danger():
    rich = estimate_carrying_capacity(
        base_cap=12,
        bonus_max=6,
        fruit_sightings_per_agent=4.0,
        predator_sightings_per_agent=0.0,
        average_energy_ratio=0.8,
        population_trend=0.0,
    )
    dangerous = estimate_carrying_capacity(
        base_cap=12,
        bonus_max=6,
        fruit_sightings_per_agent=0.2,
        predator_sightings_per_agent=2.0,
        average_energy_ratio=0.4,
        population_trend=0.0,
    )
    assert rich > dangerous


def test_predator_ttc_becomes_finite_when_distance_closes():
    c = SurvivalController(PolicyConfig())
    mem = c.memory.get_or_create(0)
    mem.begin_tick(300.0, 1.0)
    c._update_predator_memory(mem, {"distance": 100.0, "angle": 0.0, "rel_dir": 0.0})
    mem.finalize_tick(energy=300.0, age=1.0, action=ActionMemory())
    mem.begin_tick(299.0, 1.1)
    c._update_predator_memory(mem, {"distance": 90.0, "angle": 0.0, "rel_dir": 0.0})
    assert mem.predator_closing_per_tick > 0.0
    assert math.isfinite(mem.predator_ttc_ticks)


def test_microplanner_wall_constraint_penalizes_hard_clearance():
    planner = MicroPlanner(PolicyConfig())
    edge = [{"type": "Edge", "coords": ((10.0, -100.0), (10.0, 100.0))}]
    hard = planner._wall_score((9.0, 0.0), edge)
    safe = planner._wall_score((-50.0, 0.0), edge)
    assert hard < safe
    assert hard < -10.0


def test_microplanner_uses_food_progress_when_safe():
    cfg = PolicyConfig(planner_predator_weight=1.0, planner_food_weight=4.0)
    planner = MicroPlanner(cfg)
    action = planner.plan(
        preferred_move_angle=0.8,
        base_turn_angle=0.1,
        nominal_move_distance=8.0,
        normal_speed=10.0,
        sprint_speed=20.0,
        state_name="NORMAL_FORAGING",
        energy_ratio=0.7,
        chosen_fruit={"type": "Fruit", "distance": 80.0, "angle": 0.0},
        predators=[],
        edges=[],
        agents=[],
        predator_ttc_ticks=math.inf,
    )
    assert abs(action.move_direction) < 0.8


def test_v1_gate_keeps_architecture_v2_disabled():
    cfg = PolicyConfig(architecture_v2_enabled=False)
    c = SurvivalController(cfg)
    status = make_status(
        1,
        energy=300.0,
        observations=[{"type": "Fruit", "distance": 80.0, "angle": 0.2}],
    )
    step = make_step([status], sim_time=0.1)
    step["n_agents"] = cfg.population_soft_cap
    c.decide_step(step)
    mem = c.memory.agents[1]
    assert mem.fruit_tracks == {}
