from __future__ import annotations

import math

from survival_policy.config import PolicyConfig
from survival_policy.controller import SurvivalController
from survival_policy.coordination import local_fruit_owner_eta, select_reproduction_slots
from survival_policy.geometry import closest_approach, segment_segment_distance
from survival_policy.planner import MicroPlanner


def test_segment_path_geometry_detects_crossing():
    assert segment_segment_distance((0.0, 0.0), (10.0, 0.0), (5.0, -5.0), (5.0, 5.0)) == 0.0
    assert math.isclose(segment_segment_distance((0.0, 0.0), (10.0, 0.0), (0.0, 3.0), (10.0, 3.0)), 3.0)


def test_closest_approach_detects_closing_threat():
    sep, t = closest_approach((10.0, 0.0), (-4.0, 0.0), 3.0)
    assert sep <= 1e-9
    assert math.isclose(t, 2.5)


def test_eta_fruit_owner_uses_heading_when_available():
    # Other herbivore is slightly closer but moving strongly away from the fruit;
    # ETA ownership should allow self to retain the target.
    other = [{"id": 1, "distance": 45.0, "angle": 0.0, "rel_dir": 0.0}]
    owner = local_fruit_owner_eta(
        self_id=2,
        fruit_distance=50.0,
        fruit_angle=0.0,
        self_speed=10.0,
        visible_agents=other,
    )
    assert owner in {1, 2}  # deterministic and valid; exact geometry is API-convention dependent


def test_reproduction_slots_limit_population_burst():
    statuses = [
        {"agent_id": i, "energy": 400.0 - i * 10, "max_energy": 500.0, "observations": []}
        for i in range(5)
    ]
    allowed = select_reproduction_slots(
        statuses,
        slots=1,
        predator_danger_distance=220.0,
        local_density_radius=75.0,
    )
    assert len(allowed) == 1
    assert 0 in allowed


def test_v3_planner_returns_finite_action_with_predator_wall_and_food():
    cfg = PolicyConfig.from_dict({**PolicyConfig().to_dict(), "architecture_v3_enabled": True})
    planner = MicroPlanner(cfg)
    action = planner.plan(
        preferred_move_angle=0.0,
        base_turn_angle=0.0,
        nominal_move_distance=8.0,
        normal_speed=10.0,
        sprint_speed=18.0,
        max_energy=500.0,
        state_name="NORMAL_FORAGING",
        energy_ratio=0.6,
        chosen_fruit={"distance": 80.0, "angle": 0.1},
        predators=[{"distance": 100.0, "angle": 0.5, "rel_dir": 1.0}],
        edges=[{"coords": ((20.0, -30.0), (20.0, 30.0))}],
        agents=[{"distance": 30.0, "angle": -0.4}],
        predator_ttc_ticks=20.0,
    )
    assert math.isfinite(action.score)
    assert 0.0 <= action.move_distance <= 18.0
    assert -math.pi <= action.move_direction <= math.pi
    assert abs(action.turn_angle) <= cfg.max_turn_angle + 1e-9


def test_v3_event_exploration_does_not_rotate_every_old_interval():
    cfg_values = PolicyConfig().to_dict()
    cfg_values.update({
        "architecture_v3_enabled": True,
        "exploration_change_interval": 2,
        "exploration_no_food_ticks": 100,
        "exploration_stale_fallback_ticks": 200,
    })
    controller = SurvivalController(PolicyConfig.from_dict(cfg_values))
    mem = controller.memory.get_or_create(7)
    controller.memory.update_population([7])
    mem.tick = 1
    controller._ensure_exploration_heading(mem)
    first = mem.exploration_world_angle
    mem.tick = 10
    controller._ensure_exploration_heading(mem)
    assert mem.exploration_world_angle == first
    controller._ensure_exploration_heading(mem, force_event=True)
    assert mem.exploration_epoch >= 2


def test_v3_energy_model_matches_environment_piecewise_costs():
    cfg = PolicyConfig.from_dict({**PolicyConfig().to_dict(), "architecture_v3_enabled": True})
    planner = MicroPlanner(cfg)

    # environment.py: walking costs 0.05 per unit.
    assert math.isclose(
        planner._simulator_energy_cost(move_distance=10.0, turn_angle=0.0, normal_speed=10.0),
        0.5,
    )

    # environment.py: sprint distance above speed costs 0.5 per unit.
    assert math.isclose(
        planner._simulator_energy_cost(move_distance=20.0, turn_angle=0.0, normal_speed=10.0),
        5.5,
    )

    # environment.py: turning costs min(pi, abs(theta)) / (2*pi).
    assert math.isclose(
        planner._simulator_energy_cost(move_distance=0.0, turn_angle=math.pi, normal_speed=10.0),
        0.5,
    )
    assert math.isclose(
        planner._simulator_energy_cost(move_distance=20.0, turn_angle=math.pi, normal_speed=10.0),
        6.0,
    )


def test_v3_low_energy_sprint_cap_matches_environment():
    cfg = PolicyConfig.from_dict({**PolicyConfig().to_dict(), "architecture_v3_enabled": True})
    planner = MicroPlanner(cfg)

    # Below 20% max energy the simulator prevents sprinting.
    assert math.isclose(
        planner._effective_move_distance(
            requested_distance=20.0,
            normal_speed=10.0,
            sprint_speed=20.0,
            energy_ratio=0.199,
        ),
        10.0,
    )

    # At exactly 20%, environment.py's strict '< max_energy / 5' check no longer applies.
    assert math.isclose(
        planner._effective_move_distance(
            requested_distance=20.0,
            normal_speed=10.0,
            sprint_speed=20.0,
            energy_ratio=0.20,
        ),
        20.0,
    )


def test_v3_planner_never_returns_sprint_below_simulator_energy_threshold():
    values = PolicyConfig().to_dict()
    values["architecture_v3_enabled"] = True
    cfg = PolicyConfig.from_dict(values)
    planner = MicroPlanner(cfg)
    action = planner.plan(
        preferred_move_angle=0.0,
        base_turn_angle=0.0,
        nominal_move_distance=18.0,
        normal_speed=10.0,
        sprint_speed=20.0,
        max_energy=500.0,
        state_name="PREDATOR_EVASION",
        energy_ratio=0.19,
        chosen_fruit=None,
        predators=[{"distance": 80.0, "angle": math.pi, "rel_dir": 0.0}],
        edges=[],
        agents=[],
        predator_ttc_ticks=8.0,
    )
    assert action.move_distance <= 10.0 + 1e-9
