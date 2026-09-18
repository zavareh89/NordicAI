import math

from survival_policy import PolicyConfig, SurvivalController
from survival_policy.coordination import local_competition_cost
from tests.conftest import make_status


def test_predator_repulsion_points_away():
    c = SurvivalController(PolicyConfig())
    x, y = c.predator_repulsion([{"type": "Predator", "distance": 40.0, "angle": 0.0, "rel_dir": 0.0}])
    assert x < 0.0
    assert abs(y) < abs(x)


def test_wall_repulsion_points_away_from_edge():
    c = SurvivalController(PolicyConfig())
    x, y = c.wall_repulsion([{"type": "Edge", "coords": ((15.0, -20.0), (15.0, 20.0))}])
    assert x < 0.0
    assert abs(y) < 1e-9


def test_fruit_attraction_direction():
    c = SurvivalController(PolicyConfig())
    x, y = c.fruit_attraction({"type": "Fruit", "distance": 100.0, "angle": math.pi / 2})
    assert abs(x) < 1e-9
    assert y > 0.0


def test_local_fruit_deconfliction_penalizes_owned_target():
    visible = [{"type": "Agent", "id": 1, "distance": 95.0, "angle": 0.0}]
    cost = local_competition_cost(
        self_id=2,
        fruit_distance=100.0,
        fruit_angle=0.0,
        visible_agents=visible,
        competition_radius=40.0,
        base_penalty=1.0,
    )
    assert cost > 1.0


def test_target_selection_avoids_contested_fruit_when_alternative_exists():
    c = SurvivalController(PolicyConfig())
    mem = c.memory.get_or_create(2)
    fruits = [
        {"type": "Fruit", "distance": 90.0, "angle": 0.0},
        {"type": "Fruit", "distance": 95.0, "angle": 1.0},
    ]
    agents = [{"type": "Agent", "id": 1, "distance": 86.0, "angle": 0.0}]
    chosen = c._select_fruit(agent_id=2, fruits=fruits, visible_agents=agents, predators=[], mem=mem)
    assert chosen is fruits[1]
