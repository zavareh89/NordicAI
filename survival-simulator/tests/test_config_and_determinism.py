import json
from pathlib import Path

from survival_policy import PolicyConfig, SurvivalController
from tests.conftest import make_status, make_step


def test_config_serialization_hash_is_deterministic(tmp_path: Path):
    cfg = PolicyConfig()
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    cfg.to_json(p1)
    PolicyConfig.from_json(p1).to_json(p2)
    assert p1.read_text() == p2.read_text()
    assert PolicyConfig.from_json(p1).stable_hash == cfg.stable_hash


def test_seeded_exploration_reproducible():
    cfg = PolicyConfig(master_seed=12345)
    c1 = SurvivalController(cfg)
    c2 = SurvivalController(cfg)
    status = make_status(3, energy=300.0, observations=[])
    # Keep population at cap to prevent reproduction and force exploration.
    step = make_step([status], sim_time=0.1)
    step["n_agents"] = cfg.population_soft_cap
    a1 = c1.decide_step(step)[0]
    a2 = c2.decide_step(step)[0]
    assert a1 == a2


def test_different_agent_ids_diversify_exploration():
    cfg = PolicyConfig(master_seed=12345)
    c = SurvivalController(cfg)
    s0 = make_status(0, energy=300.0)
    s1 = make_status(1, energy=300.0)
    step = make_step([s0, s1], sim_time=0.1)
    step["n_agents"] = cfg.population_soft_cap
    actions = c.decide_step(step)
    assert actions[0]["move_direction"] != actions[1]["move_direction"]
