from survival_policy.memory import AgentMemory, BehaviorState, PopulationMemory
from survival_policy import PolicyConfig, SurvivalController
from tests.conftest import make_status, make_step


def test_memory_initialization_and_update():
    m = AgentMemory(agent_id=7)
    assert m.agent_id == 7
    assert m.state == BehaviorState.EXPLORATION
    m.begin_tick(150.0, 1.0)
    m.set_state(BehaviorState.NORMAL_FORAGING)
    assert m.tick == 1
    assert m.state == BehaviorState.NORMAL_FORAGING


def test_new_agent_initialization_and_dead_agent_cleanup():
    c = SurvivalController(PolicyConfig())
    c.decide_step(make_step([make_status(0)], sim_time=0.1))
    assert set(c.memory.agents) == {0}
    c.decide_step(make_step([make_status(0), make_status(5)], sim_time=0.2))
    assert set(c.memory.agents) == {0, 5}
    c.decide_step(make_step([make_status(5)], sim_time=0.3))
    assert set(c.memory.agents) == {5}


def test_target_progress_stuck_counter_uses_epsilon():
    m = AgentMemory(agent_id=1)
    m.begin_tick(150.0, 1.0)
    m.remember_target(0.0, 100.0, progress_epsilon=0.8)
    m.begin_tick(149.0, 1.1)
    m.remember_target(0.0, 99.5, progress_epsilon=0.8)
    assert m.target_stall_ticks == 1
    m.begin_tick(148.0, 1.2)
    m.remember_target(0.0, 98.0, progress_epsilon=0.8)
    assert m.target_stall_ticks == 0


def test_target_timeout_forgets_missing_target():
    cfg = PolicyConfig(target_timeout_ticks=3)
    c = SurvivalController(cfg)
    status = make_status(0, observations=[{"type": "Fruit", "distance": 30.0, "angle": 0.2}])
    c.decide_step(make_step([status], sim_time=0.1))
    assert c.memory.agents[0].target_world_bearing is not None
    empty = make_status(0, observations=[])
    for i in range(1, 6):
        c.decide_step(make_step([empty], sim_time=0.1 + i * 0.1))
    assert c.memory.agents[0].target_world_bearing is None


def test_episode_reset_on_time_rewind():
    c = SurvivalController(PolicyConfig())
    c.decide_step(make_step([make_status(0)], sim_time=10.0))
    old_obj = c.memory.agents[0]
    c.decide_step(make_step([make_status(0)], sim_time=0.0))
    assert c.memory.agents[0] is not old_obj
