from fastapi.testclient import TestClient

import agent_server
from tests.conftest import make_status, make_step


def test_predict_contract_and_one_action_per_agent():
    client = TestClient(agent_server.app)
    payload = make_step([
        make_status(0, observations=[{"type": "Fruit", "distance": 50.0, "angle": 0.1}]),
        make_status(1, observations=[{"type": "Predator", "distance": 30.0, "angle": 0.0, "rel_dir": 0.2}]),
    ], sim_time=0.1)
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"actions"}
    assert [a["agent_id"] for a in body["actions"]] == [0, 1]
    for action in body["actions"]:
        assert set(action) == {"agent_id", "move_distance", "move_direction", "turn_angle", "spawn_agent"}


def test_predict_handles_empty_population():
    client = TestClient(agent_server.app)
    response = client.post("/predict", json=make_step([], sim_time=1.0))
    assert response.status_code == 200
    assert response.json() == {"actions": []}
