from __future__ import annotations


def make_status(agent_id=0, energy=150.0, age=0.0, observations=None, speed=10.0, sprint_speed=20.0, max_energy=500.0):
    return {
        "agent_id": agent_id,
        "observations": list(observations or []),
        "energy": energy,
        "biome": "grassland",
        "age": age,
        "speed": speed,
        "sprint_speed": sprint_speed,
        "hearing_radius": 50.0,
        "vision_angle": 1.0471975512,
        "vision_range": 200.0,
        "max_energy": max_energy,
    }


def make_step(statuses, sim_time=0.0, score=0.0):
    return {
        "game_status": "ok",
        "score": score,
        "sim_time": sim_time,
        "n_agents": len(statuses),
        "agent_status": statuses,
    }
