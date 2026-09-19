from __future__ import annotations

# The downloadable overlay is intended to be copied on top of the official
# challenge repository, where ``src`` exists. For standalone overlay testing in
# a sandbox, provide the smallest possible import-only DTO/core stubs. They are
# never used when the real simulator package is available.
try:  # pragma: no cover - exercised only outside the official repository
    import src.core  # type: ignore  # noqa: F401
    import src.utils.DTOs  # type: ignore  # noqa: F401
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    import sys
    import types
    from pydantic import BaseModel, Field

    src_mod = sys.modules.setdefault("src", types.ModuleType("src"))
    utils_mod = sys.modules.setdefault("src.utils", types.ModuleType("src.utils"))
    dtos_mod = types.ModuleType("src.utils.DTOs")
    core_mod = types.ModuleType("src.core")

    class ObservationResponse(BaseModel):
        agent_id: int
        observations: list[dict] = Field(default_factory=list)
        energy: float = 0.0
        biome: str = "grassland"
        age: float = 0.0
        speed: float = 0.0
        sprint_speed: float = 0.0
        hearing_radius: float = 0.0
        vision_angle: float = 0.0
        vision_range: float = 0.0
        max_energy: float = 1.0

    class StepResponse(BaseModel):
        game_status: str = "ok"
        score: float = 0.0
        sim_time: float = 0.0
        n_agents: int = 0
        agent_status: list[ObservationResponse] = Field(default_factory=list)

    class ActionRequest(BaseModel):
        agent_id: int
        move_distance: float
        move_direction: float
        turn_angle: float
        spawn_agent: bool

    class SimulationCore:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Official SimulationCore is required for end-to-end simulation")

    dtos_mod.ObservationResponse = ObservationResponse
    dtos_mod.StepResponse = StepResponse
    dtos_mod.ActionRequest = ActionRequest
    core_mod.SimulationCore = SimulationCore
    utils_mod.DTOs = dtos_mod
    src_mod.utils = utils_mod
    src_mod.core = core_mod
    sys.modules["src.utils.DTOs"] = dtos_mod
    sys.modules["src.core"] = core_mod


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
