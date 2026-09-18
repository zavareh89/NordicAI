from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, Iterable, Optional

from .geometry import wrap_angle


class BehaviorState(str, Enum):
    EMERGENCY_ESCAPE = "EMERGENCY_ESCAPE"
    PREDATOR_EVASION = "PREDATOR_EVASION"
    WALL_AVOIDANCE = "WALL_AVOIDANCE"
    CRITICAL_ENERGY_FORAGING = "CRITICAL_ENERGY_FORAGING"
    NORMAL_FORAGING = "NORMAL_FORAGING"
    REPRODUCTION = "REPRODUCTION"
    EXPLORATION = "EXPLORATION"
    ENERGY_CONSERVATION = "ENERGY_CONSERVATION"
    RECOVERY = "RECOVERY"


@dataclass
class ActionMemory:
    move_distance: float = 0.0
    move_direction: float = 0.0
    turn_angle: float = 0.0
    spawn_agent: bool = False


@dataclass
class AgentMemory:
    agent_id: int
    state: BehaviorState = BehaviorState.EXPLORATION
    state_ticks: int = 0
    tick: int = 0

    previous_energy: Optional[float] = None
    previous_age: Optional[float] = None
    previous_action: ActionMemory = field(default_factory=ActionMemory)

    # Internal heading is relative to the unknown heading at first observation.
    heading_estimate: float = 0.0
    exploration_world_angle: float = 0.0
    next_exploration_change_tick: int = 0

    target_world_bearing: Optional[float] = None
    target_distance: Optional[float] = None
    target_age_ticks: int = 0
    target_last_seen_tick: int = -1
    target_stall_ticks: int = 0

    predator_world_bearing: Optional[float] = None
    predator_distance: Optional[float] = None
    predator_last_seen_tick: int = -1
    escape_until_tick: int = -1

    recovery_until_tick: int = -1
    reproduction_cooldown_until: int = -1
    last_spawn_tick: int = -1
    local_population_density: int = 0

    recent_energy: Deque[float] = field(default_factory=lambda: deque(maxlen=8))
    recent_target_distance: Deque[float] = field(default_factory=lambda: deque(maxlen=8))

    def begin_tick(self, energy: float, age: float) -> None:
        self.tick += 1
        self.state_ticks += 1
        self.previous_energy = energy if self.previous_energy is None else self.previous_energy
        self.previous_age = age if self.previous_age is None else self.previous_age
        self.recent_energy.append(float(energy))

    def set_state(self, state: BehaviorState) -> None:
        if state != self.state:
            self.state = state
            self.state_ticks = 0

    def current_target_local_bearing(self) -> Optional[float]:
        if self.target_world_bearing is None:
            return None
        return wrap_angle(self.target_world_bearing - self.heading_estimate)

    def remember_target(self, local_bearing: float, distance: float, progress_epsilon: float = 0.0) -> None:
        self.target_world_bearing = wrap_angle(self.heading_estimate + local_bearing)
        if self.target_distance is not None:
            progress = self.target_distance - distance
            self.target_stall_ticks = 0 if progress > progress_epsilon else self.target_stall_ticks + 1
        else:
            self.target_stall_ticks = 0
        self.target_distance = float(distance)
        self.target_last_seen_tick = self.tick
        self.target_age_ticks += 1
        self.recent_target_distance.append(float(distance))

    def forget_target(self) -> None:
        self.target_world_bearing = None
        self.target_distance = None
        self.target_age_ticks = 0
        self.target_last_seen_tick = -1
        self.target_stall_ticks = 0
        self.recent_target_distance.clear()

    def finalize_tick(self, *, energy: float, age: float, action: ActionMemory) -> None:
        self.heading_estimate = wrap_angle(self.heading_estimate + action.turn_angle)
        self.previous_action = action
        self.previous_energy = float(energy)
        self.previous_age = float(age)


@dataclass
class PopulationMemory:
    agents: Dict[int, AgentMemory] = field(default_factory=dict)
    last_sim_time: Optional[float] = None
    episode_index: int = 0

    def reset(self) -> None:
        self.agents.clear()
        self.last_sim_time = None
        self.episode_index += 1

    def get_or_create(self, agent_id: int) -> AgentMemory:
        mem = self.agents.get(agent_id)
        if mem is None:
            mem = AgentMemory(agent_id=agent_id)
            self.agents[agent_id] = mem
        return mem

    def cleanup(self, active_ids: Iterable[int]) -> None:
        active = set(int(i) for i in active_ids)
        stale = [agent_id for agent_id in self.agents if agent_id not in active]
        for agent_id in stale:
            del self.agents[agent_id]
