from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Deque, Dict, Iterable, Mapping, Optional, Sequence

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
class FruitTrack:
    """Short-lived local pseudo-identity for a fruit.

    The simulator exposes no fruit ID or global agent position. Tracks therefore
    live in one herbivore's internally consistent heading frame; they are never
    treated as globally shared object IDs.
    """

    track_id: int
    world_bearing: float
    distance: float
    last_seen_tick: int
    age_ticks: int = 1

    def local_bearing(self, heading_estimate: float) -> float:
        return wrap_angle(self.world_bearing - heading_estimate)


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
    target_track_id: Optional[int] = None
    target_age_ticks: int = 0
    target_last_seen_tick: int = -1
    target_stall_ticks: int = 0

    predator_world_bearing: Optional[float] = None
    predator_distance: Optional[float] = None
    predator_last_seen_tick: int = -1
    predator_closing_per_tick: float = 0.0
    predator_bearing_rate: float = 0.0
    predator_ttc_ticks: float = math.inf
    escape_until_tick: int = -1

    recovery_until_tick: int = -1
    reproduction_cooldown_until: int = -1
    last_spawn_tick: int = -1
    local_population_density: int = 0

    fruit_tracks: Dict[int, FruitTrack] = field(default_factory=dict)
    next_fruit_track_id: int = 1

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

    def remember_target(
        self,
        local_bearing: float,
        distance: float,
        progress_epsilon: float = 0.0,
        track_id: Optional[int] = None,
    ) -> None:
        same_track = track_id is not None and track_id == self.target_track_id
        self.target_world_bearing = wrap_angle(self.heading_estimate + local_bearing)
        if self.target_distance is not None and (same_track or self.target_track_id is None):
            progress = self.target_distance - distance
            self.target_stall_ticks = 0 if progress > progress_epsilon else self.target_stall_ticks + 1
        else:
            self.target_stall_ticks = 0
            self.target_age_ticks = 0
            self.recent_target_distance.clear()
        self.target_distance = float(distance)
        self.target_track_id = track_id
        self.target_last_seen_tick = self.tick
        self.target_age_ticks += 1
        self.recent_target_distance.append(float(distance))

    def forget_target(self) -> None:
        self.target_world_bearing = None
        self.target_distance = None
        self.target_track_id = None
        self.target_age_ticks = 0
        self.target_last_seen_tick = -1
        self.target_stall_ticks = 0
        self.recent_target_distance.clear()

    def update_fruit_tracks(
        self,
        fruits: Sequence[Mapping],
        *,
        match_angle: float,
        match_distance: float,
        ttl: int,
    ) -> list[dict]:
        """Attach stable local ``_track_id`` values to current fruit observations.

        Matching is greedy by normalized angular/distance discrepancy and bounded
        in time. The returned dictionaries are copies; official observations are
        never mutated.
        """
        stale = [tid for tid, t in self.fruit_tracks.items() if self.tick - t.last_seen_tick > ttl]
        for tid in stale:
            del self.fruit_tracks[tid]
            if self.target_track_id == tid:
                self.forget_target()

        parsed: list[tuple[int, dict, float, float]] = []
        for idx, fruit in enumerate(fruits):
            try:
                d = max(0.0, float(fruit.get("distance")))
                a = float(fruit.get("angle"))
            except (TypeError, ValueError):
                continue
            parsed.append((idx, dict(fruit), d, a))

        pairs: list[tuple[float, int, int]] = []
        for obs_idx, _, d, a in parsed:
            for tid, track in self.fruit_tracks.items():
                da = abs(wrap_angle(a - track.local_bearing(self.heading_estimate)))
                dd = abs(d - track.distance)
                if da <= match_angle and dd <= match_distance:
                    cost = da / max(match_angle, 1e-9) + dd / max(match_distance, 1e-9)
                    pairs.append((cost, obs_idx, tid))
        pairs.sort()

        assigned_obs: set[int] = set()
        assigned_tracks: set[int] = set()
        matches: dict[int, int] = {}
        for _, obs_idx, tid in pairs:
            if obs_idx in assigned_obs or tid in assigned_tracks:
                continue
            assigned_obs.add(obs_idx)
            assigned_tracks.add(tid)
            matches[obs_idx] = tid

        output: list[dict] = []
        for obs_idx, fruit, d, a in parsed:
            tid = matches.get(obs_idx)
            if tid is None:
                tid = self.next_fruit_track_id
                self.next_fruit_track_id += 1
                self.fruit_tracks[tid] = FruitTrack(
                    track_id=tid,
                    world_bearing=wrap_angle(self.heading_estimate + a),
                    distance=d,
                    last_seen_tick=self.tick,
                )
            else:
                track = self.fruit_tracks[tid]
                # Smooth enough to resist jitter without creating a long lag.
                observed_world = wrap_angle(self.heading_estimate + a)
                track.world_bearing = wrap_angle(
                    track.world_bearing + 0.25 * wrap_angle(observed_world - track.world_bearing)
                )
                track.distance = 0.65 * track.distance + 0.35 * d
                track.last_seen_tick = self.tick
                track.age_ticks += 1
            fruit["_track_id"] = tid
            output.append(fruit)
        return output

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
    active_ids: tuple[int, ...] = ()
    recent_population: Deque[int] = field(default_factory=lambda: deque(maxlen=16))

    def reset(self) -> None:
        self.agents.clear()
        self.last_sim_time = None
        self.active_ids = ()
        self.recent_population.clear()
        self.episode_index += 1

    def get_or_create(self, agent_id: int) -> AgentMemory:
        mem = self.agents.get(agent_id)
        if mem is None:
            mem = AgentMemory(agent_id=agent_id)
            self.agents[agent_id] = mem
        return mem

    def update_population(self, active_ids: Iterable[int]) -> None:
        self.active_ids = tuple(sorted(int(i) for i in active_ids))
        self.recent_population.append(len(self.active_ids))

    @property
    def population_trend(self) -> float:
        values = list(self.recent_population)
        if len(values) < 4:
            return 0.0
        half = len(values) // 2
        old = sum(values[:half]) / max(1, half)
        new = sum(values[half:]) / max(1, len(values) - half)
        return new - old

    def cleanup(self, active_ids: Iterable[int]) -> None:
        active = set(int(i) for i in active_ids)
        stale = [agent_id for agent_id in self.agents if agent_id not in active]
        for agent_id in stale:
            del self.agents[agent_id]
