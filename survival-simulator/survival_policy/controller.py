from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .config import PolicyConfig
from .coordination import local_competition_cost, local_density
from .geometry import (
    Vec2,
    add,
    angle_from_vector,
    normalize,
    polar_to_cart,
    scale,
    segment_distance_to_origin,
    stable_int_seed,
    unit_from_angle,
    wrap_angle,
)
from .memory import ActionMemory, AgentMemory, BehaviorState, PopulationMemory


@dataclass(frozen=True)
class Decision:
    agent_id: int
    move_distance: float
    move_direction: float
    turn_angle: float
    spawn_agent: bool
    state: BehaviorState

    def action_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "move_distance": float(self.move_distance),
            "move_direction": float(self.move_direction),
            "turn_angle": float(self.turn_angle),
            "spawn_agent": bool(self.spawn_agent),
        }


class SurvivalController:
    """Lightweight deterministic multi-agent controller.

    Observation geometry and action directions are in the agent-local frame.
    The controller keeps bounded memory and never performs expensive planning.
    """

    def __init__(self, config: PolicyConfig):
        config.validate()
        self.config = config
        self.memory = PopulationMemory()

    def reset(self) -> None:
        self.memory.reset()

    def _maybe_reset_episode(self, sim_time: float) -> None:
        last = self.memory.last_sim_time
        if last is not None and sim_time + 1e-9 < last:
            self.reset()
        self.memory.last_sim_time = sim_time

    @staticmethod
    def _split_observations(observations: Sequence[Mapping]) -> Dict[str, List[Mapping]]:
        groups: Dict[str, List[Mapping]] = {
            "Fruit": [], "Agent": [], "Predator": [], "Tree": [], "Edge": []
        }
        for obs in observations or []:
            if not isinstance(obs, Mapping):
                continue
            typ = obs.get("type")
            if typ in groups:
                groups[typ].append(obs)
        return groups

    def decide_step(self, step: Mapping | object) -> List[dict]:
        if hasattr(step, "model_dump"):
            data = step.model_dump()
        elif hasattr(step, "dict"):
            data = step.dict()
        else:
            data = dict(step)
        sim_time = float(data.get("sim_time", 0.0) or 0.0)
        self._maybe_reset_episode(sim_time)
        statuses = data.get("agent_status") or []
        normalized = []
        for status in statuses:
            if hasattr(status, "model_dump"):
                normalized.append(status.model_dump())
            elif hasattr(status, "dict"):
                normalized.append(status.dict())
            else:
                normalized.append(dict(status))
        active_ids = [int(s["agent_id"]) for s in normalized if "agent_id" in s]
        self.memory.cleanup(active_ids)
        n_agents = int(data.get("n_agents", len(active_ids)) or len(active_ids))
        decisions = [self._decide_agent(s, n_agents=n_agents) for s in normalized]
        return [d.action_dict() for d in decisions]

    def _agent_rng(self, agent_id: int, epoch: int) -> random.Random:
        return random.Random(stable_int_seed(self.config.master_seed, agent_id, epoch))

    def _ensure_exploration_heading(self, mem: AgentMemory) -> None:
        if mem.tick < mem.next_exploration_change_tick and mem.next_exploration_change_tick > 0:
            return
        epoch = mem.tick // max(1, self.config.exploration_change_interval)
        rng = self._agent_rng(mem.agent_id, epoch)
        # ID-stratified base phase + a small seeded offset prevents flocking.
        phase = (mem.agent_id * 2.399963229728653) % (2.0 * math.pi)
        jitter = rng.uniform(-0.65, 0.65)
        mem.exploration_world_angle = wrap_angle(phase + jitter)
        mem.next_exploration_change_tick = mem.tick + self.config.exploration_change_interval

    def _nearest_predator(self, predators: Sequence[Mapping]) -> Optional[Mapping]:
        valid = []
        for p in predators:
            try:
                valid.append((float(p.get("distance")), p))
            except (TypeError, ValueError):
                continue
        return min(valid, key=lambda x: x[0])[1] if valid else None

    def _closest_edge(self, edges: Sequence[Mapping]) -> tuple[float, Optional[Vec2]]:
        best = math.inf
        point: Optional[Vec2] = None
        for edge in edges:
            coords = edge.get("coords")
            if not coords or len(coords) != 2:
                continue
            try:
                d, p = segment_distance_to_origin(coords[0], coords[1])
            except (TypeError, ValueError, IndexError):
                continue
            if d < best:
                best, point = d, p
        return best, point

    def predator_repulsion(self, predators: Sequence[Mapping]) -> Vec2:
        cfg = self.config
        vx = vy = 0.0
        for p in predators:
            try:
                d = max(1.0, float(p.get("distance")))
                a = float(p.get("angle"))
            except (TypeError, ValueError):
                continue
            if d > cfg.predator_danger_distance * 1.4:
                continue
            strength = cfg.predator_repulsion_weight * (
                cfg.predator_danger_distance / d
            ) ** 2.0
            px, py = polar_to_cart(d, a)
            # Predict one short step using predator relative heading when available.
            rel_dir = p.get("rel_dir")
            if rel_dir is not None:
                try:
                    pred_heading = wrap_angle(a + math.pi - float(rel_dir))
                    ux, uy = unit_from_angle(pred_heading)
                    px += cfg.predator_prediction_weight * 15.0 * ux
                    py += cfg.predator_prediction_weight * 15.0 * uy
                except (TypeError, ValueError):
                    pass
            away = normalize((-px, -py))
            vx += away[0] * strength
            vy += away[1] * strength
        return vx, vy

    def wall_repulsion(self, edges: Sequence[Mapping]) -> Vec2:
        cfg = self.config
        vx = vy = 0.0
        for edge in edges:
            coords = edge.get("coords")
            if not coords or len(coords) != 2:
                continue
            try:
                d, p = segment_distance_to_origin(coords[0], coords[1])
            except (TypeError, ValueError, IndexError):
                continue
            if d >= cfg.wall_danger_distance:
                continue
            d = max(1.0, d)
            closeness = 1.0 - d / cfg.wall_danger_distance
            strength = cfg.wall_repulsion_weight * (closeness ** 2.0)
            away = normalize((-p[0], -p[1]))
            vx += away[0] * strength
            vy += away[1] * strength
        return vx, vy

    def herbivore_repulsion(self, agents: Sequence[Mapping]) -> Vec2:
        cfg = self.config
        vx = vy = 0.0
        for a in agents:
            try:
                d = max(1.0, float(a.get("distance")))
                angle = float(a.get("angle"))
            except (TypeError, ValueError):
                continue
            if d >= cfg.herbivore_repulsion_radius:
                continue
            closeness = 1.0 - d / cfg.herbivore_repulsion_radius
            ux, uy = unit_from_angle(angle + math.pi)
            vx += ux * cfg.herbivore_repulsion_weight * closeness
            vy += uy * cfg.herbivore_repulsion_weight * closeness
        return vx, vy

    def fruit_attraction(self, fruit: Optional[Mapping]) -> Vec2:
        if fruit is None:
            return 0.0, 0.0
        try:
            angle = float(fruit.get("angle"))
        except (TypeError, ValueError):
            return 0.0, 0.0
        return scale(unit_from_angle(angle), self.config.fruit_attraction_weight)

    def _fruit_utility(
        self,
        *,
        agent_id: int,
        fruit: Mapping,
        visible_agents: Sequence[Mapping],
        predators: Sequence[Mapping],
        mem: AgentMemory,
    ) -> float:
        cfg = self.config
        try:
            d = max(0.0, float(fruit.get("distance")))
            a = float(fruit.get("angle"))
        except (TypeError, ValueError):
            return -math.inf
        utility = -cfg.fruit_distance_penalty * d
        utility -= local_competition_cost(
            self_id=agent_id,
            fruit_distance=d,
            fruit_angle=a,
            visible_agents=visible_agents,
            competition_radius=cfg.fruit_competition_radius,
            base_penalty=cfg.fruit_competition_penalty,
        )
        fx, fy = polar_to_cart(d, a)
        for pred in predators:
            try:
                pd = float(pred.get("distance"))
                pa = float(pred.get("angle"))
            except (TypeError, ValueError):
                continue
            px, py = polar_to_cart(pd, pa)
            sep = max(1.0, math.hypot(fx - px, fy - py))
            if sep < cfg.predator_danger_distance:
                utility -= 2.5 * (1.0 - sep / cfg.predator_danger_distance)
        old_local = mem.current_target_local_bearing()
        if old_local is not None and mem.target_distance is not None:
            angle_match = abs(wrap_angle(a - old_local))
            distance_match = abs(d - mem.target_distance)
            if angle_match < 0.45 and distance_match < 65.0:
                utility += cfg.target_persistence_bonus
        return utility

    def _select_fruit(
        self,
        *,
        agent_id: int,
        fruits: Sequence[Mapping],
        visible_agents: Sequence[Mapping],
        predators: Sequence[Mapping],
        mem: AgentMemory,
    ) -> Optional[Mapping]:
        if not fruits:
            return None
        scored = [
            (self._fruit_utility(
                agent_id=agent_id,
                fruit=f,
                visible_agents=visible_agents,
                predators=predators,
                mem=mem,
            ), f)
            for f in fruits
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best = scored[0]
        old_local = mem.current_target_local_bearing()
        if old_local is None or mem.target_distance is None:
            return best
        old_matches = []
        for score, fruit in scored:
            try:
                if abs(wrap_angle(float(fruit["angle"]) - old_local)) < 0.45 and abs(float(fruit["distance"]) - mem.target_distance) < 65.0:
                    old_matches.append((score, fruit))
            except (KeyError, TypeError, ValueError):
                continue
        if old_matches:
            old_score, old = max(old_matches, key=lambda x: x[0])
            if best_score < old_score + self.config.target_switch_margin:
                return old
        return best

    def _state_for(
        self,
        *,
        mem: AgentMemory,
        energy: float,
        max_energy: float,
        age: float,
        n_agents: int,
        fruits: Sequence[Mapping],
        nearest_predator_distance: float,
        closest_edge_distance: float,
    ) -> BehaviorState:
        cfg = self.config
        ratio = energy / max(max_energy, 1.0)
        if mem.tick <= mem.recovery_until_tick:
            return BehaviorState.RECOVERY
        if nearest_predator_distance <= cfg.emergency_predator_distance:
            mem.escape_until_tick = mem.tick + cfg.escape_persistence_ticks
            return BehaviorState.EMERGENCY_ESCAPE
        if nearest_predator_distance <= cfg.predator_danger_distance or mem.tick <= mem.escape_until_tick:
            return BehaviorState.PREDATOR_EVASION
        if closest_edge_distance <= cfg.wall_critical_distance:
            return BehaviorState.WALL_AVOIDANCE
        if ratio <= cfg.critical_energy_ratio and fruits:
            return BehaviorState.CRITICAL_ENERGY_FORAGING
        can_spawn = (
            mem.tick >= mem.reproduction_cooldown_until
            and n_agents < cfg.population_soft_cap
            and nearest_predator_distance > cfg.reproduction_danger_distance
            and (energy >= cfg.spawn_min_energy or (age >= cfg.spawn_old_age and energy >= cfg.spawn_old_age_min_energy))
        )
        if can_spawn:
            return BehaviorState.REPRODUCTION
        if fruits:
            return BehaviorState.NORMAL_FORAGING
        if ratio <= cfg.low_energy_ratio:
            return BehaviorState.ENERGY_CONSERVATION
        return BehaviorState.EXPLORATION

    def _movement_distance(self, state: BehaviorState, status: Mapping, nearest_predator_distance: float) -> float:
        cfg = self.config
        speed = max(0.0, float(status.get("speed", 0.0) or 0.0))
        sprint = max(speed, float(status.get("sprint_speed", speed) or speed))
        energy = float(status.get("energy", 0.0) or 0.0)
        max_energy = max(1.0, float(status.get("max_energy", 1.0) or 1.0))
        if state == BehaviorState.EMERGENCY_ESCAPE:
            return sprint
        if state == BehaviorState.PREDATOR_EVASION:
            # Simulator itself prevents sprinting below 20% max energy.
            return sprint * cfg.evasion_sprint_fraction if energy >= max_energy * 0.22 else speed
        if state in (BehaviorState.CRITICAL_ENERGY_FORAGING, BehaviorState.NORMAL_FORAGING):
            return speed * cfg.forage_move_fraction
        if state == BehaviorState.ENERGY_CONSERVATION:
            return speed * cfg.conserve_move_fraction
        if state == BehaviorState.RECOVERY:
            return speed * 0.85
        if state == BehaviorState.WALL_AVOIDANCE:
            return speed * 0.60
        if state == BehaviorState.REPRODUCTION:
            return speed * 0.25
        return speed * cfg.explore_move_fraction

    def _desired_vector(
        self,
        *,
        state: BehaviorState,
        mem: AgentMemory,
        chosen_fruit: Optional[Mapping],
        groups: Dict[str, List[Mapping]],
    ) -> Vec2:
        cfg = self.config
        pred = self.predator_repulsion(groups["Predator"])
        if not groups["Predator"] and mem.predator_world_bearing is not None:
            age = mem.tick - mem.predator_last_seen_tick
            if 0 <= age <= cfg.predator_memory_ticks:
                local_bearing = wrap_angle(mem.predator_world_bearing - mem.heading_estimate)
                decay = 1.0 - age / max(1.0, float(cfg.predator_memory_ticks + 1))
                pred = add(pred, scale(unit_from_angle(local_bearing + math.pi), cfg.predator_repulsion_weight * 0.45 * decay))
        wall = self.wall_repulsion(groups["Edge"])
        herd = self.herbivore_repulsion(groups["Agent"])

        if state in (BehaviorState.EMERGENCY_ESCAPE, BehaviorState.PREDATOR_EVASION):
            v = add(scale(pred, 1.35), scale(wall, 1.0), scale(herd, 0.45))
            return normalize(v, default=(-1.0, 0.0))

        if state == BehaviorState.WALL_AVOIDANCE:
            return normalize(add(scale(wall, 1.7), scale(herd, 0.4)), default=(0.0, 1.0))

        if state in (BehaviorState.CRITICAL_ENERGY_FORAGING, BehaviorState.NORMAL_FORAGING):
            goal = self.fruit_attraction(chosen_fruit)
            pred_scale = 0.8 if state == BehaviorState.CRITICAL_ENERGY_FORAGING else 1.15
            herd_scale = 0.20 if state == BehaviorState.CRITICAL_ENERGY_FORAGING else 0.55
            return normalize(add(goal, scale(pred, pred_scale), wall, scale(herd, herd_scale)), default=(1.0, 0.0))

        if state == BehaviorState.RECOVERY:
            sign = -1.0 if mem.agent_id % 2 else 1.0
            recovery_angle = wrap_angle(mem.current_target_local_bearing() or 0.0 + sign * math.pi / 2.0)
            return normalize(add(unit_from_angle(recovery_angle), scale(wall, 1.5), herd), default=unit_from_angle(sign * math.pi / 2.0))

        self._ensure_exploration_heading(mem)
        local_explore = wrap_angle(mem.exploration_world_angle - mem.heading_estimate)
        explore = unit_from_angle(local_explore)
        tree_bias = (0.0, 0.0)
        if groups["Tree"]:
            try:
                nearest_tree = min(groups["Tree"], key=lambda t: float(t.get("distance", math.inf)))
                tree_bias = scale(unit_from_angle(float(nearest_tree.get("angle", 0.0))), 0.25)
            except (TypeError, ValueError):
                pass
        return normalize(add(explore, tree_bias, wall, herd, scale(pred, 0.8)), default=explore)

    def _turn_angle(
        self,
        *,
        state: BehaviorState,
        desired_move_angle: float,
        nearest_predator: Optional[Mapping],
    ) -> float:
        cfg = self.config
        target_angle = desired_move_angle
        if state in (BehaviorState.EMERGENCY_ESCAPE, BehaviorState.PREDATOR_EVASION) and nearest_predator is not None:
            try:
                pd = float(nearest_predator.get("distance"))
                pa = float(nearest_predator.get("angle"))
                if pd <= cfg.face_predator_distance:
                    # Move away while looking at the predator. Beyond the predator's
                    # close-range trigger this pushes its policy toward pivoting instead
                    # of a straight chase.
                    target_angle = pa
            except (TypeError, ValueError):
                pass
        turn = wrap_angle(target_angle) * 0.55
        return max(-cfg.max_turn_angle, min(cfg.max_turn_angle, turn))

    def _update_predator_memory(self, mem: AgentMemory, nearest_predator: Optional[Mapping]) -> None:
        if nearest_predator is None:
            if mem.tick - mem.predator_last_seen_tick > self.config.predator_memory_ticks:
                mem.predator_world_bearing = None
                mem.predator_distance = None
            return
        try:
            local_angle = float(nearest_predator.get("angle"))
            distance = float(nearest_predator.get("distance"))
        except (TypeError, ValueError):
            return
        mem.predator_world_bearing = wrap_angle(mem.heading_estimate + local_angle)
        mem.predator_distance = distance
        mem.predator_last_seen_tick = mem.tick

    def _decide_agent(self, status: Mapping, *, n_agents: int) -> Decision:
        cfg = self.config
        agent_id = int(status.get("agent_id"))
        energy = float(status.get("energy", 0.0) or 0.0)
        age = float(status.get("age", 0.0) or 0.0)
        max_energy = max(1.0, float(status.get("max_energy", 1.0) or 1.0))
        mem = self.memory.get_or_create(agent_id)
        mem.begin_tick(energy, age)

        groups = self._split_observations(status.get("observations") or [])
        mem.local_population_density = local_density(groups["Agent"], cfg.herbivore_repulsion_radius * 1.5)
        nearest_predator = self._nearest_predator(groups["Predator"])
        nearest_predator_distance = math.inf
        if nearest_predator is not None:
            try:
                nearest_predator_distance = float(nearest_predator.get("distance"))
            except (TypeError, ValueError):
                pass
        self._update_predator_memory(mem, nearest_predator)
        closest_edge_distance, _ = self._closest_edge(groups["Edge"])

        chosen_fruit = self._select_fruit(
            agent_id=agent_id,
            fruits=groups["Fruit"],
            visible_agents=groups["Agent"],
            predators=groups["Predator"],
            mem=mem,
        )

        if chosen_fruit is not None:
            try:
                mem.remember_target(
                    float(chosen_fruit.get("angle")),
                    float(chosen_fruit.get("distance")),
                    progress_epsilon=cfg.stuck_progress_epsilon,
                )
            except (TypeError, ValueError):
                pass
        elif mem.target_last_seen_tick >= 0:
            mem.target_age_ticks += 1
            if mem.tick - mem.target_last_seen_tick > cfg.target_timeout_ticks:
                mem.forget_target()

        # Target progress based stuck detection. No absolute position is supplied by
        # the challenge, so distance-to-persistent-target is the reliable signal.
        if mem.target_stall_ticks >= cfg.stuck_tick_threshold:
            mem.recovery_until_tick = mem.tick + cfg.recovery_ticks
            mem.forget_target()

        state = self._state_for(
            mem=mem,
            energy=energy,
            max_energy=max_energy,
            age=age,
            n_agents=n_agents,
            fruits=groups["Fruit"],
            nearest_predator_distance=nearest_predator_distance,
            closest_edge_distance=closest_edge_distance,
        )
        mem.set_state(state)

        desired = self._desired_vector(state=state, mem=mem, chosen_fruit=chosen_fruit, groups=groups)
        move_angle = angle_from_vector(desired, default=0.0)
        turn_angle = self._turn_angle(state=state, desired_move_angle=move_angle, nearest_predator=nearest_predator)
        move_distance = self._movement_distance(state, status, nearest_predator_distance)

        spawn = state == BehaviorState.REPRODUCTION
        if spawn:
            mem.last_spawn_tick = mem.tick
            mem.reproduction_cooldown_until = mem.tick + cfg.reproduction_cooldown_ticks

        action_mem = ActionMemory(
            move_distance=move_distance,
            move_direction=move_angle,
            turn_angle=turn_angle,
            spawn_agent=spawn,
        )
        mem.finalize_tick(energy=energy, age=age, action=action_mem)

        return Decision(
            agent_id=agent_id,
            move_distance=move_distance,
            move_direction=move_angle,
            turn_angle=turn_angle,
            spawn_agent=spawn,
            state=state,
        )
