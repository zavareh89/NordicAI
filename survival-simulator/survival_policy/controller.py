from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .config import PolicyConfig
from .coordination import (
    PopulationContext,
    build_population_context,
    local_competition_cost,
    local_density,
    local_fruit_owner,
    local_fruit_owner_eta,
    select_reproduction_slots,
)
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
from .planner import MicroPlanner


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
        self.planner = MicroPlanner(config)

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
        self.memory.update_population(active_ids)
        n_agents = int(data.get("n_agents", len(active_ids)) or len(active_ids))
        population_context = build_population_context(
            normalized,
            base_cap=self.config.population_soft_cap,
            bonus_max=self.config.dynamic_population_bonus,
            population_trend=self.memory.population_trend,
        )
        if self.config.architecture_v3_enabled:
            raw_capacity = float(population_context.carrying_capacity)
            if self.memory.smoothed_carrying_capacity is None:
                self.memory.smoothed_carrying_capacity = raw_capacity
            else:
                alpha = self.config.population_capacity_smoothing
                self.memory.smoothed_carrying_capacity = (
                    (1.0 - alpha) * self.memory.smoothed_carrying_capacity + alpha * raw_capacity
                )
            population_context = PopulationContext(
                n_agents=population_context.n_agents,
                average_energy_ratio=population_context.average_energy_ratio,
                fruit_sightings_per_agent=population_context.fruit_sightings_per_agent,
                predator_sightings_per_agent=population_context.predator_sightings_per_agent,
                population_trend=population_context.population_trend,
                carrying_capacity=max(2, int(round(self.memory.smoothed_carrying_capacity))),
            )
            spawn_allowed_ids = select_reproduction_slots(
                normalized,
                slots=self.config.reproduction_slots_per_tick,
                predator_danger_distance=self.config.reproduction_danger_distance,
                local_density_radius=self.config.herbivore_repulsion_radius * 1.5,
                min_energy=self.config.spawn_old_age_min_energy,
            )
        else:
            spawn_allowed_ids = set(active_ids)
        decisions = [
            self._decide_agent(
                s,
                n_agents=n_agents,
                population_context=population_context,
                spawn_allowed_ids=spawn_allowed_ids,
            )
            for s in normalized
        ]
        return [d.action_dict() for d in decisions]

    def _agent_rng(self, agent_id: int, epoch: int) -> random.Random:
        return random.Random(stable_int_seed(self.config.master_seed, agent_id, epoch))

    def _ensure_exploration_heading(self, mem: AgentMemory, *, force_event: bool = False) -> None:
        cfg = self.config
        if not cfg.architecture_v3_enabled:
            if mem.tick < mem.next_exploration_change_tick and mem.next_exploration_change_tick > 0:
                return
            epoch = mem.tick // max(1, cfg.exploration_change_interval)
        else:
            # V3 is event-driven: retain a sector while progress is plausible and
            # only rotate on a meaningful event. A long fallback prevents an agent
            # from being permanently trapped in one unproductive sector.
            stale = (
                mem.exploration_last_change_tick >= 0
                and mem.tick - mem.exploration_last_change_tick >= cfg.exploration_stale_fallback_ticks
            )
            no_food_event = mem.no_food_ticks >= cfg.exploration_no_food_ticks
            if mem.exploration_last_change_tick >= 0 and not (force_event or stale or no_food_event):
                return
            mem.exploration_epoch += 1
            epoch = mem.exploration_epoch

        rng = self._agent_rng(mem.agent_id, epoch)
        if not cfg.architecture_v2_enabled:
            phase = (mem.agent_id * 2.399963229728653) % (2.0 * math.pi)
            jitter = rng.uniform(-0.65, 0.65)
        else:
            active_ids = self.memory.active_ids
            if mem.agent_id in active_ids and active_ids:
                rank = active_ids.index(mem.agent_id)
                phase = 2.0 * math.pi * rank / len(active_ids)
                phase += (epoch % max(1, len(active_ids))) * (math.pi / max(2, len(active_ids)))
            else:
                phase = (mem.agent_id * 2.399963229728653) % (2.0 * math.pi)
            jitter = rng.uniform(-0.22, 0.22)
        mem.exploration_world_angle = wrap_angle(phase + jitter)
        mem.exploration_last_change_tick = mem.tick
        # Retained for v1/v2 compatibility/debugging; v3 does not use this as
        # its primary switching mechanism.
        mem.next_exploration_change_tick = mem.tick + cfg.exploration_change_interval

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
        self_speed: float = 10.0,
    ) -> float:
        cfg = self.config
        try:
            d = max(0.0, float(fruit.get("distance")))
            a = float(fruit.get("angle"))
        except (TypeError, ValueError):
            return -math.inf

        if cfg.architecture_v3_enabled:
            # V3 targets time-to-capture rather than raw distance. The factor of
            # ten preserves the historical utility scale for a typical speed≈10.
            travel_units = d * (10.0 / max(self_speed, 1e-6))
            utility = -cfg.fruit_distance_penalty * travel_units
        else:
            utility = -cfg.fruit_distance_penalty * d

        # Stronger local assignment than v1: if another mutually visible
        # herbivore has the clearly better claim, this fruit is heavily
        # penalized rather than merely receiving a soft crowding penalty.
        if cfg.architecture_v2_enabled:
            if cfg.architecture_v3_enabled:
                owner = local_fruit_owner_eta(
                    self_id=agent_id,
                    fruit_distance=d,
                    fruit_angle=a,
                    self_speed=self_speed,
                    visible_agents=visible_agents,
                )
            else:
                owner = local_fruit_owner(
                    self_id=agent_id,
                    fruit_distance=d,
                    fruit_angle=a,
                    visible_agents=visible_agents,
                )
            if owner != agent_id:
                utility -= 2.0 * cfg.fruit_competition_penalty

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

        track_id = fruit.get("_track_id")
        if cfg.architecture_v2_enabled and track_id is not None and track_id == mem.target_track_id:
            utility += cfg.target_persistence_bonus
        else:
            old_local = mem.current_target_local_bearing()
            if old_local is not None and mem.target_distance is not None:
                angle_match = abs(wrap_angle(a - old_local))
                distance_match = abs(d - mem.target_distance)
                if angle_match < 0.45 and distance_match < 65.0:
                    utility += cfg.target_persistence_bonus if not cfg.architecture_v2_enabled else 0.5 * cfg.target_persistence_bonus
        return utility

    def _select_fruit(
        self,
        *,
        agent_id: int,
        fruits: Sequence[Mapping],
        visible_agents: Sequence[Mapping],
        predators: Sequence[Mapping],
        mem: AgentMemory,
        self_speed: float = 10.0,
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
                self_speed=self_speed,
            ), f)
            for f in fruits
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best = scored[0]

        # Prefer an explicitly tracked target when it remains competitive.
        if mem.target_track_id is not None:
            tracked = [
                (score, fruit) for score, fruit in scored
                if fruit.get("_track_id") == mem.target_track_id
            ]
            if tracked:
                old_score, old = max(tracked, key=lambda x: x[0])
                if best_score < old_score + self.config.target_switch_margin:
                    return old

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

    def _dynamic_predator_distance(
        self,
        *,
        mem: AgentMemory,
        energy_ratio: float,
        closest_edge_distance: float,
    ) -> float:
        cfg = self.config
        if not cfg.architecture_v2_enabled:
            return cfg.predator_danger_distance
        ttc_urgency = 0.0
        if math.isfinite(mem.predator_ttc_ticks):
            ttc_urgency = max(0.0, 1.0 - mem.predator_ttc_ticks / cfg.ttc_danger_ticks)
        low_energy = max(0.0, cfg.low_energy_ratio - energy_ratio) / max(cfg.low_energy_ratio, 1e-9)
        wall_pressure = 0.0
        if math.isfinite(closest_edge_distance):
            wall_pressure = max(0.0, 1.0 - closest_edge_distance / max(2.0 * cfg.wall_danger_distance, 1.0))
        multiplier = 1.0 + 0.28 * ttc_urgency + 0.10 * low_energy + 0.10 * wall_pressure
        return cfg.predator_danger_distance * multiplier

    def _dynamic_spawn_threshold(
        self,
        *,
        context: PopulationContext,
        visible_fruits: int,
        nearest_predator_distance: float,
        local_density_count: int,
    ) -> float:
        cfg = self.config
        if not cfg.architecture_v2_enabled:
            return cfg.spawn_min_energy
        threshold = cfg.spawn_min_energy
        # Food-rich/high-energy states can reproduce more aggressively.
        threshold -= min(28.0, 8.0 * visible_fruits)
        threshold -= max(0.0, context.average_energy_ratio - 0.55) * 35.0
        if context.population_trend < -0.35:
            threshold -= 12.0
        # Density/danger raise the required reserve.
        threshold += max(0, local_density_count - 2) * 5.0
        if nearest_predator_distance < cfg.reproduction_danger_distance * 1.5:
            threshold += 45.0
        if context.n_agents >= context.carrying_capacity - 1:
            threshold += 25.0
        return max(105.0, threshold)

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
        dynamic_predator_distance: float,
        closest_edge_distance: float,
        carrying_capacity: int,
        spawn_threshold: float,
        spawn_allowed: bool,
    ) -> BehaviorState:
        cfg = self.config
        ratio = energy / max(max_energy, 1.0)
        if mem.tick <= mem.recovery_until_tick:
            return BehaviorState.RECOVERY

        emergency_distance = cfg.emergency_predator_distance
        if cfg.architecture_v2_enabled and math.isfinite(mem.predator_ttc_ticks):
            urgency = max(0.0, 1.0 - mem.predator_ttc_ticks / cfg.ttc_danger_ticks)
            emergency_distance *= 1.0 + 0.20 * urgency
        if nearest_predator_distance <= emergency_distance:
            mem.escape_until_tick = mem.tick + cfg.escape_persistence_ticks
            return BehaviorState.EMERGENCY_ESCAPE
        if nearest_predator_distance <= dynamic_predator_distance or mem.tick <= mem.escape_until_tick:
            return BehaviorState.PREDATOR_EVASION

        # Wall handling is now mostly a planner feasibility constraint. Keep only
        # a truly critical wall state for cases where the agent is already close.
        if closest_edge_distance <= cfg.wall_critical_distance:
            return BehaviorState.WALL_AVOIDANCE
        if ratio <= cfg.critical_energy_ratio and fruits:
            return BehaviorState.CRITICAL_ENERGY_FORAGING

        can_spawn = (
            spawn_allowed
            and mem.tick >= mem.reproduction_cooldown_until
            and n_agents < carrying_capacity
            and nearest_predator_distance > cfg.reproduction_danger_distance
            and (
                energy >= spawn_threshold
                or (age >= cfg.spawn_old_age and energy >= max(cfg.spawn_old_age_min_energy, spawn_threshold - 40.0))
            )
        )
        if can_spawn:
            return BehaviorState.REPRODUCTION
        if fruits:
            return BehaviorState.NORMAL_FORAGING
        if ratio <= cfg.low_energy_ratio:
            return BehaviorState.ENERGY_CONSERVATION
        return BehaviorState.EXPLORATION

    def _movement_distance(
        self,
        state: BehaviorState,
        status: Mapping,
        nearest_predator_distance: float,
        *,
        chosen_fruit: Optional[Mapping],
        mem: AgentMemory,
    ) -> float:
        cfg = self.config
        speed = max(0.0, float(status.get("speed", 0.0) or 0.0))
        sprint = max(speed, float(status.get("sprint_speed", speed) or speed))
        energy = float(status.get("energy", 0.0) or 0.0)
        max_energy = max(1.0, float(status.get("max_energy", 1.0) or 1.0))
        ratio = energy / max_energy

        if not cfg.architecture_v2_enabled:
            if state == BehaviorState.EMERGENCY_ESCAPE:
                return sprint
            if state == BehaviorState.PREDATOR_EVASION:
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

        if state == BehaviorState.EMERGENCY_ESCAPE:
            return sprint
        if state == BehaviorState.PREDATOR_EVASION:
            if energy < max_energy * 0.22:
                return speed
            urgency = 0.0
            if math.isfinite(mem.predator_ttc_ticks):
                urgency = max(0.0, 1.0 - mem.predator_ttc_ticks / cfg.ttc_danger_ticks)
            fraction = min(1.0, cfg.evasion_sprint_fraction + 0.15 * urgency)
            return sprint * fraction
        if state in (BehaviorState.CRITICAL_ENERGY_FORAGING, BehaviorState.NORMAL_FORAGING):
            fraction = cfg.forage_move_fraction
            if state == BehaviorState.CRITICAL_ENERGY_FORAGING:
                fraction = min(1.0, fraction + 0.08)
            if chosen_fruit is not None:
                try:
                    d = float(chosen_fruit.get("distance"))
                    # Slow down near food instead of overshooting it.
                    if d < 2.0 * speed:
                        fraction *= max(0.45, d / max(2.0 * speed, 1.0))
                except (TypeError, ValueError):
                    pass
            if ratio < cfg.low_energy_ratio and state != BehaviorState.CRITICAL_ENERGY_FORAGING:
                fraction *= 0.88
            return speed * fraction
        if state == BehaviorState.ENERGY_CONSERVATION:
            # Conservation becomes progressively stronger as energy falls.
            scale_factor = max(0.55, ratio / max(cfg.low_energy_ratio, 1e-9))
            return speed * cfg.conserve_move_fraction * scale_factor
        if state == BehaviorState.RECOVERY:
            return speed * 0.85
        if state == BehaviorState.WALL_AVOIDANCE:
            return speed * 0.55
        if state == BehaviorState.REPRODUCTION:
            return speed * 0.20
        # Exploration can be somewhat faster when the population is healthy and
        # slows automatically near the low-energy boundary.
        fraction = cfg.explore_move_fraction
        if ratio < cfg.low_energy_ratio + 0.08:
            fraction *= 0.82
        return speed * fraction

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
        # In architecture v2 walls are primarily handled as predicted-action
        # constraints in MicroPlanner. Retain only a weak directional prior so
        # the original controller remains a useful proposal/fallback.
        if cfg.architecture_v2_enabled and cfg.planner_enabled:
            wall = scale(wall, 0.22)
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
            recovery_angle = wrap_angle((mem.current_target_local_bearing() or 0.0) + sign * math.pi / 2.0)
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
                mem.predator_closing_per_tick = 0.0
                mem.predator_bearing_rate = 0.0
                mem.predator_ttc_ticks = math.inf
            return
        try:
            local_angle = float(nearest_predator.get("angle"))
            distance = float(nearest_predator.get("distance"))
        except (TypeError, ValueError):
            return

        world_bearing = wrap_angle(mem.heading_estimate + local_angle)
        if mem.predator_distance is not None and mem.predator_last_seen_tick == mem.tick - 1:
            closing = mem.predator_distance - distance
            bearing_rate = wrap_angle(world_bearing - (mem.predator_world_bearing or world_bearing))
            # Light smoothing avoids reacting to one noisy observation.
            mem.predator_closing_per_tick = 0.65 * mem.predator_closing_per_tick + 0.35 * closing
            mem.predator_bearing_rate = 0.65 * mem.predator_bearing_rate + 0.35 * bearing_rate
        else:
            mem.predator_closing_per_tick = 0.0
            mem.predator_bearing_rate = 0.0

        if mem.predator_closing_per_tick > 0.10:
            mem.predator_ttc_ticks = distance / mem.predator_closing_per_tick
        else:
            mem.predator_ttc_ticks = math.inf
        mem.predator_world_bearing = world_bearing
        mem.predator_distance = distance
        mem.predator_last_seen_tick = mem.tick

    def _decide_agent(
        self,
        status: Mapping,
        *,
        n_agents: int,
        population_context: PopulationContext,
        spawn_allowed_ids: set[int],
    ) -> Decision:
        cfg = self.config
        agent_id = int(status.get("agent_id"))
        energy = float(status.get("energy", 0.0) or 0.0)
        age = float(status.get("age", 0.0) or 0.0)
        max_energy = max(1.0, float(status.get("max_energy", 1.0) or 1.0))
        energy_ratio = energy / max_energy
        mem = self.memory.get_or_create(agent_id)
        mem.begin_tick(energy, age)

        groups = self._split_observations(status.get("observations") or [])
        if cfg.architecture_v3_enabled:
            mem.no_food_ticks = 0 if groups["Fruit"] else mem.no_food_ticks + 1
        if cfg.architecture_v2_enabled:
            # Replace raw fruit observations with copies carrying persistent local
            # pseudo-IDs. This affects only controller memory, never the API payload.
            groups["Fruit"] = mem.update_fruit_tracks(
                groups["Fruit"],
                match_angle=cfg.fruit_track_match_angle,
                match_distance=cfg.fruit_track_match_distance,
                ttl=cfg.fruit_track_ttl,
            )
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
            self_speed=max(1e-6, float(status.get("speed", 0.0) or 0.0)),
        )

        if chosen_fruit is not None:
            try:
                mem.remember_target(
                    float(chosen_fruit.get("angle")),
                    float(chosen_fruit.get("distance")),
                    progress_epsilon=cfg.stuck_progress_epsilon,
                    track_id=int(chosen_fruit.get("_track_id")) if chosen_fruit.get("_track_id") is not None else None,
                )
            except (TypeError, ValueError):
                pass
        elif mem.target_last_seen_tick >= 0:
            mem.target_age_ticks += 1
            if mem.tick - mem.target_last_seen_tick > cfg.target_timeout_ticks:
                mem.forget_target()

        if mem.target_stall_ticks >= cfg.stuck_tick_threshold:
            mem.recovery_until_tick = mem.tick + cfg.recovery_ticks
            mem.forget_target()

        dynamic_predator_distance = self._dynamic_predator_distance(
            mem=mem,
            energy_ratio=energy_ratio,
            closest_edge_distance=closest_edge_distance,
        )
        spawn_threshold = self._dynamic_spawn_threshold(
            context=population_context,
            visible_fruits=len(groups["Fruit"]),
            nearest_predator_distance=nearest_predator_distance,
            local_density_count=mem.local_population_density,
        )

        state = self._state_for(
            mem=mem,
            energy=energy,
            max_energy=max_energy,
            age=age,
            n_agents=n_agents,
            fruits=groups["Fruit"],
            nearest_predator_distance=nearest_predator_distance,
            dynamic_predator_distance=dynamic_predator_distance,
            closest_edge_distance=closest_edge_distance,
            carrying_capacity=(population_context.carrying_capacity if cfg.architecture_v2_enabled else cfg.population_soft_cap),
            spawn_threshold=spawn_threshold,
            spawn_allowed=(agent_id in spawn_allowed_ids),
        )
        previous_predator_state = mem.was_in_predator_state
        current_predator_state = state in (BehaviorState.EMERGENCY_ESCAPE, BehaviorState.PREDATOR_EVASION)
        if cfg.architecture_v3_enabled and state == BehaviorState.EXPLORATION:
            exploration_event = (
                (previous_predator_state and not current_predator_state)
                or mem.target_stall_ticks >= cfg.stuck_tick_threshold
                or closest_edge_distance <= cfg.wall_critical_distance
            )
            self._ensure_exploration_heading(mem, force_event=exploration_event)
        mem.was_in_predator_state = current_predator_state
        mem.set_state(state)

        # V1 hierarchy/potential field remains the proposal generator.
        desired = self._desired_vector(state=state, mem=mem, chosen_fruit=chosen_fruit, groups=groups)
        move_angle = angle_from_vector(desired, default=0.0)
        turn_angle = self._turn_angle(
            state=state,
            desired_move_angle=move_angle,
            nearest_predator=nearest_predator,
        )
        move_distance = self._movement_distance(
            state,
            status,
            nearest_predator_distance,
            chosen_fruit=chosen_fruit,
            mem=mem,
        )

        # Architecture v2: cheaply evaluate 20 nearby actions. Potential fields
        # are no longer the final decision maker, but remain the prior/fallback.
        if cfg.architecture_v2_enabled and cfg.planner_enabled and state != BehaviorState.REPRODUCTION:
            normal_speed = max(0.0, float(status.get("speed", 0.0) or 0.0))
            sprint_speed = max(normal_speed, float(status.get("sprint_speed", normal_speed) or normal_speed))
            planned = self.planner.plan(
                preferred_move_angle=move_angle,
                base_turn_angle=turn_angle,
                nominal_move_distance=move_distance,
                normal_speed=normal_speed,
                sprint_speed=sprint_speed,
                max_energy=max_energy,
                state_name=state.value,
                energy_ratio=energy_ratio,
                chosen_fruit=chosen_fruit,
                predators=groups["Predator"],
                edges=groups["Edge"],
                agents=groups["Agent"],
                predator_ttc_ticks=mem.predator_ttc_ticks,
            )
            move_distance = planned.move_distance
            move_angle = planned.move_direction
            turn_angle = planned.turn_angle

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
