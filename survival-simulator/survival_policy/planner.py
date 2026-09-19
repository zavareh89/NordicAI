from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Optional, Sequence

from .config import PolicyConfig
from .geometry import (
    Vec2,
    closest_approach,
    polar_to_cart,
    segment_distance_to_point,
    segment_segment_distance,
    unit_from_angle,
    wrap_angle,
)


@dataclass(frozen=True)
class PlannedAction:
    move_distance: float
    move_direction: float
    turn_angle: float
    score: float


@dataclass(frozen=True)
class _RolloutNode:
    position: Vec2
    heading: float
    score: float
    first_action: PlannedAction


class MicroPlanner:
    """Deterministic v2/v3 local planner.

    V2 retains the original endpoint scorer for reproducibility. V3 adds a
    bounded receding-horizon beam rollout, candidate-specific predator CPA/TTC,
    path-based wall geometry, state-adaptive action templates, and cumulative
    energy accounting. Beam width defaults to two so controller latency remains
    close to v2 despite using true sequential rollouts.
    """

    V2_ANGLE_MULTIPLIERS = (-1.0, -0.5, 0.0, 0.5, 1.0)
    V2_DISTANCE_MULTIPLIERS = (0.50, 0.70, 0.88, 1.05)

    def __init__(self, config: PolicyConfig):
        self.config = config

    @staticmethod
    def _clip_turn(turn: float, max_turn: float) -> float:
        return max(-max_turn, min(max_turn, turn))

    def _candidate_turn(
        self,
        *,
        base_turn: float,
        move_offset: float,
        predator_angle: Optional[float],
        threat_state: bool,
    ) -> float:
        cfg = self.config
        if threat_state and predator_angle is not None:
            face_turn = self._clip_turn(0.55 * wrap_angle(predator_angle), cfg.max_turn_angle)
            return self._clip_turn(face_turn + 0.20 * move_offset, cfg.max_turn_angle)
        return self._clip_turn(base_turn + 0.35 * move_offset, cfg.max_turn_angle)

    # ------------------------------------------------------------------
    # V2 scorer (kept intact for exact architecture-v2 reproduction)
    # ------------------------------------------------------------------
    def _v2_wall_score(self, endpoint: Vec2, edges: Sequence[Mapping]) -> float:
        cfg = self.config
        if not edges:
            return 0.0
        minimum = math.inf
        for edge in edges:
            coords = edge.get("coords")
            if not coords or len(coords) != 2:
                continue
            try:
                d, _ = segment_distance_to_point(endpoint, coords[0], coords[1])
            except (TypeError, ValueError, IndexError):
                continue
            minimum = min(minimum, d)
        if not math.isfinite(minimum):
            return 0.0
        if minimum < cfg.planner_wall_clearance:
            return -30.0 - 4.0 * (cfg.planner_wall_clearance - minimum)
        if minimum < cfg.planner_wall_soft_clearance:
            frac = (cfg.planner_wall_soft_clearance - minimum) / (
                cfg.planner_wall_soft_clearance - cfg.planner_wall_clearance
            )
            return -3.0 * frac * frac
        return 0.15


    # Backward-compatible test/helper alias retained for v2 callers.
    def _wall_score(self, endpoint: Vec2, edges: Sequence[Mapping]) -> float:
        return self._v2_wall_score(endpoint, edges)

    def _v2_predator_score(
        self,
        endpoint: Vec2,
        predators: Sequence[Mapping],
        *,
        ttc_ticks: float,
        normal_speed: float,
    ) -> float:
        cfg = self.config
        if not predators:
            return 0.0
        total = 0.0
        horizon = float(cfg.planner_horizon_steps)
        for pred in predators:
            try:
                d = max(1.0, float(pred.get("distance")))
                a = float(pred.get("angle"))
            except (TypeError, ValueError):
                continue
            px, py = polar_to_cart(d, a)
            rel_dir = pred.get("rel_dir")
            if rel_dir is not None:
                try:
                    heading = wrap_angle(a + math.pi - float(rel_dir))
                    step = cfg.predator_prediction_weight * normal_speed * horizon
                    px += math.cos(heading) * step
                    py += math.sin(heading) * step
                except (TypeError, ValueError):
                    pass
            sep = max(1.0, math.hypot(px - endpoint[0], py - endpoint[1]))
            safety = min(2.0, sep / max(cfg.predator_danger_distance, 1.0))
            close_penalty = max(0.0, 1.0 - sep / max(cfg.emergency_predator_distance * 1.6, 1.0))
            total += cfg.planner_predator_weight * (safety - 3.0 * close_penalty)
        if math.isfinite(ttc_ticks):
            urgency = max(0.0, 1.0 - ttc_ticks / cfg.ttc_danger_ticks)
            total -= 2.5 * cfg.planner_predator_weight * urgency
        return total

    def _food_score(self, endpoint: Vec2, fruit: Optional[Mapping], critical: bool) -> float:
        if fruit is None:
            return 0.0
        try:
            d = max(1.0, float(fruit.get("distance")))
            a = float(fruit.get("angle"))
        except (TypeError, ValueError):
            return 0.0
        fx, fy = polar_to_cart(d, a)
        after = math.hypot(fx - endpoint[0], fy - endpoint[1])
        progress = (d - after) / d
        multiplier = 1.35 if critical else 1.0
        return self.config.planner_food_weight * multiplier * progress

    def _spacing_score(self, endpoint: Vec2, agents: Sequence[Mapping]) -> float:
        cfg = self.config
        score = 0.0
        for other in agents:
            try:
                d = float(other.get("distance"))
                a = float(other.get("angle"))
            except (TypeError, ValueError):
                continue
            ox, oy = polar_to_cart(d, a)
            sep = math.hypot(ox - endpoint[0], oy - endpoint[1])
            if sep < cfg.herbivore_repulsion_radius:
                score -= cfg.planner_spacing_weight * (
                    1.0 - sep / max(cfg.herbivore_repulsion_radius, 1.0)
                )
        return score

    def _v2_energy_score(
        self,
        *,
        move_distance: float,
        turn_angle: float,
        normal_speed: float,
        sprint_speed: float,
        energy_ratio: float,
        conservation_state: bool,
    ) -> float:
        cfg = self.config
        normal = max(normal_speed, 1e-6)
        sprint = max(sprint_speed, normal)
        move_cost = move_distance / normal
        if move_distance > normal:
            excess = (move_distance - normal) / max(sprint - normal, 1e-6)
            move_cost += 2.0 * excess * excess
        turn_cost = abs(turn_angle) / max(cfg.max_turn_angle, 1e-6)
        low_energy_multiplier = 1.0 + max(0.0, cfg.low_energy_ratio - energy_ratio) * 2.0
        if conservation_state:
            low_energy_multiplier *= 1.4
        return -cfg.planner_energy_weight * low_energy_multiplier * (move_cost + 0.30 * turn_cost)

    def _plan_v2(
        self,
        *,
        preferred_move_angle: float,
        base_turn_angle: float,
        nominal_move_distance: float,
        normal_speed: float,
        sprint_speed: float,
        state_name: str,
        energy_ratio: float,
        chosen_fruit: Optional[Mapping],
        predators: Sequence[Mapping],
        edges: Sequence[Mapping],
        agents: Sequence[Mapping],
        predator_ttc_ticks: float,
    ) -> PlannedAction:
        cfg = self.config
        threat_state = state_name in {"EMERGENCY_ESCAPE", "PREDATOR_EVASION"}
        critical = state_name == "CRITICAL_ENERGY_FORAGING"
        conservation = state_name == "ENERGY_CONSERVATION"
        predator_angle: Optional[float] = None
        if predators:
            try:
                nearest = min(predators, key=lambda p: float(p.get("distance", math.inf)))
                predator_angle = float(nearest.get("angle"))
            except (TypeError, ValueError):
                predator_angle = None

        horizon = float(cfg.planner_horizon_steps)
        best: Optional[PlannedAction] = None
        for angle_mult in self.V2_ANGLE_MULTIPLIERS:
            offset = cfg.planner_angle_spread * angle_mult
            move_angle = wrap_angle(preferred_move_angle + offset)
            turn = self._candidate_turn(
                base_turn=base_turn_angle,
                move_offset=offset,
                predator_angle=predator_angle,
                threat_state=threat_state,
            )
            for dist_mult in self.V2_DISTANCE_MULTIPLIERS:
                move_distance = max(0.0, min(sprint_speed, nominal_move_distance * dist_mult))
                endpoint = polar_to_cart(move_distance * horizon, move_angle)
                score = 0.0
                score += self._v2_wall_score(endpoint, edges)
                score += self._v2_predator_score(endpoint, predators, ttc_ticks=predator_ttc_ticks, normal_speed=normal_speed)
                score += self._food_score(endpoint, chosen_fruit, critical)
                score += self._spacing_score(endpoint, agents)
                score += self._v2_energy_score(
                    move_distance=move_distance,
                    turn_angle=turn,
                    normal_speed=normal_speed,
                    sprint_speed=sprint_speed,
                    energy_ratio=energy_ratio,
                    conservation_state=conservation,
                )
                score += cfg.planner_prior_weight * math.cos(offset)
                if threat_state and predator_angle is not None:
                    post_turn_bearing = wrap_angle(predator_angle - turn)
                    score += 0.45 * math.cos(post_turn_bearing)
                candidate = PlannedAction(move_distance, move_angle, turn, score)
                if best is None or candidate.score > best.score:
                    best = candidate
        assert best is not None
        return best

    # ------------------------------------------------------------------
    # V3 rollout scorer
    # ------------------------------------------------------------------
    @staticmethod
    def _parsed_edges(edges: Sequence[Mapping]) -> tuple[tuple[Vec2, Vec2], ...]:
        out: list[tuple[Vec2, Vec2]] = []
        for edge in edges:
            coords = edge.get("coords")
            if not coords or len(coords) != 2:
                continue
            try:
                out.append(((float(coords[0][0]), float(coords[0][1])), (float(coords[1][0]), float(coords[1][1]))))
            except (TypeError, ValueError, IndexError):
                continue
        return tuple(out)

    @staticmethod
    def _parsed_agents(agents: Sequence[Mapping]) -> tuple[Vec2, ...]:
        out: list[Vec2] = []
        for other in agents:
            try:
                out.append(polar_to_cart(float(other.get("distance")), float(other.get("angle"))))
            except (TypeError, ValueError):
                pass
        return tuple(out)

    def _parsed_predators(self, predators: Sequence[Mapping], normal_speed: float) -> tuple[tuple[Vec2, Vec2], ...]:
        cfg = self.config
        out: list[tuple[Vec2, Vec2]] = []
        for pred in predators:
            try:
                d = max(1.0, float(pred.get("distance")))
                a = float(pred.get("angle"))
            except (TypeError, ValueError):
                continue
            pos = polar_to_cart(d, a)
            velocity = (0.0, 0.0)
            rel_dir = pred.get("rel_dir")
            if rel_dir is not None:
                try:
                    heading = wrap_angle(a + math.pi - float(rel_dir))
                    u = unit_from_angle(heading)
                    magnitude = cfg.predator_prediction_weight * max(normal_speed, 0.0)
                    velocity = (u[0] * magnitude, u[1] * magnitude)
                except (TypeError, ValueError):
                    pass
            out.append((pos, velocity))
        return tuple(out)

    def _state_template(self, state_name: str, first_step: bool) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Small state-dependent candidate sets; follow-up branching is tiny."""
        if not first_step:
            return (-0.35, 0.35), (0.90,)
        if state_name in {"EMERGENCY_ESCAPE", "PREDATOR_EVASION"}:
            return (-1.0, -0.50, 0.0, 0.50, 1.0), (0.88, 1.03)
        if state_name in {"CRITICAL_ENERGY_FORAGING", "NORMAL_FORAGING"}:
            return (-0.75, -0.35, 0.0, 0.35, 0.75), (0.78, 1.00)
        if state_name == "ENERGY_CONSERVATION":
            return (-0.50, 0.0, 0.50), (0.62, 0.92)
        if state_name == "WALL_AVOIDANCE":
            return (-1.10, -0.55, 0.0, 0.55, 1.10), (0.62, 0.88)
        if state_name == "RECOVERY":
            return (-1.0, -0.45, 0.0, 0.45, 1.0), (0.76, 0.98)
        return (-0.70, -0.30, 0.0, 0.30, 0.70), (0.75, 1.0)

    def _path_wall_score(self, start: Vec2, end: Vec2, edges: tuple[tuple[Vec2, Vec2], ...]) -> float:
        cfg = self.config
        if not edges:
            return 0.0
        minimum = math.inf
        for a, b in edges:
            d = segment_segment_distance(start, end, a, b)
            if d <= 1e-8:
                return -45.0
            minimum = min(minimum, d)
        if not math.isfinite(minimum):
            return 0.0
        hard = cfg.planner_wall_clearance
        soft = max(hard + 1e-6, cfg.planner_path_wall_soft_clearance)
        if minimum < hard:
            return -24.0 - 3.0 * (hard - minimum)
        if minimum < soft:
            frac = (soft - minimum) / (soft - hard)
            return -2.4 * frac * frac
        return 0.08

    def _cpa_predator_score(
        self,
        *,
        start: Vec2,
        end: Vec2,
        depth: int,
        predators: tuple[tuple[Vec2, Vec2], ...],
    ) -> float:
        cfg = self.config
        if not predators:
            return 0.0
        agent_velocity = (end[0] - start[0], end[1] - start[1])
        risks: list[float] = []
        for initial_pos, pred_vel in predators:
            pred_start = (
                initial_pos[0] + pred_vel[0] * depth,
                initial_pos[1] + pred_vel[1] * depth,
            )
            relative_pos = (pred_start[0] - start[0], pred_start[1] - start[1])
            relative_vel = (pred_vel[0] - agent_velocity[0], pred_vel[1] - agent_velocity[1])
            min_sep, t_cpa = closest_approach(relative_pos, relative_vel, 1.0)
            danger_scale = max(cfg.predator_danger_distance, 1.0)
            emergency_scale = max(cfg.emergency_predator_distance, 1.0)
            proximity = max(0.0, 1.0 - min_sep / danger_scale)
            emergency = max(0.0, 1.0 - min_sep / (1.7 * emergency_scale))
            imminence = 1.0 - 0.35 * t_cpa
            risks.append(imminence * (proximity * proximity + 2.5 * emergency * emergency))
        if not risks:
            return 0.0
        # Soft-max-like aggregation without exp(): the worst predator dominates,
        # while secondary predators still matter a little.
        worst = max(risks)
        aggregate = worst + 0.15 * (sum(risks) - worst)
        return -cfg.planner_cpa_weight * cfg.planner_predator_weight * aggregate

    def _rollout_food_score(self, start: Vec2, end: Vec2, fruit_pos: Optional[Vec2], critical: bool) -> float:
        if fruit_pos is None:
            return 0.0
        before = max(1.0, math.hypot(fruit_pos[0] - start[0], fruit_pos[1] - start[1]))
        after = math.hypot(fruit_pos[0] - end[0], fruit_pos[1] - end[1])
        progress = (before - after) / before
        return self.config.planner_food_weight * (1.35 if critical else 1.0) * progress

    def _rollout_spacing_score(self, end: Vec2, agents: tuple[Vec2, ...]) -> float:
        cfg = self.config
        score = 0.0
        for ox, oy in agents:
            sep = math.hypot(ox - end[0], oy - end[1])
            if sep < cfg.herbivore_repulsion_radius:
                score -= cfg.planner_spacing_weight * (1.0 - sep / max(cfg.herbivore_repulsion_radius, 1.0))
        return score

    @staticmethod
    def _simulator_energy_cost(
        *,
        move_distance: float,
        turn_angle: float,
        normal_speed: float,
    ) -> float:
        """Return the exact action-energy deduction used by environment.py.

        Movement is piecewise linear: movement up to ``speed`` costs 0.05 per
        distance unit, while the sprint portion above ``speed`` costs 0.5 per
        unit. Turning costs ``min(pi, abs(turn_angle)) / (2*pi)`` energy.

        Low-energy sprint suppression is handled separately by
        :meth:`_effective_move_distance`, because the simulator caps the
        requested movement before charging movement energy.
        """
        normal = max(float(normal_speed), 0.0)
        distance = max(0.0, float(move_distance))
        walking_distance = min(distance, normal)
        sprint_distance = max(0.0, distance - normal)
        movement_cost = 0.05 * walking_distance + 0.5 * sprint_distance
        turning_cost = min(math.pi, abs(float(turn_angle))) / (2.0 * math.pi)
        return movement_cost + turning_cost

    @staticmethod
    def _effective_move_distance(
        *,
        requested_distance: float,
        normal_speed: float,
        sprint_speed: float,
        energy_ratio: float,
    ) -> float:
        """Mirror environment.py's movement clamp for rollout simulation.

        The simulator clamps actions to sprint speed, and if current energy is
        below one fifth of maximum energy it prevents sprinting by capping the
        movement distance at normal speed.
        """
        normal = max(0.0, float(normal_speed))
        sprint = max(normal, float(sprint_speed))
        distance = max(0.0, min(float(requested_distance), sprint))
        if float(energy_ratio) < 0.20 and distance > normal:
            distance = normal
        return distance

    def _simulator_aligned_energy_score(
        self,
        *,
        move_distance: float,
        turn_angle: float,
        normal_speed: float,
        sprint_speed: float,
        max_energy: float,
        energy_ratio: float,
        conservation_state: bool,
    ) -> float:
        """Score the exact simulator energy cost in a planner-friendly scale.

        ``_simulator_energy_cost`` is the simulator-faithful deduction. The
        normalization and low-energy/conservation pressure below are policy
        preferences only; they do not change the modeled simulator cost.
        """
        effective_distance = self._effective_move_distance(
            requested_distance=move_distance,
            normal_speed=normal_speed,
            sprint_speed=sprint_speed,
            energy_ratio=energy_ratio,
        )
        cost = self._simulator_energy_cost(
            move_distance=effective_distance,
            turn_angle=turn_angle,
            normal_speed=normal_speed,
        )
        pressure = 1.0 + 0.8 * max(0.0, self.config.low_energy_ratio - energy_ratio)
        if conservation_state:
            pressure *= 1.35
        normalized = cost / max(1.0, 0.01 * max_energy)
        return -self.config.planner_energy_score_scale * pressure * normalized

    def _step_score(
        self,
        *,
        start: Vec2,
        end: Vec2,
        turn: float,
        move_distance: float,
        offset: float,
        depth: int,
        state_name: str,
        normal_speed: float,
        sprint_speed: float,
        max_energy: float,
        energy_ratio: float,
        fruit_pos: Optional[Vec2],
        predators: tuple[tuple[Vec2, Vec2], ...],
        edges: tuple[tuple[Vec2, Vec2], ...],
        agents: tuple[Vec2, ...],
    ) -> float:
        critical = state_name == "CRITICAL_ENERGY_FORAGING"
        conservation = state_name == "ENERGY_CONSERVATION"
        score = 0.0
        score += self._path_wall_score(start, end, edges)
        score += self._cpa_predator_score(start=start, end=end, depth=depth, predators=predators)
        score += self._rollout_food_score(start, end, fruit_pos, critical)
        score += self._rollout_spacing_score(end, agents)
        score += self._simulator_aligned_energy_score(
            move_distance=move_distance,
            turn_angle=turn,
            normal_speed=normal_speed,
            sprint_speed=sprint_speed,
            max_energy=max_energy,
            energy_ratio=energy_ratio,
            conservation_state=conservation,
        )
        score += self.config.planner_prior_weight * 0.70 * math.cos(offset)
        return score

    def _plan_v3(
        self,
        *,
        preferred_move_angle: float,
        base_turn_angle: float,
        nominal_move_distance: float,
        normal_speed: float,
        sprint_speed: float,
        max_energy: float,
        state_name: str,
        energy_ratio: float,
        chosen_fruit: Optional[Mapping],
        predators: Sequence[Mapping],
        edges: Sequence[Mapping],
        agents: Sequence[Mapping],
    ) -> PlannedAction:
        cfg = self.config
        horizon = max(1, int(cfg.planner_horizon_steps))
        beam_width = max(1, int(cfg.planner_beam_width))
        threat_state = state_name in {"EMERGENCY_ESCAPE", "PREDATOR_EVASION"}
        predator_angle: Optional[float] = None
        if predators:
            try:
                nearest = min(predators, key=lambda p: float(p.get("distance", math.inf)))
                predator_angle = float(nearest.get("angle"))
            except (TypeError, ValueError):
                pass

        fruit_pos: Optional[Vec2] = None
        if chosen_fruit is not None:
            try:
                fruit_pos = polar_to_cart(float(chosen_fruit.get("distance")), float(chosen_fruit.get("angle")))
            except (TypeError, ValueError):
                fruit_pos = None
        parsed_predators = self._parsed_predators(predators, normal_speed)
        parsed_edges = self._parsed_edges(edges)
        parsed_agents = self._parsed_agents(agents)

        initial_angles, initial_distances = self._state_template(state_name, True)
        nodes: list[_RolloutNode] = []
        for angle_mult in initial_angles:
            offset = cfg.planner_angle_spread * angle_mult
            local_move_angle = wrap_angle(preferred_move_angle + offset)
            turn = self._candidate_turn(
                base_turn=base_turn_angle,
                move_offset=offset,
                predator_angle=predator_angle,
                threat_state=threat_state,
            )
            for dist_mult in initial_distances:
                move_distance = self._effective_move_distance(
                    requested_distance=nominal_move_distance * dist_mult,
                    normal_speed=normal_speed,
                    sprint_speed=sprint_speed,
                    energy_ratio=energy_ratio,
                )
                displacement = polar_to_cart(move_distance, local_move_angle)
                end = displacement
                step_score = self._step_score(
                    start=(0.0, 0.0), end=end, turn=turn, move_distance=move_distance,
                    offset=offset, depth=0, state_name=state_name,
                    normal_speed=normal_speed, sprint_speed=sprint_speed, max_energy=max_energy,
                    energy_ratio=energy_ratio, fruit_pos=fruit_pos, predators=parsed_predators,
                    edges=parsed_edges, agents=parsed_agents,
                )
                first = PlannedAction(move_distance, local_move_angle, turn, step_score)
                nodes.append(_RolloutNode(end, turn, step_score, first))
        nodes.sort(key=lambda n: n.score, reverse=True)
        nodes = nodes[:beam_width]

        # Follow-up branching is deliberately only 2 actions per beam node.
        # With beam_width=1 and horizon=3 this evaluates ~14 action steps versus
        # v2's 20 candidates. The framework still supports wider beams, but the
        # competition default is intentionally latency-bounded.
        follow_angles, follow_distances = self._state_template(state_name, False)
        for depth in range(1, horizon):
            expanded: list[_RolloutNode] = []
            for node in nodes:
                preferred_local = wrap_angle(preferred_move_angle - node.heading)
                for angle_mult in follow_angles:
                    offset = cfg.planner_followup_angle_spread * angle_mult
                    local_move_angle = wrap_angle(preferred_local + offset)
                    world_move_angle = wrap_angle(node.heading + local_move_angle)
                    turn = self._candidate_turn(
                        base_turn=wrap_angle(base_turn_angle - node.heading),
                        move_offset=offset,
                        predator_angle=(None if predator_angle is None else wrap_angle(predator_angle - node.heading)),
                        threat_state=threat_state,
                    )
                    for dist_mult in follow_distances:
                        move_distance = self._effective_move_distance(
                            requested_distance=nominal_move_distance * dist_mult,
                            normal_speed=normal_speed,
                            sprint_speed=sprint_speed,
                            energy_ratio=energy_ratio,
                        )
                        dx, dy = polar_to_cart(move_distance, world_move_angle)
                        end = (node.position[0] + dx, node.position[1] + dy)
                        step_score = self._step_score(
                            start=node.position, end=end, turn=turn, move_distance=move_distance,
                            offset=offset, depth=depth, state_name=state_name,
                            normal_speed=normal_speed, sprint_speed=sprint_speed, max_energy=max_energy,
                            energy_ratio=energy_ratio, fruit_pos=fruit_pos, predators=parsed_predators,
                            edges=parsed_edges, agents=parsed_agents,
                        )
                        total = node.score + (cfg.planner_rollout_discount ** depth) * step_score
                        expanded.append(_RolloutNode(
                            end,
                            wrap_angle(node.heading + turn),
                            total,
                            node.first_action,
                        ))
            expanded.sort(key=lambda n: n.score, reverse=True)
            nodes = expanded[:beam_width] if expanded else nodes
        assert nodes
        winner = max(nodes, key=lambda n: n.score)
        return PlannedAction(
            winner.first_action.move_distance,
            winner.first_action.move_direction,
            winner.first_action.turn_angle,
            winner.score,
        )

    def plan(
        self,
        *,
        preferred_move_angle: float,
        base_turn_angle: float,
        nominal_move_distance: float,
        normal_speed: float,
        sprint_speed: float,
        max_energy: float = 500.0,
        state_name: str,
        energy_ratio: float,
        chosen_fruit: Optional[Mapping],
        predators: Sequence[Mapping],
        edges: Sequence[Mapping],
        agents: Sequence[Mapping],
        predator_ttc_ticks: float,
    ) -> PlannedAction:
        if self.config.architecture_v3_enabled:
            return self._plan_v3(
                preferred_move_angle=preferred_move_angle,
                base_turn_angle=base_turn_angle,
                nominal_move_distance=nominal_move_distance,
                normal_speed=normal_speed,
                sprint_speed=sprint_speed,
                max_energy=max_energy,
                state_name=state_name,
                energy_ratio=energy_ratio,
                chosen_fruit=chosen_fruit,
                predators=predators,
                edges=edges,
                agents=agents,
            )
        return self._plan_v2(
            preferred_move_angle=preferred_move_angle,
            base_turn_angle=base_turn_angle,
            nominal_move_distance=nominal_move_distance,
            normal_speed=normal_speed,
            sprint_speed=sprint_speed,
            state_name=state_name,
            energy_ratio=energy_ratio,
            chosen_fruit=chosen_fruit,
            predators=predators,
            edges=edges,
            agents=agents,
            predator_ttc_ticks=predator_ttc_ticks,
        )
