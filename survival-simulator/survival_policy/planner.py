from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Optional, Sequence

from .config import PolicyConfig
from .geometry import norm, polar_to_cart, segment_distance_to_point, wrap_angle


@dataclass(frozen=True)
class PlannedAction:
    move_distance: float
    move_direction: float
    turn_angle: float
    score: float


class MicroPlanner:
    """Tiny deterministic short-horizon action scorer.

    The hierarchy/potential field proposes a preferred direction. The planner
    evaluates only 20 nearby actions (5 directions x 4 distances), so this adds
    predictive geometry without becoming a global planner or expensive MPC.
    """

    ANGLE_MULTIPLIERS = (-1.0, -0.5, 0.0, 0.5, 1.0)
    DISTANCE_MULTIPLIERS = (0.50, 0.70, 0.88, 1.05)

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
            # Keep the body partially oriented toward the threat while allowing
            # a tangential escape path. This preserves the useful simulator-
            # specific behavior already present in the v1 controller.
            face_turn = self._clip_turn(0.55 * wrap_angle(predator_angle), cfg.max_turn_angle)
            return self._clip_turn(face_turn + 0.20 * move_offset, cfg.max_turn_angle)
        return self._clip_turn(base_turn + 0.35 * move_offset, cfg.max_turn_angle)

    def _wall_score(self, endpoint: tuple[float, float], edges: Sequence[Mapping]) -> float:
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
            # Effectively reject actions that project into a hard wall margin.
            return -30.0 - 4.0 * (cfg.planner_wall_clearance - minimum)
        if minimum < cfg.planner_wall_soft_clearance:
            frac = (cfg.planner_wall_soft_clearance - minimum) / (
                cfg.planner_wall_soft_clearance - cfg.planner_wall_clearance
            )
            return -3.0 * frac * frac
        return 0.15

    def _predator_score(
        self,
        endpoint: tuple[float, float],
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
            # Bounded separation reward plus steep close-range penalty.
            safety = min(2.0, sep / max(cfg.predator_danger_distance, 1.0))
            close_penalty = max(0.0, 1.0 - sep / max(cfg.emergency_predator_distance * 1.6, 1.0))
            total += cfg.planner_predator_weight * (safety - 3.0 * close_penalty)

        if math.isfinite(ttc_ticks):
            urgency = max(0.0, 1.0 - ttc_ticks / cfg.ttc_danger_ticks)
            total -= 2.5 * cfg.planner_predator_weight * urgency
        return total

    def _food_score(self, endpoint: tuple[float, float], fruit: Optional[Mapping], critical: bool) -> float:
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

    def _spacing_score(self, endpoint: tuple[float, float], agents: Sequence[Mapping]) -> float:
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

    def _energy_score(
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
            # Approximate the simulator's strongly increased sprint cost.
            excess = (move_distance - normal) / max(sprint - normal, 1e-6)
            move_cost += 2.0 * excess * excess
        turn_cost = abs(turn_angle) / max(cfg.max_turn_angle, 1e-6)
        low_energy_multiplier = 1.0 + max(0.0, cfg.low_energy_ratio - energy_ratio) * 2.0
        if conservation_state:
            low_energy_multiplier *= 1.4
        return -cfg.planner_energy_weight * low_energy_multiplier * (move_cost + 0.30 * turn_cost)

    def plan(
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
        for angle_mult in self.ANGLE_MULTIPLIERS:
            offset = cfg.planner_angle_spread * angle_mult
            move_angle = wrap_angle(preferred_move_angle + offset)
            turn = self._candidate_turn(
                base_turn=base_turn_angle,
                move_offset=offset,
                predator_angle=predator_angle,
                threat_state=threat_state,
            )
            for dist_mult in self.DISTANCE_MULTIPLIERS:
                move_distance = max(0.0, min(sprint_speed, nominal_move_distance * dist_mult))
                dx, dy = polar_to_cart(move_distance * horizon, move_angle)
                endpoint = (dx, dy)

                score = 0.0
                score += self._wall_score(endpoint, edges)
                score += self._predator_score(
                    endpoint,
                    predators,
                    ttc_ticks=predator_ttc_ticks,
                    normal_speed=normal_speed,
                )
                score += self._food_score(endpoint, chosen_fruit, critical)
                score += self._spacing_score(endpoint, agents)
                score += self._energy_score(
                    move_distance=move_distance,
                    turn_angle=turn,
                    normal_speed=normal_speed,
                    sprint_speed=sprint_speed,
                    energy_ratio=energy_ratio,
                    conservation_state=conservation,
                )
                score += cfg.planner_prior_weight * math.cos(offset)

                if threat_state and predator_angle is not None:
                    # Reward keeping the body oriented toward the predator while
                    # the movement vector can still be tangential/away.
                    post_turn_bearing = wrap_angle(predator_angle - turn)
                    score += 0.45 * math.cos(post_turn_bearing)

                candidate = PlannedAction(move_distance, move_angle, turn, score)
                if best is None or candidate.score > best.score:
                    best = candidate

        assert best is not None
        return best
