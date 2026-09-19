from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class PolicyConfig:
    """Centralized, optimizer-independent controller configuration.

    The original HPO parameters remain intact. Architecture-v2 parameters are
    appended with conservative defaults so older Cxxx ``params.json`` files
    continue to load without modification.
    """

    master_seed: int = 20260918

    # Predator handling (original HPO space)
    emergency_predator_distance: float = 45.0
    predator_danger_distance: float = 150.0
    predator_repulsion_weight: float = 4.5
    predator_prediction_weight: float = 0.65
    escape_persistence_ticks: int = 6

    # Walls / navigation / spacing (original HPO space)
    wall_danger_distance: float = 52.0
    wall_repulsion_weight: float = 3.2
    herbivore_repulsion_radius: float = 42.0
    herbivore_repulsion_weight: float = 0.65
    max_turn_angle: float = 0.35

    # Food / targeting (original HPO space)
    critical_energy_ratio: float = 0.23
    low_energy_ratio: float = 0.36
    fruit_attraction_weight: float = 2.4
    fruit_distance_penalty: float = 0.010
    fruit_competition_penalty: float = 1.25
    target_persistence_bonus: float = 0.70
    target_timeout_ticks: int = 24

    # Movement / exploration (original HPO space)
    forage_move_fraction: float = 0.86
    explore_move_fraction: float = 0.58
    conserve_move_fraction: float = 0.32
    evasion_sprint_fraction: float = 0.78
    exploration_change_interval: int = 45

    # Reproduction (original HPO space)
    spawn_min_energy: float = 225.0
    spawn_old_age: float = 55.0
    population_soft_cap: int = 12
    reproduction_cooldown_ticks: int = 90

    # Stuck / recovery (original HPO space)
    stuck_tick_threshold: int = 7
    recovery_ticks: int = 6

    # Master compatibility gate. Set false to reproduce the v1 controller used
    # to generate the original HPO candidates.
    architecture_v2_enabled: bool = True

    # Architecture v2: short-horizon action planner. These are intentionally
    # fixed during the existing C002+ HPO so old studies remain comparable.
    planner_enabled: bool = True
    planner_horizon_steps: int = 2
    planner_angle_spread: float = 0.42
    planner_predator_weight: float = 4.0
    planner_food_weight: float = 1.7
    planner_energy_weight: float = 0.55
    planner_wall_clearance: float = 18.0
    planner_wall_soft_clearance: float = 42.0

    # Architecture v2: predictive threat / memory / coordination.
    ttc_danger_ticks: float = 22.0
    fruit_track_ttl: int = 30
    fruit_track_match_angle: float = 0.38
    fruit_track_match_distance: float = 48.0
    dynamic_population_bonus: int = 6

    @property
    def predator_memory_ticks(self) -> int:
        return self.escape_persistence_ticks + 2

    @property
    def face_predator_distance(self) -> float:
        return self.predator_danger_distance * 1.25

    @property
    def wall_critical_distance(self) -> float:
        return self.wall_danger_distance * 0.35

    @property
    def fruit_competition_radius(self) -> float:
        return self.herbivore_repulsion_radius

    @property
    def target_switch_margin(self) -> float:
        return 0.30

    @property
    def reproduction_danger_distance(self) -> float:
        return self.predator_danger_distance * 1.20

    @property
    def spawn_old_age_min_energy(self) -> float:
        return max(105.0, self.spawn_min_energy - 40.0)

    @property
    def stuck_progress_epsilon(self) -> float:
        return 0.8

    @property
    def planner_spacing_weight(self) -> float:
        return 0.35 + 0.45 * self.herbivore_repulsion_weight

    @property
    def planner_prior_weight(self) -> float:
        return 0.55

    def validate(self) -> None:
        positive = {
            "emergency_predator_distance": self.emergency_predator_distance,
            "predator_danger_distance": self.predator_danger_distance,
            "wall_danger_distance": self.wall_danger_distance,
            "herbivore_repulsion_radius": self.herbivore_repulsion_radius,
            "spawn_min_energy": self.spawn_min_energy,
            "planner_horizon_steps": self.planner_horizon_steps,
            "planner_angle_spread": self.planner_angle_spread,
            "planner_predator_weight": self.planner_predator_weight,
            "planner_food_weight": self.planner_food_weight,
            "planner_energy_weight": self.planner_energy_weight,
            "planner_wall_clearance": self.planner_wall_clearance,
            "planner_wall_soft_clearance": self.planner_wall_soft_clearance,
            "ttc_danger_ticks": self.ttc_danger_ticks,
            "fruit_track_ttl": self.fruit_track_ttl,
            "fruit_track_match_angle": self.fruit_track_match_angle,
            "fruit_track_match_distance": self.fruit_track_match_distance,
            "dynamic_population_bonus": self.dynamic_population_bonus,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")
        if self.emergency_predator_distance >= self.predator_danger_distance:
            raise ValueError("emergency_predator_distance must be < predator_danger_distance")
        if self.planner_wall_clearance >= self.planner_wall_soft_clearance:
            raise ValueError("planner_wall_clearance must be < planner_wall_soft_clearance")
        for name in (
            "critical_energy_ratio", "low_energy_ratio", "forage_move_fraction",
            "explore_move_fraction", "conserve_move_fraction", "evasion_sprint_fraction",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1], got {value}")
        if self.critical_energy_ratio >= self.low_energy_ratio:
            raise ValueError("critical_energy_ratio must be < low_energy_ratio")
        for name in (
            "escape_persistence_ticks", "target_timeout_ticks", "exploration_change_interval",
            "population_soft_cap", "reproduction_cooldown_ticks", "stuck_tick_threshold",
            "recovery_ticks", "planner_horizon_steps", "fruit_track_ttl", "dynamic_population_bonus",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyConfig":
        valid_names = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - valid_names)
        if unknown:
            raise ValueError(f"Unknown config fields: {unknown}")
        cfg = cls(**data)
        cfg.validate()
        return cfg

    @classmethod
    def from_json(cls, path: str | Path) -> "PolicyConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @property
    def stable_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()[:16]


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "default_params.json"


def load_default_config() -> PolicyConfig:
    path = default_config_path()
    if path.exists():
        return PolicyConfig.from_json(path)
    cfg = PolicyConfig()
    cfg.validate()
    return cfg
