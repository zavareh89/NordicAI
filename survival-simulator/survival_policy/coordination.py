from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

from .geometry import polar_to_cart


@dataclass(frozen=True)
class PopulationContext:
    n_agents: int
    average_energy_ratio: float
    fruit_sightings_per_agent: float
    predator_sightings_per_agent: float
    population_trend: float
    carrying_capacity: int


def local_fruit_owner(
    *,
    self_id: int,
    fruit_distance: float,
    fruit_angle: float,
    visible_agents: Iterable[Mapping],
    tie_margin: float = 5.0,
) -> int:
    """Return the deterministic local owner of a fruit candidate.

    The closest visible herbivore owns the candidate. Near ties are broken by
    agent ID. This is intentionally local: the public API exposes neither fruit
    IDs nor globally aligned positions, so global reservations would fabricate
    unavailable state.
    """
    fx, fy = polar_to_cart(fruit_distance, fruit_angle)
    owner = self_id
    best_distance = fruit_distance
    for other in visible_agents:
        try:
            oid = int(other.get("id"))
            od = float(other.get("distance"))
            oa = float(other.get("angle"))
        except (TypeError, ValueError):
            continue
        ox, oy = polar_to_cart(od, oa)
        d_to_fruit = math.hypot(ox - fx, oy - fy)
        if d_to_fruit + tie_margin < best_distance:
            owner, best_distance = oid, d_to_fruit
        elif abs(d_to_fruit - best_distance) <= tie_margin and oid < owner:
            owner, best_distance = oid, d_to_fruit
    return owner


def local_competition_cost(
    *,
    self_id: int,
    fruit_distance: float,
    fruit_angle: float,
    visible_agents: Iterable[Mapping],
    competition_radius: float,
    base_penalty: float,
) -> float:
    """Soft competition cost retained alongside deterministic local ownership."""
    fx, fy = polar_to_cart(fruit_distance, fruit_angle)
    total = 0.0
    for other in visible_agents:
        try:
            oid = int(other.get("id"))
            od = float(other.get("distance"))
            oa = float(other.get("angle"))
        except (TypeError, ValueError):
            continue
        ox, oy = polar_to_cart(od, oa)
        d_to_fruit = math.hypot(ox - fx, oy - fy)
        if d_to_fruit >= competition_radius:
            continue
        proximity = 1.0 - d_to_fruit / max(competition_radius, 1e-9)
        ownership = 1.35 if oid < self_id else 0.70
        total += base_penalty * ownership * proximity
    return total


def local_density(visible_agents: Iterable[Mapping], radius: float) -> int:
    count = 0
    for other in visible_agents:
        try:
            if float(other.get("distance")) <= radius:
                count += 1
        except (TypeError, ValueError):
            pass
    return count


def estimate_carrying_capacity(
    *,
    base_cap: int,
    bonus_max: int,
    fruit_sightings_per_agent: float,
    predator_sightings_per_agent: float,
    average_energy_ratio: float,
    population_trend: float,
) -> int:
    """Cheap carrying-capacity estimate from signals actually observable by the API."""
    food_signal = min(1.0, fruit_sightings_per_agent / 3.0)
    energy_signal = min(1.0, max(0.0, (average_energy_ratio - 0.35) / 0.45))
    danger_signal = min(1.0, predator_sightings_per_agent / 1.5)
    collapse_signal = 1.0 if population_trend < -0.35 else 0.0
    bonus = bonus_max * (0.55 * food_signal + 0.25 * energy_signal + 0.20 * collapse_signal - 0.55 * danger_signal)
    return max(2, int(round(base_cap + max(-2.0, min(float(bonus_max), bonus)))))


def build_population_context(
    statuses: Sequence[Mapping],
    *,
    base_cap: int,
    bonus_max: int,
    population_trend: float,
) -> PopulationContext:
    n_agents = len(statuses)
    if not statuses:
        return PopulationContext(0, 0.0, 0.0, 0.0, population_trend, max(2, base_cap))

    energy_ratios: list[float] = []
    fruit_sightings = 0
    predator_sightings = 0
    for status in statuses:
        energy = float(status.get("energy", 0.0) or 0.0)
        max_energy = max(1.0, float(status.get("max_energy", 1.0) or 1.0))
        energy_ratios.append(energy / max_energy)
        for obs in status.get("observations") or []:
            if not isinstance(obs, Mapping):
                continue
            if obs.get("type") == "Fruit":
                fruit_sightings += 1
            elif obs.get("type") == "Predator":
                predator_sightings += 1

    avg_energy = sum(energy_ratios) / len(energy_ratios)
    fruit_rate = fruit_sightings / max(1, n_agents)
    predator_rate = predator_sightings / max(1, n_agents)
    capacity = estimate_carrying_capacity(
        base_cap=base_cap,
        bonus_max=bonus_max,
        fruit_sightings_per_agent=fruit_rate,
        predator_sightings_per_agent=predator_rate,
        average_energy_ratio=avg_energy,
        population_trend=population_trend,
    )
    return PopulationContext(
        n_agents=n_agents,
        average_energy_ratio=avg_energy,
        fruit_sightings_per_agent=fruit_rate,
        predator_sightings_per_agent=predator_rate,
        population_trend=population_trend,
        carrying_capacity=capacity,
    )
