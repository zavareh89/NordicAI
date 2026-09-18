from __future__ import annotations

import math
from typing import Iterable, Mapping

from .geometry import polar_to_cart


def local_competition_cost(
    *,
    self_id: int,
    fruit_distance: float,
    fruit_angle: float,
    visible_agents: Iterable[Mapping],
    competition_radius: float,
    base_penalty: float,
) -> float:
    """Approximate fruit reservation without inventing fruit IDs.

    Fruit observations contain no IDs. We therefore penalize a candidate if a
    visible herbivore is spatially near that fruit in the observer's local
    frame. Lower agent ID wins deterministic ties, creating consistent local
    ownership when two agents see the same contested fruit.
    """
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
