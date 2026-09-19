from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

Vec2 = Tuple[float, float]


def wrap_angle(angle: float) -> float:
    """Wrap radians to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def unit_from_angle(angle: float) -> Vec2:
    return math.cos(angle), math.sin(angle)


def angle_from_vector(v: Vec2, default: float = 0.0) -> float:
    if abs(v[0]) < 1e-12 and abs(v[1]) < 1e-12:
        return default
    return math.atan2(v[1], v[0])


def norm(v: Vec2) -> float:
    return math.hypot(v[0], v[1])


def normalize(v: Vec2, default: Vec2 = (0.0, 0.0)) -> Vec2:
    n = norm(v)
    if n <= 1e-12:
        return default
    return v[0] / n, v[1] / n


def scale(v: Vec2, s: float) -> Vec2:
    return v[0] * s, v[1] * s


def add(*vectors: Vec2) -> Vec2:
    x = 0.0
    y = 0.0
    for vx, vy in vectors:
        x += vx
        y += vy
    return x, y


def subtract(a: Vec2, b: Vec2) -> Vec2:
    return a[0] - b[0], a[1] - b[1]


def weighted_sum(components: Iterable[tuple[Vec2, float]]) -> Vec2:
    x = 0.0
    y = 0.0
    for (vx, vy), w in components:
        x += vx * w
        y += vy * w
    return x, y


def polar_to_cart(distance: float, angle: float) -> Vec2:
    ux, uy = unit_from_angle(angle)
    return distance * ux, distance * uy


def closest_point_on_segment(point: Sequence[float], a: Sequence[float], b: Sequence[float]) -> Vec2:
    px, py = float(point[0]), float(point[1])
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return ax, ay
    t = ((px - ax) * dx + (py - ay) * dy) / denom
    t = max(0.0, min(1.0, t))
    return ax + t * dx, ay + t * dy


def segment_distance_to_point(point: Sequence[float], a: Sequence[float], b: Sequence[float]) -> tuple[float, Vec2]:
    closest = closest_point_on_segment(point, a, b)
    return math.hypot(float(point[0]) - closest[0], float(point[1]) - closest[1]), closest


def closest_point_on_segment_to_origin(a: Sequence[float], b: Sequence[float]) -> Vec2:
    return closest_point_on_segment((0.0, 0.0), a, b)


def segment_distance_to_origin(a: Sequence[float], b: Sequence[float]) -> tuple[float, Vec2]:
    return segment_distance_to_point((0.0, 0.0), a, b)


def stable_int_seed(master_seed: int, *values: int) -> int:
    """Small deterministic integer mixer; independent of Python hash randomization."""
    x = master_seed & 0xFFFFFFFFFFFFFFFF
    for value in values:
        y = int(value) & 0xFFFFFFFFFFFFFFFF
        x ^= y + 0x9E3779B97F4A7C15 + ((x << 6) & 0xFFFFFFFFFFFFFFFF) + (x >> 2)
        x &= 0xFFFFFFFFFFFFFFFF
    return x
