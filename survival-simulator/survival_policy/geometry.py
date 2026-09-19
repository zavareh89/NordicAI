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


def cross(a: Vec2, b: Vec2) -> float:
    return a[0] * b[1] - a[1] * b[0]


def dot(a: Vec2, b: Vec2) -> float:
    return a[0] * b[0] + a[1] * b[1]


def segments_intersect(a0: Sequence[float], a1: Sequence[float], b0: Sequence[float], b1: Sequence[float], eps: float = 1e-9) -> bool:
    """Return True when two closed 2-D line segments intersect."""
    p = (float(a0[0]), float(a0[1]))
    r = (float(a1[0]) - p[0], float(a1[1]) - p[1])
    q = (float(b0[0]), float(b0[1]))
    s = (float(b1[0]) - q[0], float(b1[1]) - q[1])
    rxs = cross(r, s)
    qmp = (q[0] - p[0], q[1] - p[1])
    qmpxr = cross(qmp, r)
    if abs(rxs) <= eps and abs(qmpxr) <= eps:
        rr = dot(r, r)
        if rr <= eps:
            return segment_distance_to_point(p, b0, b1)[0] <= eps
        t0 = dot(qmp, r) / rr
        t1 = t0 + dot(s, r) / rr
        lo, hi = sorted((t0, t1))
        return hi >= -eps and lo <= 1.0 + eps
    if abs(rxs) <= eps:
        return False
    t = cross(qmp, s) / rxs
    u = cross(qmp, r) / rxs
    return -eps <= t <= 1.0 + eps and -eps <= u <= 1.0 + eps


def segment_segment_distance(
    a0: Sequence[float], a1: Sequence[float], b0: Sequence[float], b1: Sequence[float]
) -> float:
    """Minimum Euclidean distance between two closed line segments."""
    if segments_intersect(a0, a1, b0, b1):
        return 0.0
    return min(
        segment_distance_to_point(a0, b0, b1)[0],
        segment_distance_to_point(a1, b0, b1)[0],
        segment_distance_to_point(b0, a0, a1)[0],
        segment_distance_to_point(b1, a0, a1)[0],
    )


def closest_approach(relative_position: Vec2, relative_velocity: Vec2, horizon: float = 1.0) -> tuple[float, float]:
    """Return (minimum separation, time) over t in [0, horizon]."""
    vv = dot(relative_velocity, relative_velocity)
    if vv <= 1e-12:
        return norm(relative_position), 0.0
    t = -dot(relative_position, relative_velocity) / vv
    t = max(0.0, min(float(horizon), t))
    p = (
        relative_position[0] + relative_velocity[0] * t,
        relative_position[1] + relative_velocity[1] * t,
    )
    return norm(p), t
