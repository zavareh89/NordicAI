import math

from survival_policy.geometry import (
    angle_from_vector,
    normalize,
    unit_from_angle,
    weighted_sum,
    wrap_angle,
)


def test_wrap_angle_boundaries():
    assert math.isclose(wrap_angle(3 * math.pi), -math.pi, abs_tol=1e-12)
    assert math.isclose(wrap_angle(-3 * math.pi), -math.pi, abs_tol=1e-12)
    assert math.isclose(wrap_angle(0.25), 0.25, abs_tol=1e-12)


def test_angle_vector_round_trip():
    for angle in (-2.7, -1.0, 0.0, 0.8, 2.9):
        assert math.isclose(wrap_angle(angle_from_vector(unit_from_angle(angle)) - angle), 0.0, abs_tol=1e-12)


def test_zero_vector_normalization_is_safe():
    assert normalize((0.0, 0.0)) == (0.0, 0.0)
    assert normalize((0.0, 0.0), default=(1.0, 0.0)) == (1.0, 0.0)


def test_weighted_potential_combination():
    v = weighted_sum([((1.0, 0.0), 2.0), ((0.0, 1.0), 3.0), ((-1.0, 0.0), 0.5)])
    assert v == (1.5, 3.0)
