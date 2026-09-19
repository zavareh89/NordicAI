from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from drone_detector.challenge.api_adapter import build_challenge_response, detection_to_global_bbox
from drone_detector.inference.postprocess import view_pixels_to_global_normalized
from drone_detector.types import Detection


def _request() -> SimpleNamespace:
    return SimpleNamespace(
        request_id="req-1",
        frame=12,
        original_width=3840,
        original_height=2160,
        view=SimpleNamespace(
            width=960,
            height=540,
            resolution_level=0,
            source_region_xyxy=[0, 0, 3840, 2160],
        ),
    )


def test_level0_pixel_box_to_global_normalized_coordinates() -> None:
    box = view_pixels_to_global_normalized(
        (96, 54, 192, 108),
        view_width=960,
        view_height=540,
        source_region_xyxy=(0, 0, 3840, 2160),
        original_width=3840,
        original_height=2160,
    )
    assert box == pytest.approx((0.1, 0.1, 0.2, 0.2))


def test_coordinate_layer_also_supports_future_cropped_views() -> None:
    box = view_pixels_to_global_normalized(
        (0, 0, 960, 540),
        view_width=960,
        view_height=540,
        source_region_xyxy=(960, 540, 2880, 1620),
        original_width=3840,
        original_height=2160,
    )
    assert box == pytest.approx((0.25, 0.25, 0.75, 0.75))


def test_api_response_uses_global_box_and_never_requests_camera_move(monkeypatch) -> None:
    @dataclass
    class PredictionDto:
        object_id: str
        bbox: list[float]
        confidence: float

    @dataclass
    class ResponseDto:
        request_id: str
        frame: int
        annotations: list[PredictionDto]
        requested_view: object = None

    fake_dtos = types.ModuleType("dtos")
    fake_dtos.DroneFlybyPredictionDto = PredictionDto
    fake_dtos.DroneFlybyPredictResponseDto = ResponseDto
    monkeypatch.setitem(sys.modules, "dtos", fake_dtos)
    # Force api_adapter's mathematically equivalent fallback instead of importing any unrelated utils module.
    monkeypatch.setitem(sys.modules, "utils", types.ModuleType("utils"))

    request = _request()
    detection = Detection(0, "hangar", 0.42, (96.0, 54.0, 192.0, 108.0))
    response = build_challenge_response(request, [detection])
    assert response.request_id == "req-1"
    assert response.frame == 12
    assert response.requested_view is None
    assert response.annotations[0].object_id == "hangar"
    assert response.annotations[0].bbox == pytest.approx([0.1, 0.1, 0.2, 0.2])
    assert response.annotations[0].confidence == pytest.approx(0.42)
