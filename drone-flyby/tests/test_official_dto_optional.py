from __future__ import annotations

import importlib.util

import pytest


def test_official_dto_module_is_importable_when_running_inside_challenge_repo() -> None:
    if importlib.util.find_spec("dtos") is None:
        pytest.skip("Official challenge dtos.py is not present in this standalone artifact checkout")
    from dtos import OBJECT_CLASSES, DroneFlybyPredictResponseDto

    assert len(OBJECT_CLASSES) == 16
    response = DroneFlybyPredictResponseDto(
        request_id="test",
        frame=0,
        annotations=[],
        requested_view=None,
    )
    assert response.requested_view is None
