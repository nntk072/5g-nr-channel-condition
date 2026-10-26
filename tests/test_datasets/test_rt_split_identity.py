from __future__ import annotations

from typing import Any

import pytest
from datasets.rt_split_identity import rt_spatial_split_subject_id


@pytest.mark.parametrize(
    ("x", "y", "expected_x", "expected_y"),
    [
        (-0.01, -1.0, -1, -1),
        (0.0, 0.0, 0, 0),
        (0.99, 0.99, 0, 0),
        (1.0, 1.0, 1, 1),
    ],
)
def test_exact_spatial_grid_floor_handles_negative_and_boundary_points(
    x: float, y: float, expected_x: int, expected_y: int
) -> None:
    subject_id, complete = rt_spatial_split_subject_id(
        scene_asset_set_sha256="a" * 64,
        scene_asset_manifest_complete=True,
        scene_name="ignored-for-closed-assets",
        rx_position_m=(x, y, 1.5),
        block_size_m=1.0,
    )

    assert subject_id == f"rt:{'a' * 64}:block:0x1.0000000000000p+0:{expected_x}:{expected_y}"
    assert complete is True


def test_complete_identity_is_independent_of_scene_name() -> None:
    first, _ = rt_spatial_split_subject_id(
        scene_asset_set_sha256="b" * 64,
        scene_asset_manifest_complete=True,
        scene_name="renamed-a",
        rx_position_m=(2.5, -0.25, 1.5),
        block_size_m=1.0,
    )
    second, _ = rt_spatial_split_subject_id(
        scene_asset_set_sha256="b" * 64,
        scene_asset_manifest_complete=True,
        scene_name="renamed-b",
        rx_position_m=(2.5, -0.25, 1.5),
        block_size_m=1.0,
    )

    assert first == second


def test_provisional_identity_is_normalized_and_explicitly_unclosed() -> None:
    subject_id, complete = rt_spatial_split_subject_id(
        scene_asset_set_sha256="c" * 64,
        scene_asset_manifest_complete=False,
        scene_name=" scene-a ",
        rx_position_m=(-0.01, 1.0, 1.5),
        block_size_m=1.0,
    )

    assert subject_id == "rt:unclosed-scene-a:block:0x1.0000000000000p+0:-1:1"
    assert complete is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"scene_asset_manifest_complete": "true"},
        {"scene_asset_manifest_complete": True, "scene_asset_set_sha256": "A" * 64},
        {"scene_asset_manifest_complete": True, "scene_asset_set_sha256": None},
        {"block_size_m": True},
        {"block_size_m": 0.0},
        {"rx_position_m": (0.0, float("nan"), 1.5)},
        {"rx_position_m": (0.0, 1.0)},
        {"scene_name": "   "},
    ],
)
def test_invalid_identity_inputs_fail_closed(overrides) -> None:
    arguments: dict[str, Any] = {
        "scene_asset_set_sha256": "d" * 64,
        "scene_asset_manifest_complete": False,
        "scene_name": "scene-a",
        "rx_position_m": (0.0, 0.0, 1.5),
        "block_size_m": 1.0,
    }
    arguments.update(overrides)

    with pytest.raises(ValueError):
        rt_spatial_split_subject_id(**arguments)
