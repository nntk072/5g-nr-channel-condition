"""Native finite-segment visibility checks against the pinned Mitsuba scene."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mitsuba")
pytest.importorskip("sionna.rt")


@pytest.fixture(scope="module")
def screen_scene():
    import mitsuba as mi
    import sionna.rt as rt

    mi.set_variant("llvm_ad_mono_polarized")
    return rt.load_scene(rt.scene.box_one_screen, merge_shapes=False)


@pytest.mark.native_sionna
def test_independent_segment_oracle_matches_clear_and_screen_blocked_geometry(screen_scene):
    from integrations.sionna.los_oracle import SegmentVisibilityState, segment_visibility

    visible = segment_visibility(screen_scene, [-2.0, -1.0, 2.5], [-2.0, 1.0, 2.5])
    blocked = segment_visibility(screen_scene, [-2.0, 0.0, 2.5], [2.0, 0.0, 2.5])

    assert visible.state is SegmentVisibilityState.VISIBLE
    assert visible.visible is True
    assert visible.first_intersection_distance_m is None
    assert blocked.state is SegmentVisibilityState.BLOCKED
    assert blocked.visible is False
    assert blocked.first_intersection_distance_m == pytest.approx(2.0, abs=1e-5)
    assert blocked.incidence_cosine == pytest.approx(1.0, abs=1e-6)


@pytest.mark.native_sionna
def test_numpy_mesh_inventory_oracle_matches_screen_scene_geometry(screen_scene):
    from integrations.sionna.los_oracle import (
        independent_mesh_segment_visibility,
        segment_visibility,
    )
    from propagation.scene_visibility import extract_scene_triangle_inventory

    inventory = extract_scene_triangle_inventory(screen_scene)
    visible = independent_mesh_segment_visibility(
        screen_scene, [-2.0, -1.0, 2.5], [-2.0, 1.0, 2.5], inventory=inventory
    )
    blocked = independent_mesh_segment_visibility(
        screen_scene, [-2.0, 0.0, 2.5], [2.0, 0.0, 2.5], inventory=inventory
    )
    mitsuba_visible = segment_visibility(screen_scene, [-2.0, -1.0, 2.5], [-2.0, 1.0, 2.5])
    mitsuba_blocked = segment_visibility(screen_scene, [-2.0, 0.0, 2.5], [2.0, 0.0, 2.5])

    assert visible.geometry.outcome == "visible"
    assert blocked.geometry.outcome == "blocked"
    assert mitsuba_visible.visible is True
    assert mitsuba_blocked.visible is False
    assert blocked.scene_geometry_sha256 == visible.scene_geometry_sha256
    assert visible.scene_geometry_sha256 == inventory.scene_geometry_sha256
    assert blocked.triangle_count > 0
    assert blocked.object_names


@pytest.mark.native_sionna
def test_segment_touching_screen_boundary_is_ambiguous(screen_scene):
    from integrations.sionna.los_oracle import SegmentVisibilityState, segment_visibility

    result = segment_visibility(screen_scene, [-2.0, 0.0, 5.01], [2.0, 0.0, 5.01])

    assert result.state is SegmentVisibilityState.AMBIGUOUS
    assert result.visible is None
    assert len(set(result.boundary_probe_blocked)) > 1


def test_segment_visibility_rejects_invalid_positions_and_marks_coincident_points():
    from integrations.sionna.los_oracle import SegmentVisibilityState, segment_visibility

    with pytest.raises(ValueError, match="three finite"):
        segment_visibility(object(), [0.0, np.nan, 1.0], [1.0, 0.0, 1.0])

    result = segment_visibility(object(), [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert result.state is SegmentVisibilityState.COINCIDENT_ENDPOINTS
    assert result.visible is None
