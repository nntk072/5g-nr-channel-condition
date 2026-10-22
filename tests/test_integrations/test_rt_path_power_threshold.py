"""Tests for explicit path-power cutoff semantics."""

from __future__ import annotations

import numpy as np
import pytest
from integrations.sionna.path_power_threshold import (
    relative_path_power_db,
    relative_power_threshold_mask,
)


def test_relative_power_threshold_is_inclusive_and_ignores_invalid_padding():
    gains = np.asarray([1.0 + 0.0j, 0.1 + 0.0j, 0.001 + 0.0j, 1e6 + 0.0j])
    valid = np.asarray([True, True, True, False])

    levels = relative_path_power_db(gains, valid)
    np.testing.assert_allclose(levels[:3], [0.0, -20.0, -60.0], atol=1e-12)
    assert np.isneginf(levels[3])
    np.testing.assert_array_equal(
        relative_power_threshold_mask(gains, valid, -20.0),
        [True, True, False, False],
    )


@pytest.mark.parametrize(
    ("gains", "valid", "threshold", "message"),
    [
        (np.asarray([1.0 + 0j]), np.asarray([True]), 1.0, "nonpositive"),
        (np.asarray([1.0 + 0j]), np.asarray([True]), float("nan"), "nonpositive"),
        (np.asarray([0.0 + 0j]), np.asarray([True]), -20.0, "positive coefficient power"),
        (np.asarray([1.0 + 0j]), np.asarray([False]), -20.0, "positive coefficient power"),
        (np.asarray([1.0 + 0j]), np.asarray([1]), -20.0, "boolean mask"),
    ],
)
def test_relative_power_threshold_rejects_invalid_inputs(gains, valid, threshold, message):
    with pytest.raises(ValueError, match=message):
        relative_power_threshold_mask(gains, valid, threshold)


@pytest.mark.native_sionna
def test_city43_threshold_audit_preserves_full_paths_and_reports_cfr_sensitivity():
    pytest.importorskip("mitsuba")
    pytest.importorskip("sionna")
    from scripts.audit_rt_path_power_threshold import run_threshold_audit

    report = run_threshold_audit(
        scene_name="scenes/generated/research_scenes/city43/urban_00043.xml",
        tx_position_m=(0.0, -55.0, 18.0),
        rx_position_m=(24.35684045332106, -5.3057960716627575, 1.5),
        max_depth=3,
        samples_per_src=20_000,
        max_num_paths_per_src=1_000_000,
        seeds=(11,),
        thresholds_db=(-40.0, -80.0, -120.0),
        mitsuba_variant="llvm_ad_mono_polarized",
        material_scattering_overrides={
            "itu_concrete": 0.5,
            "itu_glass": 0.5,
            "itu_metal": 0.5,
        },
    )

    run = report["runs"][0]
    assert run["geometric_visibility_state"] == "blocked"
    assert run["candidate_path_cap_filled"] is False
    assert run["valid_path_count"] > 0
    assert len(run["valid_path_relative_power_db_sorted"]) == run["valid_path_count"]
    assert [row["threshold_db_relative_to_strongest_path"] for row in run["thresholds"]] == [
        -40.0,
        -80.0,
        -120.0,
    ]
    for row in run["thresholds"]:
        assert row["retained_path_count"] <= run["valid_path_count"]
        assert 0.0 <= row["retained_path_coefficient_power_fraction"] <= 1.0
        assert row["cfr_relative_l2_error"] is not None
        assert len(row["cfr_vector_real"]) == 256
        assert len(row["cfr_vector_imag"]) == 256
