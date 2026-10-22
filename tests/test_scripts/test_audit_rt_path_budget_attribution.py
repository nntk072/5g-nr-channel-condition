"""Tests for bounded path-budget attribution accounting."""

from __future__ import annotations

import numpy as np
import pytest
from scripts.audit_rt_path_budget_attribution import analyze_pair


def _path(contribution: complex) -> dict[str, object]:
    return {
        "interaction_type_masks": [0],
        "object_ids": [0],
        "primitive_ids": [0],
        "interaction_vertices_m": [[0.0, 0.0, 0.0]],
        "absolute_delay_s": 100e-9,
        "coefficient_real": 1.0,
        "coefficient_imag": 0.0,
        "theta_r_rad": 0.5,
        "phi_r_rad": -0.3,
        "theta_t_rad": 0.5,
        "phi_t_rad": -0.3,
        "cfr_contribution_real": [contribution.real, contribution.real],
        "cfr_contribution_imag": [contribution.imag, contribution.imag],
    }


def _row(samples: int, paths: list[dict[str, object]]) -> dict[str, object]:
    channel = sum(
        (
            np.asarray(path["cfr_contribution_real"])
            + 1j * np.asarray(path["cfr_contribution_imag"])
            for path in paths
        ),
        start=np.zeros(2, dtype=np.complex128),
    )
    return {
        "link_id": "los",
        "seed": 17,
        "max_depth": 10,
        "samples_per_src": samples,
        "valid_path_count": len(paths),
        "candidate_path_count_before_path_validation": len(paths),
        "candidate_path_cap_filled": False,
        "path_diagnostics": paths,
        "path_cfr_reconstruction_within_tolerance": True,
        "cfr_vector_real": channel.real.tolist(),
        "cfr_vector_imag": channel.imag.tolist(),
    }


def test_attribution_identifies_duplicate_coherent_path_growth():
    low = _row(100_000, [_path(1.0 + 0.0j)])
    high = _row(1_000_000, [_path(1.0 + 0.0j), _path(1.0 + 0.0j)])

    result = analyze_pair(low, high)

    assert result["path_counts"]["matched"] == 1
    assert result["path_counts"]["discovered_at_high_budget"] == 1
    assert result["path_counts"]["interaction_order_high"] == {"0": 2}
    assert (
        result["path_counts"]["zero_interaction_path_records"]["high"][
            "unique_delays_at_1_ps_resolution"
        ]
        == 1
    )
    assert (
        result["path_counts"]["zero_interaction_path_records"]["high"][
            "unique_coefficients_at_1e-12_resolution"
        ]
        == 1
    )
    assert result["cfr_delta_decomposition_norm_relative_to_total"][
        "new_path_contributions"
    ] == pytest.approx(1.0)
    assert result["cfr_delta_decomposition_norm_relative_to_total"][
        "additive_closure_error"
    ] == pytest.approx(0.0)
    assert result["channel_change"]["coherent_to_incoherent_power_ratio"]["high"] == pytest.approx(
        2.0
    )


def test_attribution_uses_endpoint_direction_to_match_same_topology():
    low_paths = [_path(1.0 + 0.0j), _path(1.0 + 0.0j)]
    high_paths = [_path(1.0 + 0.0j), _path(1.0 + 0.0j), _path(1.0 + 0.0j)]
    low_paths[0]["theta_t_rad"] = 1.0
    high_paths[0]["theta_t_rad"] = 1.0

    result = analyze_pair(_row(100_000, low_paths), _row(1_000_000, high_paths))

    assert result["path_counts"]["matched"] == 2
    assert result["path_counts"]["discovered_at_high_budget"] == 1
    assert result["matched_path_variation"]["maximum_arrival_departure_direction_error_rad"][
        "max"
    ] == pytest.approx(0.0)


def test_attribution_requires_same_channel_case_and_verified_path_sum():
    low = _row(100_000, [_path(1.0 + 0.0j)])
    high = _row(1_000_000, [_path(1.0 + 0.0j)])
    high["seed"] = 18
    with pytest.raises(ValueError, match="same link, seed, and depth"):
        analyze_pair(low, high)

    high["seed"] = 17
    high["path_cfr_reconstruction_within_tolerance"] = False
    with pytest.raises(ValueError, match="reconstruction must pass"):
        analyze_pair(low, high)
