"""Tests for matching-threshold sensitivity with zero-order paths as a class."""

from __future__ import annotations

import numpy as np
from scripts.audit_rt_path_match_sensitivity import analyze_report


def _path(
    interaction: int,
    vertex_x: float,
    contribution: complex,
) -> dict[str, object]:
    return {
        "interaction_type_masks": [interaction],
        "object_ids": [2 if interaction else 0],
        "primitive_ids": [3 if interaction else 0],
        "interaction_vertices_m": [[vertex_x, 0.0, 0.0]],
        "absolute_delay_s": 100e-9,
        "coefficient_real": contribution.real,
        "coefficient_imag": contribution.imag,
        "theta_r_rad": 0.5,
        "phi_r_rad": -0.3,
        "theta_t_rad": 0.5,
        "phi_t_rad": -0.3,
        "path_fingerprint": "same-empty-sequence" if not interaction else "surface-path",
        "cfr_contribution_real": [contribution.real, contribution.real],
        "cfr_contribution_imag": [contribution.imag, contribution.imag],
    }


def _row(samples: int, paths: list[dict[str, object]]) -> dict[str, object]:
    cfr = sum(
        (
            np.asarray(path["cfr_contribution_real"])
            + 1j * np.asarray(path["cfr_contribution_imag"])
            for path in paths
        ),
        start=np.zeros(2, dtype=np.complex128),
    )
    return {
        "link_id": "center-los",
        "seed": 23,
        "max_depth": 10,
        "samples_per_src": samples,
        "valid_path_count": len(paths),
        "candidate_path_count_before_path_validation": len(paths),
        "candidate_path_cap_filled": False,
        "path_diagnostics": paths,
        "path_cfr_reconstruction_within_tolerance": True,
        "cfr_vector_real": cfr.real.tolist(),
        "cfr_vector_imag": cfr.imag.tolist(),
    }


def test_threshold_sensitivity_separates_zero_order_class_from_path_ids():
    low = _row(
        100_000,
        [_path(0, 0.0, 1.0 + 0.0j), _path(1, 0.0, 0.2 + 0.0j)],
    )
    high = _row(
        1_000_000,
        [
            _path(0, 0.0, 1.0 + 0.0j),
            _path(0, 0.0, 1.0 + 0.0j),
            _path(0, 0.0, 1.0 + 0.0j),
            _path(1, 0.03, 0.2 + 0.0j),
        ],
    )
    report = {"runs": [low, high]}

    result = analyze_report(
        report,
        link_id="center-los",
        seed=23,
        max_depth=10,
        low_samples=100_000,
        high_samples=1_000_000,
        thresholds_m=(0.02, 0.05),
    )

    zero = result["zero_interaction_class"]
    assert zero["low"]["record_count"] == 1
    assert zero["high"]["record_count"] == 3
    assert zero["aggregate_delta_norm_relative_to_total_cfr_delta"] > 0.99
    sensitivity = result["nonzero_interaction_threshold_sensitivity"]
    assert sensitivity[0]["nonzero_interaction_matching"]["matched_count"] == 0
    assert sensitivity[0]["nonzero_interaction_matching"]["high_only_count"] == 1
    assert sensitivity[1]["nonzero_interaction_matching"]["matched_count"] == 1
    assert sensitivity[1]["nonzero_interaction_matching"]["high_only_count"] == 0
