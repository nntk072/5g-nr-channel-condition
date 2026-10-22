"""Numerical utility tests for the RT convergence runner."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = runpy.run_path(str(ROOT / "scripts" / "run_rt_solver_convergence.py"))
relative_l2_error = RUNNER["relative_l2_error"]
relative_l2_error_if_defined = RUNNER["relative_l2_error_if_defined"]
path_observables = RUNNER["path_observables"]
ConvergenceLink = RUNNER["ConvergenceLink"]


def test_relative_l2_error_uses_reference_norm():
    assert relative_l2_error(
        np.asarray([3 + 0j, 4 + 0j]), np.asarray([0 + 0j, 4 + 0j])
    ) == pytest.approx(0.75)


def test_relative_l2_error_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="aligned complex arrays"):
        relative_l2_error(np.asarray([1 + 0j]), np.asarray([1 + 0j, 2 + 0j]))


def test_relative_l2_error_rejects_zero_reference():
    with pytest.raises(ValueError, match="nonzero energy"):
        relative_l2_error(np.asarray([1 + 0j]), np.asarray([0 + 0j]))


def test_zero_reference_is_reported_as_undefined_for_a_convergence_matrix():
    assert relative_l2_error_if_defined(np.asarray([1 + 0j]), np.asarray([0 + 0j])) is None


def test_path_observables_use_power_weighted_delay_and_spherical_angle_spread():
    result = path_observables(
        gains=np.asarray([1 + 0j, 1 + 0j]),
        delays_s=np.asarray([1e-9, 3e-9]),
        theta_r_rad=np.asarray([np.pi / 2, np.pi / 2]),
        phi_r_rad=np.asarray([0.0, np.pi / 2]),
        valid=np.asarray([True, True]),
    )
    assert result["first_arrival_delay_ns"] == pytest.approx(1.0)
    assert result["power_weighted_rms_delay_spread_ns"] == pytest.approx(1.0)
    assert result["power_weighted_arrival_angular_spread_rad_3d"] == pytest.approx(np.pi / 4)


def test_path_observables_leave_empty_path_targets_undefined():
    result = path_observables(
        gains=np.asarray([], dtype=np.complex128),
        delays_s=np.asarray([]),
        theta_r_rad=np.asarray([]),
        phi_r_rad=np.asarray([]),
        valid=np.asarray([], dtype=bool),
    )
    assert all(value is None for value in result.values())


def test_convergence_link_requires_finite_xyz_positions():
    with pytest.raises(ValueError, match="three finite coordinates"):
        ConvergenceLink("bad", (0.0, 0.0, float("nan")), (1.0, 0.0, 0.0))


@pytest.mark.native_sionna
def test_convergence_matrix_labels_los_and_blocked_links_independently():
    pytest.importorskip("mitsuba")
    pytest.importorskip("sionna.rt")
    run_convergence = RUNNER["run_convergence"]
    result = run_convergence(
        scene_name="box_one_screen",
        tx_position=(-2.0, 0.0, 2.5),
        rx_position=(-1.0, 0.0, 2.5),
        max_depth=1,
        samples_per_src_values=(64,),
        seeds=(11,),
        n_subcarriers=8,
        mitsuba_variant="llvm_ad_mono_polarized",
        link_cases=(
            ConvergenceLink("same-side", (-2.0, 0.0, 2.5), (-1.0, 0.0, 2.5)),
            ConvergenceLink("through-screen", (-2.0, 0.0, 2.5), (2.0, 0.0, 2.5)),
        ),
    )
    by_id = {link["link_id"]: link for link in result["configuration"]["links"]}
    assert by_id["same-side"]["geometric_visibility_state"] == "visible"
    assert by_id["through-screen"]["geometric_visibility_state"] == "blocked"
    assert result["configuration"]["normalize_delays_for_cfr"] is False
    assert result["schema"].endswith(".v2")
    assert {row["link_id"] for row in result["runs"]} == set(by_id)
    assert all("power_weighted_rms_delay_spread_ns" in row for row in result["runs"])
