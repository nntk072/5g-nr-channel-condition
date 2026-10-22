"""Numerical utility tests for the RT convergence runner."""

from __future__ import annotations

import runpy
import subprocess
from pathlib import Path

import numpy as np
import pytest
from artifacts.provenance import require_file_matches_git_head
from integrations.sionna.candidate_audit import CandidatePathCounter

ROOT = Path(__file__).resolve().parents[2]
RUNNER = runpy.run_path(str(ROOT / "scripts" / "run_rt_solver_convergence.py"))
relative_l2_error = RUNNER["relative_l2_error"]
relative_l2_error_if_defined = RUNNER["relative_l2_error_if_defined"]
path_observables = RUNNER["path_observables"]
count_paths_with_interaction = RUNNER["count_paths_with_interaction"]
path_level_diagnostics = RUNNER["path_level_diagnostics"]
ConvergenceLink = RUNNER["ConvergenceLink"]


def test_candidate_path_counter_records_buffer_before_path_validation():
    pytest.importorskip("drjit")
    mi = pytest.importorskip("mitsuba")

    class CandidateBuffer:
        paths_counter = mi.UInt(17)

    sentinel = CandidateBuffer()
    recorder = CandidatePathCounter(lambda: sentinel)

    assert recorder() is sentinel
    assert recorder.last_candidate_path_count == 17
    assert recorder.last_candidate_path_counts_per_source == (17,)


def test_candidate_path_counter_recovers_each_transmitter_source_count():
    pytest.importorskip("drjit")
    mi = pytest.importorskip("mitsuba")

    class CandidateBuffer:
        paths_counter = mi.UInt(3)
        src_indices = mi.UInt([0, 1, 0])

    source_positions = mi.Point3f([0.0, 10.0, 20.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    sentinel = CandidateBuffer()
    recorder = CandidatePathCounter(lambda **_: sentinel)

    assert recorder(src_positions=source_positions) is sentinel
    assert recorder.last_total_candidate_path_count == 3
    assert recorder.last_candidate_path_counts_per_source == (2, 1, 0)
    assert recorder.last_candidate_path_count == 2


@pytest.mark.native_sionna
def test_native_candidate_path_counter_audits_each_saturated_source():
    rt = pytest.importorskip("sionna.rt")
    scene = rt.load_scene()
    scene.frequency = 3.5e9
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.add(rt.Transmitter(name="cap_audit_tx_a", position=[0.0, 0.0, 1.0]))
    scene.add(rt.Transmitter(name="cap_audit_tx_b", position=[0.0, 10.0, 1.0]))
    scene.add(rt.Receiver(name="cap_audit_rx", position=[10.0, 5.0, 1.0]))
    solver = rt.PathSolver(deterministic=True)
    recorder = CandidatePathCounter(solver._candidate_generator)
    solver._candidate_generator = recorder

    solver(
        scene,
        max_depth=0,
        max_num_paths_per_src=1,
        los=True,
        specular_reflection=False,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
        samples_per_src=1,
        seed=1,
    )

    assert recorder.last_total_candidate_path_count == 2
    assert recorder.last_candidate_path_counts_per_source == (1, 1)


def test_provenance_runner_requires_source_to_match_committed_git_head(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    source = tmp_path / "runner.py"
    source.write_text("print('committed')\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "runner.py"], check=True, capture_output=True
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Research Test",
            "-c",
            "user.email=research-test@example.invalid",
            "commit",
            "-m",
            "add runner",
        ],
        check=True,
        capture_output=True,
    )

    require_file_matches_git_head(source)
    source.write_text("print('modified')\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from Git HEAD"):
        require_file_matches_git_head(source)


def test_provenance_runner_rejects_untracked_source(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    source = tmp_path / "runner.py"
    source.write_text("print('untracked')\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="must be tracked"):
        require_file_matches_git_head(source)


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


def test_count_paths_with_interaction_uses_valid_paths_and_interaction_bits():
    interactions = np.asarray(
        [[[[0, 2, 1, 4]]], [[[0, 0, 2, 0]]]],
        dtype=np.uint32,
    )
    valid = np.asarray([True, True, False, True])
    assert count_paths_with_interaction(interactions, valid, 2) == 1


def test_path_level_diagnostics_reconstruct_cfr_and_keep_valid_zero_gain_paths():
    gains = np.asarray([1.0 + 0.25j, -0.2 + 0.4j, 0.0 + 0.0j])
    delays = np.asarray([10e-9, 25e-9, 30e-9])
    frequencies = np.asarray([-30e3, 0.0, 30e3])
    carrier_hz = 3.5e9
    interactions = np.asarray([[0, 1, 0], [0, 4, 0]], dtype=np.uint32)
    objects = np.asarray([[0, 12, 0], [0, 13, 0]], dtype=np.uint32)
    primitives = np.asarray([[0, 8, 0], [0, 9, 0]], dtype=np.uint32)
    vertices = np.zeros((2, 3, 3), dtype=np.float64)
    vertices[0, 1] = [1.25, 2.5, 3.75]
    vertices[1, 1] = [4.0, 5.0, 6.0]
    valid = np.asarray([True, True, True])
    expected = np.sum(
        gains[:, None]
        * np.exp(-2j * np.pi * delays[:, None] * (carrier_hz + frequencies[None, :])),
        axis=0,
    )

    rows, reconstruction_error = path_level_diagnostics(
        link_id="center-nlos",
        scene_tree_sha256="a" * 64,
        gains=gains,
        delays_s=delays,
        theta_r_rad=np.asarray([0.1, 0.2, 0.3]),
        phi_r_rad=np.asarray([0.4, 0.5, 0.6]),
        theta_t_rad=np.asarray([0.7, 0.8, 0.9]),
        phi_t_rad=np.asarray([1.0, 1.1, 1.2]),
        interactions=interactions,
        objects=objects,
        primitives=primitives,
        vertices_m=vertices,
        valid=valid,
        frequencies_hz=frequencies,
        carrier_hz=carrier_hz,
        reference_cfr=expected,
    )

    assert len(rows) == 3
    assert [row["path_index"] for row in rows] == [0, 1, 2]
    assert rows[2]["coefficient_power"] == 0.0
    assert rows[1]["interaction_type_masks"] == [1, 4]
    assert rows[1]["object_ids"] == [12, 13]
    assert len(rows[1]["path_fingerprint"]) == 64
    assert reconstruction_error == pytest.approx(0.0, abs=1e-12)


def test_path_level_diagnostics_uses_native_cfr_contributions_when_provided():
    native = np.asarray([[0.1 + 0.2j, -0.3 + 0.4j]], dtype=np.complex64)
    common = {
        "link_id": "native-cfr",
        "scene_tree_sha256": "c" * 64,
        "gains": np.asarray([0.5 + 0.25j]),
        "delays_s": np.asarray([370e-9]),
        "theta_r_rad": np.asarray([0.0]),
        "phi_r_rad": np.asarray([0.0]),
        "theta_t_rad": np.asarray([0.0]),
        "phi_t_rad": np.asarray([0.0]),
        "interactions": np.asarray([[1]], dtype=np.uint32),
        "objects": np.asarray([[2]], dtype=np.uint32),
        "primitives": np.asarray([[3]], dtype=np.uint32),
        "vertices_m": np.zeros((1, 1, 3)),
        "valid": np.asarray([True]),
        "frequencies_hz": np.asarray([-30e3, 0.0]),
        "carrier_hz": 3.5e9,
        "reference_cfr": native.sum(axis=0),
        "cfr_contributions": native,
    }

    rows, reconstruction_error = path_level_diagnostics(**common)

    assert reconstruction_error == pytest.approx(0.0, abs=1e-12)
    assert rows[0]["cfr_contribution_real"] == pytest.approx(native[0].real.tolist())
    assert rows[0]["cfr_contribution_imag"] == pytest.approx(native[0].imag.tolist())
    with pytest.raises(ValueError, match="align with valid paths"):
        path_level_diagnostics(**{**common, "cfr_contributions": native[:, :1]})


def test_path_fingerprint_does_not_depend_on_solver_path_slot():
    common = {
        "link_id": "same-link",
        "scene_tree_sha256": "b" * 64,
        "delays_s": np.asarray([20e-9, 10e-9]),
        "theta_r_rad": np.asarray([0.0, 0.0]),
        "phi_r_rad": np.asarray([0.0, 0.0]),
        "theta_t_rad": np.asarray([0.0, 0.0]),
        "phi_t_rad": np.asarray([0.0, 0.0]),
        "interactions": np.asarray([[2, 1]], dtype=np.uint32),
        "objects": np.asarray([[22, 11]], dtype=np.uint32),
        "primitives": np.asarray([[2, 1]], dtype=np.uint32),
        "vertices_m": np.asarray([[[2.0, 0.0, 1.0], [1.0, 0.0, 1.0]]]),
        "valid": np.asarray([True, True]),
        "frequencies_hz": np.asarray([0.0]),
        "carrier_hz": 3.5e9,
        "reference_cfr": np.asarray([1.0 + 0j]),
    }
    original, _ = path_level_diagnostics(gains=np.asarray([0.5 + 0j, 1.0 + 0j]), **common)
    reordered, _ = path_level_diagnostics(
        gains=np.asarray([1.0 + 0j, 0.5 + 0j]),
        **{
            **common,
            "delays_s": common["delays_s"][::-1],
            "interactions": common["interactions"][:, ::-1],
            "objects": common["objects"][:, ::-1],
            "primitives": common["primitives"][:, ::-1],
            "vertices_m": common["vertices_m"][:, ::-1, :],
            "theta_r_rad": common["theta_r_rad"][::-1],
            "phi_r_rad": common["phi_r_rad"][::-1],
            "theta_t_rad": common["theta_t_rad"][::-1],
            "phi_t_rad": common["phi_t_rad"][::-1],
        },
    )
    assert original[0]["path_fingerprint"] == reordered[1]["path_fingerprint"]
    assert original[1]["path_fingerprint"] == reordered[0]["path_fingerprint"]


def test_convergence_link_requires_finite_xyz_positions():
    with pytest.raises(ValueError, match="three finite coordinates"):
        ConvergenceLink("bad", (0.0, 0.0, float("nan")), (1.0, 0.0, 0.0))


def test_material_scattering_override_is_range_checked_before_runtime_setup():
    run_convergence = RUNNER["run_convergence"]
    with pytest.raises(ValueError, match=r"coefficients must be in \[0, 1\]"):
        run_convergence(
            scene_name="box_one_screen",
            tx_position=(-2.0, 0.0, 2.5),
            rx_position=(-1.0, 0.0, 2.5),
            max_depth=1,
            samples_per_src_values=(10,),
            seeds=(11,),
            material_scattering_overrides={"screen-mat": 1.1},
        )


def test_path_cap_must_be_a_positive_integer():
    run_convergence = RUNNER["run_convergence"]
    with pytest.raises(ValueError, match="max_num_paths_per_src must be a positive integer"):
        run_convergence(
            scene_name="box_one_screen",
            tx_position=(-2.0, 0.0, 2.5),
            rx_position=(-1.0, 0.0, 2.5),
            max_depth=1,
            samples_per_src_values=(10,),
            seeds=(11,),
            max_num_paths_per_src=0,
        )


def test_cfr_vector_output_flag_must_be_boolean():
    run_convergence = RUNNER["run_convergence"]
    with pytest.raises(TypeError, match="include_cfr_vectors must be a boolean"):
        run_convergence(
            scene_name="box_one_screen",
            tx_position=(-2.0, 0.0, 2.5),
            rx_position=(-1.0, 0.0, 2.5),
            max_depth=1,
            samples_per_src_values=(10,),
            seeds=(11,),
            include_cfr_vectors="yes",
        )


@pytest.mark.native_sionna
def test_convergence_matrix_labels_los_and_blocked_links_independently():
    pytest.importorskip("mitsuba")
    pytest.importorskip("sionna.rt")
    run_convergence = RUNNER["run_convergence"]
    result = run_convergence(
        scene_name="box_one_screen",
        tx_position=(-2.0, 0.0, 2.5),
        rx_position=(-1.0, 0.0, 2.5),
        max_depth=3,
        samples_per_src_values=(1000,),
        seeds=(11,),
        n_subcarriers=8,
        mitsuba_variant="llvm_ad_mono_polarized",
        max_num_paths_per_src=1000,
        link_cases=(
            ConvergenceLink("same-side", (-2.0, 0.0, 2.5), (-1.0, 0.0, 2.5)),
            ConvergenceLink("through-screen", (-2.0, 0.0, 2.5), (2.0, 0.0, 2.5)),
        ),
    )
    by_id = {link["link_id"]: link for link in result["configuration"]["links"]}
    assert by_id["same-side"]["geometric_visibility_state"] == "visible"
    assert by_id["through-screen"]["geometric_visibility_state"] == "blocked"
    assert result["configuration"]["normalize_delays_for_cfr"] is False
    assert result["schema"].endswith(".v3")
    assert result["configuration"]["max_num_paths_per_src"] == 1000
    assert result["configuration"]["materials"]["screen-mat"]["scattering_coefficient"] == [0.0]
    assert {row["link_id"] for row in result["runs"]} == set(by_id)
    assert all("power_weighted_rms_delay_spread_ns" in row for row in result["runs"])
    assert all(row["valid_diffuse_reflection_path_count"] == 0 for row in result["runs"])
    assert all("candidate_path_count_before_path_validation" in row for row in result["runs"])
    assert all("candidate_path_cap_filled" in row for row in result["runs"])
    assert result["configuration"]["candidate_path_count_stage"] == (
        "after_candidate_generation_before_invalid_path_filter"
    )

    scattered = run_convergence(
        scene_name="box_one_screen",
        tx_position=(-2.0, 0.0, 2.5),
        rx_position=(2.0, 0.0, 2.5),
        max_depth=3,
        samples_per_src_values=(1000,),
        seeds=(11,),
        n_subcarriers=8,
        mitsuba_variant="llvm_ad_mono_polarized",
        link_cases=(ConvergenceLink("through-screen", (-2.0, 0.0, 2.5), (2.0, 0.0, 2.5)),),
        material_scattering_overrides={"screen-mat": 0.5},
        max_num_paths_per_src=1000,
        include_cfr_vectors=True,
        include_path_diagnostics=True,
        path_diagnostics_max_depth_only=True,
    )
    assert scattered["configuration"]["materials"]["screen-mat"]["scattering_coefficient"] == [0.5]
    assert scattered["schema"].endswith(".v5")
    assert scattered["configuration"]["cfr_vectors_included"] is True
    assert scattered["configuration"]["path_diagnostics_included"] is True
    assert scattered["configuration"]["path_diagnostics_depths"] == [3]
    assert all(len(row["cfr_vector_real"]) == 8 for row in scattered["runs"])
    assert all(len(row["cfr_vector_imag"]) == 8 for row in scattered["runs"])
    diagnostic_runs = [row for row in scattered["runs"] if row["max_depth"] == 3]
    assert all(len(row["path_diagnostics"]) == row["valid_path_count"] for row in diagnostic_runs)
    assert all(row["path_cfr_reconstruction_within_tolerance"] for row in diagnostic_runs)
    assert all("path_diagnostics" not in row for row in scattered["runs"] if row["max_depth"] != 3)
    for row in diagnostic_runs:
        summed = sum(
            np.asarray(path["cfr_contribution_real"])
            + 1j * np.asarray(path["cfr_contribution_imag"])
            for path in row["path_diagnostics"]
        )
        recorded = np.asarray(row["cfr_vector_real"]) + 1j * np.asarray(row["cfr_vector_imag"])
        assert np.allclose(summed, recorded, rtol=1e-4, atol=1e-8)
    assert all(
        row["candidate_path_count_before_path_validation"] >= row["valid_path_count"]
        for row in scattered["runs"]
    )
    assert max(row["valid_diffuse_reflection_path_count"] for row in scattered["runs"]) > 0
