from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from integrations.sionna.srs import NRSRSMultiportResourceGrid
from integrations.sionna.srs_measurements import (
    measure_ul_srs_rsrp,
    measure_ul_srs_rsrp_from_grid,
    measure_ul_srs_rsrp_from_noisy_network_grid,
    report_gnb_rx_tx_time_difference,
    report_ul_aoa,
    report_ul_rtoa,
    report_ul_srs_rsrpp,
)


def test_ul_srs_rsrp_averages_only_configured_res_and_reports_receiver_diversity():
    power = np.array(
        [
            [[1e-3, 3e-3, 9.0], [2e-3, 6e-3, 8.0]],
            [[5e-3, 7.0, 7e-3], [10e-3, 6.0, 14e-3]],
        ]
    )
    mask = np.array([[True, True, False], [True, False, True]])

    report = measure_ul_srs_rsrp(
        power,
        mask,
        occasion_ids=["occasion-1", "occasion-2"],
        receiver_branch_ids=["branch-1", "branch-2"],
        receiver_category="terrestrial_type_1_o",
        measurement_frequency_bandwidth_hz=20e6,
        power_calibration_sha256="a" * 64,
    )

    assert report["standard"] == "3GPP_TS_38.215_V19.2.0_clause_5.2.5"
    assert report["measurement_reference_point"] == (
        "combined_antenna_elements_per_receiver_branch"
    )
    assert report["configured_re_count_per_occasion"] == [2, 2]
    assert report["receiver_branch_measurements"][0]["ul_srs_rsrp_w"] == pytest.approx(4e-3)
    assert report["receiver_branch_measurements"][1]["ul_srs_rsrp_w"] == pytest.approx(8e-3)
    assert report["diversity_ul_srs_rsrp_w"] == pytest.approx(8e-3)
    assert report["diversity_ul_srs_rsrp_dbm"] == pytest.approx(10 * np.log10(8.0))
    assert report["receiver_diversity_lower_bound_satisfied"] is True


def test_ul_srs_rsrp_zero_power_is_explicit_and_schema_valid():
    report = measure_ul_srs_rsrp(
        np.zeros((1, 1, 2), dtype=np.float64),
        np.array([[True, False]]),
        occasion_ids=["occasion-1"],
        receiver_branch_ids=["branch-1"],
        receiver_category="terrestrial_type_1_c",
        measurement_frequency_bandwidth_hz=5e6,
        power_calibration_sha256="b" * 64,
    )
    assert report["status"] == "ZERO_POWER_DIAGNOSTIC"
    assert report["diversity_ul_srs_rsrp_dbm"] is None


def test_ul_srs_rsrpp_reports_each_path_and_earliest_valid_path_per_occasion():
    response = np.array(
        [
            [[2 + 0j, 0j, 1 + 1j], [1 + 0j, 0j, 2 + 0j]],
            [[0j, 0j, 0j], [0j, 0j, 0j]],
        ],
        dtype=np.complex128,
    )
    delays = np.array([[1e-7, 0.0, 3e-7], [0.0, 0.0, 0.0]])
    valid = np.array([[True, False, True], [False, False, False]])

    report = report_ul_srs_rsrpp(
        response,
        delays,
        valid,
        occasion_ids=["occasion-1", "occasion-2"],
        receiver_branch_ids=["branch-1", "branch-2"],
        receiver_category="terrestrial_type_1_o",
        measurement_frequency_bandwidth_hz=20e6,
        watts_per_channel_response_unit_squared=1e-3,
        power_calibration_sha256="a" * 64,
        channel_estimator_sha256="b" * 64,
        path_detector_sha256="c" * 64,
    )

    assert report["status"] == "OBSERVED_NOT_QUALIFIED"
    first, no_path = report["occasion_measurements"]
    assert first["first_detected_path_index"] == 0
    assert first["detected_path_count"] == 2
    assert first["receiver_branches"][0]["path_measurements"] == [
        {
            "path_index": 0,
            "path_delay_s": 1e-7,
            "first_detected_path": True,
            "ul_srs_rsrpp_w": pytest.approx(4e-3),
            "ul_srs_rsrpp_dbm": pytest.approx(10 * np.log10(4.0)),
        },
        {
            "path_index": 2,
            "path_delay_s": 3e-7,
            "first_detected_path": False,
            "ul_srs_rsrpp_w": pytest.approx(2e-3),
            "ul_srs_rsrpp_dbm": pytest.approx(10 * np.log10(2.0)),
        },
    ]
    assert no_path["status"] == "NO_DETECTED_PATH_DIAGNOSTIC"
    assert no_path["first_detected_path_index"] is None
    assert report["occasion_measurements"][0]["receiver_branches"][1]["path_measurements"][0][
        "ul_srs_rsrpp_w"
    ] == pytest.approx(1e-3)


@pytest.mark.parametrize(
    ("response", "delays", "valid", "message"),
    (
        (np.ones((1, 1, 2)), np.zeros((1, 2)), np.ones((1, 2), dtype=bool), "complex"),
        (
            np.ones((1, 1, 2), dtype=complex),
            np.array([[1.0, 0.0]]),
            np.ones((1, 2), dtype=bool),
            "strictly increasing",
        ),
        (
            np.ones((1, 1, 2), dtype=complex),
            np.zeros((1, 2)),
            np.zeros((1, 2), dtype=bool),
            "exactly zero",
        ),
    ),
)
def test_ul_srs_rsrpp_rejects_malformed_path_inputs(response, delays, valid, message):
    with pytest.raises(ValueError, match=message):
        report_ul_srs_rsrpp(
            response,
            delays,
            valid,
            occasion_ids=["occasion-1"],
            receiver_branch_ids=["branch-1"],
            receiver_category="terrestrial_type_1_o",
            measurement_frequency_bandwidth_hz=20e6,
            watts_per_channel_response_unit_squared=1e-3,
            power_calibration_sha256="a" * 64,
            channel_estimator_sha256="b" * 64,
            path_detector_sha256="c" * 64,
        )


def test_ul_srs_rsrpp_enforces_resource_cap_and_digest_contract():
    args = dict(
        occasion_ids=["occasion-1"],
        receiver_branch_ids=["branch-1"],
        receiver_category="terrestrial_type_1_o",
        measurement_frequency_bandwidth_hz=20e6,
        watts_per_channel_response_unit_squared=1e-3,
        power_calibration_sha256="a" * 64,
        channel_estimator_sha256="b" * 64,
        path_detector_sha256="c" * 64,
    )
    response = np.ones((1, 1, 1), dtype=complex)
    delays = np.zeros((1, 1))
    valid = np.ones((1, 1), dtype=bool)
    with pytest.raises(ValueError, match="maximum_measurement_values"):
        report_ul_srs_rsrpp(
            response,
            delays,
            valid,
            maximum_measurement_values=2,
            **args,
        )
    with pytest.raises(ValueError, match="channel_estimator_sha256"):
        report_ul_srs_rsrpp(
            response,
            delays,
            valid,
            channel_estimator_sha256="not-a-digest",
            **{key: value for key, value in args.items() if key != "channel_estimator_sha256"},
        )


def test_ul_rtoa_uses_sfn_subframe_reference_and_declared_receiver_point():
    report = report_ul_rtoa(
        received_subframe_start_time_s=102.346125,
        sfn=23,
        subframe_number=4,
        sfn_zero_reference_time_s=102.0,
        receiver_category="terrestrial_type_1_o",
        timing_calibration_sha256="c" * 64,
    )

    assert report["srs_subframe_offset_from_sfn_zero_s"] == pytest.approx(0.234)
    assert report["ul_rtoa_reference_time_s"] == pytest.approx(102.234)
    assert report["ul_rtoa_s"] == pytest.approx(0.112125)
    assert report["receiver_reference_point"] == "rx_antenna_radiating_region_center"


def test_gnb_rx_tx_difference_uses_rx_minus_tx_and_separate_reference_points():
    report = report_gnb_rx_tx_time_difference(
        received_uplink_subframe_time_s=12.000004,
        transmitted_downlink_subframe_time_s=12.0,
        receiver_category="terrestrial_type_1_h",
        transmitter_category="terrestrial_type_1_c",
        timing_calibration_sha256="d" * 64,
    )

    assert report["gnb_rx_tx_time_difference_s"] == pytest.approx(4e-6)
    assert report["receiver_reference_point"] == "rx_transceiver_array_boundary_connector"
    assert report["transmitter_reference_point"] == "tx_antenna_connector"


@pytest.mark.parametrize(
    ("coordinate_system", "azimuth_reference", "vertical_reference"),
    (("GCS", "geographical_north", "zenith"), ("LCS", "local_x_axis", "local_z_axis")),
)
def test_ul_aoa_report_binds_estimate_and_coordinate_convention(
    coordinate_system, azimuth_reference, vertical_reference
):
    report = report_ul_aoa(
        sample_id="episode-01",
        azimuth_deg=359.5,
        vertical_deg=72.0,
        coordinate_system=coordinate_system,
        reference_frame_id="array-frame-fixture",
        estimated_channel_sha256="a" * 64,
        estimator_config_sha256="b" * 64,
        antenna_calibration_sha256="c" * 64,
    )

    assert report["status"] == "OBSERVED_NOT_QUALIFIED"
    assert report["azimuth_reference"] == azimuth_reference
    assert report["vertical_reference"] == vertical_reference
    assert report["measurement_location"] == "gnb_antenna"
    assert report["estimated_channel_sha256"] == "a" * 64
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "evaluation/schema/nr-ul-srs-aoa-report.v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    import jsonschema

    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(report)


@pytest.mark.parametrize(
    "overrides",
    (
        {"azimuth_deg": 360.0},
        {"vertical_deg": 180.1},
        {"coordinate_system": "ENU"},
        {"antenna_calibration_sha256": "not-a-digest"},
        {"azimuth_deg": True},
    ),
)
def test_ul_aoa_report_rejects_invalid_angle_or_identity(overrides):
    arguments = {
        "sample_id": "episode-01",
        "azimuth_deg": 10.0,
        "vertical_deg": 72.0,
        "coordinate_system": "GCS",
        "reference_frame_id": "geodetic-frame-fixture",
        "estimated_channel_sha256": "a" * 64,
        "estimator_config_sha256": "b" * 64,
        "antenna_calibration_sha256": "c" * 64,
    }
    arguments.update(overrides)
    with pytest.raises((TypeError, ValueError)):
        report_ul_aoa(**arguments)


def test_grid_observation_measures_only_configured_srs_res_and_binds_inputs():
    mapped_grid = np.zeros((1, 2, 3), dtype=np.complex128)
    mapped_grid[0, 1, [0, 2]] = [1 + 0j, -1 + 0j]
    mapped_mask = np.zeros(mapped_grid.shape, dtype=bool)
    mapped_mask[0, 1, [0, 2]] = True
    mapped = NRSRSMultiportResourceGrid(
        grid=mapped_grid,
        mask=mapped_mask,
        symbol_indices=(1,),
        subcarrier_indices=(0, 2),
    )
    received = np.full((2, 2, 2, 3), 1000 + 0j, dtype=np.complex128)
    received[:, :, 1, [0, 2]] = np.array(
        [
            [[2, 4], [6, 8]],
            [[2, 2], [4, 4]],
        ],
        dtype=np.complex128,
    )

    report = measure_ul_srs_rsrp_from_grid(
        received,
        mapped,
        occasion_ids=["occasion-1", "occasion-2"],
        receiver_branch_ids=["branch-1", "branch-2"],
        receiver_category="terrestrial_type_1_c",
        measurement_frequency_bandwidth_hz=20e6,
        power_scale_w_per_grid_unit_squared=0.5,
        power_calibration_sha256="e" * 64,
    )

    measurement = report["measurement_report"]
    assert report["status"] == "OBSERVED_NOT_QUALIFIED"
    assert len(report["received_grid_sha256"]) == 64
    assert len(report["mapped_resource_sha256"]) == 64
    assert measurement["configured_re_count_per_occasion"] == [2, 2]
    assert measurement["receiver_branch_measurements"][0]["ul_srs_rsrp_w"] == pytest.approx(3.5)
    assert measurement["receiver_branch_measurements"][1]["ul_srs_rsrp_w"] == pytest.approx(16.5)
    assert report["received_re_power_includes_noise_and_interference"] is True


def test_grid_observation_rejects_invalid_scale_shape_and_resource_bound():
    mapped = NRSRSMultiportResourceGrid(
        grid=np.array([[[1 + 0j, 1 + 0j]]]),
        mask=np.array([[[True, True]]]),
        symbol_indices=(0,),
        subcarrier_indices=(0, 1),
    )
    common = {
        "occasion_ids": ["occasion-1"],
        "receiver_branch_ids": ["branch-1"],
        "receiver_category": "terrestrial_type_1_c",
        "measurement_frequency_bandwidth_hz": 1e6,
        "power_calibration_sha256": "f" * 64,
    }
    with pytest.raises(ValueError, match="power scale"):
        measure_ul_srs_rsrp_from_grid(
            np.ones((1, 1, 1, 2), dtype=np.complex128),
            mapped,
            power_scale_w_per_grid_unit_squared=True,
            **common,
        )
    with pytest.raises(ValueError, match="dimensions"):
        measure_ul_srs_rsrp_from_grid(
            np.ones((1, 1, 2, 3), dtype=np.complex128),
            mapped,
            power_scale_w_per_grid_unit_squared=1.0,
            **common,
        )
    with pytest.raises(ValueError, match="maximum_grid_cells"):
        measure_ul_srs_rsrp_from_grid(
            np.ones((1, 1, 1, 2), dtype=np.complex128),
            mapped,
            power_scale_w_per_grid_unit_squared=1.0,
            maximum_grid_cells=1,
            **common,
        )


def test_noisy_network_grid_observation_binds_interference_and_receiver_noise():
    torch = pytest.importorskip("torch")
    from integrations.sionna.network_phy import (
        add_independent_receiver_awgn,
        compose_downlink_resource_grids,
    )

    channel = torch.zeros((1, 1, 2, 2, 1, 2, 3), dtype=torch.complex64)
    channel[:, :, 0, 0] = 1.0
    channel[:, :, 0, 1] = 0.5
    channel[:, :, 1, 0] = 0.75
    channel[:, :, 1, 1] = 0.25
    transmit_grids = torch.ones((1, 2, 1, 2, 3), dtype=torch.complex64)
    composition = compose_downlink_resource_grids(channel, transmit_grids, (1.0, 1.0))
    noisy = add_independent_receiver_awgn(composition, (0.01,), seed=29)

    mapped_grid = np.zeros((1, 2, 3), dtype=np.complex128)
    mapped_mask = np.zeros(mapped_grid.shape, dtype=bool)
    mapped_mask[0, 1, [0, 2]] = True
    mapped = NRSRSMultiportResourceGrid(
        grid=mapped_grid,
        mask=mapped_mask,
        symbol_indices=(1,),
        subcarrier_indices=(0, 2),
    )
    report = measure_ul_srs_rsrp_from_noisy_network_grid(
        noisy,
        composition,
        mapped,
        receiver_index=0,
        desired_transmitter_id="ue-1",
        transmitter_ids=["ue-1", "ue-2"],
        occasion_ids=["slot-1"],
        receiver_branch_ids=["rx-branch-1", "rx-branch-2"],
        receiver_category="terrestrial_type_1_c",
        measurement_frequency_bandwidth_hz=20e6,
        power_scale_w_per_grid_unit_squared=0.5,
        power_calibration_sha256="d" * 64,
    )

    assert report["status"] == "OBSERVED_NOT_QUALIFIED"
    assert report["transmitter_ids"] == ["ue-1", "ue-2"]
    assert report["noise_seed"] == 29
    assert report["noise_variance_w_per_re"] == [0.01]
    assert len(report["noise_grid_sha256"]) == 64
    assert len(report["transmitter_contributions_sha256"]) == 64
    assert report["measurement_report"]["configured_re_count_per_occasion"] == [2]
    assert report["measurement_report"]["receiver_branch_measurements"][0]["ul_srs_rsrp_w"] > 0
    components = report["network_components"]
    assert components["transmitter_power_w_by_id"]["ue-1"] > 0
    assert components["transmitter_power_w_by_id"]["ue-2"] > 0
    branch = components["receiver_branches"][0]
    assert branch["desired_transmitter_power_w"] == pytest.approx(0.5)
    assert branch["coherent_interference_power_w"] == pytest.approx(0.125)
    assert branch["measured_receiver_noise_power_w"] > 0
    assert branch["network_sinr_linear"] > 0
    assert components["receiver_branches"][1]["receiver_branch_id"] == "rx-branch-2"


def test_noisy_network_grid_observation_rejects_composition_noise_mismatch():
    torch = pytest.importorskip("torch")
    from dataclasses import replace

    from integrations.sionna.network_phy import (
        add_independent_receiver_awgn,
        compose_downlink_resource_grids,
    )

    channel = torch.ones((1, 1, 1, 1, 1, 1, 2), dtype=torch.complex64)
    transmit_grids = torch.ones((1, 1, 1, 1, 2), dtype=torch.complex64)
    composition = compose_downlink_resource_grids(channel, transmit_grids, (1.0,))
    noisy = add_independent_receiver_awgn(composition, (0.01,), seed=31)
    mapped = NRSRSMultiportResourceGrid(
        grid=np.ones((1, 1, 2), dtype=np.complex128),
        mask=np.ones((1, 1, 2), dtype=bool),
        symbol_indices=(0,),
        subcarrier_indices=(0, 1),
    )
    tampered = replace(noisy, received_grid=noisy.received_grid + 1)
    with pytest.raises(ValueError, match="composed grid plus its noise"):
        measure_ul_srs_rsrp_from_noisy_network_grid(
            tampered,
            composition,
            mapped,
            receiver_index=0,
            desired_transmitter_id="ue-1",
            transmitter_ids=["ue-1"],
            occasion_ids=["slot-1"],
            receiver_branch_ids=["rx-branch-1"],
            receiver_category="terrestrial_type_1_c",
            measurement_frequency_bandwidth_hz=20e6,
            power_scale_w_per_grid_unit_squared=1.0,
            power_calibration_sha256="e" * 64,
        )


@pytest.mark.parametrize(
    ("power", "mask", "receiver_category", "calibration_digest", "message"),
    [
        ([[[1.0, np.nan]]], [[True, True]], "terrestrial_type_1_c", "a" * 64, "finite"),
        ([[[1.0, -1.0]]], [[True, True]], "terrestrial_type_1_c", "a" * 64, "nonnegative"),
        ([[[1.0, 1.0]]], [[False, False]], "terrestrial_type_1_c", "a" * 64, "at least one"),
        ([[[1.0, 1.0]]], [[True, True]], "unknown", "a" * 64, "receiver_category"),
        ([[[1.0, 1.0]]], [[True, True]], "terrestrial_type_1_c", "A" * 64, "lowercase SHA-256"),
    ],
)
def test_ul_srs_rsrp_rejects_invalid_inputs(
    power, mask, receiver_category, calibration_digest, message
):
    with pytest.raises(ValueError, match=message):
        measure_ul_srs_rsrp(
            np.asarray(power),
            np.asarray(mask),
            occasion_ids=["occasion-1"],
            receiver_branch_ids=["branch-1"],
            receiver_category=receiver_category,
            measurement_frequency_bandwidth_hz=1e6,
            power_calibration_sha256=calibration_digest,
        )


@pytest.mark.parametrize(
    ("sfn", "subframe", "receiver", "digest"),
    [
        (1024, 0, "terrestrial_type_1_c", "a" * 64),
        (0, 10, "terrestrial_type_1_c", "a" * 64),
        (0, 0, "satellite_type_1_h", "a" * 64),
        (0, 0, "terrestrial_type_1_c", "A" * 64),
    ],
)
def test_ul_rtoa_rejects_invalid_timing_identity(sfn, subframe, receiver, digest):
    with pytest.raises(ValueError):
        report_ul_rtoa(
            received_subframe_start_time_s=1.0,
            sfn=sfn,
            subframe_number=subframe,
            sfn_zero_reference_time_s=0.0,
            receiver_category=receiver,
            timing_calibration_sha256=digest,
        )
