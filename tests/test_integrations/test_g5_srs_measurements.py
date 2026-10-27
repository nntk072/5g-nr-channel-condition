from __future__ import annotations

import numpy as np
import pytest
from integrations.sionna.srs_measurements import (
    measure_ul_srs_rsrp,
    report_gnb_rx_tx_time_difference,
    report_ul_rtoa,
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
