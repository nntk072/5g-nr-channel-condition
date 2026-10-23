"""Native adapter-to-record checks for standardized-channel G4 admission."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna")

from datasets import (  # noqa: E402
    StandardizedFamily,
    read_standardized_run,
    standardized_learning_example_from_record,
    standardized_run_from_channel,
    write_standardized_run,
)
from integrations.sionna.cdl import CDLArrayConfig, generate_cdl_frequency_response  # noqa: E402
from integrations.sionna.tdl import generate_tdl_frequency_response  # noqa: E402


@pytest.mark.native_sionna
def test_tdl_receive_array_and_correlation_are_admitted_with_axis_provenance() -> None:
    correlation = 0.5 + 0.2j
    channel = generate_tdl_frequency_response(
        model="A",
        seed=1201,
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=2,
        delay_spread_s=100e-9,
        num_rx_ant=2,
        rx_corr_mat=((1.0, correlation), (correlation.conjugate(), 1.0)),
        spec_version="19.2",
        device="cpu",
    )
    run = standardized_run_from_channel(channel, family=StandardizedFamily.TDL)

    assert run.records[0].h_path.shape == (1, 2, 1, 1, 23, 1)
    assert run.records[0].metadata["rx_antenna_count"] == 2
    assert run.records[0].metadata["sampled_bandwidth_hz"] == 240_000.0
    assert run.records[0].metadata["rx_corr_mat_imag"][0][1] == pytest.approx(0.2, abs=1e-7)


@pytest.mark.native_sionna
def test_cdl_panel_array_metadata_is_checked_against_native_antenna_axes() -> None:
    channel = generate_cdl_frequency_response(
        model="A",
        seed=1202,
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=2,
        delay_spread_s=100e-9,
        ut_array=CDLArrayConfig(
            num_cols_per_panel=2,
            num_cols=1,
            element_horizontal_spacing=0.5,
        ),
        spec_version="19.2",
        device="cpu",
    )
    run = standardized_run_from_channel(channel, family=StandardizedFamily.CDL)

    assert run.records[0].h_path.shape == (1, 2, 1, 1, 23, 1)
    assert run.records[0].metadata["rx_antenna_count"] == 2
    assert run.records[0].metadata["ut_array"]["num_cols_per_panel"] == 2


@pytest.mark.native_sionna
def test_out_of_scope_native_run_is_rejected_at_dataset_admission() -> None:
    diagnostic = generate_tdl_frequency_response(
        model="A",
        seed=1203,
        carrier_frequency_hz=0.49e9,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=1,
        delay_spread_s=100e-9,
        spec_version="19.2",
        device="cpu",
    )
    with pytest.raises(ValueError, match="0.5–100 GHz"):
        standardized_run_from_channel(diagnostic, family=StandardizedFamily.TDL)


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("model", "carrier_hz", "spacing_hz", "num_subcarriers", "has_profile_los"),
    (
        ("A10", 28e9, 120e3, 2048, False),
        ("A30", 3.5e9, 30e3, 8, False),
        ("B100", 3.5e9, 30e3, 8, False),
        ("C60", 28e9, 30e3, 8, False),
        ("C300", 3.5e9, 30e3, 8, False),
        ("D10", 28e9, 120e3, 2048, True),
        ("D30", 3.5e9, 30e3, 8, True),
    ),
)
def test_fixed_delay_tdl_profiles_are_admitted_and_roundtrip(
    tmp_path, model, carrier_hz, spacing_hz, num_subcarriers, has_profile_los
) -> None:
    channel = generate_tdl_frequency_response(
        model=model,
        seed=13_001,
        carrier_frequency_hz=carrier_hz,
        subcarrier_spacing_hz=spacing_hz,
        num_subcarriers=num_subcarriers,
        delay_spread_s=None,
        spec_version="19.2",
        device="cpu",
    )
    run = standardized_run_from_channel(channel, family=StandardizedFamily.TDL)

    assert run.records[0].profile_model == model
    assert run.records[0].profile_los_component is has_profile_los
    assert (
        run.records[0].metadata["delay_profile_semantics"]
        == "fixed_ts_38_101_4_absolute_tap_delays"
    )
    stored = write_standardized_run(tmp_path, run)
    restored = read_standardized_run(tmp_path, stored)
    record = restored.records[0]
    assert record.profile_model == model
    np.testing.assert_array_equal(record.tau_s, run.records[0].tau_s)
    example = standardized_learning_example_from_record(record, snr_db=30.0, seed=13_002)
    assert example.targets["profile_los_proxy"].values.tolist() == [int(has_profile_los)]


@pytest.mark.native_sionna
def test_fixed_delay_tdl_profile_is_rejected_as_cdl() -> None:
    channel = generate_tdl_frequency_response(
        model="A10",
        seed=13_003,
        carrier_frequency_hz=28e9,
        subcarrier_spacing_hz=120e3,
        num_subcarriers=2048,
        delay_spread_s=None,
        spec_version="19.2",
        device="cpu",
    )
    with pytest.raises(ValueError, match="CDL profile must be one of A–E"):
        standardized_run_from_channel(channel, family=StandardizedFamily.CDL)
