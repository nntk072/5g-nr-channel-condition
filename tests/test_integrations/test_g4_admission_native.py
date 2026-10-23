"""Native adapter-to-record checks for standardized-channel G4 admission."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna")

from datasets import StandardizedFamily, standardized_run_from_channel  # noqa: E402
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
