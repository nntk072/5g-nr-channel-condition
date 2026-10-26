"""Tests for the experimental CP-bounded delay-domain PUSCH estimator."""

from __future__ import annotations

import math
from typing import Any

import pytest
from scipy.stats import chi2

torch = pytest.importorskip("torch")
pytest.importorskip("sionna.phy")

from experiments.g5_cp_bounded_dft_estimator import (  # noqa: E402
    FiniteDelayDFTChannelEstimator,
    make_finite_delay_dft_decoder,
)
from integrations.sionna.nr_pusch import SionnaPUSCHLSReceiver  # noqa: E402
from integrations.sionna.ofdm_impairments import apply_time_varying_multipath  # noqa: E402
from receivers.nr import NRPUSCHConfig  # noqa: E402


def _case(
    additional_position: int = 0,
    noise_variance: float = 0.001,
) -> tuple[SionnaPUSCHLSReceiver, Any, Any, Any]:
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=6,
            subcarrier_spacing_khz=30,
            dmrs_additional_position=additional_position,
        ),
        device="cpu",
    )
    tx_grid, _ = receiver.transmit(4, seed=92_001)
    waveform = receiver.modulate_slot_grid(tx_grid)
    channel = apply_time_varying_multipath(
        waveform,
        path_coefficients=(0.85 + 0.0j, 0.31 - 0.22j),
        path_delays_samples=(0, 6),
        path_dopplers_hz=(0.0, 0.0),
        sampling_frequency_hz=receiver.resource_grid.fft_size * 30_000,
    )
    noise_generator = torch.Generator(device="cpu").manual_seed(92_002)
    noise = torch.complex(
        torch.randn(waveform.shape, generator=noise_generator),
        torch.randn(waveform.shape, generator=noise_generator),
    ) * math.sqrt(noise_variance / 2.0)
    received = receiver.demodulate_slot_waveform(channel + noise)
    return receiver, tx_grid, received, channel


def test_cp_bounded_dft_estimate_recovers_in_cp_static_channel() -> None:
    receiver, _, _, channel = _case()
    estimator = FiniteDelayDFTChannelEstimator(receiver, max_delay_samples=7)
    received = receiver.demodulate_slot_waveform(channel)
    estimate, error_variance = estimator(received, torch.tensor(0.0))
    fft_size = receiver.resource_grid.fft_size
    frequency_bins = torch.arange(fft_size, dtype=torch.float32) - fft_size // 2
    response = 0.85 + (0.31 - 0.22j) * torch.exp(-2j * math.pi * frequency_bins * 6 / fft_size)
    expected = response.reshape(1, 1, 1, 1, 1, 1, -1).expand_as(estimate)
    nmse = (estimate - expected).abs().square().mean() / expected.abs().square().mean()
    assert float(nmse) < 1e-5
    assert estimate.shape == (4, 1, 1, 1, 1, 14, fft_size)
    assert error_variance.shape == estimate.shape
    assert bool(torch.isfinite(error_variance).all())
    assert bool((error_variance >= 0).all())


def test_finite_delay_dft_recovers_a_signed_delay_interval() -> None:
    receiver, _, base_grid, _ = _case(noise_variance=0.0)
    fft_size = receiver.resource_grid.fft_size
    frequencies = torch.arange(fft_size, dtype=torch.float32) - fft_size // 2
    base_response = 0.85 + (0.31 - 0.22j) * torch.exp(-2j * math.pi * frequencies * 6 / fft_size)
    expected_response = (
        (0.23 + 0.10j) * torch.exp(2j * math.pi * frequencies * 3 / fft_size)
        + 0.72
        + (0.08 - 0.11j) * torch.exp(-2j * math.pi * frequencies * 7 / fft_size)
    )
    transformed_grid = base_grid * (expected_response / base_response).reshape(1, 1, 1, 1, -1)

    estimator = FiniteDelayDFTChannelEstimator(
        receiver,
        min_delay_samples=-3,
        max_delay_samples=7,
    )
    estimate, _ = estimator(transformed_grid, torch.tensor(0.0))
    expected = expected_response.reshape(1, 1, 1, 1, 1, 1, -1).expand_as(estimate)
    nmse = (estimate - expected).abs().square().mean() / expected.abs().square().mean()

    assert float(nmse) < 1e-5


def test_cp_bounded_dft_decoder_decodes_static_in_cp_waveform() -> None:
    receiver, _, received, _ = _case()
    decoder = make_finite_delay_dft_decoder(receiver, max_delay_samples=7)
    _, crc = decoder(received, torch.tensor(0.001))
    assert bool(crc.all())


def test_finite_delay_dft_estimator_handles_multiple_dmrs_symbols() -> None:
    receiver, _, _, channel = _case(additional_position=3)
    estimator = FiniteDelayDFTChannelEstimator(receiver, max_delay_samples=7)
    received = receiver.demodulate_slot_waveform(channel)
    estimate, error_variance = estimator(received, torch.tensor(0.0))
    fft_size = receiver.resource_grid.fft_size
    frequency_bins = torch.arange(fft_size, dtype=torch.float32) - fft_size // 2
    response = 0.85 + (0.31 - 0.22j) * torch.exp(-2j * math.pi * frequency_bins * 6 / fft_size)
    expected = response.reshape(1, 1, 1, 1, 1, 1, -1).expand_as(estimate)
    nmse = (estimate - expected).abs().square().mean() / expected.abs().square().mean()
    assert float(nmse) < 1e-5
    assert estimate.shape == (4, 1, 1, 1, 1, 14, fft_size)
    assert error_variance.shape == estimate.shape


def test_finite_delay_dft_reported_variance_matches_independent_noise_draws() -> None:
    receiver, _, _, channel = _case(additional_position=3)
    estimator = FiniteDelayDFTChannelEstimator(receiver, max_delay_samples=7)
    clean_grid = receiver.demodulate_slot_waveform(channel[:1])
    clean_estimate, _ = estimator(clean_grid, torch.tensor(0.0))

    draw_count = 4096
    noise_variance = 0.001
    generator = torch.Generator(device="cpu").manual_seed(93_041)
    noise = torch.complex(
        torch.randn((draw_count, *clean_grid.shape[1:]), generator=generator),
        torch.randn((draw_count, *clean_grid.shape[1:]), generator=generator),
    ) * math.sqrt(noise_variance / 2.0)
    repeated_clean = clean_grid.expand(draw_count, *clean_grid.shape[1:])
    noisy_estimate, reported_variance = estimator(
        repeated_clean + noise,
        torch.tensor(noise_variance),
    )

    symbol_index = 6
    subcarrier_index = receiver.resource_grid.fft_size // 2
    errors = (
        noisy_estimate[:, 0, 0, 0, 0, symbol_index, subcarrier_index]
        - clean_estimate[0, 0, 0, 0, 0, symbol_index, subcarrier_index]
    )
    empirical_variance = errors.sub(errors.mean()).abs().square().sum() / (draw_count - 1)
    predicted_variance = reported_variance[0, 0, 0, 0, 0, symbol_index, subcarrier_index]
    degrees_of_freedom = 2 * (draw_count - 1)
    lower_ratio = degrees_of_freedom / chi2.ppf(0.995, degrees_of_freedom)
    upper_ratio = degrees_of_freedom / chi2.ppf(0.005, degrees_of_freedom)
    ratio = float(empirical_variance / predicted_variance)

    assert float(predicted_variance) > 0.0
    assert float(lower_ratio) <= ratio <= float(upper_ratio), (
        float(empirical_variance),
        float(predicted_variance),
        ratio,
        float(lower_ratio),
        float(upper_ratio),
    )


def test_cp_bounded_dft_rejects_delay_span_not_identifiable_from_dmrs() -> None:
    receiver, _, _, _ = _case()
    with pytest.raises(ValueError, match="cannot identify"):
        FiniteDelayDFTChannelEstimator(receiver, max_delay_samples=36)
