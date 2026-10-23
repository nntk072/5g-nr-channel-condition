"""Verify NR FFT-bin ordering against the analytic frequency grid."""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from integrations.sionna.rt_to_nr import paths_to_nr_ofdm_channel  # noqa: E402


class AnalyticDelayPaths:
    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s

    def cfr(self, **kwargs):
        frequencies_hz = np.asarray(kwargs["frequencies"], dtype=np.float64)
        response = np.exp(-2j * np.pi * frequencies_hz * self.delay_s)
        response = response.astype(np.complex64)
        return torch.as_tensor(response.reshape(1, 1, 1, 1, 1, -1))


@pytest.mark.parametrize("fft_size", [5, 6, 8])
def test_rt_nr_frequency_bins_match_analytic_centered_fft_order(fft_size):
    spacing_hz = 30_000.0
    delay_s = 2.75e-6
    grid = SimpleNamespace(
        fft_size=fft_size,
        num_ofdm_symbols=1,
        subcarrier_spacing=spacing_hz,
    )

    result = paths_to_nr_ofdm_channel(AnalyticDelayPaths(delay_s), grid)

    expected_frequencies = np.fft.fftshift(
        np.fft.fftfreq(fft_size, d=1.0 / (fft_size * spacing_hz))
    )
    np.testing.assert_array_equal(result.frequencies_hz.numpy(), expected_frequencies)
    assert result.frequencies_hz[fft_size // 2].item() == 0.0

    expected_response = np.exp(-2j * np.pi * expected_frequencies * delay_s)
    actual_response = result.channel[0, 0, 0, 0, 0, 0].numpy()
    np.testing.assert_allclose(actual_response, expected_response, rtol=2e-6, atol=2e-6)
