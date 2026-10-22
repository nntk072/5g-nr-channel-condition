"""Statistical contracts for proper-complex receiver noise."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from integrations.sionna.noise import proper_complex_awgn  # noqa: E402
from integrations.sionna.rt_dmrs import thermal_noise_variance_w  # noqa: E402


def test_thermal_noise_injection_has_declared_complex_sample_covariance() -> None:
    variance_w = thermal_noise_variance_w(
        temperature_k=290.0,
        noise_figure_db=7.0,
        bandwidth_hz=30_000.0,
    )
    reference = torch.zeros((131_072, 2), dtype=torch.complex64)
    generator = torch.Generator().manual_seed(20261023)
    noise = proper_complex_awgn(reference, variance_w, generator=generator)
    normalized = noise / variance_w**0.5

    measured_complex_variance = normalized.abs().square().mean()
    measured_real_variance = normalized.real.square().mean()
    measured_imag_variance = normalized.imag.square().mean()
    cross_antenna_covariance = (normalized[:, 0] * normalized[:, 1].conj()).mean()

    assert measured_complex_variance.item() == pytest.approx(1.0, abs=0.015)
    assert measured_real_variance.item() == pytest.approx(0.5, abs=0.01)
    assert measured_imag_variance.item() == pytest.approx(0.5, abs=0.01)
    assert abs(cross_antenna_covariance.item()) < 0.01

    replay_generator = torch.Generator().manual_seed(20261023)
    replay = proper_complex_awgn(reference, variance_w, generator=replay_generator)
    torch.testing.assert_close(noise, replay, rtol=0, atol=0)


@pytest.mark.parametrize("variance", [-1.0, float("inf"), float("nan")])
def test_proper_complex_awgn_rejects_invalid_variance(variance: float) -> None:
    reference = torch.zeros((2,), dtype=torch.complex64)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        proper_complex_awgn(reference, variance, generator=torch.Generator().manual_seed(1))
