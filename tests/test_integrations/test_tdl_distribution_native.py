"""Monte Carlo check of a pinned Rayleigh TDL profile's power tails."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna")

from integrations.sionna.tdl import generate_tdl_frequency_response  # noqa: E402


@pytest.mark.native_sionna
def test_tdl_a_normalized_tap_power_tail_matches_exponential_distribution():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    batch_size = 8192
    channel = generate_tdl_frequency_response(
        model="A",
        seed=8128,
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=batch_size,
        delay_spread_s=100e-9,
        spec_version="19.2",
        device="cuda:0",
    )
    # TDL-A has no LOS component. Each complex Gaussian tap therefore has
    # exponential instantaneous power; normalize each tap by its own sample
    # mean before pooling so the assertion checks shape, not profile powers.
    path_coefficients = channel.h_path[..., 0].reshape(batch_size, -1)
    tap_power = path_coefficients.abs().square()
    normalized_power = tap_power / tap_power.mean(dim=0, keepdim=True)
    probabilities = torch.tensor([0.1, 0.5, 0.9, 0.99], device="cuda:0")
    empirical_quantiles = torch.quantile(normalized_power.flatten(), probabilities)
    exponential_quantiles = -torch.log1p(-probabilities)

    # 8,192 realizations and a fixed seed give a useful upper-tail check while
    # allowing ordinary finite-sample variation (including the 99th percentile).
    torch.testing.assert_close(
        empirical_quantiles,
        exponential_quantiles,
        rtol=0.08,
        atol=0.01,
    )
    torch.testing.assert_close(
        normalized_power.mean(),
        torch.ones((), device="cuda:0"),
        rtol=0.0,
        atol=1e-6,
    )
