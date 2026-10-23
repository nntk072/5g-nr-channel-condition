import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna")

from integrations.sionna.tdl import generate_tdl_frequency_response  # noqa: E402


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", ["A", "B", "C"])
def test_scalable_tdl_delays_have_configured_power_weighted_rms(model):
    """Scalable profiles use delay_spread_s as their nominal PDP RMS spread."""
    delay_spreads_s = (30e-9, 100e-9, 300e-9, 1_000e-9)
    channels = [
        generate_tdl_frequency_response(
            model=model,
            seed=10_000 + index,
            carrier_frequency_hz=3.5e9,
            subcarrier_spacing_hz=30_000.0,
            num_subcarriers=8,
            delay_spread_s=delay_spread_s,
            spec_version="19.2",
            device="cpu",
        )
        for index, delay_spread_s in enumerate(delay_spreads_s)
    ]

    powers = torch.as_tensor(channels[0].metadata["mean_tap_powers"]).real.to(torch.float64)
    assert torch.sum(powers).item() == pytest.approx(1.0, abs=1e-6)
    nominal_delays = [channel.tau_s[0, 0, 0].to(torch.float64) for channel in channels]

    for delay_spread_s, delays in zip(delay_spreads_s, nominal_delays, strict=True):
        mean_delay_s = torch.sum(powers * delays) / torch.sum(powers)
        rms_delay_spread_s = torch.sqrt(
            torch.sum(powers * torch.square(delays - mean_delay_s)) / torch.sum(powers)
        )
        assert rms_delay_spread_s.item() == pytest.approx(delay_spread_s, rel=2e-4)

    for reference_delays, delays, reference_spread_s, delay_spread_s in zip(
        nominal_delays[:-1],
        nominal_delays[1:],
        delay_spreads_s[:-1],
        delay_spreads_s[1:],
        strict=True,
    ):
        torch.testing.assert_close(
            delays,
            reference_delays * (delay_spread_s / reference_spread_s),
            rtol=2e-6,
            atol=1e-12,
        )


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", ["A", "B", "C", "D", "E"])
def test_tdl_19_2_profile_delays_and_average_tap_powers_match_sionna(model):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    channel = generate_tdl_frequency_response(
        model=model,
        seed=900 + ord(model),
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=4096,
        delay_spread_s=100e-9,
        spec_version="19.2",
        device="cuda:0",
    )
    empirical_powers = torch.mean(torch.abs(channel.h_path[..., 0]) ** 2, dim=(0, 1, 2, 3, 4))
    expected_powers = torch.as_tensor(
        channel.metadata["mean_tap_powers"],
        dtype=empirical_powers.dtype,
        device="cuda:0",
    )
    expected_delays = torch.as_tensor(
        channel.metadata["nominal_tap_delays_s"],
        dtype=channel.tau_s.dtype,
        device="cuda:0",
    )

    assert channel.metadata["num_profile_taps"] == len(expected_powers)
    assert channel.metadata["profile_los_component"] is (model in {"D", "E"})
    torch.testing.assert_close(channel.tau_s[0, 0, 0], expected_delays)
    torch.testing.assert_close(
        channel.tau_s[:, 0, 0], expected_delays.expand_as(channel.tau_s[:, 0, 0])
    )
    torch.testing.assert_close(empirical_powers, expected_powers, rtol=0.1, atol=0.005)
