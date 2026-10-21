import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna")

from integrations.sionna.tdl import generate_tdl_frequency_response  # noqa: E402


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
