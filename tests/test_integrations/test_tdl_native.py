import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna")
from integrations.sionna.tdl import generate_tdl_frequency_response  # noqa: E402
from sionna.phy import config as phy_config  # noqa: E402


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", ["A", "D"])
def test_pinned_tr38901_tdl_gpu_taps_map_to_centered_nr_cfr(model):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    values = {
        "model": model,
        "seed": 1234,
        "carrier_frequency_hz": 3.5e9,
        "subcarrier_spacing_hz": 30_000.0,
        "num_subcarriers": 64,
        "batch_size": 2,
        "delay_spread_s": 300e-9,
        "num_time_steps": 3,
        "sampling_frequency_hz": 15_000.0,
        "num_rx_ant": 2,
        "num_tx_ant": 1,
        "spec_version": "19.2",
        "device": "cuda:0",
    }
    rng = phy_config.torch_rng("cuda:0")
    rng_state = rng.get_state()
    channel = generate_tdl_frequency_response(**values)
    torch.testing.assert_close(rng.get_state(), rng_state)
    repeated = generate_tdl_frequency_response(**values)
    h_time_path = channel.h_path.movedim(-2, -1)
    frequencies = (torch.arange(64, device="cuda:0", dtype=channel.tau_s.dtype) - 32) * 30_000.0
    expected = torch.sum(
        h_time_path.unsqueeze(-1)
        * torch.exp(
            -2j
            * torch.pi
            * channel.tau_s[:, :, None, :, None, None, :, None]
            * frequencies.reshape(1, 1, 1, 1, 1, 1, 1, -1)
        ),
        dim=-2,
    )

    assert channel.h_path.is_cuda
    assert channel.tau_s.is_cuda
    assert channel.channel_frequency_response.is_cuda
    assert channel.channel_frequency_response.shape == (2, 1, 2, 1, 1, 3, 64)
    assert channel.metadata["spec_version"] == "19.2"
    assert channel.metadata["profile_proxy"] == model
    assert channel.metadata["geometric_los_label"] is None
    assert channel.metadata["geometric_los_label_provided"] is False
    torch.testing.assert_close(channel.channel_frequency_response, expected)
    torch.testing.assert_close(
        channel.channel_frequency_response, repeated.channel_frequency_response
    )


@pytest.mark.native_sionna
@pytest.mark.parametrize("device", ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"])
def test_tdl_a_temporal_correlation_matches_maximum_doppler_jakes_reference(device):
    """Check the moving NLOS tap against J0(2π f_D Δt), not only its CFR map."""
    carrier_frequency_hz = 3.5e9
    speed_mps = 30.0
    maximum_doppler_hz = speed_mps * carrier_frequency_hz / 299_792_458.0
    # Quarter of the maximum-Doppler period gives a well-conditioned Jakes
    # correlation (~0.472), away from both the zero-lag and first-zero cases.
    lag_s = 1.0 / (4.0 * maximum_doppler_hz)
    channel = generate_tdl_frequency_response(
        model="A",  # NLOS profile: no specular component in the reference.
        seed=4031,
        carrier_frequency_hz=carrier_frequency_hz,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=8192,
        delay_spread_s=100e-9,
        num_time_steps=2,
        sampling_frequency_hz=1.0 / lag_s,
        min_speed_mps=speed_mps,
        max_speed_mps=speed_mps,
        spec_version="19.2",
        device=device,
    )

    # h_path axes are [batch, rx, rx_ant, tx, tx_ant, path, time]. Normalize
    # by empirical tap power so the check is independent of PDP tap weight.
    tap = channel.h_path[:, 0, 0, 0, 0, 0, :]
    observed = torch.mean(tap[:, 0].conj() * tap[:, 1]) / torch.mean(torch.abs(tap[:, 0]) ** 2)
    expected = torch.special.bessel_j0(
        torch.tensor(2.0 * torch.pi * maximum_doppler_hz * lag_s, device=device)
    )

    torch.testing.assert_close(observed.real, expected, rtol=0.0, atol=0.04)
    torch.testing.assert_close(observed.imag, torch.zeros_like(observed.imag), rtol=0.0, atol=0.04)
