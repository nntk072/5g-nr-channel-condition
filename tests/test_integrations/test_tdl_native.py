import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna")
from integrations.sionna.tdl import generate_tdl_frequency_response  # noqa: E402
from sionna.phy import config as phy_config  # noqa: E402


@pytest.mark.native_sionna
def test_pinned_tr38901_tdl_gpu_taps_map_to_centered_nr_cfr():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    values = {
        "model": "A",
        "seed": 1234,
        "carrier_frequency_hz": 3.5e9,
        "subcarrier_spacing_hz": 30_000.0,
        "num_subcarriers": 64,
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
    assert channel.channel_frequency_response.shape == (1, 1, 2, 1, 1, 3, 64)
    assert channel.metadata["spec_version"] == "19.2"
    assert channel.metadata["profile_proxy"] == "A"
    assert channel.metadata["geometric_los_label"] is False
    torch.testing.assert_close(channel.channel_frequency_response, expected)
    torch.testing.assert_close(
        channel.channel_frequency_response, repeated.channel_frequency_response
    )
