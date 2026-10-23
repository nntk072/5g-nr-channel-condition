import hashlib
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna")

from integrations.sionna.tdl import generate_tdl_frequency_response  # noqa: E402
from sionna.phy.channel.tr38901 import TDL, models  # noqa: E402

_TDL_19_2_RESOURCE_SHA256 = {
    "A": "b56888b8392f2e1218adfd3b99c73cd82c8a20bbfcc68cfaccf99a8b4bf1db04",
    "B": "246d8c5930a8721acf9a6acdb2b7d1598f3420614a5d70e9b74371895c49eaad",
    "C": "6dea193df23e9c4ea83c6ba4427f97651a12ceb1d2c511757d31c849c57829ea",
    "D": "ef21c1dda98838ac74a24d7504f6d8a8bfc0bd3051279483c673f9ea3b42366a",
    "E": "5d9cf8519179a183ccfc5020efb02cf48358f5d67a5bf40520636e44a0e2f045",
}


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", ["A", "B", "C", "D", "E"])
def test_tdl_19_2_nominal_parameters_match_hash_pinned_resource_table(model):
    resource = models.parameter_file(f"TDL-{model}.json", "19.2")
    resource_bytes = resource.read_bytes()
    assert hashlib.sha256(resource_bytes).hexdigest() == _TDL_19_2_RESOURCE_SHA256[model]
    parameters = json.loads(resource_bytes)

    delay_spread_s = 100e-9
    channel = TDL(
        model=model,
        delay_spread=delay_spread_s,
        carrier_frequency=3.5e9,
        min_speed=0.0,
        max_speed=0.0,
        spec_version="19.2",
        device="cpu",
    )

    table_delays = torch.as_tensor(parameters["delays"], dtype=torch.float64)
    table_powers = torch.pow(
        10.0,
        torch.as_tensor(parameters["powers"], dtype=torch.float64) / 10.0,
    )
    table_powers /= torch.sum(table_powers)
    if parameters["los"]:
        expected_delays_s = table_delays[1:] * delay_spread_s
        expected_powers = torch.cat((table_powers[:1] + table_powers[1:2], table_powers[2:]))
    else:
        expected_delays_s = table_delays * delay_spread_s
        expected_powers = table_powers

    torch.testing.assert_close(
        channel.delays.detach().cpu().to(torch.float64), expected_delays_s, rtol=1e-6, atol=1e-13
    )
    torch.testing.assert_close(
        channel.mean_powers.detach().cpu().to(torch.float64), expected_powers, rtol=1e-6, atol=1e-7
    )
    assert channel.los is bool(parameters["los"])
    assert torch.sum(channel.mean_powers).item() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", ["A", "B", "C", "D", "E"])
def test_scalable_tdl_delays_have_configured_power_weighted_rms(model):
    """All scalable profiles preserve their normalized PDP shape as delays scale."""
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

    rms_spreads_s = []
    for delays in nominal_delays:
        mean_delay_s = torch.sum(powers * delays) / torch.sum(powers)
        rms_delay_spread_s = torch.sqrt(
            torch.sum(powers * torch.square(delays - mean_delay_s)) / torch.sum(powers)
        )
        rms_spreads_s.append(rms_delay_spread_s.item())

    normalized_table_rms = rms_spreads_s[0] / delay_spreads_s[0]
    for delay_spread_s, rms_delay_spread_s in zip(delay_spreads_s, rms_spreads_s, strict=True):
        assert rms_delay_spread_s == pytest.approx(normalized_table_rms * delay_spread_s, rel=2e-4)

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
