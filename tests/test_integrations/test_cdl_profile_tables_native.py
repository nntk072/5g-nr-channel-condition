"""Native CDL A-E delay/PDP implementation checks against pinned resources."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna.phy")

from integrations.sionna.random import call_with_sionna_seed  # noqa: E402


@pytest.mark.native_sionna
@pytest.mark.parametrize("profile", list("ABCDE"))
@pytest.mark.parametrize("delay_spread_ns", [30.0, 100.0, 300.0, 1000.0])
def test_cdl_19_2_delays_and_monte_carlo_powers_match_versioned_profile(
    profile: str, delay_spread_ns: float
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    from sionna.phy.channel.tr38901 import CDL, PanelArray
    from sionna.phy.channel.tr38901 import models as profile_resources

    carrier_hz = 3.5e9
    delay_spread_s = delay_spread_ns * 1e-9
    device = "cuda:0"
    array = PanelArray(
        num_rows_per_panel=1,
        num_cols_per_panel=1,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=carrier_hz,
        device=device,
    )
    channel = CDL(
        model=profile,
        delay_spread=delay_spread_s,
        carrier_frequency=carrier_hz,
        ut_array=array,
        bs_array=array,
        direction="downlink",
        min_speed=0.0,
        max_speed=0.0,
        device=device,
        spec_version="19.2",
    )
    parameters = profile_resources.load_json(
        profile_resources.parameter_file(f"CDL-{profile}.json", "19.2")
    )
    table_delays = (
        torch.as_tensor(parameters["delays"], dtype=channel._delays.dtype, device=device)
        * delay_spread_s
    )
    table_powers = torch.pow(
        torch.tensor(10.0, dtype=channel._powers.dtype, device=device),
        torch.as_tensor(parameters["powers"], dtype=channel._powers.dtype, device=device) / 10.0,
    )
    table_powers = table_powers / table_powers.sum()

    if bool(parameters["los"]):
        specular_power = table_powers[0]
        nlos_powers = table_powers[1:]
        k_factor = specular_power / nlos_powers.sum()
        table_delays = table_delays[1:]
        table_powers = nlos_powers / nlos_powers.sum()
        expected_powers = torch.cat(
            (
                ((k_factor + table_powers[:1]) / (k_factor + 1.0)),
                table_powers[1:] / (k_factor + 1.0),
            )
        )
    else:
        expected_powers = table_powers

    order = torch.argsort(table_delays)
    expected_delays = table_delays[order]
    expected_powers = expected_powers[order]
    with torch.no_grad():
        coefficients, delays = call_with_sionna_seed(
            device,
            31900 + ord(profile),
            lambda: channel(batch_size=4096, num_time_steps=1, sampling_frequency=30_000.0),
        )
    empirical_powers = torch.mean(torch.abs(coefficients[:, 0, 0, 0, 0, :, 0]) ** 2, dim=0)

    assert len(expected_delays) == channel._num_clusters
    assert channel._has_los is bool(parameters["los"])
    torch.testing.assert_close(delays[0, 0, 0], expected_delays, rtol=0.0, atol=1e-14)
    torch.testing.assert_close(
        empirical_powers.sum(), torch.tensor(1.0, device=device), rtol=0.02, atol=0.01
    )
    torch.testing.assert_close(empirical_powers, expected_powers, rtol=0.1, atol=0.005)
