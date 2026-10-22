"""Native CDL A-E adapter checks against the pinned profile resources."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna.phy")

from integrations.sionna.cdl import generate_cdl_frequency_response  # noqa: E402
from integrations.sionna.random import call_with_sionna_seed  # noqa: E402


@pytest.mark.native_sionna
@pytest.mark.parametrize("profile", list("ABCDE"))
@pytest.mark.parametrize("delay_spread_ns", [30.0, 100.0, 300.0, 1000.0])
def test_cdl_19_2_delays_and_monte_carlo_powers_match_versioned_profile(
    profile: str, delay_spread_ns: float
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    from sionna.phy.channel.tr38901 import models as profile_resources

    carrier_hz = 3.5e9
    delay_spread_s = delay_spread_ns * 1e-9
    device = "cuda:0"
    channel = generate_cdl_frequency_response(
        model=profile,
        seed=31900 + ord(profile),
        carrier_frequency_hz=carrier_hz,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=4096,
        delay_spread_s=delay_spread_s,
        num_time_steps=1,
        sampling_frequency_hz=30_000.0,
        direction="downlink",
        min_speed_mps=0.0,
        max_speed_mps=0.0,
        spec_version="19.2",
        device=device,
    )
    parameters = profile_resources.load_json(
        profile_resources.parameter_file(f"CDL-{profile}.json", "19.2")
    )
    table_delays_s = (
        torch.as_tensor(parameters["delays"], dtype=channel.tau_s.dtype, device=device)
        * delay_spread_s
    )
    table_powers = torch.pow(
        torch.tensor(10.0, dtype=channel.tau_s.dtype, device=device),
        torch.as_tensor(parameters["powers"], dtype=channel.tau_s.dtype, device=device) / 10.0,
    )
    table_powers = table_powers / table_powers.sum()

    if bool(parameters["los"]):
        specular_power = table_powers[0]
        nlos_powers = table_powers[1:]
        k_factor = specular_power / nlos_powers.sum()
        table_delays_s = table_delays_s[1:]
        table_powers = nlos_powers / nlos_powers.sum()
        expected_powers = torch.cat(
            (
                ((k_factor + table_powers[:1]) / (k_factor + 1.0)),
                table_powers[1:] / (k_factor + 1.0),
            )
        )
    else:
        k_factor = None
        expected_powers = table_powers

    order = torch.argsort(table_delays_s)
    expected_delays = table_delays_s[order]
    expected_powers = expected_powers[order]
    empirical_powers = torch.mean(torch.abs(channel.h_path[:, 0, 0, 0, 0, :, 0]) ** 2, dim=0)

    assert len(expected_delays) == channel.metadata["num_profile_clusters"]
    assert channel.metadata["profile_los_component"] is bool(parameters["los"])
    assert channel.metadata["profile_k_factor_linear"] == (
        None if k_factor is None else pytest.approx(float(k_factor))
    )
    torch.testing.assert_close(channel.tau_s[0, 0, 0], expected_delays, rtol=0.0, atol=1e-14)
    torch.testing.assert_close(
        torch.as_tensor(
            channel.metadata["normalized_cluster_powers"],
            dtype=expected_powers.dtype,
            device=device,
        ),
        expected_powers,
        rtol=0.0,
        atol=1e-7,
    )
    for name in ("aod", "aoa", "zod", "zoa"):
        table_angles = parameters[name][1:] if bool(parameters["los"]) else parameters[name]
        expected_angles = torch.as_tensor(table_angles, dtype=torch.float32, device=device)[order]
        torch.testing.assert_close(
            torch.as_tensor(channel.metadata["nominal_cluster_angles_deg"][name], device=device),
            expected_angles,
            rtol=0.0,
            atol=1e-6,
        )
    expected_los_angles = (
        {name: float(parameters[name][0]) for name in ("aod", "aoa", "zod", "zoa")}
        if bool(parameters["los"])
        else None
    )
    assert channel.metadata["nominal_los_angles_deg"] == expected_los_angles
    assert channel.metadata["cross_polarization_power_ratio_db"] == float(parameters["xpr"])
    torch.testing.assert_close(
        empirical_powers.sum(), torch.tensor(1.0, device=device), rtol=0.02, atol=0.01
    )
    torch.testing.assert_close(empirical_powers, expected_powers, rtol=0.1, atol=0.005)
    assert channel.metadata["parameter_table"] == f"CDL-{profile}.json"
    table_path = Path(profile_resources.parameter_file(f"CDL-{profile}.json", "19.2"))
    assert (
        channel.metadata["parameter_table_sha256"]
        == hashlib.sha256(table_path.read_bytes()).hexdigest()
    )


@pytest.mark.native_sionna
def test_pinned_cdl_false_delay_normalization_returns_unscaled_table_values():
    from sionna.phy.channel.tr38901 import CDL, PanelArray
    from sionna.phy.channel.tr38901 import models as profile_resources

    device = "cpu"
    carrier_hz = 3.5e9
    array = PanelArray(
        num_rows_per_panel=1,
        num_cols_per_panel=1,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=carrier_hz,
        device=device,
    )
    channel_model = CDL(
        model="A",
        delay_spread=250e-9,
        carrier_frequency=carrier_hz,
        ut_array=array,
        bs_array=array,
        direction="downlink",
        normalize_delays=False,
        device=device,
        spec_version="19.2",
    )
    _, delays = call_with_sionna_seed(
        device,
        41,
        lambda: channel_model(batch_size=1, num_time_steps=1, sampling_frequency=30_000.0),
    )
    parameters = profile_resources.load_json(profile_resources.parameter_file("CDL-A.json", "19.2"))
    raw_table_delays = torch.as_tensor(parameters["delays"], dtype=delays.dtype, device=device)
    raw_table_delays = raw_table_delays[torch.argsort(raw_table_delays)]

    torch.testing.assert_close(delays[0, 0, 0], raw_table_delays)
    assert float(delays.max()) > 1.0  # It cannot represent a physical delay in seconds.
