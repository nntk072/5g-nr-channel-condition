"""Native tests distinguishing independent batch draws from time samples."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna.phy")

from datasets.standardized import (  # noqa: E402
    StandardizedFamily,
    read_standardized_run,
    standardized_run_from_channel,
    write_standardized_run,
)
from integrations.sionna.cdl import generate_cdl_frequency_response  # noqa: E402
from integrations.sionna.tdl import generate_tdl_frequency_response  # noqa: E402


@pytest.mark.native_sionna
@pytest.mark.parametrize("family", ["tdl", "cdl"])
@pytest.mark.parametrize("speed_mps", [0.0, 20.0])
def test_batch_draws_and_time_samples_have_distinct_semantics(
    family: str, speed_mps: float, tmp_path
):
    carrier_frequency_hz = 3.5e9
    maximum_doppler_hz = speed_mps * carrier_frequency_hz / 299_792_458.0
    sampling_frequency_hz = max(1_000.0, 4.0 * maximum_doppler_hz)
    common = {
        "model": "A",
        "seed": 12_503,
        "carrier_frequency_hz": carrier_frequency_hz,
        "subcarrier_spacing_hz": 30_000.0,
        "num_subcarriers": 8,
        "batch_size": 4,
        "num_time_steps": 3,
        "sampling_frequency_hz": sampling_frequency_hz,
        "spec_version": "19.2",
        "device": "cpu",
    }
    if family == "tdl":
        generate = generate_tdl_frequency_response
        channel_kwargs = {
            **common,
            "delay_spread_s": 100e-9,
            "min_speed_mps": speed_mps,
            "max_speed_mps": speed_mps,
        }
    else:
        generate = generate_cdl_frequency_response
        channel_kwargs = {
            **common,
            "delay_spread_s": 100e-9,
            "ut_velocity_mps": (speed_mps, 0.0, 0.0),
        }

    channel = generate(**channel_kwargs)
    repeated = generate(**channel_kwargs)
    changed_seed = generate(**{**channel_kwargs, "seed": common["seed"] + 1})

    metadata = channel.metadata
    assert metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    assert metadata["time_axis_semantics"] == "samples_within_each_channel_draw"
    assert metadata["time_sample_interval_s"] == pytest.approx(1.0 / sampling_frequency_hz)
    assert metadata["temporal_evolution_observed"] is (speed_mps > 0.0)
    torch.testing.assert_close(channel.h_path, repeated.h_path, rtol=0.0, atol=0.0)
    torch.testing.assert_close(channel.tau_s, repeated.tau_s, rtol=0.0, atol=0.0)
    assert not torch.equal(channel.h_path[0], channel.h_path[1])
    assert not torch.equal(channel.h_path, changed_seed.h_path)

    first_draw = channel.h_path[0]
    if speed_mps > 0.0:
        assert not torch.allclose(first_draw[..., 0], first_draw[..., 1])
    else:
        torch.testing.assert_close(first_draw[..., 0], first_draw[..., 1], rtol=0.0, atol=0.0)

    standardized = standardized_run_from_channel(
        channel,
        family=StandardizedFamily.TDL if family == "tdl" else StandardizedFamily.CDL,
    )
    stored = write_standardized_run(tmp_path, standardized)
    restored = read_standardized_run(tmp_path, stored)
    assert [record.sample_index for record in restored.records] == list(range(4))
    assert all(
        record.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
        and record.metadata["time_axis_semantics"] == "samples_within_each_channel_draw"
        and record.metadata["temporal_evolution_observed"] is (speed_mps > 0.0)
        for record in restored.records
    )
