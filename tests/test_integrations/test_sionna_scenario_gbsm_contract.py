"""Native Sionna contract coverage for scenario-based TR 38.901 adapters."""

import numpy as np
import pytest
from integrations.sionna.scenario_gbsm import generate_scenario_gbsm_frequency_response


def _topology(batch_size: int = 1, *, indoor: bool = False) -> dict[str, np.ndarray]:
    return {
        "ut_loc": np.tile(np.asarray([[[50.0, 0.0, 1.5]]], dtype=np.float32), (batch_size, 1, 1)),
        "bs_loc": np.tile(np.asarray([[[0.0, 0.0, 10.0]]], dtype=np.float32), (batch_size, 1, 1)),
        "ut_velocities": np.tile(
            np.asarray([[[1.0, 0.0, 0.0]]], dtype=np.float32), (batch_size, 1, 1)
        ),
        "ut_orientations": np.zeros((batch_size, 1, 3), dtype=np.float32),
        "bs_orientations": np.zeros((batch_size, 1, 3), dtype=np.float32),
        "in_state": np.full((batch_size, 1), indoor, dtype=np.bool_),
        "los": np.ones((batch_size, 1, 1), dtype=np.bool_),
    }


@pytest.mark.parametrize("scenario", ["UMi", "UMa", "RMa", "InH", "InF"])
def test_scenario_gbsm_returns_explicit_one_link_path_and_cfr_contract(scenario: str) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("sionna")
    options = {"o2i_model": "low"} if scenario in {"UMi", "UMa"} else None

    channel = generate_scenario_gbsm_frequency_response(
        scenario=scenario,
        seed=37,
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=30e3,
        num_subcarriers=12,
        num_time_steps=2,
        sampling_frequency_hz=30e3,
        topology=_topology(indoor=scenario in {"InH", "InF"}),
        scenario_options=options,
    )

    assert channel.h_path.ndim == 7
    assert channel.tau_s.ndim == 4
    assert channel.channel_frequency_response.shape == (1, 1, 1, 1, 1, 2, 12)
    assert channel.metadata["scenario"] == scenario
    assert channel.metadata["geometric_los_label"] is None
    assert channel.metadata["scenario_los_rows"] == [True]
    assert len(channel.metadata["topology_sha256"]) == 64
    assert len(channel.metadata["configuration_sha256"]) == 64
    assert channel.metadata["status"] == (
        "NATIVE_GENERATION_CODE_ONLY_NOT_CONFORMANCE_QUALIFICATION"
    )
    import torch

    frequencies = (
        torch.arange(12, dtype=channel.tau_s.dtype) - 6
    ) * 30e3
    expected_cfr = (
        channel.h_path.movedim(-2, -1).unsqueeze(-1)
        * torch.exp(
            -2j
            * torch.pi
            * channel.tau_s[:, :, None, :, None, None, :].unsqueeze(-1)
            * frequencies.reshape(1, 1, 1, 1, 1, 1, 1, -1)
        )
    ).sum(dim=-2)
    torch.testing.assert_close(channel.channel_frequency_response, expected_cfr)


def test_scenario_gbsm_rejects_multilink_topology_before_loading_sionna() -> None:
    topology = _topology()
    topology["bs_loc"] = np.zeros((1, 2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match=r"shape \[positive batch, 1, 3\]"):
        generate_scenario_gbsm_frequency_response(
            scenario="UMi",
            seed=1,
            carrier_frequency_hz=3.5e9,
            subcarrier_spacing_hz=30e3,
            num_subcarriers=12,
            topology=topology,
            scenario_options={"o2i_model": "low"},
        )


@pytest.mark.parametrize(
    ("field", "replacement", "error", "message"),
    [
        ("ut_loc", np.full((1, 1, 3), np.nan), ValueError, "finite values"),
        ("ut_velocities", np.zeros((1, 1, 3), dtype=np.complex64), TypeError, "real numeric"),
        ("in_state", np.zeros((1, 1, 1), dtype=np.bool_), ValueError, r"shape \[1, 1\]"),
        (
            "ut_orientations",
            np.zeros((1, 2, 3), dtype=np.float32),
            ValueError,
            r"shape \[batch, 1, 3\]",
        ),
        ("los", np.ones((1, 1, 1), dtype=np.int8), TypeError, "Boolean dtype"),
    ],
)
def test_scenario_gbsm_rejects_malformed_numpy_topology_before_loading_torch(
    field: str,
    replacement: np.ndarray,
    error: type[Exception],
    message: str,
) -> None:
    topology = _topology()
    topology[field] = replacement

    with pytest.raises(error, match=message):
        generate_scenario_gbsm_frequency_response(
            scenario="UMi",
            seed=1,
            carrier_frequency_hz=3.5e9,
            subcarrier_spacing_hz=30e3,
            num_subcarriers=12,
            topology=topology,
            scenario_options={"o2i_model": "low"},
        )
