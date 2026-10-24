"""Independent joint transmit/receive spatial covariance checks for CDL-A through E."""

from __future__ import annotations

import hashlib
import math
from statistics import NormalDist
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna.phy")

from integrations.sionna.cdl import CDLArrayConfig, generate_cdl_frequency_response  # noqa: E402
from sionna.phy.channel.tr38901 import models as profile_resources  # noqa: E402

_BATCH_SIZE = 16384
_FAMILYWISE_ALPHA = 0.01
_MAX_INTERVAL_HALF_WIDTH = 0.05
_TX_RX_TOPOLOGIES = ((2, 2), (4, 4), (1, 4), (4, 1))
_CDL_TABLE_SHA256 = {
    "A": "b96f8a179646ee468717c615e8061103d1e66eb1ff1ea78301de243740254cb7",
    "B": "7c87c092cb2bb44891d301ee3c1991cdda1c6b64094a03f18746ec5fa14ee2a2",
    "C": "f2044ecc3b72a1967221805ba70b1f0dab9634bb831478c1337b737ced6e4a28",
    "D": "a8960f176347de457c0cb90fbc8cf137ac98bb1c5295b096f004d8d5c6e40176",
    "E": "04328f33d8d79b24aa3d0303fecb58b465ed7e13874930b05e76be0d97743480",
}
_PROFILE_COUNT = 5
_RAY_ANGLE_OFFSETS = (
    0.0447,
    -0.0447,
    0.1413,
    -0.1413,
    0.2492,
    -0.2492,
    0.3715,
    -0.3715,
    0.5129,
    -0.5129,
    0.6797,
    -0.6797,
    0.8844,
    -0.8844,
    1.1481,
    -1.1481,
    1.5195,
    -1.5195,
    2.1551,
    -2.1551,
)


def _unique_hermitian_components(matrix: Any) -> Any:
    components = [matrix[index, index].real for index in range(matrix.shape[0])]
    for row in range(matrix.shape[0]):
        for column in range(row + 1, matrix.shape[0]):
            components.extend((matrix[row, column].real, matrix[row, column].imag))
    return torch.stack(components).to(torch.float64)


def _one_end_covariance(
    azimuth_deg: Any,
    zenith_deg: Any,
    azimuth_spread_deg: float,
    zenith_spread_deg: float,
    offsets: Any,
    antenna_count: int,
) -> Any:
    azimuth = torch.deg2rad(azimuth_deg[:, None] + azimuth_spread_deg * offsets)
    zenith = torch.deg2rad(zenith_deg[:, None] + zenith_spread_deg * offsets)
    projection = torch.sin(zenith)[:, :, None] * torch.sin(azimuth)[:, None, :]
    indices = torch.arange(antenna_count, dtype=torch.float64)
    separation = indices[:, None] - indices[None, :]
    return torch.exp(1j * math.pi * projection[..., None, None] * separation).mean(dim=(1, 2))


def _independent_joint_covariance_reference(
    model: str, tx_count: int, rx_count: int
) -> tuple[Any, str]:
    table_path = profile_resources.parameter_file(f"CDL-{model}.json", "19.2")
    table_bytes = table_path.read_bytes()
    digest = hashlib.sha256(table_bytes).hexdigest()
    assert digest == _CDL_TABLE_SHA256[model]
    parameters = profile_resources.load_json(table_path)
    is_los = bool(parameters["los"])

    powers = torch.as_tensor(
        [10.0 ** (float(power_db) / 10.0) for power_db in parameters["powers"]],
        dtype=torch.float64,
    )
    powers = powers / powers.sum()
    offsets = torch.as_tensor(_RAY_ANGLE_OFFSETS, dtype=torch.float64)
    diffuse_start = 1 if is_los else 0

    port_count = tx_count * rx_count
    joint = torch.zeros((port_count, port_count), dtype=torch.complex128)
    if is_los:
        los_angles = {
            name: math.radians(float(parameters[name][0])) for name in ("aod", "aoa", "zod", "zoa")
        }
        rx_projection = math.sin(los_angles["zoa"]) * math.sin(los_angles["aoa"])
        tx_projection = math.sin(los_angles["zod"]) * math.sin(los_angles["aod"])
        rx_indices = torch.arange(rx_count, dtype=torch.float64)
        tx_indices = torch.arange(tx_count, dtype=torch.float64)
        rx_steering = torch.exp(1j * math.pi * rx_projection * rx_indices)
        tx_steering = torch.exp(1j * math.pi * tx_projection * tx_indices)
        los_vector = (rx_steering[:, None] * tx_steering[None, :]).reshape(port_count)
        joint += powers[0] * los_vector[:, None] * los_vector[None, :].conj()

    receive = _one_end_covariance(
        torch.as_tensor(parameters["aoa"][diffuse_start:], dtype=torch.float64),
        torch.as_tensor(parameters["zoa"][diffuse_start:], dtype=torch.float64),
        float(parameters["cASA"]),
        float(parameters["cZSA"]),
        offsets,
        rx_count,
    )
    transmit = _one_end_covariance(
        torch.as_tensor(parameters["aod"][diffuse_start:], dtype=torch.float64),
        torch.as_tensor(parameters["zod"][diffuse_start:], dtype=torch.float64),
        float(parameters["cASD"]),
        float(parameters["cZSD"]),
        offsets,
        tx_count,
    )

    # For zero-oriented omni/V panels, diffuse field gain is one. Independent
    # azimuth/zenith coupling makes the endpoint ray-angle averages separable.
    # In LOS profiles, the specular steering outer product above is kept
    # distinct from the remaining diffuse cluster terms.
    for rx in range(rx_count):
        for tx in range(tx_count):
            for rx_prime in range(rx_count):
                for tx_prime in range(tx_count):
                    row = tx_count * rx + tx
                    column = tx_count * rx_prime + tx_prime
                    joint[row, column] += torch.sum(
                        powers[diffuse_start:]
                        * receive[:, rx, rx_prime]
                        * transmit[:, tx, tx_prime]
                    )
    return joint, digest


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("model", "seed", "tx_count", "rx_count"),
    [
        (model, 6721 + index, tx_count, rx_count)
        for index, model in enumerate("ABCDE")
        for tx_count, rx_count in _TX_RX_TOPOLOGIES
    ],
)
def test_cdl_joint_tx_rx_spatial_covariance_matches_independent_ray_reference(
    model: str, seed: int, tx_count: int, rx_count: int
) -> None:
    expected, source_digest = _independent_joint_covariance_reference(model, tx_count, rx_count)
    channel = generate_cdl_frequency_response(
        model=model,
        seed=seed,
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=_BATCH_SIZE,
        delay_spread_s=100e-9,
        ut_array=CDLArrayConfig(num_cols_per_panel=rx_count, element_horizontal_spacing=0.5),
        bs_array=CDLArrayConfig(num_cols_per_panel=tx_count, element_horizontal_spacing=0.5),
        ut_orientation_rad=(0.0, 0.0, 0.0),
        bs_orientation_rad=(0.0, 0.0, 0.0),
        spec_version="19.2",
        device="cpu",
    )
    samples = (
        channel.h_path[:, 0, :, 0, :, :, 0].sum(dim=-1).reshape(_BATCH_SIZE, tx_count * rx_count)
    )
    outer_products = samples[:, :, None] * samples[:, None, :].conj()
    port_count = tx_count * rx_count
    observations = torch.stack(
        [outer_products[:, index, index].real for index in range(port_count)]
        + [
            value
            for row in range(port_count)
            for column in range(row + 1, port_count)
            for value in (outer_products[:, row, column].real, outer_products[:, row, column].imag)
        ],
        dim=1,
    )
    estimate = observations.mean(dim=0)
    expected_components = _unique_hermitian_components(expected)
    standard_error = observations.std(dim=0, unbiased=True) / math.sqrt(_BATCH_SIZE)
    # One family covers all profiles, four topologies and every unique
    # Hermitian component across 2x2, 4x4, 1x4 and 4x1 links.
    comparison_count = _PROFILE_COUNT * sum((tx * rx) ** 2 for tx, rx in _TX_RX_TOPOLOGIES)
    z_simultaneous = NormalDist().inv_cdf(1.0 - _FAMILYWISE_ALPHA / (2.0 * comparison_count))
    half_width = z_simultaneous * standard_error

    assert tuple(samples.shape) == (_BATCH_SIZE, tx_count * rx_count)
    assert expected_components.numel() == (tx_count * rx_count) ** 2
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    assert channel.metadata["profile_model"] == model
    assert channel.metadata["profile_los_component"] is (model in {"D", "E"})
    assert channel.metadata["parameter_table_sha256"] == source_digest
    assert channel.metadata["rx_antenna_count"] == rx_count
    assert channel.metadata["tx_antenna_count"] == tx_count
    assert bool((torch.abs(estimate - expected_components) <= half_width).all())
    assert float(half_width.max()) <= _MAX_INTERVAL_HALF_WIDTH
