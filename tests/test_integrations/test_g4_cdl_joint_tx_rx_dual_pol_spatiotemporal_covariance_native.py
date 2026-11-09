"""Frozen native CDL joint TX/RX dual-VH receive space-time covariance family."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import pytest
from evaluation.g4_cdl_joint_tx_rx_dual_pol_spatiotemporal_covariance import (
    cdl_joint_tx_rx_dual_pol_spatiotemporal_covariance_reference,
)

torch = pytest.importorskip("torch")
pytest.importorskip("sionna.phy")

from integrations.sionna.cdl import CDLArrayConfig, generate_cdl_frequency_response  # noqa: E402
from sionna.phy.channel.tr38901 import models as profile_resources  # noqa: E402

_MODELS = tuple("ABCDE")
_VELOCITY = (10.0, 20.0, 20.0)
_SEED_START = 2_812_000
_BATCH_SIZE = 16_384
_CARRIER_FREQUENCY_HZ = 3.5e9
_LAG_S = 0.7e-3
_TX_ELEMENT_COUNT = 2
_RX_ELEMENT_COUNT = 2
_RX_PORT_COUNT = 2 * _RX_ELEMENT_COUNT
_CHANNEL_PORT_COUNT = _TX_ELEMENT_COUNT * _RX_PORT_COUNT
_FAMILYWISE_ALPHA = 0.01
_CLAIM_COUNT = len(_MODELS) * _CHANNEL_PORT_COUNT * _CHANNEL_PORT_COUNT * 2
_MAX_INTERVAL_HALF_WIDTH = 0.05
_PROFILE_SHA256 = {
    "A": "b96f8a179646ee468717c615e8061103d1e66eb1ff1ea78301de243740254cb7",
    "B": "7c87c092cb2bb44891d301ee3c1991cdda1c6b64094a03f18746ec5fa14ee2a2",
    "C": "f2044ecc3b72a1967221805ba70b1f0dab9634bb831478c1337b737ced6e4a28",
    "D": "a8960f176347de457c0cb90fbc8cf137ac98bb1c5295b096f004d8d5c6e40176",
    "E": "04328f33d8d79b24aa3d0303fecb58b465ed7e13874930b05e76be0d97743480",
}
_REFERENCE_PATH = (
    Path(__file__).resolve().parents[2] / "docs/research/G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json"
)
_REFERENCE_SHA256 = "24ee2f8f73f26e27522a296a4a0860bdaa7f284e9138855666e09636bc97b090"


def _claims(
    estimate: Any,
    reference: tuple[tuple[complex, ...], ...],
    half_width_real: Any,
    half_width_imag: Any,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in range(_CHANNEL_PORT_COUNT):
        for column in range(_CHANNEL_PORT_COUNT):
            expected = reference[row][column]
            for component, observed, target, half_width in (
                (
                    "real",
                    float(estimate[row, column].real),
                    expected.real,
                    float(half_width_real[row, column]),
                ),
                (
                    "imag",
                    float(estimate[row, column].imag),
                    expected.imag,
                    float(half_width_imag[row, column]),
                ),
            ):
                error = abs(observed - target)
                result.append(
                    {
                        "row": row,
                        "column": column,
                        "component": component,
                        "estimate": observed,
                        "reference": float(target),
                        "absolute_error": error,
                        "half_width": half_width,
                        "passed": error <= half_width and half_width <= _MAX_INTERVAL_HALF_WIDTH,
                    }
                )
    return result


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", _MODELS)
def test_cdl_joint_tx_rx_dual_pol_spatiotemporal_covariance_matches_independent_reference(
    model: str, record_property: Any
) -> None:
    assert hashlib.sha256(_REFERENCE_PATH.read_bytes()).hexdigest() == _REFERENCE_SHA256
    expected = cdl_joint_tx_rx_dual_pol_spatiotemporal_covariance_reference(
        model,
        _VELOCITY,
        carrier_frequency_hz=_CARRIER_FREQUENCY_HZ,
        lag_s=_LAG_S,
        tx_count=_TX_ELEMENT_COUNT,
        rx_element_count=_RX_ELEMENT_COUNT,
        element_spacing_wavelengths=0.5,
    )
    table_path = profile_resources.parameter_file(f"CDL-{model}.json", "19.2")
    table_digest = hashlib.sha256(table_path.read_bytes()).hexdigest()
    assert table_digest == _PROFILE_SHA256[model]
    seed = _SEED_START + _MODELS.index(model)

    channel = generate_cdl_frequency_response(
        model=model,
        seed=seed,
        carrier_frequency_hz=_CARRIER_FREQUENCY_HZ,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=_BATCH_SIZE,
        delay_spread_s=100e-9,
        num_time_steps=2,
        sampling_frequency_hz=1.0 / _LAG_S,
        ut_velocity_mps=_VELOCITY,
        ut_array=CDLArrayConfig(
            num_cols_per_panel=_RX_ELEMENT_COUNT,
            element_horizontal_spacing=0.5,
            polarization="dual",
            polarization_type="VH",
            antenna_pattern="omni",
        ),
        bs_array=CDLArrayConfig(
            num_cols_per_panel=_TX_ELEMENT_COUNT,
            element_horizontal_spacing=0.5,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
        ),
        ut_orientation_rad=(0.0, 0.0, 0.0),
        bs_orientation_rad=(0.0, 0.0, 0.0),
        spec_version="19.2",
        device="cpu",
    )

    path = channel.h_path[:, 0, :, 0, :, :, :].to(torch.complex128)
    samples = path.sum(dim=3).reshape(_BATCH_SIZE, _CHANNEL_PORT_COUNT, 2)
    assert tuple(samples.shape) == (_BATCH_SIZE, _CHANNEL_PORT_COUNT, 2)
    h0 = samples[:, :, 0]
    h1 = samples[:, :, 1]
    cross = h0.conj()[:, :, None] * h1[:, None, :]
    estimate = cross.mean(dim=0)
    standard_error_real = cross.real.std(dim=0, unbiased=True) / math.sqrt(_BATCH_SIZE)
    standard_error_imag = cross.imag.std(dim=0, unbiased=True) / math.sqrt(_BATCH_SIZE)
    z_simultaneous = NormalDist().inv_cdf(1.0 - _FAMILYWISE_ALPHA / (2.0 * _CLAIM_COUNT))
    half_width_real = z_simultaneous * standard_error_real
    half_width_imag = z_simultaneous * standard_error_imag
    claims = _claims(estimate, expected, half_width_real, half_width_imag)

    assert channel.metadata["profile_model"] == model
    assert channel.metadata["profile_los_component"] is (model in {"D", "E"})
    assert channel.metadata["parameter_table_sha256"] == table_digest
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    assert channel.metadata["ut_velocity_mps"] == list(_VELOCITY)
    assert channel.metadata["rx_antenna_count"] == _RX_PORT_COUNT
    assert channel.metadata["tx_antenna_count"] == _TX_ELEMENT_COUNT
    assert channel.metadata["ut_array"]["polarization"] == "dual"
    assert channel.metadata["ut_array"]["polarization_type"] == "VH"
    assert channel.metadata["bs_array"]["polarization"] == "single"
    assert channel.metadata["bs_array"]["polarization_type"] == "V"

    record_property(
        "g4_cdl_joint_tx_rx_dual_pol_spatiotemporal_covariance_case",
        json.dumps(
            {
                "model": model,
                "seed": seed,
                "velocity_mps": list(_VELOCITY),
                "draw_count": _BATCH_SIZE,
                "tx_element_count": _TX_ELEMENT_COUNT,
                "rx_element_count": _RX_ELEMENT_COUNT,
                "rx_port_count": _RX_PORT_COUNT,
                "channel_port_count": _CHANNEL_PORT_COUNT,
                "family_claim_count": _CLAIM_COUNT,
                "familywise_alpha": _FAMILYWISE_ALPHA,
                "maximum_half_width": max(claim["half_width"] for claim in claims),
                "parameter_table_sha256": table_digest,
                "batch_axis_semantics": channel.metadata["batch_axis_semantics"],
                "claims": claims,
            },
            sort_keys=True,
            allow_nan=False,
        ),
    )
    assert all(claim["passed"] for claim in claims), (model, claims)
