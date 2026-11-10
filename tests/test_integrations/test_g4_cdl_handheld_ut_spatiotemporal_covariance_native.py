"""Frozen native physical-HandheldUTArray CDL space-time covariance family."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import pytest
from evaluation.g4_cdl_handheld_ut_spatiotemporal_covariance import (
    cdl_handheld_ut_spatiotemporal_covariance_reference,
)

torch = pytest.importorskip("torch")
pytest.importorskip("sionna.phy")

import sionna.phy.channel.tr38901.antenna as sionna_antenna  # noqa: E402
import sionna.phy.channel.tr38901.cdl as sionna_cdl  # noqa: E402
import sionna.phy.channel.tr38901.channel_coefficients as sionna_coefficients  # noqa: E402
from integrations.sionna.cdl import (  # noqa: E402
    CDLArrayConfig,
    CDLHandheldUTArrayConfig,
    generate_cdl_frequency_response,
)
from sionna.phy.channel.tr38901 import models as profile_resources  # noqa: E402

_MODELS = tuple("ABCDE")
_ROSTERS = ("tr38901-4", "tr38901")
_PORT_COUNTS = {"tr38901-4": 4, "tr38901": 8}
_UT_ORIENTATION = (0.2, -0.1, 0.3)
_BS_ORIENTATION = (0.0, 0.0, 0.0)
_VELOCITY = (10.0, 20.0, 20.0)
_SEED_START = 2_819_000
_BATCH_SIZE = 16_384
_CARRIER_FREQUENCY_HZ = 3.5e9
_LAG_S = 0.7e-3
_DEVICE_DEPTH_M = 0.15
_DEVICE_WIDTH_M = 0.07
_FAMILYWISE_ALPHA = 0.01
_CLAIM_COUNT = len(_MODELS) * sum(
    _PORT_COUNTS[roster] * _PORT_COUNTS[roster] * 2 for roster in _ROSTERS
)
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
_SIONNA_ANTENNA_SHA256 = "09e20b89f4c4d990bed1a3ce3d6314f9fe7e45c090737a6f5790166d4a63f926"
_SIONNA_COEFFICIENTS_SHA256 = "47059c7bf0998d2a230909e70743832ee928a738a97b9b5a25a185bb4947b5fc"
_SIONNA_CDL_SHA256 = "91e90eb7498a3725f628f2f4d4dca6c8d7438ac549bd5d68a3f45563f282361d"


def _module_sha256(module: Any) -> str:
    return hashlib.sha256(Path(module.__file__).resolve().read_bytes()).hexdigest()


def _seed_for(model: str, roster_index: int) -> int:
    return _SEED_START + _MODELS.index(model) * len(_ROSTERS) + roster_index


def _claims(
    estimate: Any,
    reference: tuple[tuple[complex, ...], ...],
    half_width_real: Any,
    half_width_imag: Any,
    port_count: int,
) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for row in range(port_count):
        for column in range(port_count):
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
                claims.append(
                    {
                        "row": row,
                        "column": column,
                        "component": component,
                        "estimate": observed,
                        "reference": float(target),
                        "absolute_error": error,
                        "half_width": half_width,
                        "passed": (error <= half_width and half_width <= _MAX_INTERVAL_HALF_WIDTH),
                    }
                )
    return claims


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("roster_index", "roster"),
    tuple(enumerate(_ROSTERS)),
)
@pytest.mark.parametrize("model", _MODELS)
def test_cdl_handheld_ut_spatiotemporal_covariance_matches_independent_reference(
    model: str,
    roster_index: int,
    roster: str,
    record_property: Any,
) -> None:
    assert hashlib.sha256(_REFERENCE_PATH.read_bytes()).hexdigest() == _REFERENCE_SHA256
    assert _module_sha256(sionna_antenna) == _SIONNA_ANTENNA_SHA256
    assert _module_sha256(sionna_coefficients) == _SIONNA_COEFFICIENTS_SHA256
    assert _module_sha256(sionna_cdl) == _SIONNA_CDL_SHA256

    port_count = _PORT_COUNTS[roster]
    expected = cdl_handheld_ut_spatiotemporal_covariance_reference(
        model,
        _VELOCITY,
        antenna_locations=roster,
        ut_orientation_rad=_UT_ORIENTATION,
        carrier_frequency_hz=_CARRIER_FREQUENCY_HZ,
        lag_s=_LAG_S,
        device_depth_m=_DEVICE_DEPTH_M,
        device_width_m=_DEVICE_WIDTH_M,
    )
    table_path = profile_resources.parameter_file(f"CDL-{model}.json", "19.2")
    table_digest = hashlib.sha256(table_path.read_bytes()).hexdigest()
    assert table_digest == _PROFILE_SHA256[model]
    seed = _seed_for(model, roster_index)

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
        ut_array=CDLHandheldUTArrayConfig(
            polarization="single",
            polarization_type="V",
            antenna_locations=roster,
            antenna_pattern="38.901-handheld",
            device_depth=_DEVICE_DEPTH_M,
            device_width=_DEVICE_WIDTH_M,
        ),
        bs_array=CDLArrayConfig(
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
        ),
        ut_orientation_rad=_UT_ORIENTATION,
        bs_orientation_rad=_BS_ORIENTATION,
        spec_version="19.2",
        device="cpu",
    )

    path = channel.h_path[:, 0, :, 0, 0, :, :].to(torch.complex128)
    assert tuple(path.shape[1:]) == (port_count, path.shape[2], 2)
    samples = path.sum(dim=2)
    h0 = samples[:, :, 0]
    h1 = samples[:, :, 1]
    cross = h0.conj()[:, :, None] * h1[:, None, :]
    estimate = cross.mean(dim=0)
    standard_error_real = cross.real.std(dim=0, unbiased=True) / math.sqrt(_BATCH_SIZE)
    standard_error_imag = cross.imag.std(dim=0, unbiased=True) / math.sqrt(_BATCH_SIZE)
    z_simultaneous = NormalDist().inv_cdf(1.0 - _FAMILYWISE_ALPHA / (2.0 * _CLAIM_COUNT))
    half_width_real = z_simultaneous * standard_error_real
    half_width_imag = z_simultaneous * standard_error_imag
    claims = _claims(
        estimate,
        expected,
        half_width_real,
        half_width_imag,
        port_count,
    )

    assert channel.metadata["profile_model"] == model
    assert channel.metadata["parameter_table_sha256"] == table_digest
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    assert channel.metadata["ut_velocity_mps"] == list(_VELOCITY)
    assert channel.metadata["ut_orientation_rad"] == list(_UT_ORIENTATION)
    assert channel.metadata["bs_orientation_rad"] == list(_BS_ORIENTATION)
    assert channel.metadata["ut_array_kind"] == "handheld_ut"
    assert channel.metadata["bs_array_kind"] == "panel"
    assert channel.metadata["rx_antenna_count"] == port_count
    assert channel.metadata["tx_antenna_count"] == 1
    assert channel.metadata["ut_array"]["antenna_locations"] == roster
    assert channel.metadata["ut_array"]["antenna_pattern"] == "38.901-handheld"
    assert channel.metadata["ut_array"]["polarization"] == "single"
    assert channel.metadata["ut_array"]["polarization_type"] == "V"
    assert channel.metadata["bs_array"]["antenna_pattern"] == "omni"

    record_property(
        "g4_cdl_handheld_ut_spatiotemporal_covariance_case",
        json.dumps(
            {
                "model": model,
                "roster_index": roster_index,
                "roster": roster,
                "seed": seed,
                "velocity_mps": list(_VELOCITY),
                "ut_orientation_rad": list(_UT_ORIENTATION),
                "bs_orientation_rad": list(_BS_ORIENTATION),
                "draw_count": _BATCH_SIZE,
                "port_count": port_count,
                "family_claim_count": _CLAIM_COUNT,
                "familywise_alpha": _FAMILYWISE_ALPHA,
                "device_depth_m": _DEVICE_DEPTH_M,
                "device_width_m": _DEVICE_WIDTH_M,
                "maximum_half_width": max(claim["half_width"] for claim in claims),
                "parameter_table_sha256": table_digest,
                "batch_axis_semantics": channel.metadata["batch_axis_semantics"],
                "claims": claims,
            },
            sort_keys=True,
            allow_nan=False,
        ),
    )
    assert all(claim["passed"] for claim in claims), (model, roster, claims)
