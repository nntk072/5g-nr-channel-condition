"""Continuous-valued four-angle CDL dependence against ETSI profile references."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna.phy")

from integrations.sionna.random import call_with_sionna_seed  # noqa: E402
from sionna.phy.channel.tr38901 import models as profile_resources  # noqa: E402

_BATCH_SIZE = 8192
_SEED = 239_700
_EXTENDED_SEED = 240_000
_BOUNDED_JOINT_SEED = 260_000
_ALPHA = 0.01
_MAX_SIMULTANEOUS_HALF_WIDTH = 0.05
_MODELS = ("A", "B", "C", "D", "E")
_ANGLE_FIELDS = ("aoa", "aod", "zoa", "zod")
_SPREAD_FIELDS = {"aoa": "cASA", "aod": "cASD", "zoa": "cZSA", "zod": "cZSD"}
_PROFILE_SHA256 = {
    "A": "b96f8a179646ee468717c615e8061103d1e66eb1ff1ea78301de243740254cb7",
    "B": "7c87c092cb2bb44891d301ee3c1991cdda1c6b64094a03f18746ec5fa14ee2a2",
    "C": "f2044ecc3b72a1967221805ba70b1f0dab9634bb831478c1337b737ced6e4a28",
    "D": "a8960f176347de457c0cb90fbc8cf137ac98bb1c5295b096f004d8d5c6e40176",
    "E": "04328f33d8d79b24aa3d0303fecb58b465ed7e13874930b05e76be0d97743480",
}
_CDL_ETSI_REFERENCE_PATH = (
    Path(__file__).resolve().parents[2] / "docs/research/G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json"
)
_RAY_OFFSETS_DEG = torch.tensor(
    (
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
    ),
    dtype=torch.float64,
)
_SIGN_TRIPLES = tuple(itertools.product((-1, 1), repeat=3))
_FOURIER_VECTORS = tuple((harmonic, *signs) for harmonic in (1, 2) for signs in _SIGN_TRIPLES)
_TERNARY_FOURIER_VECTORS = tuple(
    vector
    for vector in itertools.product((-1, 0, 1), repeat=len(_ANGLE_FIELDS))
    if sum(frequency != 0 for frequency in vector) >= 2
)
_MARGINAL_FOURIER_VECTORS = tuple(
    tuple(harmonic if index == angle else 0 for index in range(len(_ANGLE_FIELDS)))
    for angle in range(len(_ANGLE_FIELDS))
    for harmonic in (1, 2)
)
_BOUNDED_JOINT_FOURIER_VECTORS = tuple(
    vector
    for vector in itertools.product((-2, -1, 0, 1, 2), repeat=len(_ANGLE_FIELDS))
    if sum(frequency != 0 for frequency in vector) >= 2
    and next(frequency for frequency in vector if frequency != 0) > 0
)
_CLAIM_COUNT = len(_MODELS) * len(_FOURIER_VECTORS) * 2
_TERNARY_CLAIM_COUNT = len(_MODELS) * len(_TERNARY_FOURIER_VECTORS) * 2
_MARGINAL_CLAIM_COUNT = len(_MODELS) * len(_MARGINAL_FOURIER_VECTORS) * 2
_BOUNDED_JOINT_CLAIM_COUNT = len(_MODELS) * len(_BOUNDED_JOINT_FOURIER_VECTORS) * 2
_JOINT_DELAY_ANGLE_SEED = 272_000
_DELAY_SPREAD_S = 100e-9
_JOINT_DELAY_ANGLE_VECTORS = tuple(
    (delay_harmonic, angle_vector)
    for delay_harmonic in (1, 2)
    for angle_vector in (
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (1, 1, 0, 0),
        (1, -1, 1, 0),
        (1, 1, 1, 1),
    )
)
_JOINT_DELAY_ANGLE_CLAIM_COUNT = len(_MODELS) * len(_JOINT_DELAY_ANGLE_VECTORS) * 2


def _independent_profile_parameters(model: str) -> dict[str, Any]:
    """Build expected model parameters from pinned ETSI table transcription."""
    reference = json.loads(_CDL_ETSI_REFERENCE_PATH.read_text(encoding="utf-8"))
    if reference["source_id"] != "tr-138901-v19.2.0":
        raise ValueError("CDL independent reference is bound to an unexpected ETSI source")
    profile = next(row for row in reference["profiles"] if row["profile_model"] == model)
    components = profile["components"]
    spreads = profile["cluster_spreads_deg"]
    return {
        "los": profile["los"],
        "powers": [row["power_db"] for row in components],
        "delays": [row["normalized_delay"] for row in components],
        "aod": [row["angles_deg"]["aod"] for row in components],
        "aoa": [row["angles_deg"]["aoa"] for row in components],
        "zod": [row["angles_deg"]["zod"] for row in components],
        "zoa": [row["angles_deg"]["zoa"] for row in components],
        "cASD": spreads["asd"],
        "cASA": spreads["asa"],
        "cZSD": spreads["zsd"],
        "cZSA": spreads["zsa"],
    }


def _profile_diffuse_reference(parameters: dict[str, Any]) -> tuple[Any, Any]:
    """Return normalized diffuse cluster weights and table-derived angle atoms."""
    raw_powers = torch.tensor(
        [10.0 ** (float(value) / 10.0) for value in parameters["powers"]],
        dtype=torch.float64,
    )
    is_los = bool(parameters["los"])
    start = int(is_los)
    delays = torch.tensor(parameters["delays"][start:], dtype=torch.float64)
    order = torch.argsort(delays)
    powers = raw_powers[start:][order]
    powers = powers / powers.sum()
    angles = torch.stack(
        [
            torch.tensor(parameters[name][start:], dtype=torch.float64)[order, None]
            + float(parameters[_SPREAD_FIELDS[name]]) * _RAY_OFFSETS_DEG[None, :]
            for name in _ANGLE_FIELDS
        ],
        dim=-1,
    )
    return powers, angles


def _native_diffuse_rays(model: str, seed: int = _SEED) -> Any:
    """Generate native CDL ray angles and powers on CPU for one profile."""
    from sionna.phy.channel.tr38901 import CDL, PanelArray

    array = PanelArray(
        num_rows_per_panel=1,
        num_cols_per_panel=1,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        num_rows=1,
        num_cols=1,
        carrier_frequency=3.5e9,
        device="cpu",
    )
    model_instance = CDL(
        model=model,
        delay_spread=100e-9,
        carrier_frequency=3.5e9,
        ut_array=array,
        bs_array=array,
        direction="downlink",
        device="cpu",
        spec_version="19.2",
    )
    return call_with_sionna_seed(
        "cpu", seed + ord(model), lambda: model_instance._create_rays(batch_size=_BATCH_SIZE)
    )


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", _MODELS)
def test_cdl_four_angle_characteristic_matches_continuous_table_reference(model: str) -> None:
    """Check continuous four-angle coupling with simultaneous draw-level intervals."""
    table_path = profile_resources.parameter_file(f"CDL-{model}.json", "19.2")
    table_bytes = table_path.read_bytes()
    assert hashlib.sha256(table_bytes).hexdigest() == _PROFILE_SHA256[model]
    parameters = _independent_profile_parameters(model)
    expected_powers, table_angles_deg = _profile_diffuse_reference(parameters)
    rays = _native_diffuse_rays(model)

    native_angles_rad = torch.stack(
        [getattr(rays, name)[:, 0, 0] for name in _ANGLE_FIELDS], dim=-1
    ).to(torch.float64)
    native_powers = rays.powers[:, 0, 0].to(torch.float64)
    expected_clusters = int(expected_powers.numel())
    assert native_angles_rad.shape == (_BATCH_SIZE, expected_clusters, 20, 4)
    assert native_powers.shape == (_BATCH_SIZE, expected_clusters)
    native_power_sums = native_powers.sum(dim=-1)
    torch.testing.assert_close(
        native_power_sums,
        torch.ones(_BATCH_SIZE, dtype=torch.float64),
        atol=2e-7,
        rtol=0.0,
    )
    native_powers = native_powers / native_power_sums[:, None]

    z_simultaneous = NormalDist().inv_cdf(1.0 - _ALPHA / (2.0 * _CLAIM_COUNT))
    for vector in _FOURIER_VECTORS:
        frequency = torch.tensor(vector, dtype=torch.float64)
        table_cluster_moments = torch.stack(
            [
                torch.exp(
                    1j * math.pi / 180.0 * table_angles_deg[..., index] * frequency[index]
                ).mean(dim=-1)
                for index in range(len(_ANGLE_FIELDS))
            ],
            dim=-1,
        ).prod(dim=-1)
        expected = torch.sum(expected_powers * table_cluster_moments)

        native_phase = torch.sum(native_angles_rad * frequency[None, None, None, :], dim=-1)
        native_cluster_moments = torch.exp(1j * native_phase).mean(dim=-1)
        draw_moments = torch.sum(native_powers * native_cluster_moments, dim=-1)
        estimate = draw_moments.mean()
        real_se = draw_moments.real.std(unbiased=True) / math.sqrt(_BATCH_SIZE)
        imag_se = draw_moments.imag.std(unbiased=True) / math.sqrt(_BATCH_SIZE)
        real_half_width = z_simultaneous * real_se
        imag_half_width = z_simultaneous * imag_se

        assert abs(float(estimate.real - expected.real)) <= float(real_half_width) + 1e-12
        assert abs(float(estimate.imag - expected.imag)) <= float(imag_half_width) + 1e-12
        assert max(float(real_half_width), float(imag_half_width)) <= _MAX_SIMULTANEOUS_HALF_WIDTH


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", _MODELS)
def test_cdl_four_angle_ternary_fourier_family_matches_table_reference(model: str) -> None:
    """Check the frozen 72-vector joint characteristic family on fresh draws."""
    table_path = profile_resources.parameter_file(f"CDL-{model}.json", "19.2")
    table_bytes = table_path.read_bytes()
    assert hashlib.sha256(table_bytes).hexdigest() == _PROFILE_SHA256[model]
    parameters = _independent_profile_parameters(model)
    expected_powers, table_angles_deg = _profile_diffuse_reference(parameters)
    rays = _native_diffuse_rays(model, seed=_EXTENDED_SEED)

    native_angles_rad = torch.stack(
        [getattr(rays, name)[:, 0, 0] for name in _ANGLE_FIELDS], dim=-1
    ).to(torch.float64)
    native_powers = rays.powers[:, 0, 0].to(torch.float64)
    expected_clusters = int(expected_powers.numel())
    assert native_angles_rad.shape == (_BATCH_SIZE, expected_clusters, 20, 4)
    assert native_powers.shape == (_BATCH_SIZE, expected_clusters)
    native_power_sums = native_powers.sum(dim=-1)
    torch.testing.assert_close(
        native_power_sums,
        torch.ones(_BATCH_SIZE, dtype=torch.float64),
        atol=2e-7,
        rtol=0.0,
    )
    native_powers = native_powers / native_power_sums[:, None]

    z_simultaneous = NormalDist().inv_cdf(1.0 - _ALPHA / (2.0 * _TERNARY_CLAIM_COUNT))
    for vector in _TERNARY_FOURIER_VECTORS:
        frequency = torch.tensor(vector, dtype=torch.float64)
        table_cluster_moments = torch.stack(
            [
                torch.exp(
                    1j * math.pi / 180.0 * table_angles_deg[..., index] * frequency[index]
                ).mean(dim=-1)
                for index in range(len(_ANGLE_FIELDS))
            ],
            dim=-1,
        ).prod(dim=-1)
        expected = torch.sum(expected_powers * table_cluster_moments)

        native_phase = torch.sum(native_angles_rad * frequency[None, None, None, :], dim=-1)
        native_cluster_moments = torch.exp(1j * native_phase).mean(dim=-1)
        draw_moments = torch.sum(native_powers * native_cluster_moments, dim=-1)
        estimate = draw_moments.mean()
        real_half_width = (
            z_simultaneous * draw_moments.real.std(unbiased=True) / math.sqrt(_BATCH_SIZE)
        )
        imag_half_width = (
            z_simultaneous * draw_moments.imag.std(unbiased=True) / math.sqrt(_BATCH_SIZE)
        )

        assert abs(float(estimate.real - expected.real)) <= float(real_half_width) + 1e-12
        assert abs(float(estimate.imag - expected.imag)) <= float(imag_half_width) + 1e-12
        assert max(float(real_half_width), float(imag_half_width)) <= _MAX_SIMULTANEOUS_HALF_WIDTH


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", _MODELS)
def test_cdl_single_angle_fourier_marginals_match_table_reference(model: str) -> None:
    """Check first/second Fourier harmonics of each marginal on reserved fresh draws."""
    table_path = profile_resources.parameter_file(f"CDL-{model}.json", "19.2")
    assert hashlib.sha256(table_path.read_bytes()).hexdigest() == _PROFILE_SHA256[model]
    parameters = _independent_profile_parameters(model)
    expected_powers, table_angles_deg = _profile_diffuse_reference(parameters)
    rays = _native_diffuse_rays(model, seed=250_000)

    native_angles_rad = torch.stack(
        [getattr(rays, name)[:, 0, 0] for name in _ANGLE_FIELDS], dim=-1
    ).to(torch.float64)
    native_powers = rays.powers[:, 0, 0].to(torch.float64)
    expected_clusters = int(expected_powers.numel())
    assert native_angles_rad.shape == (_BATCH_SIZE, expected_clusters, 20, 4)
    assert native_powers.shape == (_BATCH_SIZE, expected_clusters)
    native_power_sums = native_powers.sum(dim=-1)
    torch.testing.assert_close(
        native_power_sums,
        torch.ones(_BATCH_SIZE, dtype=torch.float64),
        atol=2e-7,
        rtol=0.0,
    )
    native_powers = native_powers / native_power_sums[:, None]

    z_simultaneous = NormalDist().inv_cdf(1.0 - _ALPHA / (2.0 * _MARGINAL_CLAIM_COUNT))
    for vector in _MARGINAL_FOURIER_VECTORS:
        frequency = torch.tensor(vector, dtype=torch.float64)
        table_cluster_moments = torch.stack(
            [
                torch.exp(
                    1j * math.pi / 180.0 * table_angles_deg[..., index] * frequency[index]
                ).mean(dim=-1)
                for index in range(len(_ANGLE_FIELDS))
            ],
            dim=-1,
        ).prod(dim=-1)
        expected = torch.sum(expected_powers * table_cluster_moments)

        native_phase = torch.sum(native_angles_rad * frequency[None, None, None, :], dim=-1)
        native_cluster_moments = torch.exp(1j * native_phase).mean(dim=-1)
        draw_moments = torch.sum(native_powers * native_cluster_moments, dim=-1)
        estimate = draw_moments.mean()
        real_half_width = (
            z_simultaneous * draw_moments.real.std(unbiased=True) / math.sqrt(_BATCH_SIZE)
        )
        imag_half_width = (
            z_simultaneous * draw_moments.imag.std(unbiased=True) / math.sqrt(_BATCH_SIZE)
        )

        assert abs(float(estimate.real - expected.real)) <= float(real_half_width) + 1e-12
        assert abs(float(estimate.imag - expected.imag)) <= float(imag_half_width) + 1e-12
        assert max(float(real_half_width), float(imag_half_width)) <= _MAX_SIMULTANEOUS_HALF_WIDTH


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", _MODELS)
def test_cdl_four_angle_bounded_joint_fourier_family_matches_table_reference(model: str) -> None:
    """Check one vector from each conjugate pair in the bounded joint domain."""
    assert len(_BOUNDED_JOINT_FOURIER_VECTORS) == 304
    table_path = profile_resources.parameter_file(f"CDL-{model}.json", "19.2")
    assert hashlib.sha256(table_path.read_bytes()).hexdigest() == _PROFILE_SHA256[model]
    parameters = _independent_profile_parameters(model)
    expected_powers, table_angles_deg = _profile_diffuse_reference(parameters)
    rays = _native_diffuse_rays(model, seed=_BOUNDED_JOINT_SEED)

    native_angles_rad = torch.stack(
        [getattr(rays, name)[:, 0, 0] for name in _ANGLE_FIELDS], dim=-1
    ).to(torch.float64)
    native_powers = rays.powers[:, 0, 0].to(torch.float64)
    expected_clusters = int(expected_powers.numel())
    assert native_angles_rad.shape == (_BATCH_SIZE, expected_clusters, 20, 4)
    assert native_powers.shape == (_BATCH_SIZE, expected_clusters)
    native_power_sums = native_powers.sum(dim=-1)
    torch.testing.assert_close(
        native_power_sums,
        torch.ones(_BATCH_SIZE, dtype=torch.float64),
        atol=2e-7,
        rtol=0.0,
    )
    native_powers = native_powers / native_power_sums[:, None]

    z_simultaneous = NormalDist().inv_cdf(1.0 - _ALPHA / (2.0 * _BOUNDED_JOINT_CLAIM_COUNT))
    for vector in _BOUNDED_JOINT_FOURIER_VECTORS:
        frequency = torch.tensor(vector, dtype=torch.float64)
        table_cluster_moments = torch.stack(
            [
                torch.exp(
                    1j * math.pi / 180.0 * table_angles_deg[..., index] * frequency[index]
                ).mean(dim=-1)
                for index in range(len(_ANGLE_FIELDS))
            ],
            dim=-1,
        ).prod(dim=-1)
        expected = torch.sum(expected_powers * table_cluster_moments)

        native_phase = torch.sum(native_angles_rad * frequency[None, None, None, :], dim=-1)
        native_cluster_moments = torch.exp(1j * native_phase).mean(dim=-1)
        draw_moments = torch.sum(native_powers * native_cluster_moments, dim=-1)
        estimate = draw_moments.mean()
        real_half_width = (
            z_simultaneous * draw_moments.real.std(unbiased=True) / math.sqrt(_BATCH_SIZE)
        )
        imag_half_width = (
            z_simultaneous * draw_moments.imag.std(unbiased=True) / math.sqrt(_BATCH_SIZE)
        )

        assert abs(float(estimate.real - expected.real)) <= float(real_half_width) + 1e-12
        assert abs(float(estimate.imag - expected.imag)) <= float(imag_half_width) + 1e-12
        assert max(float(real_half_width), float(imag_half_width)) <= _MAX_SIMULTANEOUS_HALF_WIDTH


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", _MODELS)
def test_cdl_joint_delay_angle_characteristics_match_table_reference(model: str) -> None:
    """Check selected joint normalized-delay/four-angle moments on fresh draws."""
    table_path = profile_resources.parameter_file(f"CDL-{model}.json", "19.2")
    assert hashlib.sha256(table_path.read_bytes()).hexdigest() == _PROFILE_SHA256[model]
    parameters = _independent_profile_parameters(model)
    expected_powers, table_angles_deg = _profile_diffuse_reference(parameters)
    is_los = bool(parameters["los"])
    delay_start = int(is_los)
    table_delays = torch.tensor(parameters["delays"][delay_start:], dtype=torch.float64)
    table_delay_order = torch.argsort(table_delays)
    table_delays = table_delays[table_delay_order]
    rays = _native_diffuse_rays(model, seed=_JOINT_DELAY_ANGLE_SEED)

    native_angles_rad = torch.stack(
        [getattr(rays, name)[:, 0, 0] for name in _ANGLE_FIELDS], dim=-1
    ).to(torch.float64)
    native_powers = rays.powers[:, 0, 0].to(torch.float64)
    native_delays = rays.delays[:, 0, 0].to(torch.float64) / _DELAY_SPREAD_S
    expected_clusters = int(expected_powers.numel())
    assert native_angles_rad.shape == (_BATCH_SIZE, expected_clusters, 20, 4)
    assert native_powers.shape == (_BATCH_SIZE, expected_clusters)
    assert native_delays.shape == (_BATCH_SIZE, expected_clusters)
    native_delays, cluster_order = torch.sort(native_delays, dim=-1)
    native_powers = torch.gather(native_powers, dim=1, index=cluster_order)
    angle_order = cluster_order[:, :, None, None].expand(-1, -1, 20, len(_ANGLE_FIELDS))
    native_angles_rad = torch.gather(native_angles_rad, dim=1, index=angle_order)
    torch.testing.assert_close(
        native_delays,
        table_delays.expand_as(native_delays),
        atol=2e-7,
        rtol=2 * torch.finfo(rays.delays.dtype).eps,
    )
    native_power_sums = native_powers.sum(dim=-1)
    torch.testing.assert_close(
        native_power_sums,
        torch.ones(_BATCH_SIZE, dtype=torch.float64),
        atol=2e-7,
        rtol=0.0,
    )
    native_powers = native_powers / native_power_sums[:, None]
    z_simultaneous = NormalDist().inv_cdf(1.0 - _ALPHA / (2.0 * _JOINT_DELAY_ANGLE_CLAIM_COUNT))

    for delay_harmonic, angle_vector in _JOINT_DELAY_ANGLE_VECTORS:
        frequency = torch.tensor(angle_vector, dtype=torch.float64)
        table_angle_moment = torch.stack(
            [
                torch.exp(
                    1j * math.pi / 180.0 * table_angles_deg[..., index] * frequency[index]
                ).mean(dim=-1)
                for index in range(len(_ANGLE_FIELDS))
            ],
            dim=-1,
        ).prod(dim=-1)
        expected = torch.sum(
            expected_powers
            * torch.exp(2j * math.pi * delay_harmonic * table_delays)
            * table_angle_moment
        )

        native_phase = torch.sum(native_angles_rad * frequency[None, None, None, :], dim=-1)
        native_cluster_moments = torch.exp(1j * native_phase).mean(dim=-1)
        delay_phase = torch.exp(2j * math.pi * delay_harmonic * native_delays)
        draw_moments = torch.sum(native_powers * delay_phase * native_cluster_moments, dim=-1)
        estimate = draw_moments.mean()
        real_half_width = (
            z_simultaneous * draw_moments.real.std(unbiased=True) / math.sqrt(_BATCH_SIZE)
        )
        imag_half_width = (
            z_simultaneous * draw_moments.imag.std(unbiased=True) / math.sqrt(_BATCH_SIZE)
        )
        assert abs(float(estimate.real - expected.real)) <= float(real_half_width) + 1e-12
        assert abs(float(estimate.imag - expected.imag)) <= float(imag_half_width) + 1e-12
        assert max(float(real_half_width), float(imag_half_width)) <= _MAX_SIMULTANEOUS_HALF_WIDTH
