"""G4 phase checks for independent CDL cluster coefficient fluctuations."""

from __future__ import annotations

import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path
from statistics import NormalDist
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna.phy")

from integrations.sionna.cdl import generate_cdl_frequency_response  # noqa: E402
from integrations.sionna.random import call_with_sionna_seed  # noqa: E402
from sionna.phy.channel.tr38901 import CDL, PanelArray  # noqa: E402

_PROFILES = {
    "A": "b96f8a179646ee468717c615e8061103d1e66eb1ff1ea78301de243740254cb7",
    "B": "7c87c092cb2bb44891d301ee3c1991cdda1c6b64094a03f18746ec5fa14ee2a2",
    "C": "f2044ecc3b72a1967221805ba70b1f0dab9634bb831478c1337b737ced6e4a28",
    "D": "a8960f176347de457c0cb90fbc8cf137ac98bb1c5295b096f004d8d5c6e40176",
    "E": "04328f33d8d79b24aa3d0303fecb58b465ed7e13874930b05e76be0d97743480",
}
_DRAW_COUNT = 32_768
_FAMILYWISE_ALPHA = 0.01
_MAXIMUM_HALF_WIDTH = 0.05
_ROOT = Path(__file__).resolve().parents[2]
_TAP_ANGLE_REFERENCE_PATH = (
    _ROOT / "docs/research/G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json"
)
_TAP_ANGLE_REFERENCE_SHA256 = "24ee2f8f73f26e27522a296a4a0860bdaa7f284e9138855666e09636bc97b090"
_ETSI_SOURCE_SHA256 = "b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4"


def _independent_profile_roster() -> dict[str, dict[str, Any]]:
    reference_bytes = _TAP_ANGLE_REFERENCE_PATH.read_bytes()
    assert hashlib.sha256(reference_bytes).hexdigest() == _TAP_ANGLE_REFERENCE_SHA256
    reference = json.loads(reference_bytes)
    assert reference["source_sha256"] == _ETSI_SOURCE_SHA256
    profiles = {row["profile_model"]: row for row in reference["profiles"]}
    assert set(profiles) == set(_PROFILES)
    return profiles


def _family_claim_count(profiles: dict[str, dict[str, Any]]) -> int:
    return 2 * sum(int(profile["cluster_count"]) for profile in profiles.values())


def _sample_los_first_cluster_diffuse_component(model: str) -> Any:
    """Generate the diffuse component before Sionna combines its LOS specular ray."""
    if version("sionna") != "2.2.0":
        raise RuntimeError("LOS diffuse decomposition is pinned to Sionna 2.2.0 internals")
    carrier_frequency_hz = 3.5e9
    array = PanelArray(
        num_rows_per_panel=1,
        num_cols_per_panel=1,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        num_rows=1,
        num_cols=1,
        carrier_frequency=carrier_frequency_hz,
        device="cpu",
    )
    channel_model = CDL(
        model=model,
        delay_spread=100e-9,
        carrier_frequency=carrier_frequency_hz,
        ut_array=array,
        bs_array=array,
        direction="downlink",
        ut_orientation=[torch.pi, 0.0, 0.0],
        bs_orientation=[0.0, 0.0, 0.0],
        ut_velocity=None,
        min_speed=0.0,
        max_speed=0.0,
        normalize_delays=True,
        device="cpu",
        spec_version="19.2",
    )
    generator = channel_model._cir_gen

    def generate_diffuse_only(
        phi: Any,
        topology: object,
        k_factor: Any,
        rays: object,
        sample_times: Any,
        cluster_delay_spread: Any,
    ) -> tuple[Any, Any]:
        del k_factor
        h_full = generator._step_11_nlos(phi, topology, rays, sample_times)
        return generator._step_11_reduce_nlos(h_full, rays, cluster_delay_spread)

    generator._step_11 = generate_diffuse_only
    h_diffuse, _ = call_with_sionna_seed(
        "cpu",
        933_100 + ord(model),
        lambda: channel_model(
            batch_size=_DRAW_COUNT,
            num_time_steps=1,
            sampling_frequency=1.0,
        ),
    )
    return h_diffuse[:, 0, 0, 0, 0, 0, 0].to(torch.complex128)


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", tuple(_PROFILES))
def test_cdl_diffuse_cluster_complex_pseudocovariance_is_zero(model: str) -> None:
    """Check NLOS clusters and pure diffuse clusters in LOS CDL-D/E."""
    profiles = _independent_profile_roster()
    profile = profiles[model]
    profile_has_los = bool(profile["los"])
    assert profile_has_los is (model in {"D", "E"})

    channel = generate_cdl_frequency_response(
        model=model,
        seed=933_100 + ord(model),
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=_DRAW_COUNT,
        delay_spread_s=100e-9,
        spec_version="19.2",
        device="cpu",
    )
    cluster_coefficients = channel.h_path[:, 0, 0, 0, 0, :, 0].to(torch.complex128)
    expected_cluster_count = int(profile["cluster_count"])
    assert cluster_coefficients.shape == (_DRAW_COUNT, expected_cluster_count)
    claim_count = _family_claim_count(profiles)
    simultaneous_z = NormalDist().inv_cdf(1.0 - _FAMILYWISE_ALPHA / (2.0 * claim_count))

    for cluster_index in range(cluster_coefficients.shape[1]):
        if profile_has_los and cluster_index == 0:
            # Sionna merges the specular LOS ray into this first cluster; its
            # pseudo-covariance is not a diffuse-only phase statistic.
            continue
        samples = cluster_coefficients[:, cluster_index]
        power = samples.abs().square()
        mean_power = power.mean()
        pseudo_covariance = samples.square().mean() / mean_power
        influence = samples.square() - pseudo_covariance * power.mean()
        standard_errors = torch.stack(
            (influence.real.std(unbiased=True), influence.imag.std(unbiased=True))
        ) / (mean_power * math.sqrt(_DRAW_COUNT))
        half_widths = simultaneous_z * standard_errors
        estimate = torch.stack((pseudo_covariance.real, pseudo_covariance.imag))

        assert bool((estimate.abs() <= half_widths).all()), (
            model,
            cluster_index,
            estimate.tolist(),
            half_widths.tolist(),
        )
        assert float(half_widths.max()) <= _MAXIMUM_HALF_WIDTH

    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    assert channel.metadata["parameter_table_sha256"] == _PROFILES[model]
    assert channel.metadata["profile_los_component"] is profile_has_los


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", ("D", "E"))
def test_los_composite_cluster_diffuse_component_is_proper_complex(model: str) -> None:
    """Check the LOS tap's diffuse term before its deterministic ray is added."""
    profiles = _independent_profile_roster()
    assert profiles[model]["los"] is True
    samples = _sample_los_first_cluster_diffuse_component(model)
    assert samples.shape == (_DRAW_COUNT,)
    power = samples.abs().square()
    mean_power = power.mean()
    pseudo_covariance = samples.square().mean() / mean_power
    influence = samples.square() - pseudo_covariance * mean_power

    claim_count = _family_claim_count(profiles)
    simultaneous_z = NormalDist().inv_cdf(1.0 - _FAMILYWISE_ALPHA / (2.0 * claim_count))
    standard_errors = torch.stack(
        (influence.real.std(unbiased=True), influence.imag.std(unbiased=True))
    ) / (mean_power * math.sqrt(_DRAW_COUNT))
    half_widths = simultaneous_z * standard_errors
    estimate = torch.stack((pseudo_covariance.real, pseudo_covariance.imag))

    assert bool((estimate.abs() <= half_widths).all()), (
        model,
        estimate.tolist(),
        half_widths.tolist(),
    )
    assert float(half_widths.max()) <= _MAXIMUM_HALF_WIDTH
