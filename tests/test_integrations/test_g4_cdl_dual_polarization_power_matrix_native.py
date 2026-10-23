"""G4.07: dual-polarized CDL-A per-port powers follow the 38.901 XPR law.

The frozen case is the pinned CDL-A v19.2 table at 3.5 GHz with one dual-VH
omnidirectional port at each endpoint. Each of the four RX/TX polarization
port-pair powers is a predeclared statistic over independent channel draws.
The independent ray-law reference assigns normalized PDP power to co-polar
pairs and normalized PDP power divided by the linear XPR to cross-polar pairs.
The four two-sided intervals use Bonferroni simultaneous 99% coverage. Their
half-width is capped at one quarter of the expected cross-polar power, so the
Monte Carlo precision resolves the pinned profile's 10-dB power split.
"""

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

_BATCH_SIZE = 4096
_SEED = 407
_FAMILYWISE_ALPHA = 0.01
_PRECISION_FRACTION_OF_CROSS_POWER = 0.25
_PROFILE_SHA256 = "b96f8a179646ee468717c615e8061103d1e66eb1ff1ea78301de243740254cb7"


def _independent_cdl_a_port_power_reference() -> tuple[Any, str]:
    """Calculate the four XPR-law expectations from the pinned table alone."""
    table_path = profile_resources.parameter_file("CDL-A.json", "19.2")
    table_bytes = table_path.read_bytes()
    digest = hashlib.sha256(table_bytes).hexdigest()
    assert digest == _PROFILE_SHA256

    parameters = profile_resources.load_json(table_path)
    assert parameters["los"] == 0
    assert float(parameters["xpr"]) == 10.0
    cluster_powers = [10.0 ** (float(power_db) / 10.0) for power_db in parameters["powers"]]
    total_pdp_power = math.fsum(cluster_powers)
    normalized_pdp_sum = math.fsum(power / total_pdp_power for power in cluster_powers)
    xpr_linear = 10.0 ** (float(parameters["xpr"]) / 10.0)

    # For a dual-VH pair, the TR 38.901 ray Jones matrix has unit expected
    # co-polar power and 1/XPR expected cross-polar power. Independent ray
    # phases remove cross-ray terms; the normalized cluster PDP sums to one.
    co_polar_power = normalized_pdp_sum
    cross_polar_power = normalized_pdp_sum / xpr_linear
    expected = torch.tensor(
        (
            (co_polar_power, cross_polar_power),
            (cross_polar_power, co_polar_power),
        ),
        dtype=torch.float64,
    )
    return expected, digest


@pytest.mark.native_sionna
def test_cdl_a_dual_vh_port_powers_match_independent_xpr_ray_law() -> None:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        expected, source_digest = _independent_cdl_a_port_power_reference()
        channel = generate_cdl_frequency_response(
            model="A",
            seed=_SEED,
            carrier_frequency_hz=3.5e9,
            subcarrier_spacing_hz=30_000.0,
            num_subcarriers=8,
            batch_size=_BATCH_SIZE,
            delay_spread_s=100e-9,
            ut_array=CDLArrayConfig(
                polarization="dual",
                polarization_type="VH",
                antenna_pattern="omni",
            ),
            bs_array=CDLArrayConfig(
                polarization="dual",
                polarization_type="VH",
                antenna_pattern="omni",
            ),
            ut_orientation_rad=(0.0, 0.0, 0.0),
            bs_orientation_rad=(0.0, 0.0, 0.0),
            spec_version="19.2",
            device="cpu",
        )

        # One observation per independent draw and port pair, summing the
        # cluster tap powers. Axes after indexing: draw, RX polarization,
        # TX polarization, cluster.
        per_draw_taps = channel.h_path[:, 0, :, 0, :, :, 0].to(torch.complex128)
        assert tuple(per_draw_taps.shape[1:3]) == (2, 2)
        observations = per_draw_taps.abs().square().sum(dim=-1)
        estimates = observations.mean(dim=0)
        standard_errors = observations.std(dim=0, unbiased=True) / math.sqrt(_BATCH_SIZE)

        # Bonferroni controls the four predeclared port-pair intervals at 99%
        # simultaneous coverage; precision is tighter than the XPR power gap.
        z_simultaneous = NormalDist().inv_cdf(1.0 - _FAMILYWISE_ALPHA / (2.0 * expected.numel()))
        half_widths = z_simultaneous * standard_errors
        assert bool((torch.abs(estimates - expected) <= half_widths).all())
        assert float(half_widths.max()) <= float(expected.min()) * (
            _PRECISION_FRACTION_OF_CROSS_POWER
        )
        assert channel.metadata["parameter_table_sha256"] == source_digest
        assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    finally:
        torch.set_num_threads(previous_threads)
