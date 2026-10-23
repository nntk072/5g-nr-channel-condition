"""Dual-polarized CDL covariance trace against the TR 38.901 XPR ray law.

The fixed case uses CDL-A v19.2 and one dual-VH port at each endpoint. For a
unit omnidirectional field pattern, the dual-polarization port bases are
orthonormal. The TR 38.901 ray polarization matrix has two unit-power
co-polar terms and two cross-polar terms with power 1/XPR. Independent ray
phases remove cross-ray terms in the ensemble covariance, while unit-modulus
array phase factors preserve its Frobenius trace. Consequently the expected
trace is the sum of normalized cluster powers times 2*(1 + 1/XPR).

The 4,096 independent draws use a frozen seed. A two-sided 99% normal interval
is used for the single predeclared real-valued trace statistic (no family-wise
adjustment is needed); its half-width must be <=1.5% of the reference. The
draw-level statistic aggregates all four polarization-port covariance
diagonals and all clusters, without consulting generated channel values to
construct the reference.
"""

from __future__ import annotations

import hashlib
import math
from statistics import NormalDist

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna.phy")

from integrations.sionna.cdl import CDLArrayConfig, generate_cdl_frequency_response  # noqa: E402
from sionna.phy.channel.tr38901 import models as profile_resources  # noqa: E402

_BATCH_SIZE = 4096
_SEED = 4097
_FAMILYWISE_ALPHA = 0.01
_MAX_RELATIVE_INTERVAL_HALF_WIDTH = 0.015
_PROFILE_SHA256 = "b96f8a179646ee468717c615e8061103d1e66eb1ff1ea78301de243740254cb7"


def _cdl_a_dual_polarization_trace_reference() -> tuple[float, str, float]:
    """Derive covariance trace from the pinned CDL-A PDP and XPR table."""
    table_path = profile_resources.parameter_file("CDL-A.json", "19.2")
    table_bytes = table_path.read_bytes()
    digest = hashlib.sha256(table_bytes).hexdigest()
    assert digest == _PROFILE_SHA256
    parameters = profile_resources.load_json(table_path)
    assert parameters["los"] == 0
    assert int(parameters["num_clusters"]) == len(parameters["powers"])

    cluster_powers = [10.0 ** (float(power_db) / 10.0) for power_db in parameters["powers"]]
    cluster_power_sum = math.fsum(cluster_powers)
    normalized_cluster_powers = [power / cluster_power_sum for power in cluster_powers]
    xpr_linear = 10.0 ** (float(parameters["xpr"]) / 10.0)

    # Each of the 20 rays in a cluster has coefficient power P_cluster/20.
    # The ray's 2x2 Jones matrix contributes 1+1+1/XPR+1/XPR in squared
    # Frobenius norm in expectation. Summing the 20 ray expectations and all
    # normalized cluster powers gives the reference below.
    expected_trace = math.fsum(normalized_cluster_powers) * 2.0 * (1.0 + 1.0 / xpr_linear)
    return expected_trace, digest, float(parameters["xpr"])


@pytest.mark.native_sionna
def test_cdl_a_dual_polarized_covariance_trace_matches_pinned_xpr_reference() -> None:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        expected_trace, source_digest, xpr_db = _cdl_a_dual_polarization_trace_reference()
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

        # Axes retained: draw, RX polarization port, TX polarization port,
        # cluster. The trace of E[h h^H] is E[||h||_2^2].
        per_draw_taps = channel.h_path[:, 0, :, 0, :, :, 0].to(torch.complex128)
        assert tuple(per_draw_taps.shape[1:3]) == (2, 2)
        draw_trace = per_draw_taps.abs().square().sum(dim=(1, 2, 3))
        estimate = float(draw_trace.mean())
        standard_error = float(draw_trace.std(unbiased=True)) / math.sqrt(_BATCH_SIZE)

        # One scalar endpoint is assessed at two-sided 99% coverage. The 1.5%
        # relative-width cap makes the Monte Carlo check resolve this profile's
        # 10% cross-polar contribution to covariance trace.
        z_99 = NormalDist().inv_cdf(1.0 - _FAMILYWISE_ALPHA / 2.0)
        half_width = z_99 * standard_error
        assert abs(estimate - expected_trace) <= half_width
        assert half_width / expected_trace <= _MAX_RELATIVE_INTERVAL_HALF_WIDTH
        assert channel.metadata["parameter_table_sha256"] == source_digest
        assert channel.metadata["cross_polarization_power_ratio_db"] == xpr_db
        assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    finally:
        torch.set_num_threads(previous_threads)
