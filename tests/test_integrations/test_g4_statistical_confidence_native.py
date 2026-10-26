"""Confidence-interval checks for standardized-channel observables.

Every statistical unit below is one independently generated batch draw. Tap,
antenna, and time coordinates within a draw are treated as paired values.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna")

from datasets.standardized_specs import FIXED_TDL_DOPPLER_CASES  # noqa: E402
from evaluation.g4_cdl_temporal_reference import (  # noqa: E402
    cdl_nlos_temporal_correlation_reference,
)
from evaluation.g4_tdl_covariance_reference import (  # noqa: E402
    fixed_tdl_first_tap_power_components,
    scalable_tdl_first_tap_power_components,
)
from evaluation.g4_tdl_profile_reference import scalable_tdl_pdp_reference  # noqa: E402
from integrations.sionna.cdl import CDLArrayConfig, generate_cdl_frequency_response  # noqa: E402
from integrations.sionna.tdl import generate_tdl_frequency_response  # noqa: E402

_Z_SIMULTANEOUS = 3.5
_Z_NAMED_DOPPLER_CASE_FAMILYWISE = 3.8
_FIXED_TDL_MODELS = ("A10", "A30", "B100", "C60", "C300", "D10", "D30")
_TDL_RX_SPATIAL_MODELS = tuple("ABCDE")
_TDL_RX_SPATIAL_COMPONENT_COUNT = len(_TDL_RX_SPATIAL_MODELS) * 2 * 4 * 4
_CDL_REFERENCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs/research/G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json"
)
_CDL_REFERENCE_SHA256 = "24ee2f8f73f26e27522a296a4a0860bdaa7f284e9138855666e09636bc97b090"
_CDL_ETSI_SOURCE_SHA256 = "b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4"


def _within_interval(estimate: Any, standard_error: Any, target: Any) -> bool:
    return bool((torch.abs(estimate - target) <= _Z_SIMULTANEOUS * standard_error).all())


def _independent_cdl_pdp(
    model: str, delay_spread_s: float
) -> tuple[list[float], list[float]]:
    """Derive the ensemble cluster PDP from the pinned ETSI table fixture."""
    reference_bytes = _CDL_REFERENCE_PATH.read_bytes()
    if hashlib.sha256(reference_bytes).hexdigest() != _CDL_REFERENCE_SHA256:
        raise ValueError("CDL statistical reference bytes do not match the frozen digest")
    reference = json.loads(reference_bytes)
    if (
        reference.get("schema") != "loslab.g4_cdl_tap_angle_reference.v1"
        or reference.get("source_id") != "tr-138901-v19.2.0"
        or reference.get("source_sha256") != _CDL_ETSI_SOURCE_SHA256
    ):
        raise ValueError("CDL statistical reference is not bound to ETSI TR 38.901 V19.2.0")
    profiles = reference.get("profiles")
    if not isinstance(profiles, list):
        raise ValueError("CDL statistical reference must contain a profile list")
    matches = [row for row in profiles if row.get("profile_model") == model]
    if len(matches) != 1:
        raise ValueError(f"CDL statistical reference must contain exactly one profile {model}")
    profile = matches[0]
    components = profile.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError(f"CDL-{model} reference must contain components")

    power_by_cluster: dict[int, float] = {}
    delay_by_cluster: dict[int, float] = {}
    for component in components:
        cluster_id = component.get("cluster_index")
        delay = component.get("normalized_delay")
        power_db = component.get("power_db")
        if (
            isinstance(cluster_id, bool)
            or not isinstance(cluster_id, int)
            or cluster_id <= 0
            or isinstance(delay, bool)
            or not isinstance(delay, (int, float))
            or not math.isfinite(delay)
            or delay < 0
            or isinstance(power_db, bool)
            or not isinstance(power_db, (int, float))
            or not math.isfinite(power_db)
            or not isinstance(component.get("component"), str)
        ):
            raise ValueError(f"CDL-{model} reference contains an invalid component row")
        if (
            cluster_id in delay_by_cluster
            and delay_by_cluster[cluster_id] != float(delay)
        ):
            raise ValueError(
                f"CDL-{model} components in one cluster have inconsistent delays"
            )
        delay_by_cluster[cluster_id] = float(delay)
        power_by_cluster[cluster_id] = power_by_cluster.get(cluster_id, 0.0) + 10.0 ** (
            float(power_db) / 10.0
        )

    if len(power_by_cluster) != profile.get("cluster_count"):
        raise ValueError(
            f"CDL-{model} reference cluster count does not match its component roster"
        )
    total_power = math.fsum(power_by_cluster.values())
    if not math.isfinite(total_power) or total_power <= 0:
        raise ValueError(f"CDL-{model} reference has invalid total cluster power")
    cluster_ids = sorted(
        power_by_cluster, key=lambda cluster: (delay_by_cluster[cluster], cluster)
    )
    return (
        [power_by_cluster[cluster] / total_power for cluster in cluster_ids],
        [delay_by_cluster[cluster] * delay_spread_s for cluster in cluster_ids],
    )


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("family", "model"),
    tuple((family, model) for family in ("TDL", "CDL") for model in "ABCDE")
    + tuple(("TDL", model) for model in _FIXED_TDL_MODELS),
)
def test_delay_profile_and_rms_have_draw_level_confidence_intervals(
    family: str, model: str
) -> None:
    """Check scalable and fixed profile PDP/RMS delay with draw-level intervals."""
    batch_size = 4096
    channel: Any
    if family == "TDL":
        channel = generate_tdl_frequency_response(
            model=model,
            seed=7311,
            carrier_frequency_hz=3.5e9,
            subcarrier_spacing_hz=30_000.0,
            num_subcarriers=8,
            batch_size=batch_size,
            delay_spread_s=None if model in _FIXED_TDL_MODELS else 100e-9,
            spec_version="19.2",
            device="cpu",
        )
    else:
        channel = generate_cdl_frequency_response(
            model=model,
            seed=7311,
            carrier_frequency_hz=3.5e9,
            subcarrier_spacing_hz=30_000.0,
            num_subcarriers=8,
            batch_size=batch_size,
            delay_spread_s=100e-9,
            spec_version="19.2",
            device="cpu",
        )
    # [independent draw, profile tap/cluster]; all within-draw dimensions are
    # retained together when estimating the PDP and its uncertainty.
    tap_power = channel.h_path[:, 0, 0, 0, 0, :, 0].abs().square().to(torch.float64)
    empirical_power = tap_power.mean(dim=0)
    power_se = tap_power.std(dim=0, unbiased=True) / math.sqrt(batch_size)
    if family == "CDL":
        independent_powers, independent_delays = _independent_cdl_pdp(model, 100e-9)
        expected_power = torch.as_tensor(independent_powers, dtype=torch.float64)
        delays = torch.as_tensor(independent_delays, dtype=torch.float64)
    elif model in {"A", "B", "C", "D", "E"}:
        independent_powers, independent_delays = scalable_tdl_pdp_reference(
            model, delay_spread_s=100e-9
        )
        expected_power = torch.as_tensor(independent_powers, dtype=torch.float64)
        delays = torch.as_tensor(independent_delays, dtype=torch.float64)
    else:
        expected_values = channel.metadata.get(
            "mean_tap_powers", channel.metadata.get("normalized_cluster_powers")
        )
        expected_power = torch.as_tensor(expected_values, dtype=torch.float64)
        delays = torch.as_tensor(
            channel.metadata.get(
                "nominal_tap_delays_s", channel.metadata.get("nominal_cluster_delays_s")
            ),
            dtype=torch.float64,
        )
    assert empirical_power.shape == expected_power.shape == delays.shape
    # A 3.5-SE bound is conservative across the profile's multiple PDP taps.
    assert _within_interval(empirical_power, power_se, expected_power)

    total_power = tap_power.sum(dim=1)
    first_moment = (tap_power * delays).sum(dim=1)
    second_moment = (tap_power * delays.square()).sum(dim=1)
    mean_power = total_power.mean()
    mean_delay = first_moment.mean() / mean_power
    rms_variance = second_moment.mean() / mean_power - mean_delay.square()
    empirical_rms = torch.sqrt(rms_variance)
    expected_mean = (expected_power * delays).sum() / expected_power.sum()
    expected_rms = torch.sqrt(
        (expected_power * (delays - expected_mean).square()).sum() / expected_power.sum()
    )
    # Delta-method influence values are computed per independent realization;
    # this gives a CI for the RMS delay spread of the ensemble mean PDP.
    influence = (
        (tap_power * (delays - mean_delay).square()).sum(dim=1) - rms_variance * total_power
    ) / (2.0 * empirical_rms * mean_power)
    rms_se = influence.std(unbiased=True) / math.sqrt(batch_size)
    assert abs(float(empirical_rms - expected_rms)) <= _Z_SIMULTANEOUS * float(rms_se)
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("model", "los_angle_rad", "case_id", "case_carrier_hz", "case_doppler_hz"),
    tuple((model, math.acos(0.7), None, None, None) for model in "ABC")
    + tuple((model, angle, None, None, None) for model in "DE" for angle in (0.0, math.pi / 2.0))
    + tuple(
        (model, math.acos(0.7), None, None, None)
        for model in _FIXED_TDL_MODELS
        if model not in {"D10", "D30"}
    )
    + tuple(
        (model, angle, None, None, None)
        for model in ("D10", "D30")
        for angle in (0.0, math.pi / 2.0)
    )
    + tuple(
        (
            model,
            math.acos(0.7),
            case_id,
            3.5e9 if frequency_range == "FR1" else 28e9,
            maximum_doppler_hz,
        )
        for case_id, (model, frequency_range, maximum_doppler_hz) in FIXED_TDL_DOPPLER_CASES.items()
    ),
)
def test_doppler_correlation_confidence_interval_uses_paired_independent_draws(
    model: str,
    los_angle_rad: float,
    case_id: str | None,
    case_carrier_hz: float | None,
    case_doppler_hz: float | None,
) -> None:
    if case_id is None:
        carrier_frequency_hz = 28e9 if model in {"A10", "C60", "D10"} else 3.5e9
        speed_mps = 30.0
        maximum_doppler_hz = speed_mps * carrier_frequency_hz / 299_792_458.0
    else:
        assert case_carrier_hz is not None and case_doppler_hz is not None
        carrier_frequency_hz = case_carrier_hz
        maximum_doppler_hz = case_doppler_hz
        speed_mps = maximum_doppler_hz * 299_792_458.0 / carrier_frequency_hz
    lag_s = 1.0 / (4.0 * maximum_doppler_hz)
    batch_size = 8192
    channel = generate_tdl_frequency_response(
        model=model,
        seed=4031,
        carrier_frequency_hz=carrier_frequency_hz,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=batch_size,
        delay_spread_s=None if model in _FIXED_TDL_MODELS else 100e-9,
        num_time_steps=2,
        sampling_frequency_hz=1.0 / lag_s,
        min_speed_mps=speed_mps,
        max_speed_mps=speed_mps,
        fixed_profile_conformance_case=case_id,
        los_angle_of_arrival_rad=los_angle_rad,
        spec_version="19.2",
        device="cpu",
    )
    assert channel.metadata["maximum_doppler_hz"] == pytest.approx(maximum_doppler_hz)
    tap = channel.h_path[:, 0, 0, 0, 0, 0, :]
    cross_product = tap[:, 0].conj() * tap[:, 1]
    reference_power = tap[:, 0].abs().square()
    mean_power = reference_power.mean()
    correlation = cross_product.mean() / mean_power
    # The two times belong to one draw. The influence function is formed from
    # each paired (t0,t1) draw, not from treating time samples as independent.
    influence = cross_product - correlation * reference_power
    denominator = mean_power * math.sqrt(batch_size)
    real_se = influence.real.std(unbiased=True) / denominator
    imag_se = influence.imag.std(unbiased=True) / denominator
    # Bonferroni family: real and imaginary covariance components for all 17
    # named cases. The existing profile/angle checks retain their prior bound.
    simultaneous_z = _Z_SIMULTANEOUS if case_id is None else _Z_NAMED_DOPPLER_CASE_FAMILYWISE
    diffuse = torch.special.bessel_j0(torch.tensor(math.pi / 2.0, dtype=torch.float64))
    if model in _FIXED_TDL_MODELS:
        diffuse_power, specular_power = fixed_tdl_first_tap_power_components(model)
    else:
        diffuse_power, specular_power = scalable_tdl_first_tap_power_components(model)
    first_tap_power = diffuse_power + specular_power
    diffuse_fraction = diffuse_power / first_tap_power
    specular_fraction = specular_power / first_tap_power
    if specular_fraction == 0.0:
        expected = diffuse.to(torch.complex128)
    else:
        specular = torch.exp(
            torch.tensor(
                1j * 2.0 * math.pi * maximum_doppler_hz * math.cos(los_angle_rad) * lag_s,
                dtype=torch.complex128,
            )
        )
        expected = specular_fraction * specular + diffuse_fraction * diffuse
    assert abs(float(correlation.real - expected.real)) <= simultaneous_z * float(real_se)
    assert abs(float(correlation.imag - expected.imag)) <= simultaneous_z * float(imag_se)
    assert float(real_se) < 0.02 and float(imag_se) < 0.02


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", ("A", "B", "C"))
def test_cdl_nlos_temporal_correlation_matches_independent_ray_doppler_reference(
    model: str,
) -> None:
    """Check CDL A-C Doppler covariance against the 38.901 ray-phase equation."""
    carrier_frequency_hz = 3.5e9
    speed_mps = 30.0
    lag_s = 0.7e-3
    batch_size = 4096
    channel = generate_cdl_frequency_response(
        model=model,
        seed=1984,
        carrier_frequency_hz=carrier_frequency_hz,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=batch_size,
        delay_spread_s=100e-9,
        num_time_steps=2,
        sampling_frequency_hz=1.0 / lag_s,
        ut_velocity_mps=(speed_mps, 0.0, 0.0),
        spec_version="19.2",
        device="cpu",
    )
    # Sum distinct delay clusters to form the narrowband channel. The two
    # observations in each row share one independent CDL realization.
    path = channel.h_path[:, 0, 0, 0, 0, :, :]
    h0 = path[:, :, 0].sum(dim=1).to(torch.complex128)
    h1 = path[:, :, 1].sum(dim=1).to(torch.complex128)
    cross_product = h0.conj() * h1
    reference_power = h0.abs().square()
    mean_power = reference_power.mean()
    empirical = cross_product.mean() / mean_power
    influence = cross_product - empirical * reference_power
    real_se = influence.real.std(unbiased=True) / (mean_power * math.sqrt(batch_size))
    imag_se = influence.imag.std(unbiased=True) / (mean_power * math.sqrt(batch_size))

    # The expected value comes from pinned ETSI rows and standardized ray
    # offsets; adapter metadata is intentionally excluded from the oracle.
    metadata = channel.metadata
    expected_value = cdl_nlos_temporal_correlation_reference(
        model,
        carrier_frequency_hz=carrier_frequency_hz,
        speed_mps=speed_mps,
        lag_s=lag_s,
    )
    expected = torch.tensor(expected_value, dtype=torch.complex128)

    assert metadata["profile_los_component"] is False
    assert metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    assert abs(float(empirical.real - expected.real)) <= _Z_SIMULTANEOUS * float(real_se)
    assert abs(float(empirical.imag - expected.imag)) <= _Z_SIMULTANEOUS * float(imag_se)


@pytest.mark.native_sionna
def test_receive_spatial_covariance_confidence_intervals_match_configured_matrix() -> None:
    batch_size = 8192
    correlation = 0.55 + 0.25j
    rx_corr_mat = ((1.0 + 0.0j, correlation), (correlation.conjugate(), 1.0 + 0.0j))
    channel = generate_tdl_frequency_response(
        model="A",
        seed=9107,
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=batch_size,
        delay_spread_s=100e-9,
        num_rx_ant=2,
        rx_corr_mat=rx_corr_mat,
        spec_version="19.2",
        device="cpu",
    )
    # One vector per independent channel draw; antenna elements are paired.
    samples = channel.h_path[:, 0, :, 0, 0, 0, 0]
    outer_products = samples[:, :, None] * samples[:, None, :].conj()
    empirical_covariance = outer_products.mean(dim=0)
    covariance_real_se = outer_products.real.std(dim=0, unbiased=True) / math.sqrt(batch_size)
    covariance_imag_se = outer_products.imag.std(dim=0, unbiased=True) / math.sqrt(batch_size)
    diffuse_power, specular_power = scalable_tdl_first_tap_power_components("A")
    tap_power = diffuse_power + specular_power
    expected_covariance = tap_power * torch.as_tensor(rx_corr_mat, dtype=torch.complex64)
    target_real = expected_covariance.real
    target_imag = expected_covariance.imag
    assert _within_interval(empirical_covariance.real, covariance_real_se, target_real)
    assert _within_interval(empirical_covariance.imag, covariance_imag_se, target_imag)
    metadata_real = torch.as_tensor(channel.metadata["rx_corr_mat_real"])
    metadata_imag = torch.as_tensor(channel.metadata["rx_corr_mat_imag"])
    torch.testing.assert_close(
        metadata_real,
        torch.tensor([[1.0, correlation.real], [correlation.real, 1.0]]),
        rtol=0.0,
        atol=1e-7,
    )
    torch.testing.assert_close(
        metadata_imag,
        torch.tensor([[0.0, correlation.imag], [-correlation.imag, 0.0]]),
        rtol=0.0,
        atol=1e-7,
    )


@pytest.mark.native_sionna
def test_four_element_tdl_covariance_uses_simultaneous_draw_level_intervals() -> None:
    """Check a nontrivial complex 4x4 RX covariance with familywise intervals."""
    batch_size = 8192
    rho = 0.45
    phase_rad = 0.23
    rx_corr_mat = tuple(
        tuple(
            rho ** abs(row - col)
            * complex(math.cos(phase_rad * (row - col)), math.sin(phase_rad * (row - col)))
            for col in range(4)
        )
        for row in range(4)
    )
    channel = generate_tdl_frequency_response(
        model="A",
        seed=5217,
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=batch_size,
        delay_spread_s=100e-9,
        num_rx_ant=4,
        rx_corr_mat=rx_corr_mat,
        spec_version="19.2",
        device="cpu",
    )
    # Antenna elements remain paired within each independent channel draw.
    samples = channel.h_path[:, 0, :, 0, 0, 0, 0]
    outer_products = samples[:, :, None] * samples[:, None, :].conj()
    empirical = outer_products.mean(dim=0)
    real_se = outer_products.real.std(dim=0, unbiased=True) / math.sqrt(batch_size)
    imag_se = outer_products.imag.std(dim=0, unbiased=True) / math.sqrt(batch_size)
    independent_powers, _ = scalable_tdl_pdp_reference("A", delay_spread_s=100e-9)
    expected = independent_powers[0] * torch.as_tensor(rx_corr_mat, dtype=torch.complex64)

    # Bonferroni controls the familywise two-sided error across 32 real/imag
    # components at 1%; each interval is asymptotic under independent draws.
    component_count = 2 * expected.numel()
    z_simultaneous = NormalDist().inv_cdf(1.0 - 0.01 / (2.0 * component_count))
    assert bool((torch.abs(empirical.real - expected.real) <= z_simultaneous * real_se).all())
    assert bool((torch.abs(empirical.imag - expected.imag) <= z_simultaneous * imag_se).all())
    torch.testing.assert_close(empirical, empirical.mH, rtol=0.0, atol=1e-7)
    assert float(torch.linalg.eigvalsh(empirical).min()) >= -1e-7
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("profile_index", "model"), tuple(enumerate(_TDL_RX_SPATIAL_MODELS))
)
def test_tdl_ae_receive_covariance_matches_one_simultaneous_matrix_family(
    profile_index: int, model: str
) -> None:
    """Check a nontrivial complex receive covariance for every scalable profile."""
    batch_size = 8192
    rho = 0.45
    phase_rad = 0.23
    rx_corr_mat = tuple(
        tuple(
            rho**abs(row - col)
            * complex(math.cos(phase_rad * (row - col)), math.sin(phase_rad * (row - col)))
            for col in range(4)
        )
        for row in range(4)
    )
    channel = generate_tdl_frequency_response(
        model=model,
        seed=82_000 + profile_index,
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=batch_size,
        delay_spread_s=100e-9,
        num_rx_ant=4,
        rx_corr_mat=rx_corr_mat,
        spec_version="19.2",
        device="cpu",
    )
    # TDL-D/E are LOS profiles: use their second tap so the
    # covariance claim concerns the diffuse field rather than its specular atom.
    tap_index = 1 if model in {"D", "E"} else 0
    samples = channel.h_path[:, 0, :, 0, 0, 0, tap_index]
    outer_products = samples[:, :, None] * samples[:, None, :].conj()
    empirical = outer_products.mean(dim=0)
    real_se = outer_products.real.std(dim=0, unbiased=True) / math.sqrt(batch_size)
    imag_se = outer_products.imag.std(dim=0, unbiased=True) / math.sqrt(batch_size)
    independent_powers, _ = scalable_tdl_pdp_reference(model, delay_spread_s=100e-9)
    expected = independent_powers[tap_index] * torch.as_tensor(
        rx_corr_mat, dtype=torch.complex64
    )
    simultaneous_z = NormalDist().inv_cdf(
        1.0 - 0.01 / (2.0 * _TDL_RX_SPATIAL_COMPONENT_COUNT)
    )
    assert bool(
        (torch.abs(empirical.real - expected.real) <= simultaneous_z * real_se).all()
    )
    assert bool(
        (torch.abs(empirical.imag - expected.imag) <= simultaneous_z * imag_se).all()
    )
    torch.testing.assert_close(empirical, empirical.mH, rtol=0.0, atol=1e-7)
    assert float(torch.linalg.eigvalsh(empirical).min()) >= -1e-7
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    torch.testing.assert_close(
        torch.as_tensor(channel.metadata["rx_corr_mat_real"]),
        torch.as_tensor(rx_corr_mat, dtype=torch.complex64).real,
        rtol=0.0,
        atol=1e-7,
    )
    torch.testing.assert_close(
        torch.as_tensor(channel.metadata["rx_corr_mat_imag"]),
        torch.as_tensor(rx_corr_mat, dtype=torch.complex64).imag,
        rtol=0.0,
        atol=1e-7,
    )


@pytest.mark.native_sionna
def test_cdl_array_spatial_covariance_has_draw_level_confidence_intervals() -> None:
    """Estimate one CDL cluster's 2-element covariance with paired array draws."""
    batch_size = 4096
    channel = generate_cdl_frequency_response(
        model="A",
        seed=2381,
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=batch_size,
        delay_spread_s=100e-9,
        ut_array=CDLArrayConfig(
            num_cols_per_panel=2,
            num_cols=1,
            element_horizontal_spacing=0.5,
        ),
        spec_version="19.2",
        device="cpu",
    )
    # The receive antenna elements are paired within each independent draw.
    samples = channel.h_path[:, 0, :, 0, 0, 0, 0]
    outer_products = samples[:, :, None] * samples[:, None, :].conj()
    covariance = outer_products.mean(dim=0)
    real_se = outer_products.real.std(dim=0, unbiased=True) / math.sqrt(batch_size)
    imag_se = outer_products.imag.std(dim=0, unbiased=True) / math.sqrt(batch_size)
    assert covariance.shape == (2, 2)
    torch.testing.assert_close(covariance, covariance.mH, rtol=0.0, atol=1e-7)
    assert float(torch.linalg.eigvalsh(covariance).min()) >= -1e-7
    assert float((_Z_SIMULTANEOUS * real_se).max()) < 0.1 * float(covariance.diagonal().real.mean())
    assert float((_Z_SIMULTANEOUS * imag_se).max()) < 0.1 * float(covariance.diagonal().real.mean())

    paired_power_difference = samples[:, 0].abs().square() - samples[:, 1].abs().square()
    paired_se = paired_power_difference.std(unbiased=True) / math.sqrt(batch_size)
    assert abs(float(paired_power_difference.mean())) <= _Z_SIMULTANEOUS * float(paired_se)
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("matrix", "message"),
    (
        (((1.0,),), "shape"),
        (((1.0, math.nan), (math.nan, 1.0)), "finite"),
        (((1.0, 0.2), (0.1, 1.0)), "Hermitian"),
        (((1.0, 1.2), (1.2, 1.0)), "positive definite"),
        (((0.9, 0.0), (0.0, 1.0)), "unit diagonal"),
    ),
)
def test_tdl_rejects_invalid_receive_correlation_matrix(matrix, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        generate_tdl_frequency_response(
            model="A",
            seed=9,
            carrier_frequency_hz=3.5e9,
            subcarrier_spacing_hz=30_000.0,
            num_subcarriers=8,
            num_rx_ant=2,
            rx_corr_mat=matrix,
            spec_version="19.2",
            device="cpu",
        )
