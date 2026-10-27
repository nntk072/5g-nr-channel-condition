"""Pinned Sionna TR 38.901 TDL-to-NR frequency-response adapter."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from numbers import Real
from typing import Any, Sequence

from datasets.standardized_specs import (
    FIXED_TDL_DOPPLER_CASES,
    SPEED_OF_LIGHT_MPS,
    STANDARDIZED_PROFILE_APPLICABILITY,
    carrier_frequency_range,
    standardized_profile_applicability,
    tdl_resolved_profile_parameter_sha256,
)

from integrations.sionna_runtime import sionna_phy_version

from .random import call_with_sionna_seed

_SCALABLE_TDL_MODELS = frozenset(
    model
    for (family, model), applicability in STANDARDIZED_PROFILE_APPLICABILITY.items()
    if family == "tdl" and not applicability.fixed_profile
)
_FIXED_TDL_MODELS = frozenset(
    model
    for (family, model), applicability in STANDARDIZED_PROFILE_APPLICABILITY.items()
    if family == "tdl" and applicability.fixed_profile
)


@dataclass(frozen=True, slots=True)
class SionnaTDLChannel:
    """Path oracle and CFR at explicit channel time samples.

    ``h_path`` axes are [batch, rx, rx_ant, tx, tx_ant, path, time],
    ``tau_s`` axes are [batch, rx, tx, path], and CFR axes are
    [batch, rx, rx_ant, tx, tx_ant, time, subcarrier]. The profile name is a
    stochastic-model proxy, never a geometric LOS label.
    """

    h_path: Any
    tau_s: Any
    channel_frequency_response: Any
    metadata: dict[str, Any]


def generate_tdl_frequency_response(
    *,
    model: str,
    seed: int,
    carrier_frequency_hz: float,
    subcarrier_spacing_hz: float,
    num_subcarriers: int,
    channel_bandwidth_hz: float | None = None,
    batch_size: int = 1,
    delay_spread_s: float | None = None,
    num_time_steps: int = 1,
    sampling_frequency_hz: float = 1.0,
    num_rx_ant: int = 1,
    num_tx_ant: int = 1,
    rx_corr_mat: Sequence[Sequence[complex]] | None = None,
    los_angle_of_arrival_rad: float = math.acos(0.7),
    min_speed_mps: float = 0.0,
    max_speed_mps: float | None = None,
    fixed_profile_conformance_case: str | None = None,
    num_sinusoids: int = 20,
    spec_version: str = "19.2",
    device: str = "cpu",
) -> SionnaTDLChannel:
    """Generate a reproducible Sionna TDL realization and map taps to CFR.

    ``num_time_steps`` are channel coefficient samples spaced by
    ``1/sampling_frequency_hz``. This adapter does not claim a complete NR
    waveform, CP, synchronization, or receiver model. ``rx_corr_mat`` is an
    optional Hermitian positive-definite unit-diagonal correlation matrix;
    its real and imaginary parts are retained in JSON-compatible metadata.
    ``channel_bandwidth_hz`` is the nominal RF channel bandwidth, distinct
    from the sampled subcarrier-grid span, and is required for fixed A10/D10
    dataset admission. ``fixed_profile_conformance_case`` optionally names one
    TS 38.101-4 V19.2.2 Annex B.2.2 fixed-profile/Doppler combination; when
    present, its profile, frequency range, and maximum Doppler are enforced.
    """
    applicability = standardized_profile_applicability("tdl", model)
    if not applicability.fixed_profile and spec_version not in applicability.supported_versions:
        raise ValueError(
            f"unsupported {applicability.standard} profile-table version: {spec_version}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for name, integer_value in (
        ("num_subcarriers", num_subcarriers),
        ("batch_size", batch_size),
        ("num_time_steps", num_time_steps),
        ("num_rx_ant", num_rx_ant),
        ("num_tx_ant", num_tx_ant),
        ("num_sinusoids", num_sinusoids),
    ):
        if isinstance(integer_value, bool) or not isinstance(integer_value, int):
            raise TypeError(f"{name} must be an integer")
    if num_subcarriers < 8:
        raise ValueError("num_subcarriers must be >= 8")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_time_steps <= 0:
        raise ValueError("num_time_steps must be positive")
    if num_rx_ant <= 0 or num_tx_ant <= 0 or num_sinusoids <= 0:
        raise ValueError("antenna counts and num_sinusoids must be positive")
    for name, scalar_value in (
        ("carrier_frequency_hz", carrier_frequency_hz),
        ("subcarrier_spacing_hz", subcarrier_spacing_hz),
        ("sampling_frequency_hz", sampling_frequency_hz),
    ):
        if (
            isinstance(scalar_value, bool)
            or not isinstance(scalar_value, Real)
            or scalar_value <= 0
            or not math.isfinite(scalar_value)
        ):
            raise ValueError(f"{name} must be finite and positive")
    if channel_bandwidth_hz is not None and (
        isinstance(channel_bandwidth_hz, bool)
        or not isinstance(channel_bandwidth_hz, Real)
        or not math.isfinite(channel_bandwidth_hz)
        or channel_bandwidth_hz <= 0
    ):
        raise ValueError("channel_bandwidth_hz must be finite and positive when provided")
    if (
        isinstance(los_angle_of_arrival_rad, bool)
        or not isinstance(los_angle_of_arrival_rad, Real)
        or not math.isfinite(los_angle_of_arrival_rad)
    ):
        raise ValueError("los_angle_of_arrival_rad must be finite")
    if delay_spread_s is not None and (
        isinstance(delay_spread_s, bool)
        or not isinstance(delay_spread_s, Real)
        or delay_spread_s < 0
        or not math.isfinite(delay_spread_s)
    ):
        raise ValueError("delay_spread_s must be finite and nonnegative")
    if (
        isinstance(min_speed_mps, bool)
        or not isinstance(min_speed_mps, Real)
        or min_speed_mps < 0
        or not math.isfinite(min_speed_mps)
    ):
        raise ValueError("min_speed_mps must be finite and nonnegative")
    if max_speed_mps is not None and (
        isinstance(max_speed_mps, bool)
        or not isinstance(max_speed_mps, Real)
        or max_speed_mps < min_speed_mps
        or not math.isfinite(max_speed_mps)
    ):
        raise ValueError("max_speed_mps must be finite and >= min_speed_mps")
    effective_max_speed_mps = min_speed_mps if max_speed_mps is None else max_speed_mps
    if fixed_profile_conformance_case is not None:
        conformance = FIXED_TDL_DOPPLER_CASES.get(fixed_profile_conformance_case)
        if conformance is None:
            raise ValueError("unknown TS 38.101-4 fixed-profile Doppler case")
        expected_model, expected_frequency_range, expected_doppler_hz = conformance
        if model != expected_model:
            raise ValueError(
                f"TS 38.101-4 case {fixed_profile_conformance_case} requires TDL-{expected_model}"
            )
        if carrier_frequency_range(carrier_frequency_hz) != expected_frequency_range:
            raise ValueError(
                f"TS 38.101-4 case {fixed_profile_conformance_case} requires "
                f"{expected_frequency_range}"
            )
        case_max_speed_mps = expected_doppler_hz * SPEED_OF_LIGHT_MPS / carrier_frequency_hz
        if max_speed_mps is not None and not math.isclose(
            max_speed_mps, case_max_speed_mps, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(
                f"TS 38.101-4 case {fixed_profile_conformance_case} requires "
                f"max_speed_mps={case_max_speed_mps:.12g}"
            )
        if min_speed_mps > case_max_speed_mps:
            raise ValueError(
                f"TS 38.101-4 case {fixed_profile_conformance_case} has "
                "min_speed_mps above its maximum-Doppler speed"
            )
        effective_max_speed_mps = case_max_speed_mps
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("device must be cpu or a CUDA device such as cuda:0")

    try:
        import torch
        from sionna.phy.channel.tr38901 import TDL, models
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional Sionna 2.2 PHY dependencies to generate TDL channels"
        ) from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")

    rx_corr_tensor = None
    if rx_corr_mat is not None:
        rx_corr_tensor = torch.as_tensor(rx_corr_mat, dtype=torch.complex64, device=device)
        expected_shape = (num_rx_ant, num_rx_ant)
        if tuple(rx_corr_tensor.shape) != expected_shape:
            raise ValueError(f"rx_corr_mat must have shape {expected_shape}")
        if not bool(torch.isfinite(rx_corr_tensor).all()):
            raise ValueError("rx_corr_mat must contain only finite values")
        if not torch.allclose(rx_corr_tensor, rx_corr_tensor.mH, rtol=0.0, atol=1e-6):
            raise ValueError("rx_corr_mat must be Hermitian")
        diagonal = torch.diagonal(rx_corr_tensor)
        if not torch.allclose(diagonal, torch.ones_like(diagonal), rtol=0.0, atol=1e-6):
            raise ValueError("rx_corr_mat must have a unit diagonal")
        if bool((torch.linalg.eigvalsh(rx_corr_tensor).min() <= 1e-7).item()):
            raise ValueError("rx_corr_mat must be positive definite")

    if model in _FIXED_TDL_MODELS and delay_spread_s is not None:
        raise ValueError(
            f"fixed-delay TDL-{model} uses absolute TS 38.101-4 tap delays; "
            "omit delay_spread_s instead of requesting a rescale"
        )
    if model in _SCALABLE_TDL_MODELS and delay_spread_s is None:
        raise ValueError(f"scalable TDL-{model} requires delay_spread_s")

    tdl = TDL(
        model=model,
        delay_spread=delay_spread_s,
        carrier_frequency=carrier_frequency_hz,
        num_sinusoids=num_sinusoids,
        min_speed=min_speed_mps,
        max_speed=effective_max_speed_mps,
        los_angle_of_arrival=los_angle_of_arrival_rad,
        num_rx_ant=num_rx_ant,
        num_tx_ant=num_tx_ant,
        rx_corr_mat=rx_corr_tensor,
        spec_version=spec_version,
        device=device,
    )
    parameter_filename = f"TDL-{model}.json"
    if model in _FIXED_TDL_MODELS:
        parameter_resource = models.fixed_tdl_parameter_file(parameter_filename)
        parameter_table_source = f"TS 38.101-4:19.2.2:{parameter_filename}"
    else:
        parameter_resource = models.parameter_file(parameter_filename, spec_version)
        parameter_table_source = f"TR 38.901:{spec_version}:{parameter_filename}"
    parameter_table_sha256 = hashlib.sha256(parameter_resource.read_bytes()).hexdigest()

    # Sionna 2.2 draws from its configured per-device generator.
    h_path, tau_s = call_with_sionna_seed(
        device,
        seed,
        lambda: tdl(
            batch_size=batch_size,
            num_time_steps=num_time_steps,
            sampling_frequency=sampling_frequency_hz,
        ),
    )

    # Move the path axis behind time to broadcast delays against both time
    # samples and centered OFDM baseband frequencies.
    h_time_path = h_path.movedim(-2, -1)
    tau_for_cfr = tau_s[:, :, None, :, None, None, :]
    frequencies = (
        torch.arange(num_subcarriers, dtype=tau_s.dtype, device=device) - num_subcarriers // 2
    ) * subcarrier_spacing_hz
    phase = torch.exp(
        -2j * torch.pi * tau_for_cfr.unsqueeze(-1) * frequencies.reshape(1, 1, 1, 1, 1, 1, 1, -1)
    )
    cfr = torch.sum(h_time_path.unsqueeze(-1) * phase, dim=-2)
    try:
        sionna_version = sionna_phy_version()
    except PackageNotFoundError:
        sionna_version = "unknown"
    k_factor_linear = float(tdl.k_factor.detach().cpu()) if tdl.los else None
    metadata: dict[str, Any] = {
            "sionna_version": sionna_version,
            "spec_version": spec_version,
            "fixed_profile_spec_version": "19.2.2" if model in _FIXED_TDL_MODELS else None,
            "profile_proxy": model,
            "profile_parameter_table_source": parameter_table_source,
            "profile_parameter_table_sha256": parameter_table_sha256,
            "delay_profile_semantics": (
                "fixed_ts_38_101_4_absolute_tap_delays"
                if model in _FIXED_TDL_MODELS
                else "scalable_tr_38_901_profile"
            ),
            "profile_rms_delay_spread_s": float(tdl.delay_spread.detach().cpu()),
            "profile_los_component": bool(tdl.los),
            "num_profile_taps": int(tdl.num_clusters),
            "nominal_tap_delays_s": tdl.delays.detach().cpu().tolist(),
            "mean_tap_powers": tdl.mean_powers.detach().cpu().tolist(),
            "k_factor_linear": k_factor_linear,
            "los_angle_of_arrival_rad": los_angle_of_arrival_rad,
            "geometric_los_label": None,
            "geometric_los_label_provided": False,
            "carrier_frequency_hz": carrier_frequency_hz,
            "sampled_bandwidth_hz": num_subcarriers * subcarrier_spacing_hz,
            "channel_bandwidth_hz": channel_bandwidth_hz,
            "rx_antenna_count": int(h_path.shape[2]),
            "tx_antenna_count": int(h_path.shape[4]),
            "rx_corr_mat_real": (
                rx_corr_tensor.real.detach().cpu().tolist() if rx_corr_tensor is not None else None
            ),
            "rx_corr_mat_imag": (
                rx_corr_tensor.imag.detach().cpu().tolist() if rx_corr_tensor is not None else None
            ),
            "min_speed_mps": min_speed_mps,
            "max_speed_mps": effective_max_speed_mps,
            "maximum_doppler_hz": effective_max_speed_mps
            * carrier_frequency_hz
            / SPEED_OF_LIGHT_MPS,
            "fixed_profile_conformance_case": fixed_profile_conformance_case,
            "num_sinusoids": num_sinusoids,
            "subcarrier_spacing_hz": subcarrier_spacing_hz,
            "num_subcarriers": num_subcarriers,
            "batch_size": batch_size,
            "batch_axis_semantics": "independent_stochastic_channel_draws",
            "num_time_steps": num_time_steps,
            "sampling_frequency_hz": sampling_frequency_hz,
            "time_axis_semantics": "samples_within_each_channel_draw",
            "time_sample_interval_s": 1.0 / sampling_frequency_hz,
            "temporal_evolution_observed": (num_time_steps > 1 and effective_max_speed_mps > 0.0),
            "seed": seed,
            "device": device,
    }
    metadata["resolved_profile_parameters_sha256"] = tdl_resolved_profile_parameter_sha256(
        metadata
    )
    return SionnaTDLChannel(
        h_path=h_path,
        tau_s=tau_s,
        channel_frequency_response=cfr,
        metadata=metadata,
    )
