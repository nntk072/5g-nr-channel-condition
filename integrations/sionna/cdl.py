"""Pinned Sionna TR 38.901 CDL-to-NR frequency-response adapter."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

from .random import call_with_sionna_seed


@dataclass(frozen=True, slots=True)
class CDLArrayConfig:
    """Serializable Sionna ``PanelArray`` configuration for one link endpoint."""

    num_rows_per_panel: int = 1
    num_cols_per_panel: int = 1
    polarization: str = "single"
    polarization_type: str = "V"
    antenna_pattern: str = "omni"
    num_rows: int = 1
    num_cols: int = 1
    panel_vertical_spacing: float | None = None
    panel_horizontal_spacing: float | None = None
    element_vertical_spacing: float | None = None
    element_horizontal_spacing: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "num_rows_per_panel",
            "num_cols_per_panel",
            "num_rows",
            "num_cols",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.polarization not in {"single", "dual"}:
            raise ValueError("polarization must be 'single' or 'dual'")
        allowed_types = {"V", "H"} if self.polarization == "single" else {"VH", "cross"}
        if self.polarization_type not in allowed_types:
            raise ValueError(
                f"polarization_type must be one of {sorted(allowed_types)} "
                f"for {self.polarization} polarization"
            )
        if self.antenna_pattern not in {"omni", "38.901", "38.901-handheld"}:
            raise ValueError("antenna_pattern must be 'omni', '38.901', or '38.901-handheld'")
        for name in (
            "panel_vertical_spacing",
            "panel_horizontal_spacing",
            "element_vertical_spacing",
            "element_horizontal_spacing",
        ):
            value = getattr(self, name)
            if value is not None and (value <= 0 or not math.isfinite(value)):
                raise ValueError(f"{name} must be finite and positive when set")


@dataclass(frozen=True, slots=True)
class SionnaCDLChannel:
    """CDL clusters and CFR at explicit channel time samples.

    ``h_path`` axes are [batch, rx, rx_ant, tx, tx_ant, cluster, time],
    ``tau_s`` axes are [batch, rx, tx, cluster], and CFR axes are
    [batch, rx, rx_ant, tx, tx_ant, time, subcarrier]. Cluster coefficients
    are standardized stochastic clusters, not geometric RT ray identities.
    """

    h_path: Any
    tau_s: Any
    channel_frequency_response: Any
    metadata: dict[str, Any]


def _validate_vector(name: str, value: Sequence[float] | None) -> list[float] | None:
    if value is None:
        return None
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three components")
    result = [float(component) for component in value]
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{name} components must be finite")
    return result


def _profile_metadata(
    model: str,
    spec_version: str,
    delay_spread_s: float,
    profile_resources: Any,
    torch_module: Any,
    device: str,
    delay_dtype: Any,
) -> dict[str, Any]:
    """Read the pinned source table and derive output cluster powers/delays."""
    table_path = Path(profile_resources.parameter_file(f"CDL-{model}.json", spec_version))
    parameters = profile_resources.load_json(table_path)
    raw_powers = [10.0 ** (float(power_db) / 10.0) for power_db in parameters["powers"]]
    power_sum = sum(raw_powers)
    normalized_powers = [power / power_sum for power in raw_powers]
    is_los = bool(parameters["los"])
    los_k_factor: float | None = None
    los_angles_deg: dict[str, float] | None = None

    if is_los:
        los_k_factor = normalized_powers[0] / sum(normalized_powers[1:])
        nlos_powers = [power / sum(normalized_powers[1:]) for power in normalized_powers[1:]]
        cluster_powers = [
            (los_k_factor + nlos_powers[0]) / (los_k_factor + 1.0),
            *(power / (los_k_factor + 1.0) for power in nlos_powers[1:]),
        ]
        los_angles_deg = {name: float(parameters[name][0]) for name in ("aod", "aoa", "zod", "zoa")}
        cluster_delays = parameters["delays"][1:]
        cluster_angles = {
            name: [float(value) for value in parameters[name][1:]]
            for name in ("aod", "aoa", "zod", "zoa")
        }
    else:
        cluster_powers = normalized_powers
        cluster_delays = parameters["delays"]
        cluster_angles = {
            name: [float(value) for value in parameters[name]]
            for name in ("aod", "aoa", "zod", "zoa")
        }

    cluster_order = (
        torch_module.argsort(
            torch_module.as_tensor(cluster_delays, dtype=delay_dtype, device=device)
        )
        .detach()
        .cpu()
        .tolist()
    )
    cluster_delays = [cluster_delays[index] for index in cluster_order]
    cluster_powers = [cluster_powers[index] for index in cluster_order]
    cluster_angles = {
        name: [values[index] for index in cluster_order] for name, values in cluster_angles.items()
    }

    cluster_delays_s = [float(delay) * delay_spread_s for delay in cluster_delays]

    return {
        "profile_los_component": is_los,
        "profile_k_factor_linear": los_k_factor,
        "num_profile_clusters": len(cluster_delays),
        "rays_per_cluster": 20,
        "nominal_cluster_delays_s": cluster_delays_s,
        "normalized_cluster_powers": cluster_powers,
        "nominal_cluster_angles_deg": cluster_angles,
        "nominal_los_angles_deg": los_angles_deg,
        "cluster_angle_spreads_deg": {
            "asd": float(parameters["cASD"]),
            "asa": float(parameters["cASA"]),
            "zsd": float(parameters["cZSD"]),
            "zsa": float(parameters["cZSA"]),
        },
        "cross_polarization_power_ratio_db": float(parameters["xpr"]),
        "parameter_table": table_path.name,
        "parameter_table_sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
    }


def generate_cdl_frequency_response(
    *,
    model: str,
    seed: int,
    carrier_frequency_hz: float,
    subcarrier_spacing_hz: float,
    num_subcarriers: int,
    delay_spread_s: float,
    batch_size: int = 1,
    num_time_steps: int = 1,
    sampling_frequency_hz: float = 1.0,
    ut_array: CDLArrayConfig = CDLArrayConfig(),
    bs_array: CDLArrayConfig = CDLArrayConfig(),
    direction: str = "downlink",
    ut_orientation_rad: Sequence[float] | None = None,
    bs_orientation_rad: Sequence[float] | None = None,
    ut_velocity_mps: Sequence[float] | None = None,
    min_speed_mps: float = 0.0,
    max_speed_mps: float | None = None,
    normalize_delays: bool = True,
    spec_version: str = "19.2",
    device: str = "cpu",
) -> SionnaCDLChannel:
    """Generate reproducible Sionna CDL taps and map them to centered CFR.

    CDL delay samples are profile excess delays scaled by ``delay_spread_s``;
    this is not absolute propagation time of flight. The pinned Sionna 2.2
    ``normalize_delays=False`` path returns raw profile-table values through a
    seconds-valued API, so this adapter rejects that mode. No geometric LOS
    label is produced: ``profile_los_component`` describes only the profile.
    """
    if model not in {"A", "B", "C", "D", "E"}:
        raise ValueError("model must be one of A, B, C, D, or E")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for name, value in (
        ("num_subcarriers", num_subcarriers),
        ("batch_size", batch_size),
        ("num_time_steps", num_time_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if num_subcarriers < 8:
        raise ValueError("num_subcarriers must be >= 8")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_time_steps <= 0:
        raise ValueError("num_time_steps must be positive")
    if direction not in {"downlink", "uplink"}:
        raise ValueError("direction must be 'downlink' or 'uplink'")
    if not isinstance(normalize_delays, bool):
        raise TypeError("normalize_delays must be a boolean")
    if not normalize_delays:
        raise ValueError(
            "normalize_delays=False is unsupported: pinned Sionna 2.2 returns "
            "unscaled CDL table values through tau_s; use delay-spread-scaled delays"
        )
    for scalar_name, scalar_value in (
        ("carrier_frequency_hz", carrier_frequency_hz),
        ("subcarrier_spacing_hz", subcarrier_spacing_hz),
        ("sampling_frequency_hz", sampling_frequency_hz),
    ):
        if scalar_value <= 0 or not math.isfinite(scalar_value):
            raise ValueError(f"{scalar_name} must be finite and positive")
    if delay_spread_s < 0 or not math.isfinite(delay_spread_s):
        raise ValueError("delay_spread_s must be finite and nonnegative")
    if min_speed_mps < 0 or not math.isfinite(min_speed_mps):
        raise ValueError("min_speed_mps must be finite and nonnegative")
    if max_speed_mps is not None and (
        max_speed_mps < min_speed_mps or not math.isfinite(max_speed_mps)
    ):
        raise ValueError("max_speed_mps must be finite and >= min_speed_mps")
    effective_speed_max = 0.0 if max_speed_mps is None else max_speed_mps
    if ut_velocity_mps is None and effective_speed_max < min_speed_mps:
        raise ValueError("max_speed_mps must be specified when min_speed_mps is positive")
    ut_orientation = _validate_vector("ut_orientation_rad", ut_orientation_rad)
    bs_orientation = _validate_vector("bs_orientation_rad", bs_orientation_rad)
    ut_velocity = _validate_vector("ut_velocity_mps", ut_velocity_mps)
    if ut_velocity is not None and (min_speed_mps != 0.0 or max_speed_mps not in (None, 0.0)):
        raise ValueError("set ut_velocity_mps or a nonzero speed interval, not both")
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("device must be cpu or a CUDA device such as cuda:0")

    try:
        import torch
        from sionna.phy.channel.tr38901 import CDL, PanelArray
        from sionna.phy.channel.tr38901 import models as profile_resources
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional Sionna 2.2 PHY dependencies to generate CDL channels"
        ) from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")

    ut_orientation_effective = ut_orientation or [math.pi, 0.0, 0.0]
    bs_orientation_effective = bs_orientation or [0.0, 0.0, 0.0]
    ut_array_object = PanelArray(
        **asdict(ut_array), carrier_frequency=carrier_frequency_hz, device=device
    )
    bs_array_object = PanelArray(
        **asdict(bs_array), carrier_frequency=carrier_frequency_hz, device=device
    )
    channel_model = CDL(
        model=model,
        delay_spread=delay_spread_s,
        carrier_frequency=carrier_frequency_hz,
        ut_array=ut_array_object,
        bs_array=bs_array_object,
        direction=direction,
        ut_orientation=ut_orientation_effective,
        bs_orientation=bs_orientation_effective,
        ut_velocity=ut_velocity,
        min_speed=min_speed_mps,
        max_speed=effective_speed_max,
        normalize_delays=normalize_delays,
        device=device,
        spec_version=spec_version,
    )
    h_path, tau_s = call_with_sionna_seed(
        device,
        seed,
        lambda: channel_model(
            batch_size=batch_size,
            num_time_steps=num_time_steps,
            sampling_frequency=sampling_frequency_hz,
        ),
    )

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
        sionna_version = version("sionna")
    except PackageNotFoundError:
        sionna_version = "unknown"
    metadata = _profile_metadata(
        model,
        spec_version,
        delay_spread_s,
        profile_resources,
        torch,
        device,
        tau_s.dtype,
    )
    if ut_velocity is not None:
        effective_max_speed_mps = math.sqrt(sum(component * component for component in ut_velocity))
        effective_min_speed_mps = effective_max_speed_mps
    else:
        effective_min_speed_mps = min_speed_mps
        effective_max_speed_mps = effective_speed_max
    metadata.update(
        {
            "sionna_version": sionna_version,
            "spec_version": spec_version,
            "profile_model": model,
            "geometric_los_label": None,
            "geometric_los_label_provided": False,
            "carrier_frequency_hz": carrier_frequency_hz,
            "subcarrier_spacing_hz": subcarrier_spacing_hz,
            "num_subcarriers": num_subcarriers,
            "batch_size": batch_size,
            "num_time_steps": num_time_steps,
            "sampling_frequency_hz": sampling_frequency_hz,
            "seed": seed,
            "device": device,
            "direction": direction,
            "delay_spread_s": delay_spread_s,
            "normalize_delays": normalize_delays,
            "delay_reference": "profile_excess_delay",
            "ut_array": asdict(ut_array),
            "bs_array": asdict(bs_array),
            "ut_orientation_rad": ut_orientation_effective,
            "bs_orientation_rad": bs_orientation_effective,
            "ut_velocity_mps": ut_velocity,
            "min_speed_mps": effective_min_speed_mps,
            "max_speed_mps": effective_max_speed_mps,
            "maximum_doppler_hz": effective_max_speed_mps * carrier_frequency_hz / 299_792_458.0,
            "rx_antenna_count": int(h_path.shape[2]),
            "tx_antenna_count": int(h_path.shape[4]),
        }
    )
    return SionnaCDLChannel(h_path, tau_s, cfr, metadata)
