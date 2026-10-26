"""Explicit-topology adapters for 3GPP TR 38.901 scenario GBSMs.

This module exposes one BS--one UT link per batch row for UMi, UMa, RMa, InH,
and InF. A batch is a collection of explicit channel drops, not multiple cells
or an assertion of spatial independence. The generated model outputs are
standardized stochastic paths, not geometric RT truth or conformance evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from numbers import Real
from typing import Any, Mapping

import numpy as np

from .cdl import CDLArrayConfig
from .random import call_with_sionna_seed

_SCENARIOS = frozenset({"UMi", "UMa", "RMa", "InH", "InF"})
_SCENARIO_OPTIONS = {
    "UMi": (frozenset({"o2i_model"}), frozenset({"o2i_model"})),
    "UMa": (frozenset({"o2i_model"}), frozenset({"o2i_model"})),
    "RMa": (
        frozenset(),
        frozenset({"average_street_width", "average_building_height", "car_window_type"}),
    ),
    "InH": (frozenset(), frozenset({"indoor_scenario"})),
    "InF": (
        frozenset(),
        frozenset(
            {
                "factory_scenario",
                "hall_dimensions",
                "clutter_density",
                "clutter_size",
                "clutter_height",
            }
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class SionnaScenarioGBSMChannel:
    """Path coefficients, delays, and centered-grid CFR from a scenario GBSM.

    ``h_path`` axes are ``[batch, rx, rx_ant, tx, tx_ant, path, time]``;
    ``tau_s`` axes are ``[batch, rx, tx, path]``; CFR axes are
    ``[batch, rx, rx_ant, tx, tx_ant, time, subcarrier]``. The adapter accepts
    exactly one BS and one UT per batch row.
    """

    h_path: Any
    tau_s: Any
    channel_frequency_response: Any
    metadata: dict[str, Any]


def _topology_tensor(torch: Any, name: str, value: Any, *, device: str) -> Any:
    """Convert one finite, real explicit-topology array to float32 on device."""
    if torch.is_tensor(value):
        if value.dtype == torch.bool or value.is_complex():
            raise TypeError(f"topology {name} must be a real numeric array")
        tensor = value.to(device=device, dtype=torch.float32)
    else:
        array = np.asarray(value)
        if array.dtype.kind not in "iuf" or array.dtype.kind == "b":
            raise TypeError(f"topology {name} must be a real numeric array")
        tensor = torch.as_tensor(array, dtype=torch.float32, device=device)
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"topology {name} must contain only finite values")
    return tensor


def _topology_state(torch: Any, name: str, value: Any, *, device: str) -> Any:
    """Convert an explicit Boolean topology state without truthy coercion."""
    if torch.is_tensor(value):
        if value.dtype != torch.bool:
            raise TypeError(f"topology {name} must have Boolean dtype")
        return value.to(device=device)
    array = np.asarray(value)
    if array.dtype.kind != "b":
        raise TypeError(f"topology {name} must have Boolean dtype")
    return torch.as_tensor(array, dtype=torch.bool, device=device)


def _preflight_numpy_topology(topology: Mapping[str, Any]) -> None:
    """Reject malformed NumPy/list topology before importing native dependencies."""
    if any(type(value).__module__.split(".", maxsplit=1)[0] == "torch" for value in topology.values()):
        return

    numeric: dict[str, np.ndarray] = {}
    for name in ("ut_loc", "bs_loc", "ut_velocities"):
        value = np.asarray(topology[name])
        if value.dtype.kind not in "iuf":
            raise TypeError(f"topology {name} must be a real numeric array")
        if not np.isfinite(value).all():
            raise ValueError(f"topology {name} must contain only finite values")
        numeric[name] = value
    batch_size = numeric["ut_loc"].shape[0] if numeric["ut_loc"].ndim == 3 else 0
    if batch_size <= 0:
        raise ValueError("topology ut_loc must have shape [positive batch, 1, 3]")
    for name, value in numeric.items():
        if value.shape != (batch_size, 1, 3):
            raise ValueError(f"topology {name} must have shape [positive batch, 1, 3]")

    for name in ("ut_orientations", "bs_orientations"):
        if name not in topology or topology[name] is None:
            continue
        value = np.asarray(topology[name])
        if value.dtype.kind not in "iuf":
            raise TypeError(f"topology {name} must be a real numeric array")
        if not np.isfinite(value).all():
            raise ValueError(f"topology {name} must contain only finite values")
        if value.shape != (batch_size, 1, 3):
            raise ValueError(f"topology {name} must have shape [batch, 1, 3]")

    for name, shape in (("in_state", (batch_size, 1)), ("los", (batch_size, 1, 1))):
        value = np.asarray(topology[name])
        if value.dtype.kind != "b":
            raise TypeError(f"topology {name} must have Boolean dtype")
        if value.shape != shape:
            raise ValueError(f"topology {name} must have shape {list(shape)}")


def _json_tensor(value: Any) -> Any:
    return value.detach().cpu().tolist()


def _topology_sha256(topology: Mapping[str, Any]) -> str:
    encoded = json.dumps(topology, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _native_array_layout(
    array: Any, *, polarization: str, polarization_type: str
) -> dict[str, Any]:
    """Capture the exact normalized element positions used by the native array."""
    try:
        normalized = array.normalized_positions
        components = []
        for axis in ("x", "y", "z"):
            component = getattr(normalized, axis)
            if hasattr(component, "detach") and hasattr(component, "cpu"):
                component = component.detach().cpu().numpy()
            elif hasattr(component, "numpy"):
                component = component.numpy()
            components.append(np.asarray(component, dtype=np.float64).reshape(-1))
        positions = np.column_stack(components)
        array_size = int(array.array_size)
        port_count = int(array.num_ant)
        patterns = tuple(array.antenna_pattern.patterns)
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError("Sionna PanelArray lacks the pinned normalized layout API") from error
    if positions.shape != (array_size, 3) or not np.isfinite(positions).all():
        raise ValueError("Sionna PanelArray normalized positions have an invalid shape or values")
    if not patterns or port_count != array_size * len(patterns):
        raise ValueError("Sionna PanelArray port count disagrees with its element/pattern layout")
    if polarization == "single":
        labels = [polarization_type]
    elif polarization_type == "VH":
        labels = ["V", "H"]
    else:
        labels = ["+45", "-45"]
    if len(labels) != len(patterns):
        raise ValueError("Sionna PanelArray pattern count disagrees with polarization settings")
    return {
        "schema": "umcd.sionna_array_layout.v1",
        "normalized_element_positions_wavelengths": positions.tolist(),
        "pattern_labels_in_port_order": labels,
        "port_order": "pattern_major_element_minor",
    }


def generate_scenario_gbsm_frequency_response(
    *,
    scenario: str,
    seed: int,
    carrier_frequency_hz: float,
    subcarrier_spacing_hz: float,
    num_subcarriers: int,
    topology: Mapping[str, Any],
    ut_array: CDLArrayConfig = CDLArrayConfig(),
    bs_array: CDLArrayConfig = CDLArrayConfig(),
    direction: str = "downlink",
    num_time_steps: int = 1,
    sampling_frequency_hz: float = 1.0,
    scenario_options: Mapping[str, Any] | None = None,
    enable_pathloss: bool = True,
    enable_shadow_fading: bool = True,
    spec_version: str = "19.2",
    device: str = "cpu",
) -> SionnaScenarioGBSMChannel:
    """Generate a reproducible one-link TR 38.901 scenario channel.

    ``topology`` must provide ``ut_loc``, ``bs_loc``, ``ut_velocities``,
    ``in_state``, and ``los``. Locations and velocities use ``[batch, 1, 3]``
    axes; indoor state uses ``[batch, 1]``; LOS state uses ``[batch, 1, 1]``.
    Optional endpoint orientations use ``[batch, 1, 3]`` and default to zero.
    Scenario-specific options are passed only through an explicit allowlist.

    The returned path delays are absolute scenario-model delays. No geometric
    visibility label is inferred from the model LOS input. This adapter does
    not create multiple-BS interference, establish independent batches, or
    qualify the model against 3GPP statistics.
    """
    if not isinstance(scenario, str) or scenario not in _SCENARIOS:
        raise ValueError(f"scenario must be one of {sorted(_SCENARIOS)}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for name, value in (
        ("carrier_frequency_hz", carrier_frequency_hz),
        ("subcarrier_spacing_hz", subcarrier_spacing_hz),
        ("sampling_frequency_hz", sampling_frequency_hz),
    ):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number")
        if not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if isinstance(num_subcarriers, bool) or not isinstance(num_subcarriers, int):
        raise TypeError("num_subcarriers must be an integer")
    if num_subcarriers < 8:
        raise ValueError("num_subcarriers must be at least 8")
    if isinstance(num_time_steps, bool) or not isinstance(num_time_steps, int):
        raise TypeError("num_time_steps must be an integer")
    if num_time_steps <= 0:
        raise ValueError("num_time_steps must be positive")
    if not isinstance(direction, str) or direction not in {"downlink", "uplink"}:
        raise ValueError("direction must be downlink or uplink")
    if not isinstance(enable_pathloss, bool) or not isinstance(enable_shadow_fading, bool):
        raise TypeError("pathloss and shadow-fading controls must be Boolean")
    if not isinstance(spec_version, str) or spec_version not in {"16.1", "19.2"}:
        raise ValueError("spec_version must be a supported TR 38.901 version: 16.1 or 19.2")
    if not isinstance(device, str) or (
        device != "cpu" and not device.startswith("cuda")
    ):
        raise ValueError("device must be cpu or a CUDA device such as cuda:0")
    if not isinstance(ut_array, CDLArrayConfig) or not isinstance(bs_array, CDLArrayConfig):
        raise TypeError("ut_array and bs_array must be CDLArrayConfig values")
    if not isinstance(topology, Mapping):
        raise TypeError("topology must be a mapping of explicit native topology fields")
    required_topology = {"ut_loc", "bs_loc", "ut_velocities", "in_state", "los"}
    optional_topology = {"ut_orientations", "bs_orientations"}
    if not required_topology.issubset(topology) or set(topology) - (
        required_topology | optional_topology
    ):
        raise ValueError("topology fields must match the explicit one-link topology contract")
    if scenario_options is None:
        options: dict[str, Any] = {}
    elif isinstance(scenario_options, Mapping):
        options = dict(scenario_options)
    else:
        raise TypeError("scenario_options must be a mapping")
    try:
        json.dumps(options, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("scenario_options must be finite JSON-compatible values") from error
    required_options, allowed_options = _SCENARIO_OPTIONS[scenario]
    if not required_options.issubset(options) or set(options) - allowed_options:
        raise ValueError(
            f"scenario_options for {scenario} must include {sorted(required_options)} "
            f"and use only {sorted(allowed_options)}"
        )
    if scenario in {"UMi", "UMa"} and (
        not isinstance(options["o2i_model"], str)
        or options["o2i_model"] not in {"low", "high"}
    ):
        raise ValueError("UMi/UMa o2i_model must be low or high")

    _preflight_numpy_topology(topology)
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for scenario GBSM generation") from error

    converted: dict[str, Any] = {}
    for name in ("ut_loc", "bs_loc", "ut_velocities"):
        converted[name] = _topology_tensor(torch, name, topology[name], device=device)
    batch_size = int(converted["ut_loc"].shape[0]) if converted["ut_loc"].ndim == 3 else 0
    for name in ("ut_loc", "bs_loc", "ut_velocities"):
        if tuple(converted[name].shape) != (batch_size, 1, 3) or batch_size <= 0:
            raise ValueError(f"topology {name} must have shape [positive batch, 1, 3]")
    for name in ("ut_orientations", "bs_orientations"):
        raw_value = topology.get(name)
        if raw_value is None:
            converted[name] = torch.zeros((batch_size, 1, 3), dtype=torch.float32, device=device)
        else:
            converted[name] = _topology_tensor(torch, name, raw_value, device=device)
            if tuple(converted[name].shape) != (batch_size, 1, 3):
                raise ValueError(f"topology {name} must have shape [batch, 1, 3]")
    converted["in_state"] = _topology_state(
        torch, "in_state", topology["in_state"], device=device
    )
    converted["los"] = _topology_state(torch, "los", topology["los"], device=device)
    if tuple(converted["in_state"].shape) != (batch_size, 1):
        raise ValueError("topology in_state must have shape [batch, 1]")
    if tuple(converted["los"].shape) != (batch_size, 1, 1):
        raise ValueError("topology los must have shape [batch, 1, 1]")

    try:
        from sionna.phy.channel.tr38901 import InF, InH, PanelArray, RMa, UMa, UMi
        sionna_version = version("sionna")
    except ImportError as error:
        raise RuntimeError("Sionna 2.2 PHY is required for scenario GBSM generation") from error
    except PackageNotFoundError as error:
        raise RuntimeError("Sionna package version metadata is required") from error
    if not sionna_version.startswith("2.2."):
        raise RuntimeError(
            f"scenario GBSM adapter requires Sionna 2.2.x, found {sionna_version}"
        )
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")

    native_topology = {name: _json_tensor(value) for name, value in converted.items()}
    topology_digest = _topology_sha256(native_topology)
    topology_rows = [
        {name: values[index] for name, values in native_topology.items()}
        for index in range(batch_size)
    ]
    topology_row_digests = [_topology_sha256(row) for row in topology_rows]
    scenario_los_rows = [bool(row["los"][0][0]) for row in topology_rows]
    ut_array_object = PanelArray(
        **asdict(ut_array), carrier_frequency=carrier_frequency_hz, device=device
    )
    bs_array_object = PanelArray(
        **asdict(bs_array), carrier_frequency=carrier_frequency_hz, device=device
    )
    model_class = {
        "UMi": UMi,
        "UMa": UMa,
        "RMa": RMa,
        "InH": InH,
        "InF": InF,
    }[scenario]
    model_options: dict[str, Any] = {
        "carrier_frequency": float(carrier_frequency_hz),
        "ut_array": ut_array_object,
        "bs_array": bs_array_object,
        "direction": direction,
        "enable_pathloss": enable_pathloss,
        "enable_shadow_fading": enable_shadow_fading,
        "device": device,
        "spec_version": spec_version,
        **options,
    }
    model = model_class(**model_options)
    model.set_topology(
        ut_loc=converted["ut_loc"],
        bs_loc=converted["bs_loc"],
        ut_orientations=converted["ut_orientations"],
        bs_orientations=converted["bs_orientations"],
        ut_velocities=converted["ut_velocities"],
        in_state=converted["in_state"],
        los=converted["los"],
    )
    h_path, tau_s = call_with_sionna_seed(
        device,
        seed,
        lambda: model(
            num_time_samples=num_time_steps,
            sampling_frequency=sampling_frequency_hz,
        ),
    )
    if h_path.ndim != 7 or tau_s.ndim != 4:
        raise ValueError("Sionna scenario model returned unexpected path/delay ranks")
    if h_path.shape[0] != batch_size or tau_s.shape[0] != batch_size:
        raise ValueError("Sionna scenario model changed the declared topology batch size")
    if (
        h_path.shape[1] != 1
        or h_path.shape[3] != 1
        or tuple(tau_s.shape[1:3]) != (1, 1)
    ):
        raise ValueError("Sionna scenario model returned more than one BS/UT link per row")
    if h_path.shape[-1] != num_time_steps or h_path.shape[-2] != tau_s.shape[-1]:
        raise ValueError("Sionna scenario path and delay axes do not align")
    invalid_path_values = (
        not h_path.is_complex()
        or tau_s.is_complex()
        or not bool(torch.isfinite(h_path).all().item())
        or not bool(torch.isfinite(tau_s).all().item())
        or bool(torch.any(tau_s < 0).item())
    )
    if invalid_path_values:
        raise ValueError("Sionna scenario model returned non-finite paths or delays")

    h_time_path = h_path.movedim(-2, -1)
    tau_for_cfr = tau_s[:, :, None, :, None, None, :]
    frequencies = (
        torch.arange(num_subcarriers, dtype=tau_s.dtype, device=device) - num_subcarriers // 2
    ) * subcarrier_spacing_hz
    phase = torch.exp(
        -2j
        * torch.pi
        * tau_for_cfr.unsqueeze(-1)
        * frequencies.reshape(1, 1, 1, 1, 1, 1, 1, -1)
    )
    cfr = torch.sum(h_time_path.unsqueeze(-1) * phase, dim=-2)
    if not bool(torch.isfinite(cfr).all().item()):
        raise ValueError("scenario CFR contains non-finite values")
    configuration = {
        "scenario": scenario,
        "standard": "3GPP TR 38.901",
        "spec_version": spec_version,
        "sionna_version": sionna_version,
        "seed": seed,
        "device": device,
        "direction": direction,
        "carrier_frequency_hz": float(carrier_frequency_hz),
        "subcarrier_spacing_hz": float(subcarrier_spacing_hz),
        "num_subcarriers": num_subcarriers,
        "num_time_steps": num_time_steps,
        "sampling_frequency_hz": float(sampling_frequency_hz),
        "batch_size": batch_size,
        "ut_array": asdict(ut_array),
        "bs_array": asdict(bs_array),
        "ut_array_layout": _native_array_layout(
            ut_array_object,
            polarization=ut_array.polarization,
            polarization_type=ut_array.polarization_type,
        ),
        "bs_array_layout": _native_array_layout(
            bs_array_object,
            polarization=bs_array.polarization,
            polarization_type=bs_array.polarization_type,
        ),
        "scenario_options": options,
        "topology_sha256": topology_digest,
        "pathloss_enabled": enable_pathloss,
        "shadow_fading_enabled": enable_shadow_fading,
        "spatial_consistency_enabled": False,
        "blockage_enabled": False,
    }
    configuration_digest = _canonical_sha256(configuration)
    metadata = {
        "schema": "umcd.sionna_scenario_gbsm.v1",
        "status": "NATIVE_GENERATION_CODE_ONLY_NOT_CONFORMANCE_QUALIFICATION",
        "scenario": scenario,
        "standard": "3GPP TR 38.901",
        "spec_version": spec_version,
        "sionna_version": sionna_version,
        "seed": seed,
        "device": device,
        "direction": direction,
        "carrier_frequency_hz": float(carrier_frequency_hz),
        "subcarrier_spacing_hz": float(subcarrier_spacing_hz),
        "num_subcarriers": num_subcarriers,
        "sampled_bandwidth_hz": num_subcarriers * float(subcarrier_spacing_hz),
        "batch_size": batch_size,
        "batch_semantics": "explicit_one_bs_one_ut_drops",
        "num_time_steps": num_time_steps,
        "sampling_frequency_hz": float(sampling_frequency_hz),
        "time_axis_semantics": "coefficient_samples_within_each_drop",
        "ut_array": asdict(ut_array),
        "bs_array": asdict(bs_array),
        "scenario_options": options,
        "topology": native_topology,
        "topology_sha256": topology_digest,
        "topology_rows": topology_rows,
        "topology_row_sha256": topology_row_digests,
        "scenario_los_rows": scenario_los_rows,
        "configuration_sha256": configuration_digest,
        "generator_configuration": configuration,
        "scenario_los_input": _json_tensor(converted["los"]),
        "geometric_los_label": None,
        "geometric_los_label_provided": False,
        "pathloss_enabled": enable_pathloss,
        "shadow_fading_enabled": enable_shadow_fading,
        "spatial_consistency_enabled": False,
        "blockage_enabled": False,
        "path_count": int(h_path.shape[-2]),
        "rx_antenna_count": int(h_path.shape[2]),
        "tx_antenna_count": int(h_path.shape[4]),
    }
    return SionnaScenarioGBSMChannel(h_path, tau_s, cfr, metadata)
