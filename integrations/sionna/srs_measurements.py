"""SRS positioning measurement primitives and strict report contracts.

The UL SRS-RSRP power averaging semantics follow ETSI TS 138 215 V19.2.0,
clause 5.2.5. Callers supply received per-RE power in watts at the declared
measurement reference point; this module does not calibrate RF power.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RECEIVER_REFERENCE_POINTS = {
    "terrestrial_type_1_c": "rx_antenna_connector",
    "terrestrial_type_1_o": "combined_antenna_elements_per_receiver_branch",
    "terrestrial_type_2_o": "combined_antenna_elements_per_receiver_branch",
    "terrestrial_type_1_h": "rx_transceiver_array_boundary_connector",
    "satellite_type_1_o": "combined_antenna_elements_per_receiver_branch",
    "satellite_type_1_h": "rx_transceiver_array_boundary_connector",
}
_TIMING_REFERENCE_POINTS = {
    "terrestrial_type_1_c": "rx_antenna_connector",
    "terrestrial_type_1_o": "rx_antenna_radiating_region_center",
    "terrestrial_type_2_o": "rx_antenna_radiating_region_center",
    "terrestrial_type_1_h": "rx_transceiver_array_boundary_connector",
}
_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "evaluation/schema/nr-ul-srs-rsrp-report.v1.schema.json"
)
_TIMING_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "evaluation/schema/nr-ul-srs-timing-report.v1.schema.json"
)
_GRID_OBSERVATION_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "evaluation/schema/nr-ul-srs-rsrp-grid-observation.v1.schema.json"
)
_NETWORK_GRID_OBSERVATION_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "evaluation/schema/nr-ul-srs-rsrp-network-observation.v1.schema.json"
)


def measure_ul_srs_rsrp(
    received_re_power_w: np.ndarray,
    configured_re_mask: np.ndarray,
    *,
    occasion_ids: Sequence[str],
    receiver_branch_ids: Sequence[str],
    receiver_category: str,
    measurement_frequency_bandwidth_hz: float,
    power_calibration_sha256: str,
) -> dict[str, Any]:
    """Build a linear-average UL SRS-RSRP report from calibrated RE powers.

    ``received_re_power_w`` has axes ``[measurement occasion, receiver branch,
    resource element]``. ``configured_re_mask`` has axes ``[occasion, RE]`` and
    selects only configured SRS REs in the declared measurement bandwidth and
    time occasions. Each receiver branch is averaged independently. The
    reported diversity value uses selection combining (the largest branch
    average), which satisfies the TS 38.215 lower-bound requirement that a
    diversity report not be below any individual branch. It is not a calibrated
    front-end or propagation model.
    """

    powers = np.asarray(received_re_power_w)
    mask = np.asarray(configured_re_mask)
    if powers.ndim != 3 or any(size <= 0 for size in powers.shape):
        raise ValueError("received_re_power_w must be nonempty [occasion, branch, RE]")
    if powers.dtype.kind not in "fiu" or not np.all(np.isfinite(powers)) or np.any(powers < 0):
        raise ValueError("received_re_power_w must contain finite nonnegative real watts")
    if mask.dtype.kind != "b" or mask.shape != (powers.shape[0], powers.shape[2]):
        raise ValueError("configured_re_mask must be Boolean [occasion, RE]")
    if not np.any(mask):
        raise ValueError("configured_re_mask must select at least one configured SRS RE")
    if np.any(np.sum(mask, axis=1) == 0):
        raise ValueError("every configured measurement occasion must contain an SRS RE")

    def validate_ids(values: Sequence[str], expected_count: int, label: str) -> list[str]:
        if isinstance(values, (str, bytes)):
            raise TypeError(f"{label} must be a sequence of IDs")
        result = list(values)
        if (
            len(result) != expected_count
            or any(
                not isinstance(value, str) or not value or value != value.strip()
                for value in result
            )
            or len(set(result)) != len(result)
        ):
            raise ValueError(f"{label} must contain {expected_count} unique canonical IDs")
        return result

    occasions = validate_ids(occasion_ids, powers.shape[0], "occasion_ids")
    branches = validate_ids(receiver_branch_ids, powers.shape[1], "receiver_branch_ids")
    if receiver_category not in _RECEIVER_REFERENCE_POINTS:
        raise ValueError(
            "receiver_category must be one of the TS 38.215 terrestrial/satellite receiver types"
        )
    if (
        isinstance(measurement_frequency_bandwidth_hz, (bool, np.bool_))
        or not isinstance(measurement_frequency_bandwidth_hz, Real)
        or not math.isfinite(float(measurement_frequency_bandwidth_hz))
        or measurement_frequency_bandwidth_hz <= 0
    ):
        raise ValueError("measurement_frequency_bandwidth_hz must be finite and positive")
    if not isinstance(power_calibration_sha256, str) or not _SHA256_RE.fullmatch(
        power_calibration_sha256
    ):
        raise ValueError("power_calibration_sha256 must be a lowercase SHA-256 digest")

    # Compute each branch over the same configured occasion/RE roster.
    branch_means = [
        float(np.mean(powers[:, branch_index, :][mask])) for branch_index in range(powers.shape[1])
    ]
    combined_w = max(branch_means)
    report = {
        "schema": "loslab.nr_ul_srs_rsrp_report.v1",
        "status": "MEASURED" if combined_w > 0 else "ZERO_POWER_DIAGNOSTIC",
        "measurement": "UL_SRS_RSRP",
        "standard": "3GPP_TS_38.215_V19.2.0_clause_5.2.5",
        "receiver_category": receiver_category,
        "measurement_reference_point": _RECEIVER_REFERENCE_POINTS[receiver_category],
        "measurement_frequency_bandwidth_hz": float(measurement_frequency_bandwidth_hz),
        "power_unit": "W",
        "power_averaging": "linear_average_of_configured_srs_re_power_contributions",
        "configured_measurement_occasion_ids": occasions,
        "configured_re_count_per_occasion": [int(np.sum(row)) for row in mask],
        "configured_re_count_total": int(np.sum(mask)),
        "power_calibration_sha256": power_calibration_sha256,
        "receiver_branch_measurements": [
            {
                "receiver_branch_id": branch_id,
                "ul_srs_rsrp_w": branch_mean,
                "ul_srs_rsrp_dbm": (
                    10.0 * math.log10(branch_mean / 1e-3) if branch_mean > 0 else None
                ),
            }
            for branch_id, branch_mean in zip(branches, branch_means, strict=True)
        ],
        "diversity_combining": "selection_combining_max_receiver_branch_power",
        "diversity_ul_srs_rsrp_w": combined_w,
        "diversity_ul_srs_rsrp_dbm": (
            10.0 * math.log10(combined_w / 1e-3) if combined_w > 0 else None
        ),
        "receiver_diversity_lower_bound_satisfied": combined_w >= max(branch_means),
        "limitations": [
            "Input watts and the power calibration digest are caller supplied and not independently verified.",
            "Selection combining is a declared bounded combiner; no RF receiver or propagation qualification is implied.",
            "The report covers only the supplied configured SRS resource elements and occasions.",
        ],
    }
    _validate_report_schema(report)
    return report


def measure_ul_srs_rsrp_from_grid(
    received_grid: np.ndarray,
    mapped_resource: Any,
    *,
    occasion_ids: Sequence[str],
    receiver_branch_ids: Sequence[str],
    receiver_category: str,
    measurement_frequency_bandwidth_hz: float,
    power_scale_w_per_grid_unit_squared: float,
    power_calibration_sha256: str,
    maximum_grid_cells: int = 10_000_000,
) -> dict[str, Any]:
    """Measure calibrated receive-grid SRS RE power and bind both input grids.

    ``received_grid`` uses ``[occasion, RX branch, slot symbol, subcarrier]``
    axes. Power is measured directly from the complex receive-grid samples on
    configured SRS REs. The calibration scale maps squared grid units to watts;
    noise and interference contributions present in those REs are retained.
    This is a receiver observation/report path, not interference rejection or
    receiver qualification.
    """

    from integrations.sionna.srs import (
        NRSRSHoppedMultiportResourceGrid,
        NRSRSMultiportResourceGrid,
        NRSRSPositioningMultiportResourceGrid,
    )

    supported = (
        NRSRSMultiportResourceGrid,
        NRSRSHoppedMultiportResourceGrid,
        NRSRSPositioningMultiportResourceGrid,
    )
    if not isinstance(mapped_resource, supported):
        raise TypeError("mapped_resource must be a mapped multiport SRS resource")
    received_source = np.asarray(received_grid)
    if (
        received_source.ndim != 4
        or received_source.dtype.kind != "c"
        or any(size <= 0 for size in received_source.shape)
    ):
        raise ValueError(
            "received_grid must be nonempty complex [occasion, RX branch, symbol, subcarrier]"
        )
    mapped_grid_source = np.asarray(mapped_resource.grid)
    mapped_mask_source = np.asarray(mapped_resource.mask)
    if received_source.shape[2:] != mapped_grid_source.shape[1:]:
        raise ValueError("received_grid symbol/subcarrier dimensions must match the SRS grid")
    if isinstance(maximum_grid_cells, bool) or not isinstance(maximum_grid_cells, int):
        raise TypeError("maximum_grid_cells must be an integer")
    if maximum_grid_cells <= 0:
        raise ValueError("maximum_grid_cells must be positive")
    total_grid_cells = received_source.size + mapped_grid_source.size + mapped_mask_source.size
    if total_grid_cells > maximum_grid_cells:
        raise ValueError("received and mapped SRS grids exceed maximum_grid_cells")
    # Snapshot all measured and provenance-bound arrays so their hashes describe
    # the exact values used below, even if a caller later mutates its buffers.
    received = np.array(received_source, copy=True, order="C")
    mapped_grid = np.array(mapped_grid_source, copy=True, order="C")
    mapped_mask = np.array(mapped_mask_source, copy=True, order="C")
    if not np.isfinite(received).all():
        raise ValueError("received_grid must contain only finite complex values")
    if (
        isinstance(power_scale_w_per_grid_unit_squared, (bool, np.bool_))
        or not isinstance(power_scale_w_per_grid_unit_squared, Real)
        or not math.isfinite(float(power_scale_w_per_grid_unit_squared))
        or power_scale_w_per_grid_unit_squared <= 0
    ):
        raise ValueError("power scale must be finite and positive real watts per grid unit squared")

    configured_mask = np.any(mapped_mask, axis=0)
    if not np.any(configured_mask):
        raise ValueError("mapped SRS resource must occupy at least one resource element")
    grid64 = received.astype(np.complex128, copy=False)
    with np.errstate(over="ignore", invalid="ignore"):
        power = (grid64.real * grid64.real + grid64.imag * grid64.imag) * float(
            power_scale_w_per_grid_unit_squared
        )
    if not np.isfinite(power).all():
        raise ValueError("calibrated received-grid power is not finite")
    flattened_power = power.reshape(received.shape[0], received.shape[1], -1)
    flattened_mask = np.broadcast_to(
        configured_mask, (received.shape[0], *configured_mask.shape)
    ).reshape(received.shape[0], -1)
    flattened_power[:, :, ~flattened_mask[0]] = 0.0

    report = measure_ul_srs_rsrp(
        flattened_power,
        flattened_mask,
        occasion_ids=occasion_ids,
        receiver_branch_ids=receiver_branch_ids,
        receiver_category=receiver_category,
        measurement_frequency_bandwidth_hz=measurement_frequency_bandwidth_hz,
        power_calibration_sha256=power_calibration_sha256,
    )
    wrapper = {
        "schema": "loslab.nr_ul_srs_rsrp_grid_observation.v1",
        "status": "OBSERVED_NOT_QUALIFIED",
        "measurement_report": report,
        "received_grid_sha256": _array_sha256(received),
        "mapped_resource_sha256": _mapped_resource_sha256(
            mapped_resource, grid=mapped_grid, mask=mapped_mask
        ),
        "power_scale_w_per_grid_unit_squared": float(power_scale_w_per_grid_unit_squared),
        "power_calibration_sha256": power_calibration_sha256,
        "received_re_power_includes_noise_and_interference": True,
        "limitations": [
            "The power scale and calibration digest are caller supplied and not independently calibrated.",
            "Observed receive-grid power retains noise and interference on configured SRS REs.",
            "The workflow does not perform synchronization, interference rejection, or receiver qualification.",
        ],
    }
    _validate_grid_observation_schema(wrapper)
    return wrapper


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    header = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _mapped_resource_sha256(
    mapped_resource: Any, *, grid: np.ndarray | None = None, mask: np.ndarray | None = None
) -> str:
    identity = {
        "type": type(mapped_resource).__name__,
        "symbol_indices": list(mapped_resource.symbol_indices),
        "subcarrier_indices": getattr(mapped_resource, "subcarrier_indices", None),
        "subcarrier_indices_by_symbol": getattr(
            mapped_resource, "subcarrier_indices_by_symbol", None
        ),
        "position_indices_by_symbol": getattr(mapped_resource, "position_indices_by_symbol", None),
        "positioning_offsets": getattr(mapped_resource, "positioning_offsets", None),
        "n_srs_by_symbol": getattr(mapped_resource, "n_srs_by_symbol", None),
        "active_port_symbols": getattr(mapped_resource, "active_port_symbols", None),
        "grid_sha256": _array_sha256(mapped_resource.grid if grid is None else grid),
        "mask_sha256": _array_sha256(mapped_resource.mask if mask is None else mask),
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_grid_observation_schema(report: dict[str, Any]) -> None:
    import jsonschema

    schema = json.loads(_GRID_OBSERVATION_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(report))
    if errors:
        raise ValueError(
            "SRS receive-grid observation violates schema: "
            + "; ".join(error.message for error in errors[:10])
        )


def measure_ul_srs_rsrp_from_noisy_network_grid(
    noisy_network_grid: Any,
    network_composition: Any,
    mapped_resource: Any,
    *,
    receiver_index: int,
    desired_transmitter_id: str,
    transmitter_ids: Sequence[str],
    occasion_ids: Sequence[str],
    receiver_branch_ids: Sequence[str],
    receiver_category: str,
    measurement_frequency_bandwidth_hz: float,
    power_scale_w_per_grid_unit_squared: float,
    power_calibration_sha256: str,
    maximum_grid_cells: int = 10_000_000,
) -> dict[str, Any]:
    """Measure SRS-RSRP from the noisy multi-transmitter network-grid path.

    This adapter binds the receiver-grid observation to the coherent per-TX
    contributions and independent receiver-noise grid that produced it. Batch
    draws become measurement occasions and receive ports become branches.
    """

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Install PyTorch to measure a composed network grid") from error

    from integrations.sionna.network_phy import NetworkGridComposition, NoisyNetworkGrid

    if not isinstance(noisy_network_grid, NoisyNetworkGrid):
        raise TypeError("noisy_network_grid must come from add_independent_receiver_awgn")
    if not isinstance(network_composition, NetworkGridComposition):
        raise TypeError("network_composition must come from compose_received_grids")
    if (
        isinstance(receiver_index, bool)
        or not isinstance(receiver_index, int)
        or receiver_index < 0
    ):
        raise ValueError("receiver_index must be a nonnegative integer")

    contributions = network_composition.source_contributions
    composed_grid = network_composition.received_grid
    observed_grid = noisy_network_grid.received_grid
    noise_grid = noisy_network_grid.noise_grid
    if (
        not torch.is_tensor(contributions)
        or not torch.is_complex(contributions)
        or contributions.ndim != 6
    ):
        raise ValueError(
            "network contributions must have [batch, rx, rx_port, tx, symbol, sc] axes"
        )
    if (
        not torch.is_tensor(composed_grid)
        or not torch.is_complex(composed_grid)
        or composed_grid.ndim != 5
        or not torch.is_tensor(observed_grid)
        or not torch.is_complex(observed_grid)
        or not torch.is_tensor(noise_grid)
        or not torch.is_complex(noise_grid)
        or tuple(observed_grid.shape) != tuple(composed_grid.shape)
        or tuple(noise_grid.shape) != tuple(composed_grid.shape)
        or tuple(contributions.shape[:3] + contributions.shape[4:]) != tuple(composed_grid.shape)
    ):
        raise ValueError("network and noisy grid axes are incompatible")
    if receiver_index >= contributions.shape[1]:
        raise ValueError("receiver_index must identify an existing receiver")
    if (
        isinstance(maximum_grid_cells, bool)
        or not isinstance(maximum_grid_cells, int)
        or maximum_grid_cells <= 0
    ):
        raise ValueError("maximum_grid_cells must be a positive integer")
    if any(
        tensor.device != contributions.device
        for tensor in (composed_grid, observed_grid, noise_grid)
    ):
        raise ValueError("network and noise grids must share one device")
    if any(
        tensor.dtype != contributions.dtype for tensor in (composed_grid, observed_grid, noise_grid)
    ):
        raise ValueError("network and noise grids must share one complex dtype")
    noise_variance = noisy_network_grid.noise_variance_w_per_re
    if (
        not torch.is_tensor(noise_variance)
        or noise_variance.is_complex()
        or noise_variance.shape != (contributions.shape[1],)
        or not bool(torch.isfinite(noise_variance).all())
        or bool((noise_variance < 0).any())
    ):
        raise ValueError("noise_variance_w_per_re must be finite nonnegative values per receiver")
    noise_seed = noisy_network_grid.seed
    if isinstance(noise_seed, bool) or not isinstance(noise_seed, int) or noise_seed < 0:
        raise ValueError("noise seed must be a nonnegative integer")

    occasion_count, branch_count, transmitter_count = (
        contributions.shape[0],
        contributions.shape[2],
        contributions.shape[3],
    )
    transmitter_roster = _canonical_id_roster(transmitter_ids, transmitter_count, "transmitter_ids")
    if (
        not isinstance(desired_transmitter_id, str)
        or not desired_transmitter_id
        or desired_transmitter_id != desired_transmitter_id.strip()
        or desired_transmitter_id not in transmitter_roster
    ):
        raise ValueError("desired_transmitter_id must identify one declared transmitter")
    received_slice = observed_grid[:, receiver_index]
    composed_slice = composed_grid[:, receiver_index]
    noise_slice = noise_grid[:, receiver_index]
    contribution_slice = contributions[:, receiver_index]
    mapped_grid = np.asarray(mapped_resource.grid)
    mapped_mask = np.asarray(mapped_resource.mask)
    grid_cells = (
        received_slice.numel()
        + noise_slice.numel()
        + contribution_slice.numel()
        + mapped_grid.size
        + mapped_mask.size
    )
    if grid_cells > maximum_grid_cells:
        raise ValueError("selected network and mapped-resource grids exceed maximum_grid_cells")
    if any(
        not bool(torch.isfinite(values).all())
        for values in (composed_slice, received_slice, noise_slice, contribution_slice)
    ):
        raise ValueError("selected network and noise grids must contain finite complex values")
    if not torch.allclose(composed_slice, contribution_slice.sum(dim=2), rtol=1e-5, atol=1e-7):
        raise ValueError("network received grid must equal the coherent sum of TX contributions")
    if not torch.equal(received_slice, composed_slice + noise_slice):
        raise ValueError("noisy network grid must equal the composed grid plus its noise grid")

    received_array = _torch_array_snapshot(received_slice)
    noise_array = _torch_array_snapshot(noise_slice)
    contribution_array = _torch_array_snapshot(contribution_slice)
    observation = measure_ul_srs_rsrp_from_grid(
        received_array,
        mapped_resource,
        occasion_ids=occasion_ids,
        receiver_branch_ids=receiver_branch_ids,
        receiver_category=receiver_category,
        measurement_frequency_bandwidth_hz=measurement_frequency_bandwidth_hz,
        power_scale_w_per_grid_unit_squared=power_scale_w_per_grid_unit_squared,
        power_calibration_sha256=power_calibration_sha256,
        maximum_grid_cells=maximum_grid_cells,
    )
    configured_mask = np.any(mapped_mask, axis=0)
    selected_contributions = contribution_array[:, :, :, configured_mask].astype(
        np.complex128, copy=False
    )
    selected_noise = noise_array[:, :, configured_mask].astype(np.complex128, copy=False)
    scale = float(power_scale_w_per_grid_unit_squared)
    transmitter_power = np.mean(np.abs(selected_contributions) ** 2, axis=(0, 3)) * scale
    desired_index = transmitter_roster.index(desired_transmitter_id)
    desired_power = transmitter_power[:, desired_index]
    interference_indices = [index for index in range(transmitter_count) if index != desired_index]
    coherent_interference = (
        np.take(selected_contributions, interference_indices, axis=2).sum(axis=2)
        if interference_indices
        else np.zeros_like(selected_noise)
    )
    interference_power = np.mean(np.abs(coherent_interference) ** 2, axis=(0, 2)) * scale
    noise_power = np.mean(np.abs(selected_noise) ** 2, axis=(0, 2)) * scale
    if (
        not np.isfinite(transmitter_power).all()
        or not np.isfinite(interference_power).all()
        or not np.isfinite(noise_power).all()
    ):
        raise ValueError("network component powers are not representable in float64")
    branch_component_measurements = []
    for branch_index, branch_id in enumerate(receiver_branch_ids):
        denominator = float(interference_power[branch_index] + noise_power[branch_index])
        branch_component_measurements.append(
            {
                "receiver_branch_id": branch_id,
                "desired_transmitter_power_w": float(desired_power[branch_index]),
                "coherent_interference_power_w": float(interference_power[branch_index]),
                "measured_receiver_noise_power_w": float(noise_power[branch_index]),
                "network_sinr_linear": (
                    float(desired_power[branch_index] / denominator) if denominator > 0 else None
                ),
            }
        )
    if any(
        value["network_sinr_linear"] is not None and not math.isfinite(value["network_sinr_linear"])
        for value in branch_component_measurements
    ):
        raise ValueError("network component SINR is not representable in float64")
    report = {
        "schema": "loslab.nr_ul_srs_rsrp_network_observation.v1",
        "status": "OBSERVED_NOT_QUALIFIED",
        "receiver_index": receiver_index,
        "desired_transmitter_id": desired_transmitter_id,
        "transmitter_ids": transmitter_roster,
        "measurement_occasion_count": occasion_count,
        "receiver_branch_count": branch_count,
        "noise_seed": noise_seed,
        "noise_variance_w_per_re": _torch_array_snapshot(noise_variance).tolist(),
        "received_grid_sha256": observation["received_grid_sha256"],
        "mapped_resource_sha256": observation["mapped_resource_sha256"],
        "noise_grid_sha256": _array_sha256(noise_array),
        "transmitter_contributions_sha256": _array_sha256(contribution_array),
        "measurement_report": observation["measurement_report"],
        "network_components": {
            "measurement": "OBSERVED_CONFIGURED_SRS_RE_POWER",
            "combiner": "coherent_sum_of_non_desired_transmitters",
            "transmitter_power_w_by_id": {
                tx_id: float(transmitter_power[:, index].mean())
                for index, tx_id in enumerate(transmitter_roster)
            },
            "receiver_branches": branch_component_measurements,
            "limitations": [
                "Component powers use the caller-supplied watts-per-grid-unit-squared scale.",
                "Interference is coherently summed before power measurement; reported SRS-RSRP uses the full received grid.",
                "Component decomposition is simulated network truth for this composition, not measured RF separation or receiver qualification.",
            ],
        },
        "limitations": [
            "The watts-per-grid-unit-squared scale and calibration digest are caller supplied.",
            "Per-transmitter contributions are provenance only; SRS-RSRP is measured from their coherent sum plus receiver noise.",
            "The network composer assumes synchronized same-numerology grids and does not qualify an RF receiver.",
        ],
    }
    _validate_network_grid_observation_schema(report)
    return report


def _canonical_id_roster(values: Sequence[str], count: int, name: str) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of IDs")
    result = list(values)
    if (
        len(result) != count
        or any(
            not isinstance(value, str) or not value or value != value.strip() for value in result
        )
        or len(set(result)) != len(result)
    ):
        raise ValueError(f"{name} must contain {count} unique canonical IDs")
    return result


def _torch_array_snapshot(values: Any) -> np.ndarray:
    return np.array(values.detach().cpu().numpy(), copy=True, order="C")


def _validate_network_grid_observation_schema(report: dict[str, Any]) -> None:
    import jsonschema

    schema = json.loads(_NETWORK_GRID_OBSERVATION_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(report))
    if errors:
        raise ValueError(
            "SRS network-grid observation violates schema: "
            + "; ".join(error.message for error in errors[:10])
        )


def report_ul_rtoa(
    *,
    received_subframe_start_time_s: float,
    sfn: int,
    subframe_number: int,
    sfn_zero_reference_time_s: float,
    receiver_category: str,
    timing_calibration_sha256: str,
) -> dict[str, Any]:
    """Report UL-RTOA from a separately detected received SRS subframe start.

    TS 38.215 defines the reference instant as ``T0 + (10*SFN + subframe)*1 ms``.
    The received start time and SFN-zero time must use the same calibrated clock.
    This function binds the timing result; it does not detect the first path or
    establish clock calibration.
    """

    if receiver_category not in _TIMING_REFERENCE_POINTS:
        raise ValueError("receiver_category must be a terrestrial TS 38.104 base-station type")
    if isinstance(sfn, bool) or not isinstance(sfn, int) or not 0 <= sfn <= 1023:
        raise ValueError("sfn must be an integer from 0 through 1023")
    if (
        isinstance(subframe_number, bool)
        or not isinstance(subframe_number, int)
        or not 0 <= subframe_number <= 9
    ):
        raise ValueError("subframe_number must be an integer from 0 through 9")
    received = _finite_real(received_subframe_start_time_s, "received_subframe_start_time_s")
    t0 = _finite_real(sfn_zero_reference_time_s, "sfn_zero_reference_time_s")
    digest = _validate_digest(timing_calibration_sha256, "timing_calibration_sha256")
    srs_time = (10 * sfn + subframe_number) * 1e-3
    reference_time = t0 + srs_time
    report = {
        "schema": "loslab.nr_ul_srs_timing_report.v1",
        "measurement": "UL_RTOA",
        "standard": "3GPP_TS_38.215_V19.2.0_clause_5.2.2",
        "status": "REPORTED",
        "receiver_category": receiver_category,
        "receiver_reference_point": _TIMING_REFERENCE_POINTS[receiver_category],
        "sfn": sfn,
        "subframe_number": subframe_number,
        "sfn_zero_reference_time_s": t0,
        "srs_subframe_offset_from_sfn_zero_s": srs_time,
        "ul_rtoa_reference_time_s": reference_time,
        "received_subframe_start_time_s": received,
        "ul_rtoa_s": received - reference_time,
        "timing_calibration_sha256": digest,
        "limitations": [
            "The received subframe start and SFN-zero reference are caller supplied.",
            "First-path detection, SRS acquisition, and clock calibration are not performed or verified.",
            "This report is arithmetic evidence only and does not qualify a positioning receiver.",
        ],
    }
    _validate_timing_report_schema(report)
    return report


def report_gnb_rx_tx_time_difference(
    *,
    received_uplink_subframe_time_s: float,
    transmitted_downlink_subframe_time_s: float,
    receiver_category: str,
    transmitter_category: str,
    timing_calibration_sha256: str,
) -> dict[str, Any]:
    """Report gNB Rx-Tx timing from caller-supplied TRP timing observations."""

    if receiver_category not in _TIMING_REFERENCE_POINTS:
        raise ValueError("receiver_category must be a terrestrial TS 38.104 base-station type")
    if transmitter_category not in _TIMING_REFERENCE_POINTS:
        raise ValueError("transmitter_category must be a terrestrial TS 38.104 base-station type")
    received = _finite_real(received_uplink_subframe_time_s, "received_uplink_subframe_time_s")
    transmitted = _finite_real(
        transmitted_downlink_subframe_time_s, "transmitted_downlink_subframe_time_s"
    )
    digest = _validate_digest(timing_calibration_sha256, "timing_calibration_sha256")
    report = {
        "schema": "loslab.nr_ul_srs_timing_report.v1",
        "measurement": "GNB_RX_TX_TIME_DIFFERENCE",
        "standard": "3GPP_TS_38.215_V19.2.0_clause_5.2.3",
        "status": "REPORTED",
        "receiver_category": receiver_category,
        "receiver_reference_point": _TIMING_REFERENCE_POINTS[receiver_category],
        "transmitter_category": transmitter_category,
        "transmitter_reference_point": _TIMING_REFERENCE_POINTS[transmitter_category].replace(
            "rx_", "tx_", 1
        ),
        "received_uplink_subframe_time_s": received,
        "transmitted_downlink_subframe_time_s": transmitted,
        "gnb_rx_tx_time_difference_s": received - transmitted,
        "timing_calibration_sha256": digest,
        "limitations": [
            "The selected uplink and downlink TRP timing observations are caller supplied.",
            "The closest downlink subframe selection and first-path detection are not verified.",
            "This report is arithmetic evidence only and does not qualify a positioning receiver.",
        ],
    }
    _validate_timing_report_schema(report)
    return report


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _validate_digest(value: str, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_report_schema(report: dict[str, Any]) -> None:
    import jsonschema

    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(report))
    if errors:
        raise ValueError(
            "UL SRS-RSRP report violates schema: "
            + "; ".join(error.message for error in errors[:10])
        )


def _validate_timing_report_schema(report: dict[str, Any]) -> None:
    import jsonschema

    schema = json.loads(_TIMING_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(report))
    if errors:
        raise ValueError(
            "UL SRS timing report violates schema: "
            + "; ".join(error.message for error in errors[:10])
        )
