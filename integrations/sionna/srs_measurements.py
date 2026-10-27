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
_AOA_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "evaluation/schema/nr-ul-srs-aoa-report.v1.schema.json"
)
_RSRPP_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "evaluation/schema/nr-ul-srs-rsrpp-report.v1.schema.json"
)
_TDCT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "evaluation/schema/nr-ul-srs-tdct-tdcp-report.v1.schema.json"
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


def report_ul_aoa(
    *,
    sample_id: str,
    azimuth_deg: float,
    vertical_deg: float,
    coordinate_system: str,
    reference_frame_id: str,
    estimated_channel_sha256: str,
    estimator_config_sha256: str,
    antenna_calibration_sha256: str,
) -> dict[str, Any]:
    """Bind a supplied UL-AoA estimate to its channel, estimator, and array calibration.

    TS 38.215 defines the angle directions and places the estimate at the gNB
    antenna. This report does not estimate the angles, validate the array
    calibration, or claim receiver qualification.
    """

    if not isinstance(sample_id, str) or not sample_id or sample_id != sample_id.strip():
        raise ValueError("sample_id must be a canonical non-empty string")
    if coordinate_system not in {"GCS", "LCS"}:
        raise ValueError("coordinate_system must be GCS or LCS")
    if (
        not isinstance(reference_frame_id, str)
        or not reference_frame_id
        or (reference_frame_id != reference_frame_id.strip())
    ):
        raise ValueError("reference_frame_id must be a canonical non-empty string")
    azimuth = _finite_real(azimuth_deg, "azimuth_deg")
    vertical = _finite_real(vertical_deg, "vertical_deg")
    if not 0.0 <= azimuth < 360.0:
        raise ValueError("azimuth_deg must be in [0, 360)")
    if not 0.0 <= vertical <= 180.0:
        raise ValueError("vertical_deg must be in [0, 180]")
    source_digest = _validate_digest(estimated_channel_sha256, "estimated_channel_sha256")
    estimator_digest = _validate_digest(estimator_config_sha256, "estimator_config_sha256")
    calibration_digest = _validate_digest(antenna_calibration_sha256, "antenna_calibration_sha256")
    report = {
        "schema": "loslab.nr_ul_aoa_report.v1",
        "measurement": "UL_AOA",
        "standard": "3GPP_TS_38.215_V19.2.0_clause_5.2.4",
        "status": "OBSERVED_NOT_QUALIFIED",
        "sample_id": sample_id,
        "measurement_location": "gnb_antenna",
        "coordinate_system": coordinate_system,
        "reference_frame_id": reference_frame_id,
        "azimuth_reference": "geographical_north" if coordinate_system == "GCS" else "local_x_axis",
        "azimuth_positive_direction": "counter_clockwise",
        "vertical_reference": "zenith" if coordinate_system == "GCS" else "local_z_axis",
        "vertical_positive_direction": "toward_horizontal_plane",
        "azimuth_deg": azimuth,
        "vertical_deg": vertical,
        "estimated_channel_sha256": source_digest,
        "estimator_config_sha256": estimator_digest,
        "antenna_calibration_sha256": calibration_digest,
        "limitations": [
            "Angles are caller-supplied estimates; this function does not run an AoA estimator.",
            "The reference frame and antenna calibration are identified by caller-supplied declarations.",
            "This report does not validate calibration, synchronization, uncertainty, or receiver performance.",
        ],
    }
    _validate_aoa_report_schema(report)
    return report


def report_ul_srs_rsrpp(
    path_channel_response: np.ndarray,
    path_delay_s: np.ndarray,
    path_valid_mask: np.ndarray,
    *,
    occasion_ids: Sequence[str],
    receiver_branch_ids: Sequence[str],
    receiver_category: str,
    measurement_frequency_bandwidth_hz: float,
    watts_per_channel_response_unit_squared: float,
    power_calibration_sha256: str,
    channel_estimator_sha256: str,
    path_detector_sha256: str,
    maximum_measurement_values: int = 1_000_000,
) -> dict[str, Any]:
    """Report UL SRS-RSRPP from caller-detected, calibrated path responses.

    Inputs have axes ``[occasion, receiver branch, path candidate]`` for the
    complex path response and ``[occasion, path candidate]`` for delays/mask.
    Each complex response is the caller's linear average of estimated channel
    response at that path delay over the configured SRS resource elements for
    one occasion. This function converts its squared magnitude to watts and
    marks the earliest valid delay as first path. It does not estimate the
    channel, detect paths, associate path identities across occasions, or
    validate the calibration/estimator declarations.
    """
    raw_response = np.asarray(path_channel_response)
    raw_delays = np.asarray(path_delay_s)
    raw_mask = np.asarray(path_valid_mask)
    if (
        raw_response.ndim != 3
        or any(size <= 0 for size in raw_response.shape[:2])
        or raw_response.shape[2] <= 0
        or raw_response.dtype.kind != "c"
    ):
        raise ValueError("path_channel_response must be nonempty complex [occasion, branch, path]")
    occasion_count, branch_count, path_count = raw_response.shape
    if raw_delays.dtype.kind not in "iuf" or raw_delays.shape != (occasion_count, path_count):
        raise ValueError("path_delay_s must be real [occasion, path]")
    if raw_mask.dtype.kind != "b" or raw_mask.shape != (occasion_count, path_count):
        raise ValueError("path_valid_mask must be Boolean [occasion, path]")
    if isinstance(maximum_measurement_values, bool) or not isinstance(
        maximum_measurement_values, int
    ):
        raise TypeError("maximum_measurement_values must be an integer")
    if maximum_measurement_values <= 0:
        raise ValueError("maximum_measurement_values must be positive")
    total_values = raw_response.size + raw_delays.size + raw_mask.size
    if total_values > maximum_measurement_values:
        raise ValueError("UL SRS-RSRPP inputs exceed maximum_measurement_values")

    # Hash and score immutable canonical snapshots of the exact admitted arrays.
    with np.errstate(over="ignore", invalid="ignore"):
        response = np.array(raw_response, dtype=np.complex128, copy=True, order="C")
        delays = np.array(raw_delays, dtype=np.float64, copy=True, order="C")
    valid = np.array(raw_mask, dtype=np.bool_, copy=True, order="C")
    if not np.isfinite(response).all() or not np.isfinite(delays).all():
        raise ValueError("path responses and delays must be finite")
    if np.any(delays < 0):
        raise ValueError("path delays must be nonnegative")
    if np.any(response * np.logical_not(valid[:, None, :]) != 0):
        raise ValueError("invalid path response entries must be exactly zero")
    if np.any(delays[~valid] != 0):
        raise ValueError("invalid path delay entries must be exactly zero")
    for occasion_index in range(occasion_count):
        active_delays = delays[occasion_index, valid[occasion_index]]
        if np.any(np.diff(active_delays) <= 0):
            raise ValueError("valid path delays must be strictly increasing per occasion")

    occasions = _canonical_id_roster(occasion_ids, occasion_count, "occasion_ids")
    branches = _canonical_id_roster(receiver_branch_ids, branch_count, "receiver_branch_ids")
    if receiver_category not in _RECEIVER_REFERENCE_POINTS:
        raise ValueError("receiver_category must be a TS 38.104 terrestrial/satellite type")
    bandwidth_hz = _finite_real(
        measurement_frequency_bandwidth_hz, "measurement_frequency_bandwidth_hz"
    )
    if bandwidth_hz <= 0:
        raise ValueError("measurement_frequency_bandwidth_hz must be positive")
    scale = _finite_real(
        watts_per_channel_response_unit_squared,
        "watts_per_channel_response_unit_squared",
    )
    if scale <= 0:
        raise ValueError("watts_per_channel_response_unit_squared must be positive")
    calibration_digest = _validate_digest(power_calibration_sha256, "power_calibration_sha256")
    estimator_digest = _validate_digest(channel_estimator_sha256, "channel_estimator_sha256")
    detector_digest = _validate_digest(path_detector_sha256, "path_detector_sha256")

    with np.errstate(over="ignore", invalid="ignore"):
        path_power_w = (response.real * response.real + response.imag * response.imag) * scale
    if not np.isfinite(path_power_w).all():
        raise ValueError("calibrated UL SRS path power is not finite")
    occasion_rows: list[dict[str, Any]] = []
    for occasion_index, occasion_id in enumerate(occasions):
        active_path_indices = np.flatnonzero(valid[occasion_index]).tolist()
        first_path_index = active_path_indices[0] if active_path_indices else None
        branch_rows = []
        for branch_index, branch_id in enumerate(branches):
            measurements = []
            for path_index in active_path_indices:
                power_w = float(path_power_w[occasion_index, branch_index, path_index])
                measurements.append(
                    {
                        "path_index": path_index,
                        "path_delay_s": float(delays[occasion_index, path_index]),
                        "first_detected_path": path_index == first_path_index,
                        "ul_srs_rsrpp_w": power_w,
                        "ul_srs_rsrpp_dbm": (
                            10.0 * math.log10(power_w / 1e-3) if power_w > 0 else None
                        ),
                    }
                )
            branch_rows.append({"receiver_branch_id": branch_id, "path_measurements": measurements})
        occasion_rows.append(
            {
                "occasion_id": occasion_id,
                "status": "MEASURED" if active_path_indices else "NO_DETECTED_PATH_DIAGNOSTIC",
                "first_detected_path_index": first_path_index,
                "detected_path_count": len(active_path_indices),
                "receiver_branches": branch_rows,
            }
        )

    report = {
        "schema": "loslab.nr_ul_srs_rsrpp_report.v1",
        "status": (
            "OBSERVED_NOT_QUALIFIED"
            if any(row["detected_path_count"] for row in occasion_rows)
            else "NO_DETECTED_PATH_DIAGNOSTIC"
        ),
        "measurement": "UL_SRS_RSRPP",
        "standard": "3GPP_TS_38.215_V19.2.0_clause_5.2.6",
        "receiver_category": receiver_category,
        "measurement_reference_point": _RECEIVER_REFERENCE_POINTS[receiver_category],
        "measurement_frequency_bandwidth_hz": bandwidth_hz,
        "power_unit": "W",
        "occasion_ids": occasions,
        "receiver_branch_ids": branches,
        "first_path_rule": "earliest_valid_caller_detected_path_delay_per_occasion",
        "path_power_rule": "squared_magnitude_of_linear_averaged_path_response_times_calibration",
        "watts_per_channel_response_unit_squared": scale,
        "power_calibration_sha256": calibration_digest,
        "channel_estimator_sha256": estimator_digest,
        "path_detector_sha256": detector_digest,
        "path_channel_response_sha256": _array_sha256(response),
        "path_delay_sha256": _array_sha256(delays),
        "path_valid_mask_sha256": _array_sha256(valid),
        "occasion_measurements": occasion_rows,
        "limitations": [
            "Channel estimates, path detections, and calibration are caller supplied and not independently verified.",
            "Path indices are local to each occasion; no cross-occasion path identity association is claimed.",
            "The report preserves each receiver branch and does not synthesize a diversity-combined path measurement.",
            "This is an observed measurement contract, not synchronized receiver or scientific qualification.",
        ],
    }
    _validate_rsrpp_report_schema(report)
    return report


def report_ul_srs_tdct_tdcp(
    linear_averaged_channel_response: np.ndarray,
    candidate_time_ticks: np.ndarray,
    first_detected_path_time_ticks: np.ndarray,
    *,
    occasion_ids: Sequence[str],
    receiver_branch_ids: Sequence[str],
    receiver_category: str,
    sfn: int,
    subframe_number: int,
    sfn_zero_reference_time_s: float,
    timing_reporting_granularity_factor: int,
    reported_time_count: int,
    watts_per_channel_response_unit_squared: float,
    power_calibration_sha256: str,
    channel_estimator_sha256: str,
    path_detector_sha256: str,
    maximum_measurement_values: int = 1_000_000,
) -> dict[str, Any]:
    """Report the coupled UL SRS-TDCT and TDCP values from caller estimates.

    The complex input is the caller's linear average of estimated channel
    response at each candidate time, with axes ``[occasion, branch, time]``.
    Candidate times and first-path times are integer multiples of NR ``Tc``
    from the declared SFN-zero epoch. This function constructs the
    ``2**k * Tc`` grid beginning at the first detected path rounded down,
    chooses the requested highest-power grid points, reports timing relative
    to the SRS reference instant, and reports TDCP power on the receiver branch
    selected by the strongest chosen first time. It performs no channel
    estimation, first-path detection, clock calibration, or receiver
    qualification.
    """
    tc_s = 1.0 / (480_000.0 * 4096.0)  # TS 38.211 V19.2.0, clause 4.1.
    raw_response = np.asarray(linear_averaged_channel_response)
    raw_times = np.asarray(candidate_time_ticks)
    raw_first = np.asarray(first_detected_path_time_ticks)
    if (
        raw_response.ndim != 3
        or any(size <= 0 for size in raw_response.shape)
        or raw_response.dtype.kind != "c"
    ):
        raise ValueError(
            "linear_averaged_channel_response must be nonempty complex [occasion, branch, time]"
        )
    occasion_count, branch_count, time_count = raw_response.shape
    if raw_times.dtype.kind not in "iu" or raw_times.shape != (occasion_count, time_count):
        raise ValueError("candidate_time_ticks must be integer [occasion, time]")
    if raw_first.dtype.kind not in "iu" or raw_first.shape != (occasion_count,):
        raise ValueError("first_detected_path_time_ticks must be integer [occasion]")
    int64_max = np.iinfo(np.int64).max
    if (
        np.any(raw_times < 0)
        or np.any(raw_first < 0)
        or np.any(raw_times > int64_max)
        or np.any(raw_first > int64_max)
    ):
        raise ValueError("candidate and first-path time ticks must fit nonnegative int64")
    if isinstance(maximum_measurement_values, bool) or not isinstance(
        maximum_measurement_values, int
    ):
        raise TypeError("maximum_measurement_values must be an integer")
    if maximum_measurement_values <= 0:
        raise ValueError("maximum_measurement_values must be positive")
    if raw_response.size + raw_times.size + raw_first.size > maximum_measurement_values:
        raise ValueError("UL SRS-TDCT/TDCP inputs exceed maximum_measurement_values")
    if isinstance(timing_reporting_granularity_factor, bool) or not isinstance(
        timing_reporting_granularity_factor, int
    ):
        raise TypeError("timing_reporting_granularity_factor must be an integer")
    if not 0 <= timing_reporting_granularity_factor <= 5:
        raise ValueError("timing_reporting_granularity_factor must be in [0, 5]")
    if isinstance(reported_time_count, bool) or not isinstance(reported_time_count, int):
        raise TypeError("reported_time_count must be an integer")
    if not 1 <= reported_time_count <= time_count:
        raise ValueError("reported_time_count must be in [1, number of candidate times]")
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

    response = np.array(raw_response, dtype=np.complex128, copy=True, order="C")
    times = np.array(raw_times, dtype=np.int64, copy=True, order="C")
    first_times = np.array(raw_first, dtype=np.int64, copy=True, order="C")
    if not np.isfinite(response).all():
        raise ValueError("linear_averaged_channel_response must be finite")
    granularity_ticks = 1 << timing_reporting_granularity_factor
    for occasion_index in range(occasion_count):
        expected_start = (int(first_times[occasion_index]) // granularity_ticks) * granularity_ticks
        expected_end = expected_start + (time_count - 1) * granularity_ticks
        if expected_end > int64_max:
            raise ValueError("candidate time grid exceeds the supported int64 tick range")
        expected = expected_start + np.arange(time_count, dtype=np.int64) * granularity_ticks
        if not np.array_equal(times[occasion_index], expected):
            raise ValueError(
                "candidate_time_ticks must be consecutive 2**k*Tc instances beginning "
                "at the rounded-down first detected path"
            )
    occasions = _canonical_id_roster(occasion_ids, occasion_count, "occasion_ids")
    branches = _canonical_id_roster(receiver_branch_ids, branch_count, "receiver_branch_ids")
    t0 = _finite_real(sfn_zero_reference_time_s, "sfn_zero_reference_time_s")
    scale = _finite_real(
        watts_per_channel_response_unit_squared,
        "watts_per_channel_response_unit_squared",
    )
    if scale <= 0:
        raise ValueError("watts_per_channel_response_unit_squared must be positive")
    calibration_digest = _validate_digest(power_calibration_sha256, "power_calibration_sha256")
    estimator_digest = _validate_digest(channel_estimator_sha256, "channel_estimator_sha256")
    detector_digest = _validate_digest(path_detector_sha256, "path_detector_sha256")
    with np.errstate(over="ignore", invalid="ignore"):
        power_by_branch_w = (response.real**2 + response.imag**2) * scale
    if not np.isfinite(power_by_branch_w).all():
        raise ValueError("calibrated UL SRS time-domain channel power is not finite")

    srs_offset_s = (10 * sfn + subframe_number) * 1e-3
    reference_time_s = t0 + srs_offset_s
    occasion_measurements: list[dict[str, Any]] = []
    for occasion_index, occasion_id in enumerate(occasions):
        strongest_branch_power_by_time = np.max(power_by_branch_w[occasion_index], axis=0)
        selected = sorted(
            range(time_count),
            key=lambda time_index: (
                -float(strongest_branch_power_by_time[time_index]),
                int(times[occasion_index, time_index]),
            ),
        )[:reported_time_count]
        first_selected_time_index = selected[0]
        selected_branch_index = int(
            np.argmax(power_by_branch_w[occasion_index, :, first_selected_time_index])
        )
        selected_times = []
        for power_rank, time_index in enumerate(selected):
            tick = int(times[occasion_index, time_index])
            absolute_time_s = t0 + tick * tc_s
            branch_powers = [
                float(power_by_branch_w[occasion_index, branch_index, time_index])
                for branch_index in range(branch_count)
            ]
            selected_power_w = branch_powers[selected_branch_index]
            selected_times.append(
                {
                    "power_rank": power_rank,
                    "candidate_index": time_index,
                    "time_tick": tick,
                    "absolute_time_s": absolute_time_s,
                    "tdct_relative_to_reference_s": absolute_time_s - reference_time_s,
                    "receiver_branch_power_w": branch_powers,
                    "selected_branch_power_w": selected_power_w,
                    "selected_branch_power_dbm": (
                        10.0 * math.log10(selected_power_w / 1e-3) if selected_power_w > 0 else None
                    ),
                }
            )
        occasion_measurements.append(
            {
                "occasion_id": occasion_id,
                "first_detected_path_time_tick": int(first_times[occasion_index]),
                "candidate_time_ticks": times[occasion_index].tolist(),
                "selected_receiver_branch_id": branches[selected_branch_index],
                "selected_times": selected_times,
            }
        )

    report = {
        "schema": "loslab.nr_ul_srs_tdct_tdcp_report.v1",
        "status": "OBSERVED_NOT_QUALIFIED",
        "measurements": ["UL_SRS_TDCT", "UL_SRS_TDCP"],
        "standard": "3GPP_TS_38.215_V19.2.0_clauses_5.2.9_5.2.10",
        "receiver_category": receiver_category,
        "measurement_reference_point": _TIMING_REFERENCE_POINTS[receiver_category],
        "tc_s": tc_s,
        "timing_reporting_granularity_factor": timing_reporting_granularity_factor,
        "timing_granularity_s": granularity_ticks * tc_s,
        "candidate_time_count": time_count,
        "reported_time_count": reported_time_count,
        "sfn": sfn,
        "subframe_number": subframe_number,
        "sfn_zero_reference_time_s": t0,
        "srs_subframe_offset_from_sfn_zero_s": srs_offset_s,
        "timing_reference_time_s": reference_time_s,
        "occasion_ids": occasions,
        "receiver_branch_ids": branches,
        "power_unit": "W",
        "watts_per_channel_response_unit_squared": scale,
        "power_calibration_sha256": calibration_digest,
        "channel_estimator_sha256": estimator_digest,
        "path_detector_sha256": detector_digest,
        "linear_averaged_channel_response_sha256": _array_sha256(response),
        "candidate_time_ticks_sha256": _array_sha256(times),
        "first_detected_path_time_ticks_sha256": _array_sha256(first_times),
        "occasion_measurements": occasion_measurements,
        "limitations": [
            "Channel estimates, first-path times, calibration, and timing epoch are caller supplied and not independently verified.",
            "The report selects the highest-power candidate times and uses the branch selected at the strongest time for all TDCP values.",
            "No receiver synchronization, channel estimation, first-path detection, or clock calibration is performed.",
            "This is a measurement arithmetic contract, not scientific or receiver qualification.",
        ],
    }
    _validate_tdct_tdcp_report_schema(report)
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


def _validate_aoa_report_schema(report: dict[str, Any]) -> None:
    import jsonschema

    schema = json.loads(_AOA_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(report))
    if errors:
        raise ValueError(
            "UL AoA report violates schema: " + "; ".join(error.message for error in errors[:10])
        )


def _validate_rsrpp_report_schema(report: dict[str, Any]) -> None:
    import jsonschema

    schema = json.loads(_RSRPP_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(report))
    if errors:
        raise ValueError(
            "UL SRS-RSRPP report violates schema: "
            + "; ".join(error.message for error in errors[:10])
        )


def _validate_tdct_tdcp_report_schema(report: dict[str, Any]) -> None:
    import jsonschema

    schema = json.loads(_TDCT_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(report))
    if errors:
        raise ValueError(
            "UL SRS-TDCT/TDCP report violates schema: "
            + "; ".join(error.message for error in errors[:10])
        )
