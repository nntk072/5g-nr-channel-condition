"""Channel estimation and demapping for NR PUSCH intra-slot hopping."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np

from receivers.nr.pusch_core.dmrs import (
    verify_pusch_intra_slot_hop_dmrs_grid,
    verify_pusch_intra_slot_hop_dmrs_plan,
)
from receivers.nr.pusch_core.hopping import (
    _array_is_finite_c_order,
    _sha256_array_c_order,
)
from receivers.nr.pusch_core.mapping import (
    _nearest_sorted_positions,
    _normalized_hop_noise_variance,
)
from receivers.nr.pusch_core.types import (
    NRPUSCHHoppedEstimateDemap,
    NRPUSCHIntraSlotHopDMRSGrid,
)


def estimate_demap_pusch_intra_slot_hop(
    plan_metadata: Mapping[str, Any],
    dmrs_grid: NRPUSCHIntraSlotHopDMRSGrid,
    *,
    data_mask: np.ndarray,
    allocation_metadata: Mapping[str, Any],
    received_allocation_grid: np.ndarray,
    noise_variance: Any,
    num_cdm_groups_without_data: int = 1,
    pt_rs_enabled: bool = False,
    puncture_mask: np.ndarray | None = None,
    max_output_elements: int = 64_000_000,
) -> NRPUSCHHoppedEstimateDemap:
    """Estimate and demap one caller-resolved two-hop allocation without shared state.

    The active allocation is consumed exactly as mapped, with axes
    [batch, rx, rx_ant, allocated_symbol, allocated_subcarrier]. The
    standards-derived DM-RS grid is replay-verified, the data mask must match
    the mapper's digest and unpunctured complement of the pilot mask, and each
    hop is estimated independently using nearest-DM-RS least squares. Data REs
    are emitted in the mapper's frequency-subcarrier-then-symbol order per hop.

    This bounded route supports one port/layer and exactly one CDM group without
    data. PT-RS and puncturing remain unsupported. It performs no shared Sionna
    resource-grid mutation and does not qualify receiver performance.
    """
    verify_pusch_intra_slot_hop_dmrs_plan(plan_metadata)
    if not isinstance(dmrs_grid, NRPUSCHIntraSlotHopDMRSGrid):
        raise TypeError("dmrs_grid must be a generated NRPUSCHIntraSlotHopDMRSGrid")
    dmrs_metadata = dmrs_grid.metadata
    if not isinstance(dmrs_metadata, Mapping) or dmrs_metadata.get("schema") != (
        "loslab.nr_pusch_intra_slot_hop_dmrs_grid.v1"
    ):
        raise ValueError("unsupported hop DM-RS grid metadata")
    plan_digest = hashlib.sha256(
        json.dumps(
            dict(plan_metadata),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if dmrs_metadata.get("dmrs_plan_sha256") != plan_digest:
        raise ValueError("DM-RS grid was generated for a different timing plan")
    dmrs_replay = verify_pusch_intra_slot_hop_dmrs_grid(
        plan_metadata,
        dmrs_grid.pilot_mask,
        dmrs_grid.pilot_grid,
        allocation_rb_count=dmrs_metadata.get("allocation_rb_count"),
        dmrs_config_type=dmrs_metadata.get("dmrs_config_type"),
        n_cell_id=dmrs_metadata.get("n_id"),
        dmrs_n_id=(dmrs_metadata.get("n_id"), dmrs_metadata.get("n_id")),
        dmrs_n_scid=dmrs_metadata.get("n_scid"),
        subcarrier_spacing_khz=dmrs_metadata.get("subcarrier_spacing_khz"),
        slot_index=dmrs_metadata.get("slot_index"),
    )
    for digest_key in ("pilot_mask_sha256", "pilot_grid_sha256"):
        if dmrs_metadata.get(digest_key) != dmrs_replay[digest_key]:
            raise ValueError(f"DM-RS {digest_key} metadata does not match replay")

    if isinstance(num_cdm_groups_without_data, bool) or not isinstance(
        num_cdm_groups_without_data, int
    ):
        raise TypeError("num_cdm_groups_without_data must be an integer")
    if num_cdm_groups_without_data != 1:
        raise ValueError("multiple CDM groups without data are unsupported by the hop estimator")
    if type(pt_rs_enabled) is not bool:
        raise TypeError("pt_rs_enabled must be a Boolean")
    if pt_rs_enabled:
        raise ValueError("PT-RS is unsupported by the hop estimator")

    if not isinstance(data_mask, np.ndarray) or data_mask.dtype != np.bool_:
        raise TypeError("data_mask must be a Boolean NumPy array")
    if puncture_mask is not None:
        if not isinstance(puncture_mask, np.ndarray) or puncture_mask.dtype != np.bool_:
            raise TypeError("puncture_mask must be a Boolean NumPy array or None")
        if puncture_mask.shape != data_mask.shape:
            raise ValueError("puncture_mask must match data_mask shape")
        if bool(puncture_mask.any()):
            raise ValueError("punctured REs are unsupported by the hop estimator")

    if not isinstance(allocation_metadata, Mapping) or allocation_metadata.get("schema") != (
        "loslab.nr_pusch_hopped_allocation_grid.v2"
    ):
        raise ValueError("unsupported hopped-allocation metadata")
    if allocation_metadata.get("qualification") != "NOT_QUALIFIED":
        raise ValueError("hopped-allocation metadata must preserve NOT_QUALIFIED status")
    if allocation_metadata.get("dmrs_plan_sha256") != plan_digest:
        raise ValueError("hopped-allocation metadata belongs to a different timing plan")
    if allocation_metadata.get("dmrs_grid_sha256") != dmrs_replay["pilot_grid_sha256"]:
        raise ValueError("hopped-allocation metadata belongs to a different DM-RS grid")
    if allocation_metadata.get("one_port_zero_cdm_group_without_data") is not True:
        raise ValueError("hopped-allocation metadata does not preserve the one-CDM-group scope")
    if allocation_metadata.get("pt_rs_and_other_puncturing") != "not_supported":
        raise ValueError("hopped-allocation metadata does not preserve the no-puncturing scope")

    allocation_start = plan_metadata["allocation_start_symbol"]
    first_count = plan_metadata["first_hop_symbol_count"]
    second_count = plan_metadata["second_hop_symbol_count"]
    allocation_count = first_count + second_count
    allocation_rb_count = dmrs_metadata["allocation_rb_count"]
    subcarrier_count = allocation_rb_count * 12
    expected_shape = (allocation_count, subcarrier_count)
    expected_layout = {
        "allocation_start_symbol": allocation_start,
        "first_hop_symbol_count": first_count,
        "second_hop_symbol_count": second_count,
        "allocation_rb_count": allocation_rb_count,
        "allocation_grid_shape": list(expected_shape),
    }
    if any(allocation_metadata.get(key) != value for key, value in expected_layout.items()):
        raise ValueError("hopped-allocation metadata does not match the active allocation span")
    if data_mask.shape != expected_shape:
        raise ValueError("data_mask does not match the active allocation span")
    expected_data_mask = ~dmrs_grid.pilot_mask[
        allocation_start : allocation_start + allocation_count
    ]
    expected_mapper_contract = {
        "status": "MAPPED_NOT_QUALIFIED",
        "standard": "3GPP TS 38.211 V19.2.0",
        "data_mapping_clause": "6.3.1.6",
        "data_mapping_order": "increasing_frequency_subcarrier_then_ofdm_symbol_per_hop",
        "first_hop_data_symbol_count": int(expected_data_mask[:first_count].sum()),
        "second_hop_data_symbol_count": int(expected_data_mask[first_count:].sum()),
    }
    if any(
        allocation_metadata.get(key) != value for key, value in expected_mapper_contract.items()
    ):
        raise ValueError("hopped-allocation metadata does not preserve the mapper contract")
    if bool((data_mask & ~expected_data_mask).any()):
        raise ValueError("data mask overlaps DM-RS or leaves the active allocation contract")
    if not np.array_equal(data_mask, expected_data_mask):
        raise ValueError("data mask contains unsupported puncturing or muting")
    data_mask_digest = hashlib.sha256(np.packbits(data_mask, bitorder="big").tobytes()).hexdigest()
    if allocation_metadata.get("data_mask_sha256") != data_mask_digest:
        raise ValueError("data mask digest does not match hopped-allocation metadata")

    if not isinstance(received_allocation_grid, np.ndarray):
        raise TypeError("received_allocation_grid must be a NumPy array")
    received = received_allocation_grid
    if received.ndim != 5 or any(size < 1 for size in received.shape[:3]):
        raise ValueError(
            "received_allocation_grid must have [batch, rx, rx_ant, symbol, subcarrier] axes"
        )
    if tuple(received.shape[-2:]) != expected_shape:
        raise ValueError("received_allocation_grid does not preserve the active allocation span")
    if received.dtype.kind != "c" or received.dtype.itemsize not in {8, 16}:
        raise ValueError("received_allocation_grid must contain complex64 or complex128 values")
    if not _array_is_finite_c_order(received):
        raise ValueError("received_allocation_grid must contain only finite values")
    if (
        isinstance(max_output_elements, bool)
        or not isinstance(max_output_elements, int)
        or max_output_elements < 1
    ):
        raise ValueError("max_output_elements must be a positive integer")

    leading_shape = (
        int(received.shape[0]),
        int(received.shape[1]),
        int(received.shape[2]),
    )
    noise = _normalized_hop_noise_variance(noise_variance, leading_shape=leading_shape)
    data_count = int(data_mask.sum())
    projected_elements = received.size + (3 * math.prod(leading_shape) * data_count)
    if projected_elements > max_output_elements:
        raise ValueError("hop estimator/demapper inputs and outputs exceed max_output_elements")

    local_pilot_mask = dmrs_grid.pilot_mask[allocation_start : allocation_start + allocation_count]
    local_pilot_grid = dmrs_grid.pilot_grid[allocation_start : allocation_start + allocation_count]
    data_parts: list[np.ndarray] = []
    estimate_parts: list[np.ndarray] = []
    variance_factors: list[np.ndarray] = []
    hop_slices = ((0, first_count), (first_count, second_count))
    for hop_offset, hop_count in hop_slices:
        hop_data_mask = data_mask[hop_offset : hop_offset + hop_count]
        coordinates = np.argwhere(hop_data_mask.T)
        if coordinates.size == 0:
            raise ValueError("each hop must expose at least one data RE")
        data_subcarriers = coordinates[:, 0]
        data_symbols = coordinates[:, 1]
        absolute_local_symbols = hop_offset + data_symbols
        data_parts.append(received[..., absolute_local_symbols, data_subcarriers])

        hop_pilot_mask = local_pilot_mask[hop_offset : hop_offset + hop_count]
        pilot_symbols = np.flatnonzero(hop_pilot_mask.any(axis=1))
        chosen_symbols = _nearest_sorted_positions(pilot_symbols, data_symbols)
        hop_estimate = np.empty(
            (*leading_shape, coordinates.shape[0]),
            dtype=received.dtype,
        )
        hop_variance_factor = np.empty(coordinates.shape[0], dtype=np.float64)
        for pilot_symbol in np.unique(chosen_symbols):
            selected = np.flatnonzero(chosen_symbols == pilot_symbol)
            pilot_subcarriers = np.flatnonzero(hop_pilot_mask[pilot_symbol])
            chosen_subcarriers = _nearest_sorted_positions(
                pilot_subcarriers,
                data_subcarriers[selected],
            )
            rx_symbol = hop_offset + int(pilot_symbol)
            reference = local_pilot_grid[rx_symbol, chosen_subcarriers].astype(
                received.dtype,
                copy=False,
            )
            if bool((np.abs(reference) == 0).any()):
                raise ValueError("DM-RS reference contains a zero-valued pilot")
            hop_estimate[..., selected] = received[..., rx_symbol, chosen_subcarriers] / reference
            hop_variance_factor[selected] = 1.0 / np.square(np.abs(reference))
        estimate_parts.append(hop_estimate)
        variance_factors.append(hop_variance_factor)

    received_data = np.concatenate(data_parts, axis=-1)
    channel_estimate = np.concatenate(estimate_parts, axis=-1)
    variance_factor = np.concatenate(variance_factors)
    error_variance = noise[..., np.newaxis] * variance_factor
    for output in (received_data, channel_estimate, error_variance):
        output.setflags(write=False)

    metadata = {
        "schema": "loslab.nr_pusch_hopped_estimate_demap.v1",
        "status": "ESTIMATED_DEMAPPED_NOT_QUALIFIED",
        "scope": "per_call_single_port_nearest_dmrs_ls_and_data_re_extraction",
        "qualification": "NOT_QUALIFIED",
        "dmrs_plan_sha256": plan_digest,
        "dmrs_grid_sha256": dmrs_replay["pilot_grid_sha256"],
        "data_mask_sha256": data_mask_digest,
        "allocation_start_symbol": allocation_start,
        "allocation_symbol_count": allocation_count,
        "first_hop_symbol_count": first_count,
        "second_hop_symbol_count": second_count,
        "allocation_rb_count": allocation_rb_count,
        "active_allocation_shape": list(expected_shape),
        "received_grid_shape": [int(size) for size in received.shape],
        "data_re_count": data_count,
        "data_mapping_order": "increasing_frequency_subcarrier_then_ofdm_symbol_per_hop",
        "estimator": "nearest_dmrs_re_ls_per_hop",
        "num_cdm_groups_without_data": 1,
        "pt_rs": "not_supported",
        "puncturing": "not_supported",
        "received_grid_sha256": _sha256_array_c_order(received),
        "received_data_sha256": _sha256_array_c_order(received_data),
        "channel_estimate_sha256": _sha256_array_c_order(channel_estimate),
        "error_variance_sha256": _sha256_array_c_order(error_variance),
    }
    return NRPUSCHHoppedEstimateDemap(
        received_data_symbols=received_data,
        channel_estimate=channel_estimate,
        error_variance=error_variance,
        metadata=metadata,
    )
