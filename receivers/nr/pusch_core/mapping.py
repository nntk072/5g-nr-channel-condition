"""Data-symbol mapping and allocation checks for NR PUSCH intra-slot hopping."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from receivers.nr.pusch_core.dmrs import (
    verify_pusch_intra_slot_hop_dmrs_grid,
    verify_pusch_intra_slot_hop_dmrs_plan,
)
from receivers.nr.pusch_core.hopping import (
    _array_is_finite_c_order,
)
from receivers.nr.pusch_core.types import (
    NRPUSCHHoppedAllocationGrid,
    NRPUSCHIntraSlotHopDMRSGrid,
)


def map_pusch_intra_slot_hop_data_symbols(
    plan_metadata: Mapping[str, Any],
    dmrs_grid: NRPUSCHIntraSlotHopDMRSGrid,
    *,
    first_hop_data_symbols: np.ndarray,
    second_hop_data_symbols: np.ndarray,
    max_output_elements: int = 64_000_000,
) -> NRPUSCHHoppedAllocationGrid:
    """Map one-layer hop data into a local BWP grid beside the generated DM-RS.

    Each hop receives its own modulation-symbol vector. Within each hop, REs
    are visited in increasing frequency-subcarrier index and then OFDM symbol,
    following TS 38.211 V19.2.0 clause 6.3.1.6. This bounded mapper assumes
    one DM-RS port-0 layer, one CDM group without data, and no PT-RS, muting,
    co-scheduled UE DM-RS, or other punctures. It does not encode UL-SCH bits,
    derive grants, apply multi-port precoding, or qualify a receiver.
    """
    verify_pusch_intra_slot_hop_dmrs_plan(plan_metadata)
    if not isinstance(dmrs_grid, NRPUSCHIntraSlotHopDMRSGrid):
        raise TypeError("dmrs_grid must be a generated NRPUSCHIntraSlotHopDMRSGrid")
    dmrs_config = dmrs_grid.metadata
    if not isinstance(dmrs_config, Mapping) or dmrs_config.get("schema") != (
        "loslab.nr_pusch_intra_slot_hop_dmrs_grid.v1"
    ):
        raise ValueError("unsupported hop DM-RS grid metadata")
    dmrs_plan_digest = hashlib.sha256(
        json.dumps(dict(plan_metadata), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if dmrs_config.get("dmrs_plan_sha256") != dmrs_plan_digest:
        raise ValueError("DM-RS grid was generated for a different timing plan")
    dmrs_replay = verify_pusch_intra_slot_hop_dmrs_grid(
        plan_metadata,
        dmrs_grid.pilot_mask,
        dmrs_grid.pilot_grid,
        allocation_rb_count=dmrs_config.get("allocation_rb_count"),
        dmrs_config_type=dmrs_config.get("dmrs_config_type"),
        n_cell_id=dmrs_config.get("n_id"),
        dmrs_n_id=(dmrs_config.get("n_id"), dmrs_config.get("n_id")),
        dmrs_n_scid=dmrs_config.get("n_scid"),
        subcarrier_spacing_khz=dmrs_config.get("subcarrier_spacing_khz"),
        slot_index=dmrs_config.get("slot_index"),
    )
    for digest_key in ("pilot_mask_sha256", "pilot_grid_sha256"):
        if dmrs_config.get(digest_key) != dmrs_replay[digest_key]:
            raise ValueError(f"DM-RS {digest_key} metadata does not match replay")

    data_vectors = (first_hop_data_symbols, second_hop_data_symbols)
    for name, values in zip(
        ("first_hop_data_symbols", "second_hop_data_symbols"), data_vectors, strict=True
    ):
        if not isinstance(values, np.ndarray) or values.ndim != 1:
            raise TypeError(f"{name} must be a one-dimensional NumPy array")
        if values.dtype.kind != "c" or values.dtype.itemsize not in {8, 16}:
            raise ValueError(f"{name} must contain complex64 or complex128 values")
        if not _array_is_finite_c_order(values):
            raise ValueError(f"{name} must contain only finite values")
    if first_hop_data_symbols.dtype != second_hop_data_symbols.dtype:
        raise ValueError("both hop data vectors must use the same complex dtype")
    if (
        isinstance(max_output_elements, bool)
        or not isinstance(max_output_elements, int)
        or max_output_elements < 1
    ):
        raise ValueError("max_output_elements must be a positive integer")

    allocation_start = plan_metadata["allocation_start_symbol"]
    first_count = plan_metadata["first_hop_symbol_count"]
    second_count = plan_metadata["second_hop_symbol_count"]
    allocation_count = first_count + second_count
    allocation_rb_count = dmrs_config["allocation_rb_count"]
    subcarrier_count = allocation_rb_count * 12
    expected_shape = (14, subcarrier_count)
    if (
        tuple(dmrs_grid.pilot_mask.shape) != expected_shape
        or tuple(dmrs_grid.pilot_grid.shape) != expected_shape
    ):
        raise ValueError("DM-RS grid axes do not match the declared allocation width")
    output_elements = allocation_count * subcarrier_count
    input_elements = first_hop_data_symbols.size + second_hop_data_symbols.size
    if input_elements + output_elements > max_output_elements:
        raise ValueError("hop-data inputs and allocation grid exceed max_output_elements")
    output = np.zeros((allocation_count, subcarrier_count), dtype=first_hop_data_symbols.dtype)
    output_data_mask = np.zeros((allocation_count, subcarrier_count), dtype=np.bool_)
    data_counts: list[int] = []
    hop_parameters = (
        (0, first_count, plan_metadata["first_hop_dmrs_symbols"], first_hop_data_symbols),
        (
            first_count,
            second_count,
            plan_metadata["second_hop_dmrs_symbols"],
            second_hop_data_symbols,
        ),
    )
    for hop_offset, hop_count, absolute_dmrs_symbols, data_symbols in hop_parameters:
        absolute_start = allocation_start + hop_offset
        absolute_stop = absolute_start + hop_count
        relative_dmrs_symbols = [
            symbol - allocation_start
            for symbol in absolute_dmrs_symbols
            if absolute_start <= symbol < absolute_stop
        ]
        hop_mask = dmrs_grid.pilot_mask[absolute_start:absolute_stop]
        hop_pilots = dmrs_grid.pilot_grid[absolute_start:absolute_stop]
        local_slice = slice(hop_offset, hop_offset + hop_count)
        output[local_slice] = np.where(hop_mask, hop_pilots, 0).astype(output.dtype, copy=False)
        # The only occupied pilot symbols must be the declared symbols for this hop.
        actual_dmrs = np.flatnonzero(hop_mask.any(axis=1)) + absolute_start - allocation_start
        if tuple(int(value) for value in actual_dmrs) != tuple(relative_dmrs_symbols):
            raise ValueError("DM-RS grid contains pilots outside its declared hop positions")
        data_mask = ~hop_mask
        ordered_coordinates = np.argwhere(data_mask.T)
        if len(ordered_coordinates) != data_symbols.size:
            raise ValueError(
                "hop data-symbol count must equal available non-DM-RS REs "
                f"({len(ordered_coordinates)} expected, {data_symbols.size} received)"
            )
        subcarrier_indices = ordered_coordinates[:, 0]
        local_symbol_indices = ordered_coordinates[:, 1]
        output[hop_offset + local_symbol_indices, subcarrier_indices] = data_symbols
        output_data_mask[hop_offset + local_symbol_indices, subcarrier_indices] = True
        data_counts.append(int(data_symbols.size))

    def digest(values: np.ndarray) -> str:
        return hashlib.sha256(values.tobytes(order="C")).hexdigest()

    metadata = {
        "schema": "loslab.nr_pusch_hopped_allocation_grid.v2",
        "status": "MAPPED_NOT_QUALIFIED",
        "standard": "3GPP TS 38.211 V19.2.0",
        "data_mapping_clause": "6.3.1.6",
        "dmrs_plan_sha256": dmrs_plan_digest,
        "dmrs_grid_sha256": dmrs_replay["pilot_grid_sha256"],
        "allocation_start_symbol": allocation_start,
        "first_hop_symbol_count": first_count,
        "second_hop_symbol_count": second_count,
        "allocation_rb_count": allocation_rb_count,
        "allocation_grid_shape": list(output.shape),
        "complex_dtype": str(output.dtype),
        "data_mapping_order": "increasing_frequency_subcarrier_then_ofdm_symbol_per_hop",
        "first_hop_data_symbol_count": data_counts[0],
        "second_hop_data_symbol_count": data_counts[1],
        "one_port_zero_cdm_group_without_data": True,
        "pt_rs_and_other_puncturing": "not_supported",
        "data_symbols_sha256": hashlib.sha256(
            bytes.fromhex(digest(first_hop_data_symbols))
            + bytes.fromhex(digest(second_hop_data_symbols))
        ).hexdigest(),
        "allocation_grid_sha256": digest(output),
        "data_mask_sha256": hashlib.sha256(
            np.packbits(output_data_mask, bitorder="big").tobytes()
        ).hexdigest(),
        "qualification": "NOT_QUALIFIED",
    }
    return NRPUSCHHoppedAllocationGrid(output, output_data_mask, metadata)


def verify_pusch_intra_slot_hop_allocation_grid(
    plan_metadata: Mapping[str, Any],
    dmrs_grid: NRPUSCHIntraSlotHopDMRSGrid,
    allocation_grid: np.ndarray,
    data_mask: np.ndarray,
    metadata: Mapping[str, Any],
    *,
    first_hop_data_symbols: np.ndarray,
    second_hop_data_symbols: np.ndarray,
    max_output_elements: int = 64_000_000,
) -> dict[str, Any]:
    """Re-map and verify a serialized one-layer two-hop allocation grid."""
    if not isinstance(allocation_grid, np.ndarray):
        raise TypeError("allocation_grid must be a NumPy array")
    if not isinstance(data_mask, np.ndarray) or data_mask.dtype != np.bool_:
        raise TypeError("data_mask must be a Boolean NumPy array")
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    expected = map_pusch_intra_slot_hop_data_symbols(
        plan_metadata,
        dmrs_grid,
        first_hop_data_symbols=first_hop_data_symbols,
        second_hop_data_symbols=second_hop_data_symbols,
        max_output_elements=max_output_elements,
    )
    if not np.array_equal(allocation_grid, expected.allocation_grid):
        raise ValueError("hopped allocation grid does not replay from its DM-RS/data inputs")
    if not np.array_equal(data_mask, expected.data_mask):
        raise ValueError("hopped allocation data mask does not replay from its DM-RS/data inputs")
    if dict(metadata) != dict(expected.metadata):
        raise ValueError("hopped allocation metadata does not replay from its declared inputs")
    return {
        "status": "PASS_HOPPED_ALLOCATION_REPLAY",
        "allocation_grid_sha256": expected.metadata["allocation_grid_sha256"],
        "qualification": "NOT_QUALIFIED",
    }


def _normalized_hop_noise_variance(
    noise_variance: Any,
    *,
    leading_shape: tuple[int, int, int],
) -> np.ndarray:
    """Validate and broadcast real nonnegative variance over [batch, rx, rx_ant]."""
    raw = np.asarray(noise_variance)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError("noise_variance must contain real numeric values")
    if raw.ndim > 3:
        raise ValueError("noise_variance must be scalar or have at most three dimensions")
    if raw.ndim == 0:
        shaped = raw.reshape(1, 1, 1)
    elif raw.ndim == 1:
        if raw.shape[0] not in {1, leading_shape[0]}:
            raise ValueError("one-dimensional noise_variance must be scalar-like or per batch")
        shaped = raw.reshape(raw.shape[0], 1, 1)
    elif raw.ndim == 2:
        if raw.shape[0] not in {1, leading_shape[0]} or raw.shape[1] not in {
            1,
            leading_shape[1],
        }:
            raise ValueError("two-dimensional noise_variance must broadcast over [batch, rx]")
        shaped = raw.reshape(raw.shape[0], raw.shape[1], 1)
    else:
        shaped = raw
    try:
        broadcast = np.broadcast_to(shaped, leading_shape)
    except ValueError as error:
        raise ValueError("noise_variance must broadcast over [batch, rx, rx_ant]") from error
    values = np.asarray(broadcast, dtype=np.float64)
    if not bool(np.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("noise_variance must be finite and nonnegative")
    return values


def _nearest_sorted_positions(source_positions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return nearest source positions for sorted integer coordinates, preferring lower ties."""
    if source_positions.ndim != 1 or source_positions.size == 0:
        raise ValueError("each hop must contain at least one DM-RS position")
    insertions = np.searchsorted(source_positions, targets, side="left")
    right_index = np.clip(insertions, 0, source_positions.size - 1)
    left_index = np.clip(insertions - 1, 0, source_positions.size - 1)
    left = source_positions[left_index]
    right = source_positions[right_index]
    choose_right = np.abs(right - targets) < np.abs(targets - left)
    return np.where(choose_right, right, left)
