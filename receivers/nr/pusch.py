"""Framework-independent contracts for NR PUSCH receiver observations."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

MAX_CARRIER_GRID_START_CRB = 2199
MAX_CRB_END_EXCLUSIVE = 2474


@dataclass(frozen=True, slots=True)
class NRPUSCHFrequencyHopGrid:
    """Carrier-grid composition of two caller-mapped intra-slot hop segments."""

    frequency_grid: np.ndarray
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class NRPUSCHDehoppedGrid:
    """Received carrier grid rehomed onto the first-hop BWP allocation."""

    frequency_grid: np.ndarray
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class NRPUSCHIntraSlotHopDMRSPlan:
    """Table-derived single-symbol DM-RS positions for two PUSCH hops."""

    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class NRPUSCHIntraSlotHopDMRSGrid:
    """Single-port DM-RS reference grid for a replayed two-hop timing plan."""

    pilot_mask: np.ndarray
    pilot_grid: np.ndarray
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class NRPUSCHHoppedAllocationGrid:
    """Single-layer local-BWP grid with hop-specific DM-RS and mapped data."""

    allocation_grid: np.ndarray
    data_mask: np.ndarray
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class NRPUSCHHoppedEstimateDemap:
    """Per-call hop-aware LS estimates and data REs in mapper order."""

    received_data_symbols: np.ndarray
    channel_estimate: np.ndarray
    error_variance: np.ndarray
    metadata: Mapping[str, Any]


def derive_pusch_intra_slot_hop_dmrs_plan(
    *,
    mapping_type: str,
    allocation_start_symbol: int,
    first_hop_symbol_count: int,
    second_hop_symbol_count: int,
    dmrs_additional_position: int,
    dmrs_type_a_position: int = 2,
) -> NRPUSCHIntraSlotHopDMRSPlan:
    """Derive single-symbol PUSCH hopping DM-RS symbols from TS 38.211 V19.2.0.

    This implements the two-hop rows of Table 6.4.1.1.3-6 for per-hop
    durations through seven symbols. As required by clause 6.4.1.1.3, any
    configured additional position above pos0 uses pos1 for each hop. Mapping
    type A is restricted to a slot-start allocation so its first-hop ``l0``
    origin is unambiguous. This is a timing plan only; it does not generate
    DM-RS values, resolve a grant, or qualify a receiver.
    """
    for name, value in {
        "allocation_start_symbol": allocation_start_symbol,
        "first_hop_symbol_count": first_hop_symbol_count,
        "second_hop_symbol_count": second_hop_symbol_count,
        "dmrs_additional_position": dmrs_additional_position,
        "dmrs_type_a_position": dmrs_type_a_position,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if mapping_type not in {"A", "B"}:
        raise ValueError("mapping_type must be 'A' or 'B'")
    if not 0 <= allocation_start_symbol <= 13:
        raise ValueError("allocation_start_symbol must be in [0, 13]")
    if first_hop_symbol_count < 1 or second_hop_symbol_count < 1:
        raise ValueError("both hop symbol counts must be positive")
    total_symbols = first_hop_symbol_count + second_hop_symbol_count
    if allocation_start_symbol + total_symbols > 14:
        raise ValueError("two-hop allocation must fit in the 14-symbol slot")
    if dmrs_additional_position not in {0, 1, 2, 3}:
        raise ValueError("dmrs_additional_position must be in [0, 3]")
    if dmrs_type_a_position not in {2, 3}:
        raise ValueError("dmrs_type_a_position must be 2 or 3")
    if mapping_type == "A" and allocation_start_symbol != 0:
        raise ValueError("mapping type A hopping is bounded to slot-start allocations")

    pos1 = dmrs_additional_position > 0
    l0 = dmrs_type_a_position if mapping_type == "A" else 0

    if max(first_hop_symbol_count, second_hop_symbol_count) > 7:
        raise ValueError("TS 38.211 hopping DM-RS table supports at most 7 symbols per hop")
    duration_group = lambda count: (  # noqa: E731 - concise TS table row selector
        "le3" if count <= 3 else "4" if count == 4 else "5_6" if count <= 6 else "7"
    )
    if duration_group(first_hop_symbol_count) != duration_group(second_hop_symbol_count):
        raise ValueError(
            "TS 38.211 Table 6.4.1.1.3-6 does not define this asymmetric hop-duration pair"
        )
    row = duration_group(first_hop_symbol_count)
    first_offsets: tuple[int, ...]
    second_offsets: tuple[int, ...]
    if mapping_type == "A":
        if row == "le3":
            raise ValueError(
                "mapping type A hopping has no table-defined DM-RS for hops shorter than 4 symbols"
            )
        if row in {"4", "5_6"}:
            first_offsets = (l0,)
            second_offsets = (0, 4) if row == "5_6" and pos1 else (0,)
        else:
            first_offsets = (l0, 6 if dmrs_type_a_position == 2 else 3) if pos1 else (l0,)
            second_offsets = (0, 4) if pos1 else (0,)
    elif row == "le3" or row == "4" or not pos1:
        first_offsets = (0,)
        second_offsets = (0,)
    else:
        first_offsets = (0, 4)
        second_offsets = (0, 4)
    first_symbols = tuple(allocation_start_symbol + value for value in first_offsets)
    second_start = allocation_start_symbol + first_hop_symbol_count
    second_symbols = tuple(second_start + value for value in second_offsets)
    if any(
        symbol >= allocation_start_symbol + total_symbols
        for symbol in (*first_symbols, *second_symbols)
    ):
        raise ValueError("derived DM-RS symbol falls outside its hop allocation")
    metadata = {
        "schema": "loslab.nr_pusch_intra_slot_hop_dmrs_plan.v1",
        "status": "DERIVED_NOT_QUALIFIED",
        "standard": "3GPP TS 38.211 V19.2.0",
        "table": "6.4.1.1.3-6",
        "mapping_type": mapping_type,
        "allocation_start_symbol": allocation_start_symbol,
        "first_hop_symbol_count": first_hop_symbol_count,
        "second_hop_symbol_count": second_hop_symbol_count,
        "dmrs_additional_position_configured": dmrs_additional_position,
        "dmrs_additional_position_used_per_hop": 0 if not pos1 else 1,
        "dmrs_type_a_position": dmrs_type_a_position if mapping_type == "A" else None,
        "first_hop_dmrs_symbols": list(first_symbols),
        "second_hop_dmrs_symbols": list(second_symbols),
        "scope": "single_symbol_dmrs_two_hop_time_positions_only",
        "qualification": "NOT_QUALIFIED",
    }
    return NRPUSCHIntraSlotHopDMRSPlan(metadata=metadata)


def verify_pusch_intra_slot_hop_dmrs_plan(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Re-derive and verify a serialized table-derived two-hop DM-RS plan."""
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    required = {
        "schema",
        "status",
        "standard",
        "table",
        "mapping_type",
        "allocation_start_symbol",
        "first_hop_symbol_count",
        "second_hop_symbol_count",
        "dmrs_additional_position_configured",
        "dmrs_additional_position_used_per_hop",
        "dmrs_type_a_position",
        "first_hop_dmrs_symbols",
        "second_hop_dmrs_symbols",
        "scope",
        "qualification",
    }
    if set(metadata) != required:
        raise ValueError("DM-RS plan metadata fields do not match the v1 contract")
    expected = derive_pusch_intra_slot_hop_dmrs_plan(
        mapping_type=metadata["mapping_type"],
        allocation_start_symbol=metadata["allocation_start_symbol"],
        first_hop_symbol_count=metadata["first_hop_symbol_count"],
        second_hop_symbol_count=metadata["second_hop_symbol_count"],
        dmrs_additional_position=metadata["dmrs_additional_position_configured"],
        dmrs_type_a_position=metadata["dmrs_type_a_position"] or 2,
    ).metadata
    if dict(metadata) != expected:
        raise ValueError("DM-RS plan does not replay from its declared PUSCH configuration")
    return {
        "status": "PASS_TABLE_DERIVATION_REPLAY",
        "standard": expected["standard"],
        "table": expected["table"],
        "qualification": "NOT_QUALIFIED",
    }


def _pusch_dmrs_gold_sequence(
    *, n_id: int, n_scid: int, slot_index: int, symbol_index: int, count: int
) -> np.ndarray:
    """Generate the TS 38.211 5.2.1/6.4.1.1.1 sequence for one DM-RS symbol."""
    c_init = (
        2**17 * (14 * slot_index + symbol_index + 1) * (2 * n_id + 1) + 2 * n_id + n_scid
    ) % 2**31
    nc = 1600
    length = 2 * count
    x1 = [0] * (nc + length + 31)
    x2 = [0] * (nc + length + 31)
    x1[0] = 1
    for index in range(31):
        x2[index] = (c_init >> index) & 1
    for index in range(nc + length):
        x1[index + 31] = (x1[index + 3] + x1[index]) & 1
        x2[index + 31] = (x2[index + 3] + x2[index + 2] + x2[index + 1] + x2[index]) & 1
    chips = [(x1[index + nc] + x2[index + nc]) & 1 for index in range(length)]
    scale = 1.0 / math.sqrt(2.0)
    return np.asarray(
        [
            complex(scale * (1 - 2 * chips[2 * index]), scale * (1 - 2 * chips[2 * index + 1]))
            for index in range(count)
        ],
        dtype=np.complex128,
    )


def build_pusch_intra_slot_hop_dmrs_grid(
    plan_metadata: Mapping[str, Any],
    *,
    allocation_rb_count: int,
    dmrs_config_type: int,
    n_cell_id: int,
    dmrs_n_id: tuple[int, int] | None,
    dmrs_n_scid: int,
    subcarrier_spacing_khz: int,
    slot_index: int,
) -> NRPUSCHIntraSlotHopDMRSGrid:
    """Generate/replay one-port PUSCH DM-RS for caller-resolved hop positions.

    Only DM-RS port 0 is generated. The result is a timing/sequence reference
    `[slot-symbol, BWP-subcarrier]` grid and pilot mask, not a complete PUSCH
    resource mapper: other ports, CDM data reservation, UL-SCH mapping, grant
    decoding, transform precoding, and receiver qualification are out of scope.
    ``n_cell_id`` and optional two-value ``dmrs_n_id`` follow PUSCHConfig; the
    latter is selected by ``dmrs_n_scid``. ``slot_index`` is the absolute NR
    slot number within the radio frame and is used with the absolute symbol
    index in the Gold-sequence initialization.
    """
    verify_pusch_intra_slot_hop_dmrs_plan(plan_metadata)
    for name, value in {
        "allocation_rb_count": allocation_rb_count,
        "dmrs_config_type": dmrs_config_type,
        "n_cell_id": n_cell_id,
        "dmrs_n_scid": dmrs_n_scid,
        "subcarrier_spacing_khz": subcarrier_spacing_khz,
        "slot_index": slot_index,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if not 1 <= allocation_rb_count <= 275:
        raise ValueError("allocation_rb_count must be in [1, 275]")
    if dmrs_config_type not in {1, 2}:
        raise ValueError("dmrs_config_type must be 1 or 2")
    if not 0 <= n_cell_id <= 1007:
        raise ValueError("n_cell_id must be in [0, 1007]")
    if dmrs_n_id is not None:
        if (
            not isinstance(dmrs_n_id, tuple)
            or len(dmrs_n_id) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 65535
                for value in dmrs_n_id
            )
        ):
            raise ValueError("dmrs_n_id must be a pair of integers in [0, 65535]")
    if dmrs_n_scid not in {0, 1}:
        raise ValueError("dmrs_n_scid must be 0 or 1")
    if subcarrier_spacing_khz not in {15, 30, 60, 120, 240, 480, 960}:
        raise ValueError("subcarrier_spacing_khz is unsupported")
    mu = int(math.log2(subcarrier_spacing_khz // 15))
    if not 0 <= slot_index < 10 * (2**mu):
        raise ValueError("slot_index must be in the subcarrier-spacing frame range")

    bwp_subcarriers = 12 * allocation_rb_count
    pilot_mask = np.zeros((14, bwp_subcarriers), dtype=bool)
    pilot_grid = np.zeros((14, bwp_subcarriers), dtype=np.complex128)
    subcarrier_offsets = (0, 2, 4, 6, 8, 10) if dmrs_config_type == 1 else (0, 1, 6, 7)
    relative_subcarriers = np.asarray(
        [12 * rb + offset for rb in range(allocation_rb_count) for offset in subcarrier_offsets],
        dtype=np.int64,
    )
    n_id = n_cell_id if dmrs_n_id is None else dmrs_n_id[dmrs_n_scid]
    hop_symbols = (
        ("first", plan_metadata["first_hop_dmrs_symbols"]),
        ("second", plan_metadata["second_hop_dmrs_symbols"]),
    )
    for _, symbols in hop_symbols:
        for symbol in symbols:
            if pilot_mask[symbol].any():
                raise ValueError("overlapping hop DM-RS positions are unsupported")
            sequence = _pusch_dmrs_gold_sequence(
                n_id=n_id,
                n_scid=dmrs_n_scid,
                slot_index=slot_index,
                symbol_index=symbol,
                count=len(relative_subcarriers),
            )
            pilot_mask[symbol, relative_subcarriers] = True
            pilot_grid[symbol, relative_subcarriers] = sequence

    def digest(value: np.ndarray) -> str:
        return hashlib.sha256(value.tobytes(order="C")).hexdigest()

    metadata = {
        "schema": "loslab.nr_pusch_intra_slot_hop_dmrs_grid.v1",
        "status": "GENERATED_NOT_QUALIFIED",
        "standard": "3GPP TS 38.211 V19.2.0",
        "sequence_clause": "6.4.1.1.1.1",
        "sequence_definition": "5.2.1",
        "dmrs_plan_sha256": hashlib.sha256(
            json.dumps(dict(plan_metadata), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "dmrs_config_type": dmrs_config_type,
        "port": 0,
        "n_id": n_id,
        "n_scid": dmrs_n_scid,
        "slot_index": slot_index,
        "subcarrier_spacing_khz": subcarrier_spacing_khz,
        "allocation_rb_count": allocation_rb_count,
        "grid_shape": [14, bwp_subcarriers],
        "pilot_mask_sha256": digest(pilot_mask),
        "pilot_grid_sha256": digest(pilot_grid),
        "scope": "single_port_dmrs_sequence_and_re_mask_only",
        "qualification": "NOT_QUALIFIED",
    }
    return NRPUSCHIntraSlotHopDMRSGrid(pilot_mask, pilot_grid, metadata)


def verify_pusch_intra_slot_hop_dmrs_grid(
    plan_metadata: Mapping[str, Any],
    pilot_mask: np.ndarray,
    pilot_grid: np.ndarray,
    **configuration: Any,
) -> dict[str, Any]:
    """Regenerate and byte-verify a bounded single-port hopping DM-RS grid."""
    if not isinstance(pilot_mask, np.ndarray) or pilot_mask.dtype != np.bool_:
        raise TypeError("pilot_mask must be a NumPy Boolean array")
    if not isinstance(pilot_grid, np.ndarray) or pilot_grid.dtype != np.complex128:
        raise TypeError("pilot_grid must be a NumPy complex128 array")
    expected = build_pusch_intra_slot_hop_dmrs_grid(plan_metadata, **configuration)
    if not np.array_equal(pilot_mask, expected.pilot_mask):
        raise ValueError("DM-RS pilot mask does not replay from the declared hopping plan")
    if not np.array_equal(pilot_grid, expected.pilot_grid):
        raise ValueError("DM-RS pilot values do not replay from the declared slot/configuration")
    return {
        "status": "PASS_DMRS_GRID_REPLAY",
        "pilot_mask_sha256": expected.metadata["pilot_mask_sha256"],
        "pilot_grid_sha256": expected.metadata["pilot_grid_sha256"],
        "qualification": "NOT_QUALIFIED",
    }


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


def compose_pusch_intra_slot_hops(
    bwp_allocation_grid: np.ndarray,
    *,
    carrier_grid_size_rb: int,
    carrier_grid_start_rb: int,
    bwp_start_rb: int,
    bwp_size_rb: int,
    allocation_start_symbol: int,
    first_hop_rb_start: int,
    second_hop_rb_start: int,
    first_hop_symbol_count: int,
    max_output_elements: int = 64_000_000,
) -> NRPUSCHFrequencyHopGrid:
    """Map two already-generated PUSCH hop segments into a carrier slot grid.

    ``bwp_allocation_grid`` has the Sionna transmitter layout
    ``[batch, tx, stream, allocated_symbol, allocated_subcarrier]``. The first
    ``first_hop_symbol_count`` symbols go to ``first_hop_rb_start`` and the
    remaining symbols go to ``second_hop_rb_start``; hop starts are RB offsets
    from the BWP start. The symbol boundary and frequency allocations must be
    resolved by the caller. This composes resource grids only; it does not
    generate hop-specific DMRS, decode RRC/DCI, or enable the pinned Sionna
    receiver's hopping mode. ``max_output_elements`` bounds the input and dense
    carrier-grid output together before allocation. Finiteness checks and
    hashes stream through a fixed-size buffer, avoiding grid-sized temporaries.
    """
    values = {
        "carrier_grid_size_rb": carrier_grid_size_rb,
        "carrier_grid_start_rb": carrier_grid_start_rb,
        "bwp_start_rb": bwp_start_rb,
        "bwp_size_rb": bwp_size_rb,
        "allocation_start_symbol": allocation_start_symbol,
        "first_hop_rb_start": first_hop_rb_start,
        "second_hop_rb_start": second_hop_rb_start,
        "first_hop_symbol_count": first_hop_symbol_count,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if (
        isinstance(max_output_elements, bool)
        or not isinstance(max_output_elements, int)
        or max_output_elements <= 0
    ):
        raise ValueError("max_output_elements must be a positive integer")
    if carrier_grid_size_rb < 1 or carrier_grid_size_rb > 275:
        raise ValueError("carrier_grid_size_rb must be in [1, 275]")
    if bwp_size_rb < 1 or bwp_size_rb > 275:
        raise ValueError("bwp_size_rb must be in [1, 275]")
    carrier_end_rb = carrier_grid_start_rb + carrier_grid_size_rb
    bwp_end_rb = bwp_start_rb + bwp_size_rb
    if (
        carrier_grid_start_rb < 0
        or carrier_grid_start_rb > MAX_CARRIER_GRID_START_CRB
        or carrier_end_rb > MAX_CRB_END_EXCLUSIVE
    ):
        raise ValueError(
            "carrier grid must fit within the supported absolute CRB range "
            f"[0, {MAX_CRB_END_EXCLUSIVE - 1}] and start no later than "
            f"{MAX_CARRIER_GRID_START_CRB}"
        )
    if bwp_start_rb >= MAX_CRB_END_EXCLUSIVE or bwp_end_rb > MAX_CRB_END_EXCLUSIVE:
        raise ValueError(
            f"BWP must fit within absolute CRB indices [0, {MAX_CRB_END_EXCLUSIVE - 1}]"
        )
    if bwp_start_rb < carrier_grid_start_rb or bwp_end_rb > carrier_end_rb:
        raise ValueError("BWP must lie within the carrier grid")

    if not isinstance(bwp_allocation_grid, np.ndarray):
        raise TypeError("bwp_allocation_grid must be a NumPy array")
    grid = bwp_allocation_grid
    if grid.ndim != 5 or any(size < 1 for size in grid.shape[:3]) or grid.shape[3] < 2:
        raise ValueError("bwp_allocation_grid must have five axes and at least two symbols")
    if grid.dtype.kind != "c" or grid.dtype.itemsize not in {8, 16}:
        raise ValueError("bwp_allocation_grid must contain complex64 or complex128 values")
    allocation_rb_count, remainder = divmod(grid.shape[4], 12)
    if remainder or not 1 <= allocation_rb_count <= bwp_size_rb:
        raise ValueError("allocated subcarrier width must be 12 times an in-BWP RB count")
    output_shape = (*grid.shape[:3], 14, carrier_grid_size_rb * 12)
    work_elements = grid.size + math.prod(output_shape)
    if work_elements > max_output_elements:
        raise ValueError(
            "PUSCH hop-grid input and output exceed max_output_elements; "
            f"required {work_elements}, limit {max_output_elements}"
        )
    if not _array_is_finite_c_order(grid):
        raise ValueError("bwp_allocation_grid must contain only finite values")
    if not 0 <= allocation_start_symbol < 14 or allocation_start_symbol + grid.shape[3] > 14:
        raise ValueError("PUSCH symbol allocation must fit within the 14-symbol slot")
    if not 1 <= first_hop_symbol_count < grid.shape[3]:
        raise ValueError("first_hop_symbol_count must split the allocation into two nonempty hops")
    if any(
        hop_start < 0 or hop_start + allocation_rb_count > bwp_size_rb
        for hop_start in (first_hop_rb_start, second_hop_rb_start)
    ):
        raise ValueError("each hop allocation must fit within the configured BWP")

    output = np.zeros(output_shape, dtype=grid.dtype)
    bwp_offset_in_carrier = bwp_start_rb - carrier_grid_start_rb
    first_start = (bwp_offset_in_carrier + first_hop_rb_start) * 12
    second_start = (bwp_offset_in_carrier + second_hop_rb_start) * 12
    allocation_end_symbol = allocation_start_symbol + grid.shape[3]
    hop_boundary_symbol = allocation_start_symbol + first_hop_symbol_count
    output[
        ..., allocation_start_symbol:hop_boundary_symbol, first_start : first_start + grid.shape[4]
    ] = grid[..., :first_hop_symbol_count, :]
    output[
        ..., hop_boundary_symbol:allocation_end_symbol, second_start : second_start + grid.shape[4]
    ] = grid[..., first_hop_symbol_count:, :]
    source_digest = _sha256_array_c_order(grid)
    output_digest = _sha256_array_c_order(output)
    metadata = {
        "schema": "loslab.nr_pusch_intra_slot_hop_grid.v1",
        "scope": "frequency_domain_grid_composition_only",
        "carrier_grid_start_rb": carrier_grid_start_rb,
        "carrier_grid_size_rb": carrier_grid_size_rb,
        "bwp_start_rb": bwp_start_rb,
        "bwp_size_rb": bwp_size_rb,
        "allocation_rb_count": allocation_rb_count,
        "allocation_start_symbol": allocation_start_symbol,
        "allocation_symbol_count": int(grid.shape[3]),
        "allocation_grid_shape": [int(size) for size in grid.shape],
        "complex_dtype": str(grid.dtype),
        "carrier_grid_shape": [int(size) for size in output.shape],
        "first_hop_symbol_count": first_hop_symbol_count,
        "hop_boundary_symbol": hop_boundary_symbol,
        "first_hop_start_crb": bwp_start_rb + first_hop_rb_start,
        "second_hop_start_crb": bwp_start_rb + second_hop_rb_start,
        "source_grid_sha256": source_digest,
        "carrier_grid_sha256": output_digest,
        "receiver_hopping_qualification": "NOT_QUALIFIED",
    }
    return NRPUSCHFrequencyHopGrid(output, metadata)


def verify_pusch_intra_slot_hops(
    bwp_allocation_grid: np.ndarray,
    carrier_slot_grid: np.ndarray,
    metadata: Mapping[str, Any],
    *,
    max_output_elements: int = 64_000_000,
) -> dict[str, Any]:
    """Replay and verify a saved PUSCH hop-grid composition and its digests."""
    if not isinstance(carrier_slot_grid, np.ndarray):
        raise TypeError("carrier_slot_grid must be a NumPy array")
    if not isinstance(metadata, Mapping) or metadata.get("schema") != (
        "loslab.nr_pusch_intra_slot_hop_grid.v1"
    ):
        raise ValueError("unsupported PUSCH hop-grid metadata schema")
    declared_grid_shape = metadata.get("carrier_grid_shape")
    if (
        not isinstance(declared_grid_shape, list)
        or len(declared_grid_shape) != 5
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in declared_grid_shape
        )
    ):
        raise ValueError("PUSCH hop-grid metadata contains a malformed carrier-grid shape")
    if tuple(declared_grid_shape) != carrier_slot_grid.shape:
        raise ValueError("observed carrier grid shape differs from replay metadata")
    if carrier_slot_grid.dtype.kind != "c" or carrier_slot_grid.dtype.itemsize not in {8, 16}:
        raise ValueError("carrier_slot_grid must contain complex64 or complex128 values")
    if metadata.get("complex_dtype") != str(carrier_slot_grid.dtype):
        raise ValueError("observed carrier grid dtype differs from replay metadata")
    integer_fields = (
        "carrier_grid_size_rb",
        "carrier_grid_start_rb",
        "bwp_start_rb",
        "bwp_size_rb",
        "allocation_start_symbol",
        "allocation_rb_count",
        "first_hop_symbol_count",
        "first_hop_start_crb",
        "second_hop_start_crb",
    )
    if any(
        isinstance(metadata.get(field), bool) or not isinstance(metadata.get(field), int)
        for field in integer_fields
    ):
        raise ValueError("PUSCH hop-grid metadata contains malformed layout values")
    first_offset = metadata["first_hop_start_crb"] - metadata["bwp_start_rb"]
    second_offset = metadata["second_hop_start_crb"] - metadata["bwp_start_rb"]
    replay = compose_pusch_intra_slot_hops(
        bwp_allocation_grid,
        carrier_grid_size_rb=metadata["carrier_grid_size_rb"],
        carrier_grid_start_rb=metadata["carrier_grid_start_rb"],
        bwp_start_rb=metadata["bwp_start_rb"],
        bwp_size_rb=metadata["bwp_size_rb"],
        allocation_start_symbol=metadata["allocation_start_symbol"],
        first_hop_rb_start=first_offset,
        second_hop_rb_start=second_offset,
        first_hop_symbol_count=metadata["first_hop_symbol_count"],
        max_output_elements=max_output_elements,
    )
    if dict(metadata) != replay.metadata:
        raise ValueError("PUSCH hop-grid metadata does not replay from the source grid")
    observed = carrier_slot_grid
    if observed.dtype != replay.frequency_grid.dtype or not np.array_equal(
        observed, replay.frequency_grid
    ):
        raise ValueError("PUSCH carrier grid does not replay from the source grid")
    return {
        "schema": "loslab.nr_pusch_hop_grid_replay.v1",
        "status": "PASS_INPUT_SCOPED_REPLAY",
        "qualification_status": "NOT_QUALIFIED",
        "source_grid_sha256": replay.metadata["source_grid_sha256"],
        "carrier_grid_sha256": replay.metadata["carrier_grid_sha256"],
    }


def dehop_pusch_intra_slot_received_grid(
    received_carrier_grid: np.ndarray,
    hop_metadata: Mapping[str, Any],
    *,
    max_output_elements: int = 64_000_000,
) -> NRPUSCHDehoppedGrid:
    """Rehome the two received PUSCH hop segments onto the first-hop allocation.

    The result retains the full carrier-grid shape and is suitable for the
    ordinary non-hopping BWP estimator when its configured BWP starts at
    ``first_hop_start_crb`` and has ``allocation_rb_count`` RBs. Samples from
    each source hop are copied at their recorded OFDM symbols; other symbols
    and out-of-allocation subcarriers are zero-filled. This handles resource
    relocation only: it does not generate normative hop schedules, hop-specific
    DMRS, channel truth, or qualification evidence.
    """
    if not isinstance(received_carrier_grid, np.ndarray):
        raise TypeError("received_carrier_grid must be a NumPy array")
    grid = received_carrier_grid
    if grid.ndim != 5 or any(size < 1 for size in grid.shape[:3]):
        raise ValueError(
            "received_carrier_grid must have [batch, rx, rx_ant, symbol, subcarrier] axes"
        )
    if grid.shape[3] != 14:
        raise ValueError("received_carrier_grid must contain exactly one 14-symbol slot")
    if grid.dtype.kind != "c" or grid.dtype.itemsize not in {8, 16}:
        raise ValueError("received_carrier_grid must contain complex64 or complex128 values")
    if not _array_is_finite_c_order(grid):
        raise ValueError("received_carrier_grid must contain only finite values")
    if not isinstance(hop_metadata, Mapping) or hop_metadata.get("schema") != (
        "loslab.nr_pusch_intra_slot_hop_grid.v1"
    ):
        raise ValueError("unsupported PUSCH hop-grid metadata schema")
    integer_fields = (
        "carrier_grid_start_rb",
        "carrier_grid_size_rb",
        "bwp_start_rb",
        "bwp_size_rb",
        "allocation_rb_count",
        "allocation_start_symbol",
        "allocation_symbol_count",
        "first_hop_symbol_count",
        "hop_boundary_symbol",
        "first_hop_start_crb",
        "second_hop_start_crb",
    )
    if any(
        isinstance(hop_metadata.get(name), bool) or not isinstance(hop_metadata.get(name), int)
        for name in integer_fields
    ):
        raise ValueError("PUSCH hop-grid metadata contains malformed layout values")
    if hop_metadata.get("receiver_hopping_qualification") != "NOT_QUALIFIED":
        raise ValueError("PUSCH hop-grid metadata does not preserve its unqualified status")
    carrier_start = hop_metadata["carrier_grid_start_rb"]
    carrier_size = hop_metadata["carrier_grid_size_rb"]
    allocation_start_symbol = hop_metadata["allocation_start_symbol"]
    allocation_symbol_count = hop_metadata["allocation_symbol_count"]
    hop_count = hop_metadata["first_hop_symbol_count"]
    boundary = hop_metadata["hop_boundary_symbol"]
    allocation_rbs = hop_metadata["allocation_rb_count"]
    first_start_crb = hop_metadata["first_hop_start_crb"]
    second_start_crb = hop_metadata["second_hop_start_crb"]
    width = allocation_rbs * 12
    carrier_width = carrier_size * 12
    end_symbol = allocation_start_symbol + allocation_symbol_count
    if grid.shape[-1] != carrier_width:
        raise ValueError("received carrier-grid width differs from hop metadata")
    if (
        not 1 <= carrier_size <= 275
        or not 1 <= hop_metadata["bwp_size_rb"] <= 275
        or hop_metadata["bwp_start_rb"] < carrier_start
        or hop_metadata["bwp_start_rb"] + hop_metadata["bwp_size_rb"] > carrier_start + carrier_size
    ):
        raise ValueError("PUSCH hop metadata contains an invalid carrier/BWP allocation")
    if (
        allocation_start_symbol < 0
        or allocation_symbol_count < 2
        or end_symbol > 14
        or not 1 <= hop_count < allocation_symbol_count
        or boundary != allocation_start_symbol + hop_count
    ):
        raise ValueError("PUSCH hop-grid metadata contains an invalid symbol layout")
    if not 1 <= allocation_rbs <= hop_metadata["bwp_size_rb"]:
        raise ValueError("PUSCH hop-grid metadata contains an invalid allocation width")
    if (
        first_start_crb < hop_metadata["bwp_start_rb"]
        or second_start_crb < hop_metadata["bwp_start_rb"]
        or first_start_crb + allocation_rbs
        > (hop_metadata["bwp_start_rb"] + hop_metadata["bwp_size_rb"])
        or second_start_crb + allocation_rbs
        > (hop_metadata["bwp_start_rb"] + hop_metadata["bwp_size_rb"])
    ):
        raise ValueError("PUSCH hop allocations must fit within the declared BWP")
    relative_offsets = (
        first_start_crb - carrier_start,
        second_start_crb - carrier_start,
    )
    if any(
        offset < 0 or (offset + allocation_rbs) * 12 > carrier_width for offset in relative_offsets
    ):
        raise ValueError("PUSCH hop allocations must fit within the declared carrier grid")
    allocation_shape = hop_metadata.get("allocation_grid_shape")
    carrier_shape = hop_metadata.get("carrier_grid_shape")
    if (
        not isinstance(allocation_shape, list)
        or len(allocation_shape) != 5
        or tuple(allocation_shape[3:]) != (allocation_symbol_count, width)
        or not isinstance(carrier_shape, list)
        or len(carrier_shape) != 5
        or tuple(carrier_shape) != tuple(grid.shape)
        or hop_metadata.get("complex_dtype") != str(grid.dtype)
    ):
        raise ValueError("PUSCH hop metadata shapes disagree with its resource allocation")
    output_shape = grid.shape
    work_elements = grid.size + math.prod(output_shape)
    if (
        isinstance(max_output_elements, bool)
        or not isinstance(max_output_elements, int)
        or max_output_elements <= 0
    ):
        raise ValueError("max_output_elements must be a positive integer")
    if work_elements > max_output_elements:
        raise ValueError(
            "PUSCH dehop input and output exceed max_output_elements; "
            f"required {work_elements}, limit {max_output_elements}"
        )

    output = np.zeros(output_shape, dtype=grid.dtype)
    first_source = relative_offsets[0] * 12
    second_source = relative_offsets[1] * 12
    target_start = first_source
    target_end = target_start + width
    first_symbol_end = allocation_start_symbol + hop_count
    output[..., allocation_start_symbol:first_symbol_end, target_start:target_end] = grid[
        ..., allocation_start_symbol:first_symbol_end, first_source : first_source + width
    ]
    output[..., first_symbol_end:end_symbol, target_start:target_end] = grid[
        ..., first_symbol_end:end_symbol, second_source : second_source + width
    ]
    input_digest = _sha256_array_c_order(grid)
    output_digest = _sha256_array_c_order(output)
    hop_layout_digest = hashlib.sha256(
        json.dumps(
            dict(hop_metadata), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    metadata = {
        "schema": "loslab.nr_pusch_intra_slot_dehopped_grid.v1",
        "scope": "received_grid_relocation_only",
        "receiver_hopping_qualification": "NOT_QUALIFIED",
        "input_carrier_grid_sha256": input_digest,
        "output_carrier_grid_sha256": output_digest,
        "hop_composition_sha256": hop_layout_digest,
        "input_shape": [int(size) for size in grid.shape],
        "output_shape": [int(size) for size in output.shape],
        "carrier_grid_start_rb": carrier_start,
        "carrier_grid_size_rb": carrier_size,
        "receiver_bwp_start_rb": first_start_crb,
        "receiver_bwp_size_rb": allocation_rbs,
        "allocation_start_symbol": allocation_start_symbol,
        "allocation_symbol_count": allocation_symbol_count,
        "first_hop_symbol_count": hop_count,
        "hop_boundary_symbol": boundary,
        "first_hop_start_crb": first_start_crb,
        "second_hop_start_crb": second_start_crb,
        "allocation_rb_count": allocation_rbs,
        "target_start_subcarrier": target_start,
        "allocation_subcarrier_count": width,
        "complex_dtype": str(grid.dtype),
    }
    output.setflags(write=False)
    return NRPUSCHDehoppedGrid(output, metadata)


def verify_pusch_intra_slot_dehopped_grid(
    received_carrier_grid: np.ndarray,
    hop_metadata: Mapping[str, Any],
    dehopped_carrier_grid: np.ndarray,
    dehopped_metadata: Mapping[str, Any],
    *,
    max_output_elements: int = 64_000_000,
) -> dict[str, Any]:
    """Replay a claimed received-grid relocation from exact source arrays."""
    replay = dehop_pusch_intra_slot_received_grid(
        received_carrier_grid,
        hop_metadata,
        max_output_elements=max_output_elements,
    )
    if not isinstance(dehopped_carrier_grid, np.ndarray) or not isinstance(
        dehopped_metadata, Mapping
    ):
        raise TypeError("dehopped grid and metadata must be NumPy array and mapping values")
    if dict(dehopped_metadata) != dict(replay.metadata):
        raise ValueError("PUSCH dehopped-grid metadata does not replay")
    if dehopped_carrier_grid.dtype != replay.frequency_grid.dtype or not np.array_equal(
        dehopped_carrier_grid, replay.frequency_grid
    ):
        raise ValueError("PUSCH dehopped carrier grid does not replay")
    return {
        "schema": "loslab.nr_pusch_dehopped_grid_replay.v1",
        "status": "PASS_INPUT_SCOPED_REPLAY",
        "qualification_status": "NOT_QUALIFIED",
        "input_carrier_grid_sha256": replay.metadata["input_carrier_grid_sha256"],
        "output_carrier_grid_sha256": replay.metadata["output_carrier_grid_sha256"],
        "hop_composition_sha256": replay.metadata["hop_composition_sha256"],
    }


def _sha256_array_c_order(array: np.ndarray) -> str:
    """Hash array bytes in C order using a bounded iterator buffer."""
    digest = hashlib.sha256()
    for chunk in np.nditer(  # type: ignore[call-overload]
        array,
        flags=["external_loop", "buffered"],
        op_flags=["readonly"],
        order="C",
        buffersize=65_536,
    ):
        digest.update(np.asarray(chunk).tobytes(order="C"))
    return digest.hexdigest()


def _array_is_finite_c_order(array: np.ndarray) -> bool:
    """Check finiteness with bounded temporary storage for strided arrays."""
    for chunk in np.nditer(  # type: ignore[call-overload]
        array,
        flags=["external_loop", "buffered"],
        op_flags=["readonly"],
        order="C",
        buffersize=65_536,
    ):
        if not bool(np.isfinite(chunk).all()):
            return False
    return True


@dataclass(frozen=True, slots=True)
class NRPUSCHConfig:
    """PUSCH/DMRS resource-grid and channel-estimator configuration.

    Subcarrier spacing is in kHz and BWP size is in resource blocks.
    Sionna validates additional cross-field constraints when the optional
    adapter is constructed.
    """

    n_size_bwp: int = 4
    subcarrier_spacing_khz: int = 30
    num_antenna_ports: int = 1
    num_layers: int = 1
    n_cell_id: int = 1
    dmrs_n_id: tuple[int, int] | None = None
    dmrs_n_scid: int = 0
    dmrs_config_type: int = 1
    dmrs_length: int = 1
    dmrs_additional_position: int = 0
    num_cdm_groups_without_data: int = 1
    mapping_type: str = "A"
    symbol_allocation: tuple[int, int] = (0, 14)
    precoding: str = "non-codebook"
    interpolation_type: str = "lin"
    n_size_grid: int | None = None
    n_start_grid: int = 0
    n_start_bwp: int = 0
    frequency_hopping: str = "neither"
    transform_precoding: bool = False

    def __post_init__(self) -> None:
        integer_fields = (
            "n_size_bwp",
            "subcarrier_spacing_khz",
            "n_start_grid",
            "n_start_bwp",
            "num_antenna_ports",
            "num_layers",
            "n_cell_id",
            "dmrs_n_scid",
            "dmrs_config_type",
            "dmrs_length",
            "dmrs_additional_position",
            "num_cdm_groups_without_data",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if not 1 <= self.n_size_bwp <= 275:
            raise ValueError("n_size_bwp must be in [1, 275]")
        n_size_grid = self.n_size_grid
        if n_size_grid is None:
            n_size_grid = self.n_size_bwp
            object.__setattr__(self, "n_size_grid", n_size_grid)
        elif isinstance(n_size_grid, bool) or not isinstance(n_size_grid, int):
            raise TypeError("n_size_grid must be an integer or None")
        if not 1 <= n_size_grid <= 275:
            raise ValueError("n_size_grid must be in [1, 275]")
        if not 0 <= self.n_start_grid <= 2199:
            raise ValueError("n_start_grid must be in [0, 2199]")
        if not 0 <= self.n_start_bwp <= 2473:
            raise ValueError("n_start_bwp must be in [0, 2473]")
        bwp_start_in_grid = self.n_start_bwp - self.n_start_grid
        if bwp_start_in_grid < 0 or bwp_start_in_grid + self.n_size_bwp > n_size_grid:
            raise ValueError("BWP must lie within the configured carrier resource grid")
        if self.subcarrier_spacing_khz not in {15, 30, 60, 120, 240, 480, 960}:
            raise ValueError("Unsupported NR subcarrier spacing")
        if self.num_antenna_ports not in {1, 2, 4}:
            raise ValueError("num_antenna_ports must be 1, 2, or 4")
        if not 1 <= self.num_layers <= self.num_antenna_ports:
            raise ValueError("num_layers must be between 1 and num_antenna_ports")
        if not 0 <= self.n_cell_id <= 1007:
            raise ValueError("n_cell_id must be in [0, 1007]")
        if self.dmrs_n_id is not None:
            if not isinstance(self.dmrs_n_id, tuple) or len(self.dmrs_n_id) != 2:
                raise TypeError("dmrs_n_id must be a two-element tuple or None")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 65535
                for value in self.dmrs_n_id
            ):
                raise ValueError("each dmrs_n_id value must be an integer in [0, 65535]")
        if self.dmrs_n_scid not in {0, 1}:
            raise ValueError("dmrs_n_scid must be 0 or 1")
        if self.dmrs_config_type not in {1, 2}:
            raise ValueError("dmrs_config_type must be 1 or 2")
        if self.dmrs_length not in {1, 2}:
            raise ValueError("dmrs_length must be 1 or 2")
        max_position = 3 if self.dmrs_length == 1 else 1
        if not 0 <= self.dmrs_additional_position <= max_position:
            raise ValueError(f"dmrs_additional_position must be in [0, {max_position}]")
        if not 1 <= self.num_cdm_groups_without_data <= 3:
            raise ValueError("num_cdm_groups_without_data must be in [1, 3]")
        if self.mapping_type not in {"A", "B"}:
            raise ValueError("mapping_type must be A or B")
        if not isinstance(self.frequency_hopping, str) or self.frequency_hopping != "neither":
            raise ValueError(
                "frequency_hopping currently supports only 'neither'; "
                "PUSCH frequency hopping is not implemented by this receiver"
            )
        if type(self.transform_precoding) is not bool:
            raise TypeError("transform_precoding must be a Boolean")
        if self.transform_precoding:
            raise ValueError(
                "transform_precoding is not supported: pinned Sionna PHY 2.2.0 "
                "does not implement DFT-s-OFDM PUSCH"
            )
        if (
            not isinstance(self.symbol_allocation, (tuple, list))
            or len(self.symbol_allocation) != 2
        ):
            raise TypeError("symbol_allocation must be a two-element tuple or list")
        allocation = tuple(self.symbol_allocation)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in allocation):
            raise TypeError("symbol_allocation values must be integers")
        object.__setattr__(self, "symbol_allocation", allocation)
        start_symbol, duration = allocation
        if not 0 <= start_symbol < 14 or not 1 <= duration <= 14 - start_symbol:
            raise ValueError("symbol_allocation must fit within the 14-symbol slot")
        if self.mapping_type == "A" and start_symbol != 0:
            raise ValueError("PUSCH mapping type A requires symbol allocation to start at 0")
        if self.precoding not in {"codebook", "non-codebook"}:
            raise ValueError("precoding must be codebook or non-codebook")
        if self.interpolation_type not in {"nn", "lin", "lin_time_avg"}:
            raise ValueError("Unsupported Sionna PUSCH interpolation type")

    @property
    def carrier_grid_size(self) -> int:
        """Carrier resource-block count, defaulting to the complete configured BWP."""
        return self.n_size_bwp if self.n_size_grid is None else self.n_size_grid


@dataclass(frozen=True, slots=True)
class NRChannelEstimate:
    """Estimated channel, broadcastable estimator variance, pilot mask, and metadata.

    Channel axes remain [batch, rx, rx antenna, tx, stream, OFDM symbol,
    effective subcarrier]; no antenna or time axis is flattened.
    """

    h_hat: Any
    error_variance: Any
    pilot_mask: Any
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        h_shape = tuple(getattr(self.h_hat, "shape", ()))
        error_shape = tuple(getattr(self.error_variance, "shape", ()))
        mask_shape = tuple(getattr(self.pilot_mask, "shape", ()))
        if len(h_shape) != 7:
            raise ValueError("h_hat must have seven named channel axes")
        if any(size <= 0 for size in h_shape):
            raise ValueError("h_hat axes must be non-empty")
        if len(error_shape) != 7 or any(
            variance_axis not in (1, estimate_axis)
            for variance_axis, estimate_axis in zip(error_shape, h_shape, strict=True)
        ):
            raise ValueError("error_variance must be broadcastable to h_hat")
        if any(size <= 0 for size in error_shape):
            raise ValueError("error_variance axes must be non-empty")
        if len(mask_shape) != 4 or mask_shape != h_shape[3:]:
            raise ValueError("pilot_mask must have [tx, stream, symbol, subcarrier] axes")
        if not isinstance(self.metadata, Mapping) or not self.metadata:
            raise ValueError("estimate metadata must not be empty")

        torch_module: Any | None = None
        if any(
            type(value).__module__.partition(".")[0] == "torch"
            for value in (self.h_hat, self.error_variance, self.pilot_mask)
        ):
            try:
                import torch as imported_torch
            except ImportError:
                pass
            else:
                torch_module = imported_torch

        def array_domain(value: Any, name: str) -> tuple[str, bool, bool]:
            if isinstance(value, np.ndarray):
                kind = value.dtype.kind
                if kind not in "biufc":
                    raise TypeError(f"{name} must be a dense numeric array")
                finite = bool(np.isfinite(value).all())
                negative = bool(np.any(value < 0)) if kind in "iuf" else False
                return kind, finite, negative
            if torch_module is not None and torch_module.is_tensor(value):
                if value.layout != torch_module.strided:
                    raise TypeError(f"{name} must be a dense numeric tensor")
                is_complex = value.is_complex()
                is_bool = value.dtype == torch_module.bool
                is_real = value.is_floating_point() or (not is_complex and not is_bool)
                if not (is_complex or is_bool or is_real):
                    raise TypeError(f"{name} must be a dense numeric tensor")
                finite = bool(torch_module.isfinite(value).all())
                negative = bool((value < 0).any()) if is_real and not is_bool else False
                if is_complex:
                    kind = "c"
                elif is_bool:
                    kind = "b"
                elif value.is_floating_point():
                    kind = "f"
                else:
                    kind = "i"
                return kind, finite, negative
            raise TypeError(f"{name} must be a NumPy array or dense PyTorch tensor")

        h_kind, h_finite, _ = array_domain(self.h_hat, "h_hat")
        if h_kind != "c":
            raise TypeError("h_hat must be complex-valued")
        if not h_finite:
            raise ValueError("h_hat must contain only finite values")
        variance_kind, variance_finite, variance_negative = array_domain(
            self.error_variance, "error_variance"
        )
        if variance_kind not in "uf":
            raise TypeError("error_variance must be real-valued")
        if not variance_finite:
            raise ValueError("error_variance must contain only finite values")
        if variance_negative:
            raise ValueError("error_variance must be nonnegative")
        mask_kind, mask_finite, _ = array_domain(self.pilot_mask, "pilot_mask")
        if mask_kind != "b":
            raise TypeError("pilot_mask must be Boolean-valued")
        if not mask_finite:
            raise ValueError("pilot_mask must contain only finite values")
