"""DM-RS plans and grids for NR PUSCH intra-slot hopping."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np

from receivers.nr.pusch_core.types import (
    NRPUSCHIntraSlotHopDMRSGrid,
    NRPUSCHIntraSlotHopDMRSPlan,
)


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
