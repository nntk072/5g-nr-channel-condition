"""NR SRS configuration: comb, cyclic shift, bandwidth, and frequency allocation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import cast

import numpy as np

from .srs_constants import (
    _SRS_BANDWIDTH_TABLE,
    _SRS_SYMBOL_COUNTS,
    NRSRSMultiportResourceGrid,
    NRSRSPositioningMultiportResourceGrid,
    NRSRSPositioningResourceGrid,
    _normalized_multiport_beta,
    _nr_gold_bits,
    _nr_srs_dci_0_3_antenna_ports_width,
    _nr_srs_dci_0_3_pdcch_monitoring_adaptation_width,
    _nr_srs_dci_0_3_precoding_width,
    _nr_srs_dci_0_3_scell_dormancy_width,
    _nr_srs_dci_0_3_sri_width,
    _validate_srs_hopping_position,
    nr_srs_dci_0_3_frequency_assignment_widths,
    nr_srs_dci_0_3_scheduled_cell_ids,
)
from .srs_lookups import (
    map_nr_srs_resource,
    nr_dci_0_3_field_values_from_layout,
    nr_low_papr_sequence_type1,
    nr_srs_dci_0_3_ul_bwp_selection,
    nr_srs_frequency_position_indices,
    nr_srs_group_hopping_index,
    nr_srs_port_symbol_activity,
    nr_srs_positioning_symbol_offset,
    nr_srs_sequence_hopping_index,
    nr_srs_triggered_resource_set_ids,
    srs_cyclic_shift_count,
)


@dataclass(frozen=True, slots=True)
class NRSRSTransmissionCombConfiguration:
    transmission_comb: int
    comb_offset: int
    base_cyclic_shift: int

    def __post_init__(self) -> None:
        for name in ("transmission_comb", "comb_offset", "base_cyclic_shift"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
        comb = int(self.transmission_comb)
        maximum = srs_cyclic_shift_count(comb)
        if not 0 <= self.comb_offset < comb:
            raise ValueError("comb_offset must lie inside the transmission comb")
        if not 0 <= self.base_cyclic_shift < maximum:
            raise ValueError(f"base_cyclic_shift must be in [0, {maximum})")
        object.__setattr__(self, "transmission_comb", comb)
        object.__setattr__(self, "comb_offset", int(self.comb_offset))
        object.__setattr__(self, "base_cyclic_shift", int(self.base_cyclic_shift))


def nr_srs_transmission_comb_from_rrc(
    decoded_choice: Mapping[str, object], *, positioning_resource: bool = False
) -> NRSRSTransmissionCombConfiguration:
    """Resolve a decoded TS 38.331 transmissionComb CHOICE."""
    if not isinstance(decoded_choice, Mapping) or len(decoded_choice) != 1:
        raise ValueError("transmissionComb must be a single-choice decoded mapping")
    if not isinstance(positioning_resource, bool):
        raise TypeError("positioning_resource must be a boolean")
    choice, raw_value = next(iter(decoded_choice.items()))
    suffix = "-r16" if positioning_resource else ""
    if choice not in {f"n{comb}{suffix}" for comb in (2, 4, 8)}:
        raise ValueError("transmissionComb choice must be n2, n4, or n8 for this resource")
    if not isinstance(raw_value, Mapping):
        raise ValueError("transmissionComb choice value must be a mapping")
    comb = int(choice[1:].removesuffix(suffix))
    fields = {f"combOffset-n{comb}{suffix}", f"cyclicShift-n{comb}{suffix}"}
    if set(raw_value) != fields:
        raise ValueError("transmissionComb choice has missing or unsupported fields")
    return NRSRSTransmissionCombConfiguration(
        transmission_comb=comb,
        comb_offset=raw_value[f"combOffset-n{comb}{suffix}"],  # type: ignore[arg-type]
        base_cyclic_shift=raw_value[f"cyclicShift-n{comb}{suffix}"],  # type: ignore[arg-type]
    )


def nr_srs_port_cyclic_shift_indices(
    antenna_ports: tuple[int, ...],
    *,
    transmission_comb: int,
    base_cyclic_shift: int,
    ports8_tdm: bool = False,
) -> tuple[int, ...]:
    """Resolve TS 38.211 antenna-port cyclic shifts for an SRS resource.

    ``antenna_ports`` contains the configured SRS antenna-port numbers
    (1000 onward), in the desired output order. ``base_cyclic_shift`` is the
    ``n_SRS^cs`` value configured by ``transmissionComb``. For the eight-port
    TDM option, the standard's effective four-port roster and port remapping
    are applied before deriving the cyclic shifts.
    """
    if not isinstance(antenna_ports, tuple) or not antenna_ports:
        raise TypeError("antenna_ports must be a nonempty tuple of integers")
    if not isinstance(ports8_tdm, bool):
        raise TypeError("ports8_tdm must be a boolean")
    for name, value in (
        ("transmission_comb", transmission_comb),
        ("base_cyclic_shift", base_cyclic_shift),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    maximum = srs_cyclic_shift_count(int(transmission_comb))
    base_cyclic_shift = int(base_cyclic_shift)
    if not 0 <= base_cyclic_shift < maximum:
        raise ValueError(f"base_cyclic_shift must be in [0, {maximum})")
    if any(isinstance(port, bool) or not isinstance(port, Integral) for port in antenna_ports):
        raise TypeError("antenna_ports must contain integers")
    ports = tuple(int(port) for port in antenna_ports)
    port_count = 8 if ports8_tdm else len(ports)
    if port_count not in {1, 2, 4, 8}:
        raise ValueError("SRS antenna-port count must be 1, 2, 4, or 8")
    if ports != tuple(range(1000, 1000 + port_count)):
        raise ValueError("antenna_ports must list the complete configured SRS port roster")
    effective_count = 4 if ports8_tdm else port_count
    shifts: list[int] = []
    for port in ports:
        offset = port - 1000
        mapped_port = 1000 + offset % 2 + (2 if offset >= 4 else 0) if ports8_tdm else port
        mapped_index = mapped_port - 1000
        if effective_count == 8 and maximum == 6:
            increment = maximum * (mapped_index // 4) // (effective_count // 4)
        elif (effective_count == 4 and maximum == 6) or (effective_count == 8 and maximum == 12):
            increment = maximum * (mapped_index // 2) // (effective_count // 2)
        else:
            increment = maximum * mapped_index // effective_count
        shifts.append((base_cyclic_shift + increment) % maximum)
    return tuple(shifts)


def nr_srs_cyclic_shift_hopping_offset(
    *,
    transmission_comb: int,
    cyclic_shift_hopping_id: int,
    radio_frame_index: int,
    slot_index: int,
    slots_per_frame: int,
    slot_symbol_count: int,
    start_symbol_index: int,
    resource_symbol_index: int,
    hopping_subset: tuple[bool, ...] | None = None,
    hopping_finer_granularity: bool = False,
) -> int:
    """Resolve TS 38.211 V19.2.0 ``f_csh`` for one SRS resource symbol.

    The slot index is local to its radio frame. A subset is supplied as the
    configured ``n_SRS^cs,max``-bit bitmap; finer granularity and a subset are
    mutually exclusive, as specified for the cyclicShiftHopping IE.
    """
    for name, value in (
        ("transmission_comb", transmission_comb),
        ("cyclic_shift_hopping_id", cyclic_shift_hopping_id),
        ("radio_frame_index", radio_frame_index),
        ("slot_index", slot_index),
        ("slots_per_frame", slots_per_frame),
        ("slot_symbol_count", slot_symbol_count),
        ("start_symbol_index", start_symbol_index),
        ("resource_symbol_index", resource_symbol_index),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    if not isinstance(hopping_finer_granularity, bool):
        raise TypeError("hopping_finer_granularity must be a boolean")
    maximum = srs_cyclic_shift_count(int(transmission_comb))
    identity = int(cyclic_shift_hopping_id)
    frame = int(radio_frame_index)
    slot = int(slot_index)
    slots = int(slots_per_frame)
    symbols = int(slot_symbol_count)
    start = int(start_symbol_index)
    resource_symbol = int(resource_symbol_index)
    if not 0 <= identity <= 65535:
        raise ValueError("cyclic_shift_hopping_id must be in [0, 65535]")
    if frame < 0 or slots not in {10, 20, 40, 80, 160, 320} or not 0 <= slot < slots:
        raise ValueError("radio frame and frame-local slot indices are invalid")
    if symbols not in {12, 14}:
        raise ValueError("slot_symbol_count must be 12 or 14")
    if not 0 <= start < symbols or not 0 <= resource_symbol < symbols - start:
        raise ValueError("SRS symbol coordinates must lie inside the slot")

    granularity = 2 if hopping_finer_granularity else 1
    if hopping_subset is None:
        subset = tuple(range(granularity * maximum))
    else:
        if not isinstance(hopping_subset, tuple) or len(hopping_subset) != maximum:
            raise ValueError("hopping_subset must be a bitmap matching the comb shift count")
        if any(not isinstance(bit, (bool, np.bool_)) for bit in hopping_subset):
            raise TypeError("hopping_subset must contain boolean bitmap entries")
        if hopping_finer_granularity:
            raise ValueError("hopping_subset and finer granularity cannot both be configured")
        subset = tuple(index for index, enabled in enumerate(hopping_subset) if enabled)
        if not 1 < len(subset) <= maximum:
            raise ValueError("hopping_subset must enable between 2 and n_cs,max shifts")

    symbol_offset = ((frame % 128) * slots + slot) * symbols + start + resource_symbol
    bits = _nr_gold_bits(identity, 8 * symbol_offset + 8)[8 * symbol_offset :]
    hop_word = sum(int(bit) << bit_index for bit_index, bit in enumerate(bits))
    return subset[hop_word % len(subset)]


def nr_srs_cyclic_shift_hopping_offsets(
    *,
    transmission_comb: int,
    cyclic_shift_hopping_id: int,
    radio_frame_index: int,
    slot_index: int,
    slots_per_frame: int,
    slot_symbol_count: int,
    start_symbol_index: int,
    srs_symbol_count: int,
    hopping_subset: tuple[bool, ...] | None = None,
    hopping_finer_granularity: bool = False,
) -> tuple[int, ...]:
    """Resolve ``f_csh`` for every resource-local SRS symbol in one slot."""
    for name, value in (
        ("slot_symbol_count", slot_symbol_count),
        ("start_symbol_index", start_symbol_index),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    slot_symbol_count = int(slot_symbol_count)
    start_symbol_index = int(start_symbol_index)
    if slot_symbol_count not in {12, 14} or not 0 <= start_symbol_index < slot_symbol_count:
        raise ValueError("slot symbol coordinates are invalid")
    if (
        isinstance(srs_symbol_count, bool)
        or not isinstance(srs_symbol_count, Integral)
        or int(srs_symbol_count) not in _SRS_SYMBOL_COUNTS
    ):
        raise ValueError("srs_symbol_count must be one of the supported SRS lengths")
    if int(srs_symbol_count) > slot_symbol_count - start_symbol_index:
        raise ValueError("SRS resource symbols extend past the slot")
    return tuple(
        nr_srs_cyclic_shift_hopping_offset(
            transmission_comb=transmission_comb,
            cyclic_shift_hopping_id=cyclic_shift_hopping_id,
            radio_frame_index=radio_frame_index,
            slot_index=slot_index,
            slots_per_frame=slots_per_frame,
            slot_symbol_count=slot_symbol_count,
            start_symbol_index=start_symbol_index,
            resource_symbol_index=resource_symbol_index,
            hopping_subset=hopping_subset,
            hopping_finer_granularity=hopping_finer_granularity,
        )
        for resource_symbol_index in range(int(srs_symbol_count))
    )


@dataclass(frozen=True, slots=True)
class NRSRSBandwidthConfiguration:
    """Resolved fixed SRS sequence extent for one standardized c-SRS/b-SRS pair."""

    c_srs: int
    b_srs: int
    transmission_comb: int
    bandwidth_rb: int
    subband_count: int
    sequence_length: int
    allocated_bandwidth_subcarriers: int

    def __post_init__(self) -> None:
        values = {
            "c_srs": self.c_srs,
            "b_srs": self.b_srs,
            "transmission_comb": self.transmission_comb,
            "bandwidth_rb": self.bandwidth_rb,
            "subband_count": self.subband_count,
            "sequence_length": self.sequence_length,
            "allocated_bandwidth_subcarriers": self.allocated_bandwidth_subcarriers,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
        c_srs = int(self.c_srs)
        b_srs = int(self.b_srs)
        transmission_comb = int(self.transmission_comb)
        if not 0 <= c_srs < len(_SRS_BANDWIDTH_TABLE):
            raise ValueError("c_srs must be in [0, 64)")
        if not 0 <= b_srs <= 3:
            raise ValueError("b_srs must be in [0, 3]")
        srs_cyclic_shift_count(transmission_comb)
        bandwidth_rb, subband_count = _SRS_BANDWIDTH_TABLE[c_srs][b_srs]
        expected_subcarriers = bandwidth_rb * 12
        expected_sequence_length = expected_subcarriers // transmission_comb
        if self.bandwidth_rb != bandwidth_rb:
            raise ValueError("bandwidth_rb disagrees with the standardized c_srs row")
        if self.subband_count != subband_count:
            raise ValueError("subband_count disagrees with the standardized c_srs/b_srs row")
        if self.allocated_bandwidth_subcarriers != expected_subcarriers:
            raise ValueError("allocated bandwidth disagrees with the standardized c_srs row")
        if self.sequence_length != expected_sequence_length:
            raise ValueError("sequence_length disagrees with the bandwidth and transmission comb")
        for name, value in values.items():
            object.__setattr__(self, name, int(value))


@dataclass(frozen=True, slots=True)
class NRSRSFixedFrequencyAllocation:
    """Resolved SRS allocation for one frequency-hop counter in a grid frame."""

    bandwidth: NRSRSBandwidthConfiguration
    position_indices: tuple[int, ...]
    frequency_domain_position: int
    reference_subcarrier: int
    frequency_domain_shift_rb: int
    comb_offset: int
    start_subcarrier: int
    b_hop: int = 3
    n_srs: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.bandwidth, NRSRSBandwidthConfiguration):
            raise TypeError("bandwidth must be an NRSRSBandwidthConfiguration")
        indices = tuple(self.position_indices)
        level_count = self.bandwidth.b_srs + 1
        if len(indices) != level_count:
            raise ValueError("position_indices must have one entry per bandwidth level")
        for name in (
            "frequency_domain_position",
            "reference_subcarrier",
            "frequency_domain_shift_rb",
            "comb_offset",
            "start_subcarrier",
            "b_hop",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, Integral) or index < 0:
                raise ValueError("position indices must be nonnegative integers")
        expected_indices = nr_srs_frequency_position_indices(
            self.bandwidth.c_srs,
            b_srs=self.bandwidth.b_srs,
            b_hop=int(self.b_hop),
            frequency_domain_position=int(self.frequency_domain_position),
            n_srs=self.n_srs,
        )
        if tuple(indices) != expected_indices:
            raise ValueError("position_indices disagree with the resolved SRS frequency position")
        if self.comb_offset >= self.bandwidth.transmission_comb:
            raise ValueError("comb_offset must be smaller than transmission_comb")
        expected_start = (
            self.reference_subcarrier
            + 12 * self.frequency_domain_shift_rb
            + sum(
                12 * _SRS_BANDWIDTH_TABLE[self.bandwidth.c_srs][level][0] * index
                for level, index in enumerate(indices)
            )
        )
        if self.start_subcarrier != expected_start:
            raise ValueError("start_subcarrier disagrees with the resolved SRS allocation")
        if self.n_srs is not None and (
            isinstance(self.n_srs, bool) or not isinstance(self.n_srs, Integral) or self.n_srs < 0
        ):
            raise ValueError("n_srs must be a nonnegative integer when provided")
        object.__setattr__(self, "position_indices", tuple(int(index) for index in indices))
        for name in (
            "frequency_domain_position",
            "reference_subcarrier",
            "frequency_domain_shift_rb",
            "comb_offset",
            "start_subcarrier",
            "b_hop",
        ):
            object.__setattr__(self, name, int(getattr(self, name)))
        if self.n_srs is not None:
            object.__setattr__(self, "n_srs", int(self.n_srs))


def nr_srs_type1_bandwidth_configuration(
    c_srs: int, *, transmission_comb: int, b_srs: int = 0
) -> NRSRSBandwidthConfiguration:
    """Resolve the TS 38.211 type-1 SRS bandwidth table entry.

    ``b_srs`` selects one of the four standardized bandwidth levels. The
    returned extent does not by itself derive frequency position or resource
    hopping; :func:`nr_srs_fixed_frequency_allocation` resolves one position,
    using an explicit SRS transmission counter when hopping is enabled.
    """
    if isinstance(c_srs, bool) or not isinstance(c_srs, Integral):
        raise TypeError("c_srs must be an integer")
    c_srs = int(c_srs)
    if not 0 <= c_srs < len(_SRS_BANDWIDTH_TABLE):
        raise ValueError("c_srs must be in [0, 64)")
    if isinstance(b_srs, bool) or not isinstance(b_srs, Integral):
        raise TypeError("b_srs must be an integer")
    b_srs = int(b_srs)
    if not 0 <= b_srs <= 3:
        raise ValueError("b_srs must be in [0, 3]")
    srs_cyclic_shift_count(transmission_comb)
    bandwidth_rb, subband_count = _SRS_BANDWIDTH_TABLE[c_srs][b_srs]
    allocated_bandwidth_subcarriers = bandwidth_rb * 12
    if allocated_bandwidth_subcarriers % transmission_comb:
        raise ValueError("standardized SRS bandwidth must contain an integer sequence length")
    return NRSRSBandwidthConfiguration(
        c_srs=c_srs,
        b_srs=b_srs,
        transmission_comb=int(transmission_comb),
        bandwidth_rb=bandwidth_rb,
        subband_count=subband_count,
        sequence_length=allocated_bandwidth_subcarriers // int(transmission_comb),
        allocated_bandwidth_subcarriers=allocated_bandwidth_subcarriers,
    )


def nr_srs_dci_0_3_request_list_index_from_layout(
    payload_bits: Sequence[int | bool],
    *,
    field_widths_bits: Mapping[str, int],
    request_list_entry_count: int,
) -> int | None:
    """Locate and decode the SRS request selector using the ordered 0_3 layout.

    ``field_widths_bits`` must contain every field in the TS 38.212 clause
    7.3.1.1.4 order, including fields absent in a particular configuration as
    zero-width entries. Width derivation from decoded RRC/higher-layer state is
    caller supplied; this function owns the normative ordering, cumulative
    MSB-first offset, and selector-width consistency checks. Extra payload bits
    are accepted only as zero padding after the complete layout.
    """
    if isinstance(request_list_entry_count, bool) or not isinstance(
        request_list_entry_count, Integral
    ):
        raise TypeError("request_list_entry_count must be an integer")
    entry_count = int(request_list_entry_count)
    if not 0 <= entry_count <= 16:
        raise ValueError("request_list_entry_count must be in [0, 16]")
    decoded = nr_dci_0_3_field_values_from_layout(
        payload_bits,
        field_widths_bits=field_widths_bits,
    )
    expected_srs_width = (entry_count - 1).bit_length() if entry_count else 0
    if field_widths_bits["srs_request"] != expected_srs_width:
        raise ValueError(
            "srs_request field width does not match the configured request-list length"
        )
    if entry_count == 0:
        return None
    selector = decoded["srs_request"]
    if selector is None:
        return 0
    if selector >= entry_count:
        raise ValueError("SRS request selector uses an unconfigured reserved codepoint")
    return selector


def nr_srs_dci_0_3_field_widths_from_configuration(
    *,
    cell_set_configs: Sequence[Mapping[str, object]],
    configured_ul_bwp_ids_by_cell: Mapping[str, Sequence[int]],
    active_ul_bwp_id_by_cell: Mapping[str, int],
    ul_bwp_frequency_allocation_by_cell: Mapping[str, Mapping[int, Mapping[str, object]]],
    request_list_entry_count: int,
    srs_offset_list: Sequence[Sequence[int]] | None,
    report_trigger_size: int,
    tdra_field_index_list_entry_count: int,
    frequency_hopping_enabled_by_cell: Mapping[str, bool],
    transform_precoding_enabled_by_cell: Mapping[str, bool],
    beta_offsets_semi_static_by_cell: Mapping[str, bool],
    channel_access_config_entry_count_by_cell: Mapping[str, int],
    harq_process_number_size_bits_by_cell: Mapping[str, int],
    max_schedulable_pusch_count_by_cell: Mapping[str, int],
    redundancy_version_bits_by_cell: Mapping[str, int],
    sri_mode: str,
    tpmi_mode: str,
    antenna_ports_mode: str,
    tx_config_by_cell: Mapping[str, str],
    srs_resource_count_by_cell: Mapping[str, int],
    max_noncodebook_layers_by_cell: Mapping[str, int | None],
    ue_supports_max_mimo_layers_by_cell: Mapping[str, bool],
    dmrs_port_configuration_by_cell: Mapping[str, Sequence[Mapping[str, object]]],
    codebook_srs_antenna_port_counts_by_cell: Mapping[str, Sequence[int]],
    codebook_subset_by_cell: Mapping[str, str | None],
    max_rank_for_codebook_by_cell: Mapping[str, int | None],
    ul_full_power_transmission_by_cell: Mapping[str, str | None],
    harq_ack_codebook_types: Sequence[str],
    dynamic_harq_ack_two_sub_codebooks_by_codebook: Sequence[bool],
    priority_indicator_configured: bool,
    minimum_scheduling_offset_configured: bool,
    selected_cell_set_id: int | None = None,
    scheduled_cells_indicator_value: int | None = None,
    initial_bwp_id: int = 0,
    supports_active_bwp_change: bool = True,
    pdcch_monitoring_adaptation_enabled: bool = False,
    pdcch_skipping_duration_count: int | None = None,
    search_space_group_ids: Sequence[int] | None = None,
    dormancy_configured: bool = False,
    dormancy_group_ids: Sequence[int] | None = None,
    carried_on_primary_cell_within_drx_active_time: bool = False,
    scell_has_multiple_dl_bwps: bool = False,
    ptrs_uplink_configured_by_cell: Mapping[str, bool] | None = None,
    max_pusch_rank_by_cell: Mapping[str, int] | None = None,
    p0_pusch_set_list_configured_by_cell: Mapping[str, bool] | None = None,
    sri_field_present_by_cell: Mapping[str, bool] | None = None,
    olpc_parameter_set_dci_0_1_bits_by_cell: Mapping[str, int] | None = None,
    ue_max_noncodebook_layers_by_cell: Mapping[str, int | None] | None = None,
    measurement_gap_cancellation_configured: bool = False,
) -> dict[str, int]:
    """Derive the DCI 0_3 fixed/cell/BWP/SRS selector widths from decoded state.

    This derives the format identifier, scheduled-cell-set, scheduled-cells,
    BWP-indicator, frequency-domain resource assignment, time-domain assignment,
    frequency-hopping, DMRS sequence initialization, UL-SCH, priority,
    minimum-scheduling-offset, SCell dormancy, PDCCH monitoring adaptation,
    beta-offset,
    channel-access, scheduled-PUSCH TPC, HARQ-process-number, MCS, NDI, RV,
    first and second downlink assignment index, SRS-request, SRS-offset-indicator,
    SRS-resource-indicator, TPMI, antenna-port, and CSI-request widths. Antenna-port
    width is derived from the configured DMRS mapping-type roster, DMRS type,
    enhanced-DMRS state, maximum length, and transform-precoding state. The TDRA
    width uses the configured
    ``tdra-FieldIndexListDCI-0-3`` entry count; the hopping flag is present
    when any cell in the complete scheduled-cell list has frequency hopping
    configured; and the DMRS sequence initialization field is present when
    transform precoding is disabled for any cell in that list. The UL-SCH
    indicator is one bit. Channel-access entry counts are zero when the field
    is not applicable to a scheduled cell and otherwise count that cell's
    decoded access-configuration list entries; the resulting field width is
    the maximum per-cell ``ceil(log2(count))``. Frequency assignment is
    computed from each scheduled cell's active UL BWP and its decoded
    allocation/interlace configuration. Codebook TPMI width is derived from
    SRS antenna-port count, subset, maximum rank, full-power-transmission mode,
    and transform-precoding state for the supported one-, two-, and four-port
    branches, then aggregated by type1a/type2. Every field in the returned
    ordered layout is derived here; callers cannot override a field width.
    Beta-offset width is zero when every scheduled cell uses semi-static beta
    offsets and two bits otherwise. TPC width is two bits per cell in the
    decoded scheduled-cell combination. MCS is five bits per selected cell.
    HARQ-process width is the sum of each selected cell's decoded 0–5-bit
    configuration. NDI width sums the maximum schedulable PUSCH count per
    selected cell; RV width sums that count times the cell's configured 0–2-bit
    RV width. Those per-cell maps cover the full list; multi-PUSCH count is one
    when no multi-PUSCH TDRA list is configured. The first and second DAI widths
    are the maximum widths over the one or two configured HARQ-ACK codebooks,
    with each codebook's mode and
    dynamic sub-codebook state supplied in matching ordered sequences.
    SCell dormancy width depends on its unique configured group roster and is
    zero outside the required primary-cell/DRX-Active-Time/multiple-DL-BWP
    occasion. PDCCH monitoring adaptation width is derived from its configured
    skipping-duration and search-space-group lists.
    Open-loop power-control indication width depends on per-cell P0 set-list
    presence, SRI field presence, and the configured one/two-bit OLPC width.
    Non-codebook SRI width uses configured ``maxMIMO-Layers`` only when the
    UE supports that parameter; otherwise ``ue_max_noncodebook_layers_by_cell``
    must supply the UE's supported maximum for each non-codebook cell.
    The function rejects omitted and extra field names so its result can be
    passed directly to :func:`nr_srs_dci_0_3_request_list_index_from_layout`.
    """
    if not isinstance(cell_set_configs, Sequence) or isinstance(cell_set_configs, (str, bytes)):
        raise TypeError("cell_set_configs must be an ordered sequence")
    # Reuse the cell-set adapter for complete roster and row validation.
    if selected_cell_set_id is None:
        if len(cell_set_configs) != 1:
            raise ValueError("multiple cell sets require selected_cell_set_id")
        selected_config = cell_set_configs[0]
        if not isinstance(selected_config, Mapping):
            raise TypeError("each cell-set config must be a mapping")
        selected_cell_set_id = selected_config.get("set_id")  # type: ignore[assignment]
    selected_configs = [
        config
        for config in cell_set_configs
        if isinstance(config, Mapping) and config.get("set_id") == selected_cell_set_id
    ]
    if len(selected_configs) != 1:
        raise ValueError("selected_cell_set_id must identify exactly one configured cell set")
    full_cell_roster = selected_configs[0]["scheduled_cell_ids"]
    if not isinstance(full_cell_roster, Sequence) or isinstance(full_cell_roster, (str, bytes)):
        raise TypeError("scheduled_cell_ids must be an ordered sequence")
    combos = selected_configs[0].get("scheduled_cell_combos")
    if combos is None:
        combo_count = 1
    elif isinstance(combos, Sequence) and not isinstance(combos, (str, bytes)):
        combo_count = len(combos)
    else:
        raise TypeError("scheduled_cell_combos must be an ordered sequence")
    selected_cell_roster = nr_srs_dci_0_3_scheduled_cell_ids(
        cell_set_configs,
        scheduled_cells_indicator=scheduled_cells_indicator_value,
        default_set_id=selected_cell_set_id,
    )
    antenna_ports_width = _nr_srs_dci_0_3_antenna_ports_width(
        mode=antenna_ports_mode,
        cell_ids=selected_cell_roster,
        transform_precoding_enabled_by_cell=transform_precoding_enabled_by_cell,
        dmrs_port_configuration_by_cell=dmrs_port_configuration_by_cell,
        full_cell_roster=full_cell_roster,
    )
    precoding_width = _nr_srs_dci_0_3_precoding_width(
        mode=tpmi_mode,
        cell_ids=selected_cell_roster,
        tx_config_by_cell=tx_config_by_cell,
        transform_precoding_enabled_by_cell=transform_precoding_enabled_by_cell,
        codebook_srs_antenna_port_counts_by_cell=(codebook_srs_antenna_port_counts_by_cell),
        codebook_subset_by_cell=codebook_subset_by_cell,
        max_rank_for_codebook_by_cell=max_rank_for_codebook_by_cell,
        ul_full_power_transmission_by_cell=ul_full_power_transmission_by_cell,
        full_cell_roster=full_cell_roster,
    )
    sri_width = _nr_srs_dci_0_3_sri_width(
        mode=sri_mode,
        cell_ids=selected_cell_roster,
        tx_config_by_cell=tx_config_by_cell,
        srs_resource_count_by_cell=srs_resource_count_by_cell,
        max_noncodebook_layers_by_cell=max_noncodebook_layers_by_cell,
        ue_supports_max_mimo_layers_by_cell=ue_supports_max_mimo_layers_by_cell,
        ue_max_noncodebook_layers_by_cell=ue_max_noncodebook_layers_by_cell,
        full_cell_roster=full_cell_roster,
    )
    if not isinstance(configured_ul_bwp_ids_by_cell, Mapping):
        raise TypeError("configured_ul_bwp_ids_by_cell must be a mapping")
    if set(configured_ul_bwp_ids_by_cell) != set(full_cell_roster):
        raise ValueError("configured_ul_bwp_ids_by_cell must cover the full scheduled cell list")
    for cell in full_cell_roster:
        ids = configured_ul_bwp_ids_by_cell[cell]
        if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)):
            raise TypeError(f"configured UL BWP IDs for {cell!r} must be an ordered sequence")
    bwp_field_present = any(
        len(configured_ul_bwp_ids_by_cell[cell]) > 1 for cell in full_cell_roster
    )
    bwp_selection = nr_srs_dci_0_3_ul_bwp_selection(
        full_cell_roster,
        configured_ul_bwp_ids_by_cell,
        active_ul_bwp_id_by_cell,
        bwp_indicator=0 if bwp_field_present else None,
        initial_bwp_id=initial_bwp_id,
        supports_active_bwp_change=supports_active_bwp_change,
    )
    frequency_assignment_widths = nr_srs_dci_0_3_frequency_assignment_widths(
        selected_cell_roster,
        configured_ul_bwp_ids_by_cell,
        active_ul_bwp_id_by_cell,
        ul_bwp_frequency_allocation_by_cell,
    )
    if isinstance(tdra_field_index_list_entry_count, bool) or not isinstance(
        tdra_field_index_list_entry_count, Integral
    ):
        raise TypeError("tdra_field_index_list_entry_count must be an integer")
    tdra_entry_count = int(tdra_field_index_list_entry_count)
    if not 1 <= tdra_entry_count <= 4096:
        raise ValueError("tdra_field_index_list_entry_count must be in [1, 4096]")
    if not isinstance(frequency_hopping_enabled_by_cell, Mapping):
        raise TypeError("frequency_hopping_enabled_by_cell must be a mapping")
    if set(frequency_hopping_enabled_by_cell) != set(full_cell_roster):
        raise ValueError(
            "frequency_hopping_enabled_by_cell must cover the full scheduled cell list"
        )
    if any(type(value) is not bool for value in frequency_hopping_enabled_by_cell.values()):
        raise TypeError("frequency-hopping configuration values must be booleans")
    if not isinstance(transform_precoding_enabled_by_cell, Mapping):
        raise TypeError("transform_precoding_enabled_by_cell must be a mapping")
    if set(transform_precoding_enabled_by_cell) != set(full_cell_roster):
        raise ValueError(
            "transform_precoding_enabled_by_cell must cover the full scheduled cell list"
        )
    if any(type(value) is not bool for value in transform_precoding_enabled_by_cell.values()):
        raise TypeError("transform-precoding configuration values must be booleans")
    if ptrs_uplink_configured_by_cell is None and max_pusch_rank_by_cell is None:
        ptrs_dmrs_width_by_cell = dict.fromkeys(full_cell_roster, 0)
    elif not isinstance(ptrs_uplink_configured_by_cell, Mapping) or not isinstance(
        max_pusch_rank_by_cell, Mapping
    ):
        raise TypeError(
            "PTRS configuration and maximum PUSCH rank must both be supplied as mappings"
        )
    else:
        if set(ptrs_uplink_configured_by_cell) != set(full_cell_roster):
            raise ValueError("PTRS configuration must cover the full scheduled cell list")
        if set(max_pusch_rank_by_cell) != set(full_cell_roster):
            raise ValueError("maximum PUSCH rank must cover the full scheduled cell list")
        if any(type(value) is not bool for value in ptrs_uplink_configured_by_cell.values()):
            raise TypeError("PTRS-UplinkConfig states must be booleans")
        max_pusch_rank: dict[str, int] = {}
        for cell, raw_rank in max_pusch_rank_by_cell.items():
            if isinstance(raw_rank, bool) or not isinstance(raw_rank, Integral):
                raise TypeError(f"maximum PUSCH rank for {cell!r} must be an integer")
            rank = int(raw_rank)
            if not 1 <= rank <= 8:
                raise ValueError("maximum PUSCH rank must be in [1, 8]")
            max_pusch_rank[cell] = rank
        ptrs_dmrs_width_by_cell = {
            cell: 2
            if ptrs_uplink_configured_by_cell[cell]
            and not transform_precoding_enabled_by_cell[cell]
            and max_pusch_rank[cell] > 1
            else 0
            for cell in full_cell_roster
        }
    olpc_inputs = (
        p0_pusch_set_list_configured_by_cell,
        sri_field_present_by_cell,
        olpc_parameter_set_dci_0_1_bits_by_cell,
    )
    if all(value is None for value in olpc_inputs):
        olpc_width_by_cell = dict.fromkeys(full_cell_roster, 0)
    elif any(not isinstance(value, Mapping) for value in olpc_inputs):
        raise TypeError("all open-loop power-control configuration maps must be supplied")
    else:
        p0_configured = cast(Mapping[str, bool], p0_pusch_set_list_configured_by_cell)
        sri_present = cast(Mapping[str, bool], sri_field_present_by_cell)
        olpc_bits_config = cast(Mapping[str, int], olpc_parameter_set_dci_0_1_bits_by_cell)
        for name, roster in (
            ("p0_pusch_set_list_configured_by_cell", p0_configured),
            ("sri_field_present_by_cell", sri_present),
            ("olpc_parameter_set_dci_0_1_bits_by_cell", olpc_bits_config),
        ):
            if set(roster) != set(full_cell_roster):
                raise ValueError(f"{name} must cover the full scheduled cell list")
        if any(type(value) is not bool for value in p0_configured.values()):
            raise TypeError("P0-PUSCH set-list configuration values must be booleans")
        if any(type(value) is not bool for value in sri_present.values()):
            raise TypeError("SRI field-presence values must be booleans")
        olpc_bits: dict[str, int] = {}
        for cell, raw_width in olpc_bits_config.items():
            if isinstance(raw_width, bool) or not isinstance(raw_width, Integral):
                raise TypeError(f"OLPC indication width for {cell!r} must be an integer")
            width = int(raw_width)
            if width not in {1, 2}:
                raise ValueError("configured OLPC indication widths must be one or two bits")
            olpc_bits[cell] = width
        olpc_width_by_cell = {
            cell: (0 if not p0_configured[cell] else 1 if sri_present[cell] else olpc_bits[cell])
            for cell in full_cell_roster
        }
    if not isinstance(beta_offsets_semi_static_by_cell, Mapping):
        raise TypeError("beta_offsets_semi_static_by_cell must be a mapping")
    if set(beta_offsets_semi_static_by_cell) != set(full_cell_roster):
        raise ValueError("beta_offsets_semi_static_by_cell must cover the full scheduled cell list")
    if any(type(value) is not bool for value in beta_offsets_semi_static_by_cell.values()):
        raise TypeError("beta-offset configuration values must be booleans")
    if not isinstance(harq_ack_codebook_types, Sequence) or isinstance(
        harq_ack_codebook_types, (str, bytes)
    ):
        raise TypeError("harq_ack_codebook_types must be an ordered sequence")
    if not 1 <= len(harq_ack_codebook_types) <= 2:
        raise ValueError("one or two HARQ-ACK codebook types must be supplied")
    if any(
        not isinstance(value, str) or value not in {"semi_static", "dynamic"}
        for value in harq_ack_codebook_types
    ):
        raise ValueError("HARQ-ACK codebook types must be 'semi_static' or 'dynamic'")
    if not isinstance(dynamic_harq_ack_two_sub_codebooks_by_codebook, Sequence) or isinstance(
        dynamic_harq_ack_two_sub_codebooks_by_codebook, (str, bytes)
    ):
        raise TypeError(
            "dynamic_harq_ack_two_sub_codebooks_by_codebook must be an ordered sequence"
        )
    if len(dynamic_harq_ack_two_sub_codebooks_by_codebook) != len(harq_ack_codebook_types):
        raise ValueError("HARQ-ACK codebook state sequences must have equal lengths")
    if any(type(value) is not bool for value in dynamic_harq_ack_two_sub_codebooks_by_codebook):
        raise TypeError("two-sub-codebook flags must be booleans")
    if any(
        codebook_type == "semi_static" and two_sub_codebooks
        for codebook_type, two_sub_codebooks in zip(
            harq_ack_codebook_types,
            dynamic_harq_ack_two_sub_codebooks_by_codebook,
            strict=True,
        )
    ):
        raise ValueError("semi-static HARQ-ACK cannot configure dynamic sub-codebooks")
    if not isinstance(channel_access_config_entry_count_by_cell, Mapping):
        raise TypeError("channel_access_config_entry_count_by_cell must be a mapping")
    if set(channel_access_config_entry_count_by_cell) != set(full_cell_roster):
        raise ValueError(
            "channel_access_config_entry_count_by_cell must cover the full scheduled cell list"
        )
    channel_access_counts: list[int] = []
    for cell, raw_count in channel_access_config_entry_count_by_cell.items():
        if isinstance(raw_count, bool) or not isinstance(raw_count, Integral):
            raise TypeError(f"channel-access entry count for {cell!r} must be an integer")
        count = int(raw_count)
        if not 0 <= count <= 64:
            raise ValueError("channel-access entry counts must be in [0, 64]")
        channel_access_counts.append(count)
    if not isinstance(harq_process_number_size_bits_by_cell, Mapping):
        raise TypeError("harq_process_number_size_bits_by_cell must be a mapping")
    if set(harq_process_number_size_bits_by_cell) != set(full_cell_roster):
        raise ValueError(
            "harq_process_number_size_bits_by_cell must cover the full scheduled cell list"
        )
    harq_process_widths: dict[str, int] = {}
    for cell, raw_width in harq_process_number_size_bits_by_cell.items():
        if isinstance(raw_width, bool) or not isinstance(raw_width, Integral):
            raise TypeError(f"HARQ-process width for {cell!r} must be an integer")
        width = int(raw_width)
        if not 0 <= width <= 5:
            raise ValueError("HARQ-process widths must be in [0, 5]")
        harq_process_widths[cell] = width
    if not isinstance(max_schedulable_pusch_count_by_cell, Mapping):
        raise TypeError("max_schedulable_pusch_count_by_cell must be a mapping")
    if set(max_schedulable_pusch_count_by_cell) != set(full_cell_roster):
        raise ValueError(
            "max_schedulable_pusch_count_by_cell must cover the full scheduled cell list"
        )
    schedulable_pusch_counts: dict[str, int] = {}
    for cell, raw_count in max_schedulable_pusch_count_by_cell.items():
        if isinstance(raw_count, bool) or not isinstance(raw_count, Integral):
            raise TypeError(f"schedulable PUSCH count for {cell!r} must be an integer")
        count = int(raw_count)
        if not 1 <= count <= 64:
            raise ValueError("schedulable PUSCH counts must be in [1, 64]")
        schedulable_pusch_counts[cell] = count
    if not isinstance(redundancy_version_bits_by_cell, Mapping):
        raise TypeError("redundancy_version_bits_by_cell must be a mapping")
    if set(redundancy_version_bits_by_cell) != set(full_cell_roster):
        raise ValueError("redundancy_version_bits_by_cell must cover the full scheduled cell list")
    redundancy_widths: dict[str, int] = {}
    for cell, raw_width in redundancy_version_bits_by_cell.items():
        if isinstance(raw_width, bool) or not isinstance(raw_width, Integral):
            raise TypeError(f"redundancy-version width for {cell!r} must be an integer")
        width = int(raw_width)
        if not 0 <= width <= 2:
            raise ValueError("redundancy-version widths must be in [0, 2]")
        redundancy_widths[cell] = width
    if type(priority_indicator_configured) is not bool:
        raise TypeError("priority_indicator_configured must be a boolean")
    if type(minimum_scheduling_offset_configured) is not bool:
        raise TypeError("minimum_scheduling_offset_configured must be a boolean")
    if type(measurement_gap_cancellation_configured) is not bool:
        raise TypeError("measurement_gap_cancellation_configured must be a boolean")
    if isinstance(request_list_entry_count, bool) or not isinstance(
        request_list_entry_count, Integral
    ):
        raise TypeError("request_list_entry_count must be an integer")
    entry_count = int(request_list_entry_count)
    if not 0 <= entry_count <= 16:
        raise ValueError("request_list_entry_count must be in [0, 16]")
    if srs_offset_list is None:
        offset_count = 0
    else:
        if not isinstance(srs_offset_list, Sequence) or isinstance(srs_offset_list, (str, bytes)):
            raise TypeError("srs_offset_list must be an ordered sequence or None")
        offset_count = len(srs_offset_list)
        if not 1 <= offset_count <= 8:
            raise ValueError("srs_offset_list must contain 1..8 configured entries")
        for row_index, row in enumerate(srs_offset_list):
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                raise TypeError(f"srs_offset_list[{row_index}] must be an ordered sequence")
            if len(row) != len(full_cell_roster):
                raise ValueError(
                    f"srs_offset_list[{row_index}] must contain one index per scheduled cell"
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or not 0 <= int(value) <= 7
                for value in row
            ):
                raise ValueError("SRS offset-list indexes must be integers in [0, 7]")
    if isinstance(report_trigger_size, bool) or not isinstance(report_trigger_size, Integral):
        raise TypeError("report_trigger_size must be an integer")
    csi_width = int(report_trigger_size)
    if not 0 <= csi_width <= 6:
        raise ValueError("report_trigger_size must be in [0, 6]")
    cell_set_count = len(cell_set_configs)
    return {
        "identifier_for_dci_formats": 1,
        "scheduled_cell_set_indicator": (cell_set_count - 1).bit_length(),
        "scheduled_cells_indicator": (combo_count - 1).bit_length(),
        "bandwidth_part_indicator": bwp_selection.indicator_bit_width,
        "frequency_domain_resource_assignment": sum(
            frequency_assignment_widths[cell] for cell in selected_cell_roster
        ),
        "time_domain_resource_assignment": (tdra_entry_count - 1).bit_length(),
        "frequency_hopping_flag": int(any(frequency_hopping_enabled_by_cell.values())),
        "dmrs_sequence_initialization": int(not all(transform_precoding_enabled_by_cell.values())),
        "ul_sch_indicator": 1,
        "priority_indicator": int(priority_indicator_configured),
        "minimum_applicable_scheduling_offset_indicator": int(minimum_scheduling_offset_configured),
        "beta_offset_indicator": 0 if all(beta_offsets_semi_static_by_cell.values()) else 2,
        "tpc_command_for_scheduled_pusch": 2 * len(selected_cell_roster),
        "harq_process_number": sum(harq_process_widths[cell] for cell in selected_cell_roster),
        "modulation_and_coding_scheme": 5 * len(selected_cell_roster),
        "new_data_indicator": sum(schedulable_pusch_counts[cell] for cell in selected_cell_roster),
        "redundancy_version": sum(
            schedulable_pusch_counts[cell] * redundancy_widths[cell]
            for cell in selected_cell_roster
        ),
        "first_downlink_assignment_index": max(
            1 if codebook_type == "semi_static" else 2 for codebook_type in harq_ack_codebook_types
        ),
        "second_downlink_assignment_index": max(
            2 if codebook_type == "dynamic" and two_sub_codebooks else 0
            for codebook_type, two_sub_codebooks in zip(
                harq_ack_codebook_types,
                dynamic_harq_ack_two_sub_codebooks_by_codebook,
                strict=True,
            )
        ),
        "channel_access_cpext_capc": max(
            ((count - 1).bit_length() if count else 0 for count in channel_access_counts),
            default=0,
        ),
        "srs_request": (entry_count - 1).bit_length() if entry_count else 0,
        "srs_offset_indicator": (offset_count - 1).bit_length() if offset_count else 0,
        "srs_resource_indicator": sri_width,
        "precoding_information_and_number_of_layers": precoding_width,
        "antenna_ports": antenna_ports_width,
        "csi_request": csi_width,
        "ptrs_dmrs_association": sum(
            ptrs_dmrs_width_by_cell[cell] for cell in selected_cell_roster
        ),
        "open_loop_power_control_parameter_set_indication": max(
            (olpc_width_by_cell[cell] for cell in selected_cell_roster),
            default=0,
        ),
        "scell_dormancy_indication": _nr_srs_dci_0_3_scell_dormancy_width(
            dormancy_configured=dormancy_configured,
            dormancy_group_ids=dormancy_group_ids,
            carried_on_primary_cell_within_drx_active_time=(
                carried_on_primary_cell_within_drx_active_time
            ),
            scell_has_multiple_dl_bwps=scell_has_multiple_dl_bwps,
        ),
        "pdcch_monitoring_adaptation_indication": (
            _nr_srs_dci_0_3_pdcch_monitoring_adaptation_width(
                enabled=pdcch_monitoring_adaptation_enabled,
                skipping_duration_count=pdcch_skipping_duration_count,
                search_space_group_ids=search_space_group_ids,
            )
        ),
        "measurement_gap_cancellation": int(measurement_gap_cancellation_configured),
    }


def nr_srs_multicell_request_resource_set_ids(
    request_list_index: int,
    request_list: Sequence[Sequence[int]],
    serving_cell_ids: Sequence[str],
    resource_sets_by_cell: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    scheduled_cell_ids: Sequence[str] | None = None,
    supplementary_uplink_cells: Sequence[str] = (),
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Resolve a decoded DCI 0_3/1_3 request-list entry for each scheduled cell.

    ``request_list`` is the already-decoded ``srs-RequestListDCI-0-3`` or
    ``srs-RequestListDCI-1-3`` roster. Each entry contains one per-cell SRS
    request codepoint, ordered by ascending serving-cell index as specified by
    TS 38.212. ``serving_cell_ids`` must use that same order. Codepoints use
    the per-cell two-bit request, with the leading SUL selector included when
    that cell is listed in ``supplementary_uplink_cells``. The returned tuple
    preserves serving-cell order and contains only matching aperiodic set IDs.
    When ``scheduled_cell_ids`` is supplied, request entries for configured but
    unscheduled cells are returned as empty and are not interpreted.

    This resolves the decoded request-list indirection only; it does not parse
    DCI bits, decode ASN.1 RRC, resolve scheduled-cell combinations, or handle
    typeA antenna-switching semantics.
    """
    if isinstance(request_list_index, bool) or not isinstance(request_list_index, Integral):
        raise TypeError("request_list_index must be an integer")
    if not isinstance(request_list, Sequence) or isinstance(request_list, (str, bytes)):
        raise TypeError("request_list must be a sequence of per-cell request entries")
    if not request_list:
        raise ValueError("request_list must contain at least one configured entry")
    request_list_index = int(request_list_index)
    if not 0 <= request_list_index < len(request_list):
        raise ValueError("request_list_index does not select a configured entry")
    if not isinstance(serving_cell_ids, Sequence) or isinstance(serving_cell_ids, (str, bytes)):
        raise TypeError("serving_cell_ids must be an ordered sequence")
    cell_ids = tuple(serving_cell_ids)
    if not cell_ids or any(not isinstance(cell, str) or not cell for cell in cell_ids):
        raise ValueError("serving_cell_ids must contain nonempty cell identifiers")
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("serving_cell_ids must be unique")
    if scheduled_cell_ids is None:
        scheduled_cells = cell_ids
    else:
        if not isinstance(scheduled_cell_ids, Sequence) or isinstance(
            scheduled_cell_ids, (str, bytes)
        ):
            raise TypeError("scheduled_cell_ids must be an ordered sequence")
        scheduled_cells = tuple(scheduled_cell_ids)
        if not scheduled_cells or any(
            not isinstance(cell, str) or not cell for cell in scheduled_cells
        ):
            raise ValueError("scheduled_cell_ids must contain nonempty cell IDs")
        if len(set(scheduled_cells)) != len(scheduled_cells) or not set(scheduled_cells).issubset(
            cell_ids
        ):
            raise ValueError("scheduled_cell_ids must be a unique subset of serving_cell_ids")
        if tuple(cell for cell in cell_ids if cell in set(scheduled_cells)) != scheduled_cells:
            raise ValueError("scheduled_cell_ids must follow serving-cell order")
    if not isinstance(resource_sets_by_cell, Mapping):
        raise TypeError("resource_sets_by_cell must be a mapping")
    if set(resource_sets_by_cell) != set(cell_ids):
        raise ValueError("resource_sets_by_cell keys must exactly match serving_cell_ids")
    if not isinstance(supplementary_uplink_cells, Sequence) or isinstance(
        supplementary_uplink_cells, (str, bytes)
    ):
        raise TypeError("supplementary_uplink_cells must be a sequence")
    sul_cells = tuple(supplementary_uplink_cells)
    if any(not isinstance(cell, str) for cell in sul_cells):
        raise TypeError("supplementary_uplink_cells entries must be strings")
    if len(set(sul_cells)) != len(sul_cells) or not set(sul_cells).issubset(cell_ids):
        raise ValueError("supplementary_uplink_cells must be unique scheduled cell IDs")

    selected_entry = request_list[request_list_index]
    if not isinstance(selected_entry, Sequence) or isinstance(selected_entry, (str, bytes)):
        raise TypeError("selected request-list entry must be a per-cell sequence")
    if len(selected_entry) != len(cell_ids):
        raise ValueError("selected request-list entry must contain one codepoint per cell")
    scheduled_set = set(scheduled_cells)

    return tuple(
        (
            cell_id,
            (
                nr_srs_triggered_resource_set_ids(
                    codepoint,
                    resource_sets_by_cell[cell_id],
                    dci_format="0_1",
                    supplementary_uplink=cell_id in sul_cells,
                )
                if cell_id in scheduled_set
                else ()
            ),
        )
        for cell_id, codepoint in zip(cell_ids, selected_entry, strict=True)
    )


def nr_srs_sequence_type1(
    sequence_length: int,
    *,
    group_index: int,
    base_sequence_index: int = 0,
    cyclic_shift_index: int = 0,
    cyclic_shift_hopping_offset: int = 0,
    hopping_finer_granularity: bool = False,
    transmission_comb: int,
) -> np.ndarray:
    """Generate a bounded SRS type-1 sequence with comb-correct cyclic shift.

    Group/base-sequence selection and hopping remain caller-owned. The comb
    determines the TS 38.211 cyclic-shift count through
    :func:`srs_cyclic_shift_count`.
    """
    return nr_low_papr_sequence_type1(
        sequence_length,
        group_index=group_index,
        base_sequence_index=base_sequence_index,
        cyclic_shift_index=cyclic_shift_index,
        cyclic_shift_count=srs_cyclic_shift_count(transmission_comb),
        cyclic_shift_hopping_offset=cyclic_shift_hopping_offset,
        hopping_finer_granularity=hopping_finer_granularity,
    )


def nr_srs_hopped_sequence_type1(
    sequence_length: int,
    *,
    sequence_id: int,
    slot_index: int,
    start_symbol_index: int,
    resource_symbol_index: int,
    hopping_mode: str,
    transmission_comb: int,
    cyclic_shift_index: int = 0,
    cyclic_shift_hopping_offset: int = 0,
    hopping_finer_granularity: bool = False,
    slot_symbol_count: int = 14,
) -> np.ndarray:
    """Generate a type-1 SRS sequence using the configured hopping mode.

    ``slot_index`` is frame-local, ``start_symbol_index`` is the absolute slot
    symbol ``l0``, and ``resource_symbol_index`` is the resource-local symbol
    ``l'``. Sequence hopping applies only where the configured sequence-length
    eligibility threshold is met.
    """
    (
        sequence_id,
        slot_index,
        start_symbol_index,
        resource_symbol_index,
        slot_symbol_count,
    ) = _validate_srs_hopping_position(
        sequence_id,
        slot_index,
        start_symbol_index,
        resource_symbol_index,
        slot_symbol_count,
    )
    if hopping_mode == "neither":
        group_index = sequence_id % 30
        base_sequence_index = 0
    elif hopping_mode == "groupHopping":
        group_index = nr_srs_group_hopping_index(
            sequence_id,
            slot_index,
            start_symbol_index,
            resource_symbol_index,
            slot_symbol_count=slot_symbol_count,
        )
        base_sequence_index = 0
    elif hopping_mode == "sequenceHopping":
        group_index = sequence_id % 30
        base_sequence_index = nr_srs_sequence_hopping_index(
            sequence_id,
            slot_index,
            start_symbol_index,
            resource_symbol_index,
            sequence_length,
            slot_symbol_count=slot_symbol_count,
        )
    else:
        raise ValueError("hopping_mode must be neither, groupHopping, or sequenceHopping")
    return nr_srs_sequence_type1(
        sequence_length,
        group_index=group_index,
        base_sequence_index=base_sequence_index,
        cyclic_shift_index=cyclic_shift_index,
        cyclic_shift_hopping_offset=cyclic_shift_hopping_offset,
        hopping_finer_granularity=hopping_finer_granularity,
        transmission_comb=transmission_comb,
    )


def map_nr_srs_positioning_resource(
    sequence: np.ndarray,
    *,
    slot_symbol_count: int,
    subcarrier_count: int,
    start_position_offset: int,
    srs_symbol_count: int,
    comb_size: int,
    comb_offset: int,
    frequency_start_subcarrier: int,
    allocated_bandwidth_subcarriers: int,
    beta_srs: float = 1.0,
) -> NRSRSPositioningResourceGrid:
    """Map a single-port SRS-PosResource with the table offset per symbol.

    The sequence is one row reused or one row per resource symbol. The
    allocation start and width must already be resolved. Each resource-local
    symbol uses TS 38.211 Table 6.4.1.4.3-2 before applying the transmission
    comb, so the occupied subcarrier roster can vary by symbol.
    """
    integer_values = {
        "slot_symbol_count": slot_symbol_count,
        "subcarrier_count": subcarrier_count,
        "start_position_offset": start_position_offset,
        "srs_symbol_count": srs_symbol_count,
        "comb_size": comb_size,
        "comb_offset": comb_offset,
        "frequency_start_subcarrier": frequency_start_subcarrier,
        "allocated_bandwidth_subcarriers": allocated_bandwidth_subcarriers,
    }
    for name, value in integer_values.items():
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    slot_symbol_count = int(slot_symbol_count)
    subcarrier_count = int(subcarrier_count)
    start_position_offset = int(start_position_offset)
    srs_symbol_count = int(srs_symbol_count)
    comb_size = int(comb_size)
    comb_offset = int(comb_offset)
    frequency_start_subcarrier = int(frequency_start_subcarrier)
    allocated_bandwidth_subcarriers = int(allocated_bandwidth_subcarriers)
    if slot_symbol_count not in {12, 14}:
        raise ValueError("slot_symbol_count must be 12 or 14")
    if subcarrier_count < 1:
        raise ValueError("subcarrier_count must be positive")
    if not 1 <= srs_symbol_count <= slot_symbol_count:
        raise ValueError("symbol count is unsupported for this SRS-PosResource comb")
    offsets = tuple(
        nr_srs_positioning_symbol_offset(
            comb_size,
            srs_symbol_count=srs_symbol_count,
            resource_symbol_index=symbol_index,
        )
        for symbol_index in range(srs_symbol_count)
    )
    if start_position_offset < srs_symbol_count - 1 or start_position_offset >= slot_symbol_count:
        raise ValueError("start_position_offset must fit the SRS symbols in the slot")
    if not 0 <= comb_offset < comb_size:
        raise ValueError("comb_offset must be in [0, comb_size)")
    if frequency_start_subcarrier < 0:
        raise ValueError("frequency_start_subcarrier must be nonnegative")
    if allocated_bandwidth_subcarriers < 1 or (
        allocated_bandwidth_subcarriers % comb_size
        or frequency_start_subcarrier + allocated_bandwidth_subcarriers > subcarrier_count
    ):
        raise ValueError("allocated bandwidth must be positive, comb-aligned, and fit the grid")
    if (
        isinstance(beta_srs, bool)
        or not isinstance(beta_srs, Real)
        or not np.isfinite(float(beta_srs))
        or beta_srs <= 0
    ):
        raise ValueError("beta_srs must be a finite positive real number")

    values = np.asarray(sequence)
    if not np.iscomplexobj(values):
        raise TypeError("sequence must be complex-valued")
    if values.ndim not in {1, 2}:
        raise ValueError("sequence must have shape [occupied REs] or [SRS symbols, occupied REs]")
    expected_length = allocated_bandwidth_subcarriers // comb_size
    if values.shape[-1] != expected_length:
        raise ValueError("sequence length does not match the positioning SRS comb")
    if values.ndim == 2 and values.shape[0] != srs_symbol_count:
        raise ValueError("per-symbol SRS sequence must include exactly srs_symbol_count rows")
    if not np.isfinite(values).all():
        raise ValueError("sequence must contain only finite values")

    first_symbol = slot_symbol_count - 1 - start_position_offset
    symbol_indices = tuple(range(first_symbol, first_symbol + srs_symbol_count))
    subcarriers_by_symbol = tuple(
        tuple(
            frequency_start_subcarrier + (comb_offset + offset) % comb_size + comb_size * index
            for index in range(expected_length)
        )
        for offset in offsets
    )
    grid = np.zeros((slot_symbol_count, subcarrier_count), dtype=np.complex128)
    mask = np.zeros((slot_symbol_count, subcarrier_count), dtype=np.bool_)
    rows = (
        values if values.ndim == 2 else np.broadcast_to(values, (srs_symbol_count, expected_length))
    )
    for row_index, (symbol, subcarriers) in enumerate(
        zip(symbol_indices, subcarriers_by_symbol, strict=True)
    ):
        grid[symbol, list(subcarriers)] = beta_srs * rows[row_index]
        mask[symbol, list(subcarriers)] = True
    return NRSRSPositioningResourceGrid(
        grid=grid,
        mask=mask,
        symbol_indices=symbol_indices,
        subcarrier_indices_by_symbol=subcarriers_by_symbol,
        positioning_offsets=offsets,
    )


def map_nr_srs_positioning_multiport_resource(
    sequences: np.ndarray,
    *,
    slot_symbol_count: int,
    subcarrier_count: int,
    start_position_offset: int,
    srs_symbol_count: int,
    comb_size: int,
    comb_offset: int,
    frequency_start_subcarrier: int,
    allocated_bandwidth_subcarriers: int,
    beta_srs: float = 1.0,
    four_port_srs_3tx: bool = False,
    ports8_tdm: bool = False,
) -> NRSRSPositioningMultiportResourceGrid:
    """Map caller-scheduled positioning-SRS rows with TS per-port scaling."""
    values = np.asarray(sequences)
    if not np.iscomplexobj(values):
        raise TypeError("sequences must be complex-valued")
    if values.ndim != 3 or values.shape[0] < 1:
        raise ValueError("sequences must have shape [ports, SRS symbols, occupied REs]")
    if values.shape[1] != srs_symbol_count:
        raise ValueError("each port must have exactly srs_symbol_count rows")
    if not isinstance(ports8_tdm, bool):
        raise TypeError("ports8_tdm must be a boolean")
    if ports8_tdm and (values.shape[0] != 8 or four_port_srs_3tx):
        raise ValueError("ports8_tdm requires eight ports and cannot combine with four-port 3Tx")
    normalized_beta = _normalized_multiport_beta(beta_srs, values.shape[0], four_port_srs_3tx)
    mapped = [
        map_nr_srs_positioning_resource(
            port_sequences,
            slot_symbol_count=slot_symbol_count,
            subcarrier_count=subcarrier_count,
            start_position_offset=start_position_offset,
            srs_symbol_count=srs_symbol_count,
            comb_size=comb_size,
            comb_offset=comb_offset,
            frequency_start_subcarrier=frequency_start_subcarrier,
            allocated_bandwidth_subcarriers=allocated_bandwidth_subcarriers,
            beta_srs=normalized_beta,
        )
        for port_sequences in values
    ]
    grid = np.stack([item.grid for item in mapped], axis=0)
    mask = np.stack([item.mask for item in mapped], axis=0)
    activity = (
        nr_srs_port_symbol_activity(
            tuple(range(1000, 1008)),
            srs_symbol_count=srs_symbol_count,
            ports8_tdm=True,
        )
        if ports8_tdm
        else None
    )
    if activity is not None:
        for port_index, active_symbols in enumerate(activity):
            for symbol_index, is_active in zip(
                mapped[0].symbol_indices, active_symbols, strict=True
            ):
                if not is_active:
                    grid[port_index, symbol_index, :] = 0
                    mask[port_index, symbol_index, :] = False
    return NRSRSPositioningMultiportResourceGrid(
        grid=grid,
        mask=mask,
        symbol_indices=mapped[0].symbol_indices,
        subcarrier_indices_by_symbol=mapped[0].subcarrier_indices_by_symbol,
        positioning_offsets=mapped[0].positioning_offsets,
        active_port_symbols=activity,
    )


def map_nr_srs_multiport_resource(
    sequences: np.ndarray,
    *,
    slot_symbol_count: int,
    subcarrier_count: int,
    start_position_offset: int,
    srs_symbol_count: int,
    comb_size: int,
    comb_offset: int,
    frequency_start_subcarrier: int = 0,
    allocated_bandwidth_subcarriers: int | None = None,
    beta_srs: float = 1.0,
    four_port_srs_3tx: bool = False,
    ports8_tdm: bool = False,
) -> NRSRSMultiportResourceGrid:
    """Map per-port SRS rows with TS 38.211 amplitude normalization."""
    values = np.asarray(sequences)
    if not np.iscomplexobj(values):
        raise TypeError("sequences must be complex-valued")
    if values.ndim != 3 or values.shape[0] < 1:
        raise ValueError("sequences must have shape [ports, SRS symbols, occupied REs]")
    if values.shape[1] != srs_symbol_count:
        raise ValueError("each port must have exactly srs_symbol_count rows")
    if not isinstance(ports8_tdm, bool):
        raise TypeError("ports8_tdm must be a boolean")
    if ports8_tdm and (values.shape[0] != 8 or four_port_srs_3tx):
        raise ValueError("ports8_tdm requires eight ports and cannot combine with four-port 3Tx")
    normalized_beta = _normalized_multiport_beta(beta_srs, values.shape[0], four_port_srs_3tx)
    mapped = [
        map_nr_srs_resource(
            port_sequences,
            slot_symbol_count=slot_symbol_count,
            subcarrier_count=subcarrier_count,
            start_position_offset=start_position_offset,
            srs_symbol_count=srs_symbol_count,
            comb_size=comb_size,
            comb_offset=comb_offset,
            frequency_start_subcarrier=frequency_start_subcarrier,
            allocated_bandwidth_subcarriers=allocated_bandwidth_subcarriers,
            beta_srs=normalized_beta,
        )
        for port_sequences in values
    ]
    grid = np.stack([item.grid for item in mapped], axis=0)
    activity = (
        nr_srs_port_symbol_activity(
            tuple(range(1000, 1008)),
            srs_symbol_count=srs_symbol_count,
            ports8_tdm=True,
        )
        if ports8_tdm
        else None
    )
    mask = np.stack([item.mask for item in mapped], axis=0)
    if activity is not None:
        for port_index, active_symbols in enumerate(activity):
            for symbol_index, is_active in zip(
                mapped[0].symbol_indices, active_symbols, strict=True
            ):
                if not is_active:
                    grid[port_index, symbol_index, :] = 0
                    mask[port_index, symbol_index, :] = False
    return NRSRSMultiportResourceGrid(
        grid=grid,
        mask=mask,
        symbol_indices=mapped[0].symbol_indices,
        subcarrier_indices=mapped[0].subcarrier_indices,
        active_port_symbols=activity,
    )
