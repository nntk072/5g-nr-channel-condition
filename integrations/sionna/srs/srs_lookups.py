"""Lookups that derive SRS parameters from constants."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral, Real

import numpy as np

from .srs_constants import (
    _DCI_0_3_CONFIG_DERIVED_FIELDS,
    _DCI_0_3_FIELD_ORDER,
    _LOW_PAPR_SHORT_PHASES,
    _SRS_BANDWIDTH_TABLE,
    _SRS_COMB_SIZES,
    _SRS_CYCLIC_SHIFT_COUNTS,
    _SRS_PERIODICITY_EXT_OFFSETS,
    _SRS_PERIODICITY_OFFSETS,
    _SRS_POSITIONING_SYMBOL_OFFSETS,
    _SRS_REQUEST_DCI_FORMATS,
    _SRS_SYMBOL_COUNTS,
    NRSRSBWPSelection,
    NRSRSHoppedMultiportResourceGrid,
    NRSRSHoppedResourceGrid,
    NRSRSMultiportChannelEstimate,
    NRSRSMultiportResourceGrid,
    NRSRSPortConfiguration,
    NRSRSPositioningMultiportResourceGrid,
    NRSRSPositioningResourceGrid,
    NRSRSResourceGrid,
    NRSRSSinglePortChannelEstimate,
    _largest_prime_below,
    _nr_gold_bits,
    _srs_gold_bit_offset,
    _validate_srs_hopping_position,
)

if set(_DCI_0_3_FIELD_ORDER) != _DCI_0_3_CONFIG_DERIVED_FIELDS:
    raise RuntimeError("every DCI 0_3 field must have a configuration-derived width")


def srs_cyclic_shift_count(transmission_comb: int) -> int:
    """Return TS 38.211's maximum SRS cyclic shifts for a transmission comb."""
    if isinstance(transmission_comb, bool) or not isinstance(transmission_comb, Integral):
        raise TypeError("transmission_comb must be an integer")
    try:
        return _SRS_CYCLIC_SHIFT_COUNTS[int(transmission_comb)]
    except KeyError as error:
        raise ValueError("transmission_comb must be 2, 4, or 8") from error


def nr_srs_port_configuration_from_rrc(
    nrof_srs_ports: str | None,
    *,
    nrof_srs_ports_n8: str | None = None,
    cyclic_shift_hopping_enabled: bool = False,
    comb_offset_hopping_enabled: bool = False,
) -> NRSRSPortConfiguration:
    """Resolve Release 19 decoded SRS port enums and ports8tdm constraints."""
    if not isinstance(cyclic_shift_hopping_enabled, bool) or not isinstance(
        comb_offset_hopping_enabled, bool
    ):
        raise TypeError("RRC hopping-presence flags must be booleans")
    if nrof_srs_ports_n8 is not None:
        if not isinstance(nrof_srs_ports_n8, str):
            raise TypeError("nrofSRS-Ports-n8 must be a string enum")
        if nrof_srs_ports_n8 not in {"ports8", "ports8tdm"}:
            raise ValueError("nrofSRS-Ports-n8 must be ports8 or ports8tdm")
        ports8_tdm = nrof_srs_ports_n8 == "ports8tdm"
        if ports8_tdm and (cyclic_shift_hopping_enabled or comb_offset_hopping_enabled):
            raise ValueError(
                "ports8tdm cannot be combined with cyclic-shift or comb-offset hopping"
            )
        return NRSRSPortConfiguration(tuple(range(1000, 1008)), ports8_tdm)
    if nrof_srs_ports is None:
        nrof_srs_ports = "port1"
    if not isinstance(nrof_srs_ports, str):
        raise TypeError("nrofSRS-Ports must be a string enum")
    counts = {"port1": 1, "ports2": 2, "ports4": 4}
    try:
        port_count = counts[nrof_srs_ports]
    except (KeyError, TypeError) as error:
        raise ValueError("nrofSRS-Ports must be port1, ports2, or ports4") from error
    return NRSRSPortConfiguration(tuple(range(1000, 1000 + port_count)), False)


def nr_srs_port_symbol_activity(
    antenna_ports: tuple[int, ...],
    *,
    srs_symbol_count: int,
    ports8_tdm: bool = False,
) -> tuple[tuple[bool, ...], ...]:
    """Return TS 38.211 activity by antenna port and resource-local symbol."""
    if not isinstance(antenna_ports, tuple) or not antenna_ports:
        raise TypeError("antenna_ports must be a nonempty tuple of integers")
    if isinstance(srs_symbol_count, bool) or not isinstance(srs_symbol_count, Integral):
        raise TypeError("srs_symbol_count must be an integer")
    if srs_symbol_count not in _SRS_SYMBOL_COUNTS:
        raise ValueError("srs_symbol_count is unsupported")
    if not isinstance(ports8_tdm, bool):
        raise TypeError("ports8_tdm must be a boolean")
    if any(isinstance(port, bool) or not isinstance(port, Integral) for port in antenna_ports):
        raise TypeError("antenna_ports must contain integers")
    ports = tuple(int(port) for port in antenna_ports)
    if len(set(ports)) != len(ports) or any(port not in range(1000, 1008) for port in ports):
        raise ValueError("antenna_ports must be unique SRS ports in [1000, 1008)")
    if ports8_tdm and ports != tuple(range(1000, 1008)):
        raise ValueError("ports8_tdm requires the complete ordered port roster 1000 through 1007")
    if not ports8_tdm:
        return tuple((True,) * int(srs_symbol_count) for _ in ports)
    even_symbol_ports = {1000, 1001, 1004, 1005}
    return tuple(
        tuple(
            (symbol_index % 2 == 0) == (port in even_symbol_ports)
            for symbol_index in range(int(srs_symbol_count))
        )
        for port in ports
    )


def nr_srs_positioning_symbol_offset(
    transmission_comb: int,
    *,
    srs_symbol_count: int,
    resource_symbol_index: int,
) -> int:
    """Resolve ``k'_offset`` for one SRS-PosResource symbol.

    TS 38.211 Table 6.4.1.4.3-2 applies this offset to SRS-PosResource. An
    ordinary SRS-Resource uses zero and should not call this helper. Symbol
    counts unsupported by the table are rejected rather than approximated.
    """
    for name, value in (
        ("transmission_comb", transmission_comb),
        ("srs_symbol_count", srs_symbol_count),
        ("resource_symbol_index", resource_symbol_index),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    transmission_comb = int(transmission_comb)
    srs_symbol_count = int(srs_symbol_count)
    resource_symbol_index = int(resource_symbol_index)
    if transmission_comb not in _SRS_POSITIONING_SYMBOL_OFFSETS:
        raise ValueError("transmission_comb must be 2, 4, or 8")
    offsets = _SRS_POSITIONING_SYMBOL_OFFSETS[transmission_comb].get(srs_symbol_count)
    if offsets is None:
        raise ValueError("symbol count is unsupported for this SRS-PosResource comb")
    if not 0 <= resource_symbol_index < srs_symbol_count:
        raise ValueError("resource_symbol_index must be inside the SRS resource")
    return offsets[resource_symbol_index]


def nr_srs_frequency_position_indices(
    c_srs: int,
    *,
    b_srs: int,
    b_hop: int,
    frequency_domain_position: int = 0,
    n_srs: int | None = None,
) -> tuple[int, ...]:
    """Resolve type-1 SRS frequency-tree indices for one transmission.

    ``n_srs`` is the TS 38.211 transmission counter. Periodicity and
    higher-layer scheduling are intentionally not inferred by this primitive.
    """
    for name, value in (
        ("c_srs", c_srs),
        ("b_srs", b_srs),
        ("b_hop", b_hop),
        ("frequency_domain_position", frequency_domain_position),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    c_srs = int(c_srs)
    b_srs = int(b_srs)
    b_hop = int(b_hop)
    frequency_domain_position = int(frequency_domain_position)
    if not 0 <= c_srs < len(_SRS_BANDWIDTH_TABLE):
        raise ValueError("c_srs must be in [0, 64)")
    if not 0 <= b_srs <= 3:
        raise ValueError("b_srs must be in [0, 3]")
    if not 0 <= b_hop <= 3:
        raise ValueError("b_hop must be in [0, 3]")
    if frequency_domain_position < 0:
        raise ValueError("frequency_domain_position must be nonnegative")
    if n_srs is not None:
        if isinstance(n_srs, bool) or not isinstance(n_srs, Integral):
            raise TypeError("n_srs must be an integer")
        n_srs = int(n_srs)
        if n_srs < 0:
            raise ValueError("n_srs must be nonnegative")
    hopping = b_hop < b_srs
    if hopping and n_srs is None:
        raise ValueError("n_srs is required when SRS frequency hopping is enabled")

    table_row = _SRS_BANDWIDTH_TABLE[c_srs]
    radices = [table_row[level][1] for level in range(b_srs + 1)]
    if hopping:
        # TS 38.211 fixes N_(b_hop)=1 for the hopping recurrence.
        radices[b_hop] = 1
    indices: list[int] = []
    product_previous = 1
    for level in range(b_srs + 1):
        bandwidth_rb = table_row[level][0]
        radix = radices[level]
        base_position = (4 * frequency_domain_position // bandwidth_rb) % radix
        if not hopping or level <= b_hop:
            indices.append(base_position)
        else:
            assert n_srs is not None
            product_through_level = product_previous * radix
            if radix % 2 == 0:
                hop_position = (radix // 2) * (
                    (n_srs % product_through_level) // product_previous
                ) + (n_srs % product_through_level) // (2 * product_previous)
            else:
                hop_position = (radix // 2) * (n_srs // product_previous)
            indices.append((base_position + hop_position) % radix)
        product_previous *= radix
    return tuple(indices)


def nr_srs_transmission_counter(
    *,
    resource_type: str,
    srs_symbol_count: int,
    resource_symbol_index: int,
    repetition_factor: int | None = None,
    ports8_tdm: bool = False,
    absolute_slot_index: int | None = None,
    periodicity_slots: int | None = None,
    slot_offset: int | None = None,
    positioning_resource: bool = False,
    numerology: int | None = None,
    hyper_frame_number: int | None = None,
    hyper_sfn_index: int | None = None,
) -> int | None:
    """Resolve the TS 38.211 ``n_SRS`` counter from resolved scheduling data.

    ``absolute_slot_index`` is the linear slot number within the radio-frame
    sequence, while periodicity and offset are already resolved in slots by
    the caller. ``None`` means the periodic/semi-persistent resource does not
    transmit in the supplied slot. For periodic positioning SRS, the optional
    Hyper-SFN index applies the TS 38.211 long-period gate using the supplied
    numerology and hyper-frame number.
    """
    if resource_type not in {"aperiodic", "periodic", "semi-persistent"}:
        raise ValueError("resource_type must be aperiodic, periodic, or semi-persistent")
    for name, value in (
        ("srs_symbol_count", srs_symbol_count),
        ("resource_symbol_index", resource_symbol_index),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    srs_symbol_count = int(srs_symbol_count)
    resource_symbol_index = int(resource_symbol_index)
    if srs_symbol_count not in _SRS_SYMBOL_COUNTS:
        raise ValueError("srs_symbol_count is unsupported")
    if not 0 <= resource_symbol_index < srs_symbol_count:
        raise ValueError("resource_symbol_index must be inside the SRS resource")
    if not isinstance(ports8_tdm, bool):
        raise TypeError("ports8_tdm must be a boolean")
    if not isinstance(positioning_resource, bool):
        raise TypeError("positioning_resource must be a boolean")
    port_factor = 2 if ports8_tdm else 1
    if srs_symbol_count % port_factor:
        raise ValueError("SRS symbol count must be divisible by the port factor")
    symbols_per_port_group = srs_symbol_count // port_factor
    if repetition_factor is None:
        repetition_factor = symbols_per_port_group
    elif isinstance(repetition_factor, bool) or not isinstance(repetition_factor, Integral):
        raise TypeError("repetition_factor must be an integer")
    repetition_factor = int(repetition_factor)
    if (
        repetition_factor < 1
        or repetition_factor > symbols_per_port_group
        or symbols_per_port_group % repetition_factor
    ):
        raise ValueError("repetition_factor must divide the SRS symbols per port group")
    counter_per_resource = symbols_per_port_group // repetition_factor
    symbol_counter = resource_symbol_index // (port_factor * repetition_factor)
    if resource_type == "aperiodic":
        if any(
            value is not None for value in (absolute_slot_index, periodicity_slots, slot_offset)
        ):
            raise ValueError("aperiodic SRS does not use periodic slot parameters")
        if any(value is not None for value in (numerology, hyper_frame_number, hyper_sfn_index)):
            raise ValueError("aperiodic SRS does not use Hyper-SFN parameters")
        return symbol_counter

    for name, periodic_value in (
        ("absolute_slot_index", absolute_slot_index),
        ("periodicity_slots", periodicity_slots),
        ("slot_offset", slot_offset),
    ):
        if isinstance(periodic_value, bool) or not isinstance(periodic_value, Integral):
            raise TypeError(f"{name} must be an integer for periodic SRS")
    assert absolute_slot_index is not None
    assert periodicity_slots is not None
    assert slot_offset is not None
    absolute_slot_index = int(absolute_slot_index)
    periodicity_slots = int(periodicity_slots)
    slot_offset = int(slot_offset)
    if absolute_slot_index < 0 or periodicity_slots < 1 or not 0 <= slot_offset < periodicity_slots:
        raise ValueError("periodic SRS requires nonnegative slots and offset inside its period")
    elapsed_slots = absolute_slot_index - slot_offset
    if hyper_sfn_index is not None:
        if not positioning_resource:
            raise ValueError("Hyper-SFN gating is only defined for positioning SRS")
        if isinstance(numerology, bool) or not isinstance(numerology, Integral):
            raise TypeError("numerology must be an integer when Hyper-SFN gating is configured")
        if not 0 <= int(numerology) <= 6:
            raise ValueError("numerology must be between 0 and 6")
        if periodicity_slots < (1 << int(numerology)) * 10240:
            raise ValueError("Hyper-SFN index requires a period of at least 2^mu * 10240 slots")
        if isinstance(hyper_frame_number, bool) or not isinstance(hyper_frame_number, Integral):
            raise TypeError("hyper_frame_number must be a nonnegative integer")
        if not 0 <= int(hyper_frame_number) <= 1023:
            raise ValueError("hyper_frame_number must be in [0, 1023]")
        if isinstance(hyper_sfn_index, bool) or not isinstance(hyper_sfn_index, Integral):
            raise TypeError("hyper_sfn_index must be 0 or 1")
        if hyper_sfn_index not in {0, 1}:
            raise ValueError("hyper_sfn_index must be 0 or 1")
        if (int(hyper_frame_number) + int(hyper_sfn_index)) % 2:
            return None
    elif numerology is not None or hyper_frame_number is not None:
        raise ValueError("numerology and hyper_frame_number require hyper_sfn_index")
    if elapsed_slots < 0 or elapsed_slots % periodicity_slots:
        return None
    occasion_index = elapsed_slots // periodicity_slots
    return occasion_index * counter_per_resource + symbol_counter


def nr_srs_periodicity_and_offset_slots(
    periodicity_and_offset: Mapping[str, object] | None,
    *,
    periodicity_and_offset_ext: Mapping[str, object] | None = None,
) -> tuple[int, int]:
    """Decode an RRC SRS periodicity CHOICE into slot period and offset.

    The arguments are single-entry mappings as commonly produced by ASN.1
    decoders, for example ``{"sl20": 7}``; ``sl1`` uses the ASN.1 NULL value
    ``None``. Release 16's extension CHOICE is accepted separately and takes
    precedence over the base CHOICE, as specified for the paired RRC fields.
    This resolves only the time-domain CHOICE; it does not parse RRC messages,
    decide set-level triggering, or apply positioning Hyper-SFN gating.
    """
    selected = (
        (periodicity_and_offset_ext, _SRS_PERIODICITY_EXT_OFFSETS)
        if periodicity_and_offset_ext is not None
        else (periodicity_and_offset, _SRS_PERIODICITY_OFFSETS)
    )
    choice, valid_choices = selected
    if not isinstance(choice, Mapping) or len(choice) != 1:
        raise ValueError("SRS periodicity CHOICE must contain exactly one alternative")
    name, offset = next(iter(choice.items()))
    if not isinstance(name, str):
        raise TypeError("SRS periodicity alternative name must be a string")
    normalized_name = name.casefold()
    if normalized_name not in valid_choices:
        raise ValueError("unsupported SRS periodicity CHOICE alternative")
    period_slots = valid_choices[normalized_name]
    if normalized_name == "sl1":
        if offset is not None:
            raise ValueError("sl1 SRS periodicity uses the ASN.1 NULL value")
        return period_slots, 0
    if isinstance(offset, bool) or not isinstance(offset, Integral):
        raise TypeError("SRS periodicity offset must be an integer")
    offset_slots = int(offset)
    if not 0 <= offset_slots < period_slots:
        raise ValueError("SRS periodicity offset must be inside its period")
    return period_slots, offset_slots


def nr_srs_triggered_resource_set_ids(
    srs_request_codepoint: int,
    resource_sets: Sequence[Mapping[str, object]],
    *,
    dci_format: str,
    srs_tpc_pdcch_group: str = "typeB",
    srs_request_mode: int | None = None,
    supplementary_uplink: bool = False,
) -> tuple[int, ...]:
    """Resolve a DCI SRS-request bit field to configured aperiodic sets.

    For DCI 0_1/1_1, the two low bits select triggers 0..3. For DCI
    0_2/1_2, ``srs_request_mode`` is the configured ``srs-RequestDCI-*``
    value (1 selects the one-bit table; 2 selects the two-bit table). When
    supplementary uplink is configured, the most-significant field bit
    selects non-SUL (0) or SUL (1), and the remaining bits carry the request.
    Resource-set records may declare ``ul_carrier`` as ``non_sul`` or
    ``sul``; this field is required when supplementary uplink is enabled.
    Resource-set records also use ``resource_set_id``, ``resource_type``,
    ``aperiodic_trigger``, and optional ``aperiodic_trigger_list``. Complete
    DCI/RRC parsing, typeA handling, and decoded multi-cell request-list
    indirection are handled outside this single-cell resolver.
    """
    if not isinstance(dci_format, str) or dci_format not in _SRS_REQUEST_DCI_FORMATS:
        raise ValueError("unsupported DCI format for this SRS request resolver")
    if not isinstance(srs_tpc_pdcch_group, str) or srs_tpc_pdcch_group != "typeB":
        raise ValueError("only typeB SRS TPC PDCCH group is supported")
    if not isinstance(supplementary_uplink, bool):
        raise TypeError("supplementary_uplink must be a boolean")
    if dci_format in {"0_2", "1_2"}:
        if isinstance(srs_request_mode, bool) or not isinstance(srs_request_mode, Integral):
            raise ValueError("DCI 0_2/1_2 requires configured srs_request_mode 1 or 2")
        srs_request_mode = int(srs_request_mode)
        if srs_request_mode not in {1, 2}:
            raise ValueError("srs_request_mode must be 1 or 2 for DCI 0_2/1_2")
        request_bit_count = srs_request_mode
    else:
        if srs_request_mode is not None:
            raise ValueError("srs_request_mode applies only to DCI 0_2/1_2")
        request_bit_count = 2
    if isinstance(srs_request_codepoint, bool) or not isinstance(srs_request_codepoint, Integral):
        raise TypeError("srs_request_codepoint must be an integer")
    codepoint = int(srs_request_codepoint)
    field_bit_count = request_bit_count + int(supplementary_uplink)
    maximum_codepoint = (1 << field_bit_count) - 1
    if not 0 <= codepoint <= maximum_codepoint:
        raise ValueError(
            f"{field_bit_count}-bit SRS request field must be in [0, {maximum_codepoint}] "
            f"for DCI format {dci_format}"
        )
    carrier = "non_sul"
    if supplementary_uplink:
        carrier = "sul" if codepoint >> request_bit_count else "non_sul"
    request = codepoint & ((1 << request_bit_count) - 1)
    if not isinstance(resource_sets, Sequence) or isinstance(resource_sets, (str, bytes)):
        raise TypeError("resource_sets must be a sequence of decoded resource-set mappings")

    seen_ids: set[int] = set()
    matched_ids: list[int] = []
    for index, resource_set in enumerate(resource_sets):
        if not isinstance(resource_set, Mapping):
            raise TypeError(f"resource_sets[{index}] must be a mapping")
        resource_set_id = resource_set.get("resource_set_id")
        if isinstance(resource_set_id, bool) or not isinstance(resource_set_id, Integral):
            raise TypeError(f"resource_sets[{index}].resource_set_id must be an integer")
        resource_set_id = int(resource_set_id)
        if resource_set_id < 0 or resource_set_id in seen_ids:
            raise ValueError("resource-set IDs must be unique nonnegative integers")
        seen_ids.add(resource_set_id)
        resource_type = resource_set.get("resource_type")
        if not isinstance(resource_type, str) or resource_type not in {
            "aperiodic",
            "periodic",
            "semi-persistent",
        }:
            raise ValueError(f"resource_sets[{index}].resource_type is unsupported")
        if resource_type != "aperiodic":
            continue
        configured_carrier = resource_set.get("ul_carrier")
        if supplementary_uplink and (
            not isinstance(configured_carrier, str) or configured_carrier not in {"non_sul", "sul"}
        ):
            raise ValueError(f"resource_sets[{index}].ul_carrier must identify non_sul or sul")
        if configured_carrier is not None and (
            not isinstance(configured_carrier, str) or configured_carrier not in {"non_sul", "sul"}
        ):
            raise ValueError(f"resource_sets[{index}].ul_carrier is invalid")
        if configured_carrier is not None and configured_carrier != carrier:
            continue
        primary_trigger = resource_set.get("aperiodic_trigger")
        if isinstance(primary_trigger, bool) or not isinstance(primary_trigger, Integral):
            raise TypeError(f"resource_sets[{index}].aperiodic_trigger must be an integer")
        primary_trigger = int(primary_trigger)
        if not 1 <= primary_trigger <= 3:
            raise ValueError(f"resource_sets[{index}].aperiodic_trigger must be in [1, 3]")
        additional = resource_set.get("aperiodic_trigger_list", ())
        if additional is None:
            additional = ()
        if not isinstance(additional, Sequence) or isinstance(additional, (str, bytes)):
            raise TypeError(f"resource_sets[{index}].aperiodic_trigger_list must be a sequence")
        triggers = [primary_trigger]
        for value in additional:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError("aperiodic trigger-list entries must be integers")
            value = int(value)
            if not 1 <= value <= 3:
                raise ValueError("aperiodic trigger-list entries must be in [1, 3]")
            triggers.append(value)
        if request in triggers:
            matched_ids.append(resource_set_id)
    return tuple(matched_ids)


def nr_srs_dci_0_3_ul_bwp_selection(
    scheduled_cell_ids: Sequence[str],
    configured_ul_bwp_ids_by_cell: Mapping[str, Sequence[int]],
    active_ul_bwp_id_by_cell: Mapping[str, int],
    *,
    bwp_indicator: int | None,
    supports_active_bwp_change: bool = True,
    initial_bwp_id: int = 0,
) -> NRSRSBWPSelection:
    """Resolve the decoded DCI 0_3 BWP indicator independently per cell.

    The configuration maps cover the full ``scheduledCellListDCI-0-3`` for
    the selected cell set, not just the combination selected by DCI. Each
    configured-ID tuple includes the initial UL BWP and is normalized in
    ascending BWP-ID order. For up to three noninitial BWPs, the indicator
    indexes that ordered BWP list. With four noninitial BWPs, TS 38.212
    Table 7.3.1.1.2-1 maps codepoints 0..3 to BWP IDs 1..4. A codepoint with
    no corresponding BWP in an individual scheduled cell leaves that cell on
    its active BWP. A UE that does not support active BWP changes via DCI also
    remains on each cell's active BWP.

    This resolves a decoded field only; it does not parse DCI bits or perform
    resource-set lookup for the selected BWP.
    """
    if not isinstance(scheduled_cell_ids, Sequence) or isinstance(scheduled_cell_ids, (str, bytes)):
        raise TypeError("scheduled_cell_ids must be an ordered sequence")
    cells = tuple(scheduled_cell_ids)
    if not cells or any(not isinstance(cell, str) or not cell for cell in cells):
        raise ValueError("scheduled_cell_ids must contain nonempty cell IDs")
    if len(set(cells)) != len(cells):
        raise ValueError("scheduled_cell_ids must be unique")
    for name, value in (
        ("configured_ul_bwp_ids_by_cell", configured_ul_bwp_ids_by_cell),
        ("active_ul_bwp_id_by_cell", active_ul_bwp_id_by_cell),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping")
        if set(value) != set(cells):
            raise ValueError(f"{name} keys must exactly match scheduled_cell_ids")
    if not isinstance(supports_active_bwp_change, bool):
        raise TypeError("supports_active_bwp_change must be a boolean")
    if isinstance(initial_bwp_id, bool) or not isinstance(initial_bwp_id, Integral):
        raise TypeError("initial_bwp_id must be an integer")
    initial_bwp_id = int(initial_bwp_id)
    if not 0 <= initial_bwp_id <= 4:
        raise ValueError("initial_bwp_id must be in [0, 4]")

    configured_by_cell: dict[str, tuple[int, ...]] = {}
    active_by_cell: dict[str, int] = {}
    for cell in cells:
        raw_ids = configured_ul_bwp_ids_by_cell[cell]
        if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes)):
            raise TypeError(f"configured UL BWP IDs for {cell!r} must be an ordered sequence")
        bwp_ids = tuple(raw_ids)
        if not bwp_ids or any(
            isinstance(bwp_id, bool) or not isinstance(bwp_id, Integral) or not 0 <= bwp_id <= 4
            for bwp_id in bwp_ids
        ):
            raise ValueError(f"configured UL BWP IDs for {cell!r} must lie in [0, 4]")
        normalized_ids = tuple(sorted(int(bwp_id) for bwp_id in bwp_ids))
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError(f"configured UL BWP IDs for {cell!r} must be unique")
        if initial_bwp_id not in normalized_ids:
            raise ValueError(f"configured UL BWP IDs for {cell!r} must include initial BWP")
        if len(normalized_ids) > 5:
            raise ValueError(f"configured UL BWP IDs for {cell!r} exceed the TS limit")
        active_id = active_ul_bwp_id_by_cell[cell]
        if (
            isinstance(active_id, bool)
            or not isinstance(active_id, Integral)
            or int(active_id) not in normalized_ids
        ):
            raise ValueError(f"active UL BWP for {cell!r} must be configured")
        configured_by_cell[cell] = normalized_ids
        active_by_cell[cell] = int(active_id)

    max_noninitial = max(len(bwp_ids) - 1 for bwp_ids in configured_by_cell.values())
    indicator_domain_size = max_noninitial + 1 if max_noninitial <= 3 else max_noninitial
    bit_width = (indicator_domain_size - 1).bit_length()
    if bit_width == 0:
        if bwp_indicator is not None:
            raise ValueError("bwp_indicator must be absent when the field has zero bits")
    else:
        if isinstance(bwp_indicator, bool) or not isinstance(bwp_indicator, Integral):
            raise TypeError("bwp_indicator must be a decoded integer codepoint")
        if not 0 <= int(bwp_indicator) < (1 << bit_width):
            raise ValueError("bwp_indicator does not fit the configured field width")

    indicator_value = None if bwp_indicator is None else int(bwp_indicator)
    selected: list[tuple[str, int]] = []
    outcomes: list[tuple[str, str]] = []
    for cell in cells:
        bwp_ids = configured_by_cell[cell]
        active_id = active_by_cell[cell]
        outcome: str
        selected_id = active_id
        if bit_width == 0:
            outcome = "field_absent"
        elif len(bwp_ids) == 1:
            outcome = "single_bwp"
        elif not supports_active_bwp_change:
            outcome = "active_bwp_change_unsupported"
        else:
            if max_noninitial > 3:
                assert indicator_value is not None
                indicated_id = indicator_value + 1
            else:
                assert indicator_value is not None
                indicated_id = bwp_ids[indicator_value] if indicator_value < len(bwp_ids) else -1
            if indicated_id in bwp_ids:
                selected_id = indicated_id
                outcome = "selected_by_indicator"
            else:
                outcome = "codepoint_not_configured_for_cell"
        selected.append((cell, selected_id))
        outcomes.append((cell, outcome))

    return NRSRSBWPSelection(bit_width, tuple(selected), tuple(outcomes))


def nr_dci_0_3_field_values_from_layout(
    payload_bits: Sequence[int | bool],
    *,
    field_widths_bits: Mapping[str, int],
) -> dict[str, int | None]:
    """Decode all bounded MSB-first DCI 0_3 fields from a width roster.

    Widths must cover the complete canonical clause 7.3.1.1.4 field order,
    including zero-width absent fields. A nonzero field is returned as its
    unsigned integer codepoint; a zero-width field is returned as ``None``.
    Zero padding after the declared layout is accepted, while nonzero trailing
    bits, truncated payloads, malformed bits, and resource-heavy layouts fail
    closed. This handles the already-decoded DCI information payload only; it
    does not perform CRC/polar decoding, RRC/PER parsing, or field-specific
    configuration interpretation.
    """
    if not isinstance(payload_bits, Sequence) or isinstance(payload_bits, (str, bytes)):
        raise TypeError("payload_bits must be a sequence of decoded bits")
    if not isinstance(field_widths_bits, Mapping):
        raise TypeError("field_widths_bits must be a mapping")
    if any(not isinstance(name, str) for name in field_widths_bits):
        raise TypeError("DCI 0_3 field names must be strings")
    expected_names = set(_DCI_0_3_FIELD_ORDER)
    if set(field_widths_bits) != expected_names:
        missing = sorted(expected_names - set(field_widths_bits))
        extra = sorted(set(field_widths_bits) - expected_names)
        raise ValueError(f"DCI 0_3 field-width roster mismatch; missing={missing}, extra={extra}")
    normalized_widths: dict[str, int] = {}
    total_width = 0
    for field_name in _DCI_0_3_FIELD_ORDER:
        width = field_widths_bits[field_name]
        if isinstance(width, bool) or not isinstance(width, Integral):
            raise TypeError(f"field width for {field_name} must be an integer")
        width = int(width)
        if not 0 <= width <= 512:
            raise ValueError(f"field width for {field_name} must be in [0, 512]")
        normalized_widths[field_name] = width
        total_width += width
    if total_width > 4096:
        raise ValueError("DCI 0_3 decoded layout exceeds the 4096-bit resource bound")
    if len(payload_bits) < total_width:
        raise ValueError("DCI payload is shorter than the declared format 0_3 field layout")
    if len(payload_bits) > total_width and any(payload_bits[total_width:]):
        raise ValueError("DCI payload has nonzero trailing bits beyond the declared layout")

    for index, bit in enumerate(payload_bits):
        if isinstance(bit, (bool, np.bool_)):
            continue
        if isinstance(bit, Integral) and int(bit) in {0, 1}:
            continue
        raise ValueError(f"payload_bits[{index}] must be a Boolean or integer bit")

    decoded: dict[str, int | None] = {}
    offset = 0
    for field_name in _DCI_0_3_FIELD_ORDER:
        width = normalized_widths[field_name]
        if width == 0:
            decoded[field_name] = None
            continue
        value = 0
        for bit in payload_bits[offset : offset + width]:
            value = (value << 1) | int(bit)
        decoded[field_name] = value
        offset += width
    return decoded


def nr_srs_group_hopping_index(
    sequence_id: int,
    slot_index: int,
    start_symbol_index: int,
    resource_symbol_index: int,
    *,
    slot_symbol_count: int = 14,
) -> int:
    """Return the TS 38.211 SRS group index for a resource symbol.

    The pseudo-random sequence restarts at each radio frame. ``slot_index`` is
    the zero-based slot within that frame; ``start_symbol_index`` is the
    absolute SRS start symbol ``l0`` within the slot; and
    ``resource_symbol_index`` is the resource-local symbol index ``l'``.
    Set ``slot_symbol_count=12`` for extended-CP slots.
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
    first_bit = _srs_gold_bit_offset(
        slot_index, start_symbol_index, resource_symbol_index, slot_symbol_count
    )
    hopping_bits = _nr_gold_bits(sequence_id, first_bit + 8)
    group_hopping_value = sum(int(hopping_bits[first_bit + bit]) << bit for bit in range(8))
    return (group_hopping_value + sequence_id) % 30


def nr_srs_sequence_hopping_index(
    sequence_id: int,
    slot_index: int,
    start_symbol_index: int,
    resource_symbol_index: int,
    sequence_length: int,
    *,
    slot_symbol_count: int = 14,
) -> int:
    """Return the SRS base-sequence number under TS 38.211 sequence hopping.

    Sequence hopping is enabled only for ``M_ZC >= 6 N_sc^RB``; with the NR
    resource-block width ``N_sc^RB=12``, shorter sequences use base sequence 0.
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
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, Integral):
        raise TypeError("sequence_length must be an integer")
    sequence_length = int(sequence_length)
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if sequence_length < 72:
        return 0
    bit_index = slot_symbol_count * slot_index + start_symbol_index + resource_symbol_index
    return int(_nr_gold_bits(sequence_id, bit_index + 1)[bit_index])


def nr_low_papr_sequence_type1(
    sequence_length: int,
    *,
    group_index: int,
    base_sequence_index: int = 0,
    cyclic_shift_index: int = 0,
    cyclic_shift_count: int = 1,
    cyclic_shift_hopping_offset: int = 0,
    hopping_finer_granularity: bool = False,
) -> np.ndarray:
    """Generate the type-1 low-PAPR sequence for supported TS 38.211 lengths.

    This implements TS 38.211 clause 5.2.2.1 for explicitly supplied group,
    base-sequence, and cyclic-shift indices. ``cyclic_shift_count`` is the
    denominator in the normalized cyclic shift; for SRS it is 8, 12, or 6 for
    transmission comb sizes 2, 4, or 8, respectively. Lengths 6, 12, 18,
    and 24 use standardized phase tables; length 30 and lengths at least 36
    use their respective closed-form definitions.
    """
    for name, value in (
        ("sequence_length", sequence_length),
        ("group_index", group_index),
        ("base_sequence_index", base_sequence_index),
        ("cyclic_shift_index", cyclic_shift_index),
        ("cyclic_shift_count", cyclic_shift_count),
        ("cyclic_shift_hopping_offset", cyclic_shift_hopping_offset),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    sequence_length = int(sequence_length)
    group_index = int(group_index)
    base_sequence_index = int(base_sequence_index)
    cyclic_shift_index = int(cyclic_shift_index)
    cyclic_shift_count = int(cyclic_shift_count)
    cyclic_shift_hopping_offset = int(cyclic_shift_hopping_offset)
    if not isinstance(hopping_finer_granularity, bool):
        raise TypeError("hopping_finer_granularity must be a boolean")
    if (
        sequence_length not in _LOW_PAPR_SHORT_PHASES
        and sequence_length != 30
        and sequence_length < 36
    ):
        raise ValueError("supported type-1 sequence lengths are 6, 12, 18, 24, 30, or >=36")
    if not 0 <= group_index <= 29:
        raise ValueError("group_index must be in [0, 29]")
    if base_sequence_index not in {0, 1}:
        raise ValueError("base_sequence_index must be 0 or 1")
    if sequence_length < 72 and base_sequence_index != 0:
        raise ValueError("base_sequence_index must be 0 for sequence lengths below 72")

    if cyclic_shift_count < 1:
        raise ValueError("cyclic_shift_count must be positive")
    if not 0 <= cyclic_shift_index < cyclic_shift_count:
        raise ValueError("cyclic_shift_index must be in [0, cyclic_shift_count)")
    hop_denominator = 2 if hopping_finer_granularity else 1
    if not 0 <= cyclic_shift_hopping_offset < hop_denominator * cyclic_shift_count:
        raise ValueError("cyclic_shift_hopping_offset is outside the configured shift range")
    indices = np.arange(sequence_length, dtype=np.float64)
    if sequence_length in _LOW_PAPR_SHORT_PHASES:
        phases = np.asarray(_LOW_PAPR_SHORT_PHASES[sequence_length][group_index])
        base = np.exp(1j * np.pi * phases / 4.0)
    elif sequence_length == 30:
        base = np.exp(-1j * np.pi * group_index * (indices + 1.0) * (indices + 2.0) / 31.0)
    else:
        zc_length = _largest_prime_below(sequence_length)
        q_numerator = zc_length * (group_index + 1)
        q_bar = (2 * q_numerator + 31) // 62
        sign_exponent = 2 * q_numerator // 31
        q = q_bar + base_sequence_index * (-1 if sign_exponent % 2 else 1)
        residues = np.remainder(indices, zc_length)
        base = np.exp(-1j * np.pi * q * residues * (residues + 1.0) / zc_length)
    normalized_shift = cyclic_shift_index / cyclic_shift_count + (
        cyclic_shift_hopping_offset / (hop_denominator * cyclic_shift_count)
    )
    shift = np.exp(2j * np.pi * normalized_shift * indices)
    return np.asarray(base * shift, dtype=np.complex128)


def map_nr_srs_resource(
    sequence: np.ndarray,
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
) -> NRSRSResourceGrid:
    """Map one pre-generated single-port, single-hop SRS sequence into a slot.

    ``start_position_offset`` counts symbols backward from the last slot symbol,
    as in the TS 38.211 ``startPosition`` field. The caller supplies either one
    sequence row reused for every SRS symbol or one row per SRS symbol. A
    symbol row contains one value for each occupied comb subcarrier. When
    ``allocated_bandwidth_subcarriers`` is supplied, the comb is confined to
    that many subcarriers beginning at ``frequency_start_subcarrier``; without
    it, the allocation extends to the end of the grid.

    This function does not generate the sequence or derive the allocation from
    ``c-SRS``/``b-SRS``; ``frequency_start_subcarrier`` and ``comb_offset`` are
    already-resolved mapping parameters.
    """
    integer_values = {
        "slot_symbol_count": slot_symbol_count,
        "subcarrier_count": subcarrier_count,
        "start_position_offset": start_position_offset,
        "srs_symbol_count": srs_symbol_count,
        "comb_size": comb_size,
        "comb_offset": comb_offset,
        "frequency_start_subcarrier": frequency_start_subcarrier,
    }
    if allocated_bandwidth_subcarriers is not None:
        integer_values["allocated_bandwidth_subcarriers"] = allocated_bandwidth_subcarriers
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
    if allocated_bandwidth_subcarriers is not None:
        allocated_bandwidth_subcarriers = int(allocated_bandwidth_subcarriers)
    if slot_symbol_count not in {12, 14}:
        raise ValueError("slot_symbol_count must be 12 or 14")
    if subcarrier_count < 1:
        raise ValueError("subcarrier_count must be positive")
    if srs_symbol_count not in _SRS_SYMBOL_COUNTS or srs_symbol_count > slot_symbol_count:
        raise ValueError("unsupported SRS symbol count for the declared slot")
    if start_position_offset < srs_symbol_count - 1 or start_position_offset >= slot_symbol_count:
        raise ValueError("start_position_offset must fit the SRS symbols in the slot")
    if comb_size not in _SRS_COMB_SIZES:
        raise ValueError("comb_size must be 2, 4, or 8")
    if not 0 <= comb_offset < comb_size:
        raise ValueError("comb_offset must be in [0, comb_size)")
    if frequency_start_subcarrier < 0:
        raise ValueError("frequency_start_subcarrier must be nonnegative")
    if allocated_bandwidth_subcarriers is not None and (
        allocated_bandwidth_subcarriers < 1
        or frequency_start_subcarrier + allocated_bandwidth_subcarriers > subcarrier_count
    ):
        raise ValueError("allocated bandwidth must be positive and fit inside the grid")
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
    frequency_width = (
        subcarrier_count - frequency_start_subcarrier
        if allocated_bandwidth_subcarriers is None
        else allocated_bandwidth_subcarriers
    )
    remaining_width = frequency_width - comb_offset
    occupied_count = (remaining_width + comb_size - 1) // comb_size
    if occupied_count < 1:
        raise ValueError("SRS comb has no resource elements inside the subcarrier grid")
    if values.shape[-1] != occupied_count:
        raise ValueError("sequence length does not match the occupied SRS comb")
    if values.ndim == 2 and values.shape[0] != srs_symbol_count:
        raise ValueError("per-symbol SRS sequence must include exactly srs_symbol_count rows")
    if not np.isfinite(values).all():
        raise ValueError("sequence must contain only finite values")

    first_symbol = slot_symbol_count - 1 - start_position_offset
    symbol_indices = tuple(range(first_symbol, first_symbol + srs_symbol_count))
    subcarrier_indices = tuple(
        frequency_start_subcarrier + comb_offset + comb_size * index
        for index in range(occupied_count)
    )
    grid = np.zeros((slot_symbol_count, subcarrier_count), dtype=np.complex128)
    mask = np.zeros((slot_symbol_count, subcarrier_count), dtype=np.bool_)
    rows = (
        values if values.ndim == 2 else np.broadcast_to(values, (srs_symbol_count, occupied_count))
    )
    for symbol_offset, symbol_index in enumerate(symbol_indices):
        grid[symbol_index, list(subcarrier_indices)] = beta_srs * rows[symbol_offset]
        mask[symbol_index, list(subcarrier_indices)] = True

    return NRSRSResourceGrid(
        grid=grid,
        mask=mask,
        symbol_indices=symbol_indices,
        subcarrier_indices=subcarrier_indices,
    )


def nr_srs_single_port_ls_channel_estimate(
    mapped_resource: NRSRSResourceGrid | NRSRSHoppedResourceGrid | NRSRSPositioningResourceGrid,
    received_grid: np.ndarray,
) -> NRSRSSinglePortChannelEstimate:
    """Estimate the effective channel on mapped single-port SRS REs.

    The estimator computes ``Y/X`` only where the transmitted resource mask
    is true. It requires the receiver grid to use the same slot/subcarrier
    coordinates and the mapped transmit values to be nonzero on every active
    RE. The output is an un-interpolated per-RE estimate; synchronization,
    receive combining, multiport separation, and noise-variance estimation
    remain receiver-level responsibilities.
    """
    if not isinstance(
        mapped_resource,
        (NRSRSResourceGrid, NRSRSHoppedResourceGrid, NRSRSPositioningResourceGrid),
    ):
        raise TypeError("mapped_resource must be a mapped single-port SRS resource")
    received = np.asarray(received_grid)
    if received.ndim != 2 or not np.iscomplexobj(received):
        raise ValueError("received_grid must be complex [slot symbols, subcarriers]")
    if received.shape != mapped_resource.grid.shape:
        raise ValueError("received_grid shape must match the mapped SRS grid")
    if not np.isfinite(received).all():
        raise ValueError("received_grid must contain only finite values")
    transmit = mapped_resource.grid
    mask = mapped_resource.mask
    if not mask.any():
        raise ValueError("mapped SRS resource must occupy at least one resource element")
    if np.any(transmit[mask] == 0):
        raise ValueError("mapped SRS transmit values must be nonzero on active resource elements")
    estimate = np.zeros(transmit.shape, dtype=np.complex128)
    estimate[mask] = received[mask] / transmit[mask]
    return NRSRSSinglePortChannelEstimate(
        channel_by_re=estimate,
        mask=mask,
        symbol_indices=mapped_resource.symbol_indices,
    )


def nr_srs_multiport_ls_channel_estimate(
    mapped_resource: (
        NRSRSMultiportResourceGrid
        | NRSRSHoppedMultiportResourceGrid
        | NRSRSPositioningMultiportResourceGrid
    ),
    received_grid: np.ndarray,
    *,
    noise_variance_by_rx_ant: float | Sequence[float] | None = None,
    noise_variance_by_rx_symbol: np.ndarray | None = None,
) -> NRSRSMultiportChannelEstimate:
    """Separate mapped SRS transmit ports with per-subcarrier least squares.

    The receiver grid is ``[rx antennas, slot symbols, subcarriers]`` and the
    mapped transmit grid is ``[TX ports, slot symbols, subcarriers]``. For each
    subcarrier this solves ``Y = X H`` over the mapped SRS symbols. A rank
    deficient port signature is retained as an unavailable subcarrier rather
    than producing an underdetermined estimate. Per-RX covariance is computed
    from the conditional weighted-LS equations. The optional variance profile
    has shape ``[rx antennas, slot symbols]`` and weights each observed SRS
    symbol independently; it can represent a caller-calibrated changing noise
    or interference level. No variance is inferred from the received grid.
    """
    supported = (
        NRSRSMultiportResourceGrid,
        NRSRSHoppedMultiportResourceGrid,
        NRSRSPositioningMultiportResourceGrid,
    )
    if not isinstance(mapped_resource, supported):
        raise TypeError("mapped_resource must be a mapped multiport SRS resource")
    received = np.asarray(received_grid)
    transmit = mapped_resource.grid
    transmit_mask = mapped_resource.mask
    if received.ndim != 3 or not np.iscomplexobj(received):
        raise ValueError("received_grid must be complex [rx antennas, slot symbols, subcarriers]")
    if received.shape[0] < 1:
        raise ValueError("received_grid must contain at least one RX antenna")
    if received.shape[1:] != transmit.shape[1:]:
        raise ValueError("received_grid symbol/subcarrier axes must match the mapped SRS grid")
    if not np.isfinite(received).all():
        raise ValueError("received_grid must contain only finite values")
    if not transmit_mask.any():
        raise ValueError("mapped SRS resource must occupy at least one resource element")
    if np.any(transmit[transmit_mask] == 0):
        raise ValueError("mapped SRS transmit values must be nonzero on active resource elements")

    rx_count, symbol_count, _ = received.shape
    if noise_variance_by_rx_symbol is not None and noise_variance_by_rx_ant is not None:
        raise ValueError("provide per-RX or per-RX/per-symbol noise variances, not both")
    symbol_noise: np.ndarray | None
    symbol_weights: np.ndarray | None = None
    noise_weights: np.ndarray | None = None
    if noise_variance_by_rx_symbol is not None:
        raw_symbol_noise = np.asarray(noise_variance_by_rx_symbol)
        if raw_symbol_noise.dtype.kind not in "iuf" or raw_symbol_noise.dtype.kind == "b":
            raise TypeError("noise_variance_by_rx_symbol must be real-valued")
        if raw_symbol_noise.shape != (rx_count, symbol_count):
            raise ValueError("noise_variance_by_rx_symbol must have shape [rx, slot symbols]")
        symbol_noise = np.array(raw_symbol_noise, dtype=np.float64, copy=True)
        if not np.isfinite(symbol_noise).all() or np.any(symbol_noise <= 0):
            raise ValueError("per-symbol noise variances must be finite and positive")
        with np.errstate(over="ignore", divide="ignore", under="ignore"):
            symbol_weights = np.reciprocal(symbol_noise)
        if not np.isfinite(symbol_weights).all() or np.any(symbol_weights <= 0):
            raise ValueError("per-symbol noise variance reciprocals must be representable")
        noise = np.mean(symbol_noise, axis=1)
        if not np.isfinite(noise).all() or np.any(noise <= 0):
            raise ValueError("mean per-RX noise variance must be finite and positive")
    else:
        if noise_variance_by_rx_ant is None:
            raise ValueError("per-RX noise variance is required")
        raw_noise = np.asarray(noise_variance_by_rx_ant)
        if raw_noise.dtype.kind not in "iuf" or raw_noise.dtype.kind == "b":
            raise TypeError("noise_variance_by_rx_ant must be real-valued")
        if raw_noise.ndim == 0:
            noise = np.full(rx_count, float(raw_noise), dtype=np.float64)
        elif raw_noise.shape == (rx_count,):
            noise = np.array(raw_noise, dtype=np.float64, copy=True)
        else:
            raise ValueError("noise_variance_by_rx_ant must be scalar or one value per RX antenna")
        if not np.isfinite(noise).all() or np.any(noise <= 0):
            raise ValueError("noise variances must be finite and positive")
        with np.errstate(over="ignore", divide="ignore", under="ignore"):
            noise_weights = np.reciprocal(noise)
        if not np.isfinite(noise_weights).all() or np.any(noise_weights <= 0):
            raise ValueError("noise variance reciprocals must be representable")
        symbol_noise = None

    port_count, _, subcarrier_count = transmit.shape
    channel = np.zeros((rx_count, port_count, subcarrier_count), dtype=np.complex128)
    covariance = np.zeros((rx_count, subcarrier_count, port_count, port_count), dtype=np.complex128)
    valid = np.zeros(subcarrier_count, dtype=np.bool_)
    ranks = np.zeros(subcarrier_count, dtype=np.int64)
    condition_numbers: list[float | None] = [None] * subcarrier_count
    reasons: list[str | None] = ["not_occupied"] * subcarrier_count
    symbols = tuple(mapped_resource.symbol_indices)

    for subcarrier in range(subcarrier_count):
        design = transmit[:, symbols, subcarrier].T
        occupied_rows = np.any(transmit_mask[:, symbols, subcarrier], axis=0)
        design = design[occupied_rows]
        if design.shape[0] == 0:
            continue
        rank = int(np.linalg.matrix_rank(design))
        ranks[subcarrier] = rank
        if rank < port_count:
            reasons[subcarrier] = "rank_deficient_port_signature"
            continue
        condition = float(np.linalg.cond(design))
        if not np.isfinite(condition):
            reasons[subcarrier] = "numerically_singular_port_signature"
            continue
        condition_numbers[subcarrier] = condition
        observed = received[:, np.asarray(symbols)[occupied_rows], subcarrier].T
        selected_symbols = np.asarray(symbols)[occupied_rows]
        for rx_index in range(rx_count):
            if symbol_noise is None:
                assert noise_weights is not None
                weights = np.full(design.shape[0], noise_weights[rx_index])
            else:
                assert symbol_weights is not None
                weights = symbol_weights[rx_index, selected_symbols]
            weighted_design = design * np.sqrt(weights)[:, None]
            weighted_observed = observed[:, rx_index] * np.sqrt(weights)
            try:
                coefficients, _, solved_rank, _ = np.linalg.lstsq(
                    weighted_design, weighted_observed, rcond=None
                )
            except np.linalg.LinAlgError:
                channel[:, :, subcarrier] = 0
                covariance[:, subcarrier] = 0
                reasons[subcarrier] = "weighted_solver_failure"
                condition_numbers[subcarrier] = None
                break
            if solved_rank != port_count:
                channel[:, :, subcarrier] = 0
                covariance[:, subcarrier] = 0
                ranks[subcarrier] = int(solved_rank)
                reasons[subcarrier] = "rank_deficient_port_signature"
                condition_numbers[subcarrier] = None
                break
            gram = design.conj().T @ (weights[:, None] * design)
            try:
                coefficient_covariance = np.linalg.inv(gram)
            except np.linalg.LinAlgError:
                channel[:, :, subcarrier] = 0
                covariance[:, subcarrier] = 0
                reasons[subcarrier] = "singular_weighted_port_signature"
                condition_numbers[subcarrier] = None
                break
            if not np.isfinite(coefficients).all() or not np.isfinite(coefficient_covariance).all():
                channel[:, :, subcarrier] = 0
                covariance[:, subcarrier] = 0
                reasons[subcarrier] = "nonfinite_weighted_estimate"
                condition_numbers[subcarrier] = None
                break
            covariance[rx_index, subcarrier] = coefficient_covariance
            channel[rx_index, :, subcarrier] = coefficients
        else:
            valid[subcarrier] = True
            reasons[subcarrier] = None

    return NRSRSMultiportChannelEstimate(
        channel_by_rx_port_subcarrier=channel,
        noise_covariance_by_rx_subcarrier=covariance,
        valid_subcarrier_mask=valid,
        rank_by_subcarrier=ranks,
        condition_number_by_subcarrier=tuple(condition_numbers),
        reason_codes=tuple(reasons),
        noise_variance_by_rx_ant=noise,
        symbol_indices=symbols,
        noise_variance_by_rx_symbol=symbol_noise,
    )
