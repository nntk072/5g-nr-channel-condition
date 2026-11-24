"""DCI format 0_3 SRS request fields, widths, and scheduled cells."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import comb
from numbers import Integral
from typing import cast

import numpy as np

_DCI_0_3_FIELD_ORDER = (
    "identifier_for_dci_formats",
    "scheduled_cell_set_indicator",
    "scheduled_cells_indicator",
    "bandwidth_part_indicator",
    "frequency_domain_resource_assignment",
    "time_domain_resource_assignment",
    "frequency_hopping_flag",
    "modulation_and_coding_scheme",
    "new_data_indicator",
    "redundancy_version",
    "harq_process_number",
    "first_downlink_assignment_index",
    "second_downlink_assignment_index",
    "tpc_command_for_scheduled_pusch",
    "srs_resource_indicator",
    "precoding_information_and_number_of_layers",
    "antenna_ports",
    "srs_request",
    "srs_offset_indicator",
    "csi_request",
    "ptrs_dmrs_association",
    "beta_offset_indicator",
    "dmrs_sequence_initialization",
    "ul_sch_indicator",
    "channel_access_cpext_capc",
    "open_loop_power_control_parameter_set_indication",
    "priority_indicator",
    "minimum_applicable_scheduling_offset_indicator",
    "scell_dormancy_indication",
    "pdcch_monitoring_adaptation_indication",
    "measurement_gap_cancellation",
)


_DCI_0_3_CONFIG_DERIVED_FIELDS = frozenset(
    {
        "identifier_for_dci_formats",
        "scheduled_cell_set_indicator",
        "scheduled_cells_indicator",
        "bandwidth_part_indicator",
        "frequency_domain_resource_assignment",
        "time_domain_resource_assignment",
        "frequency_hopping_flag",
        "dmrs_sequence_initialization",
        "ul_sch_indicator",
        "priority_indicator",
        "minimum_applicable_scheduling_offset_indicator",
        "beta_offset_indicator",
        "channel_access_cpext_capc",
        "tpc_command_for_scheduled_pusch",
        "harq_process_number",
        "modulation_and_coding_scheme",
        "new_data_indicator",
        "redundancy_version",
        "first_downlink_assignment_index",
        "second_downlink_assignment_index",
        "srs_resource_indicator",
        "precoding_information_and_number_of_layers",
        "antenna_ports",
        "srs_request",
        "srs_offset_indicator",
        "csi_request",
        "ptrs_dmrs_association",
        "open_loop_power_control_parameter_set_indication",
        "scell_dormancy_indication",
        "pdcch_monitoring_adaptation_indication",
        "measurement_gap_cancellation",
    }
)


def nr_srs_aperiodic_target_slot_index(
    triggering_dci_slot: int,
    *,
    slot_offset: int | None = None,
    available_slot_offsets: Sequence[int] | None = None,
    selected_available_offset_index: int | None = None,
) -> int:
    """Resolve the target slot for one triggered aperiodic SRS resource set.

    The optional list contains offsets measured from the reference slot
    ``triggering_dci_slot + slot_offset``. The selected index is supplied by
    the caller after interpreting the applicable DCI/resource-set trigger
    mapping; this function does not decode that mapping.
    """
    if isinstance(triggering_dci_slot, bool) or not isinstance(triggering_dci_slot, Integral):
        raise TypeError("triggering_dci_slot must be an integer")
    if triggering_dci_slot < 0:
        raise ValueError("triggering_dci_slot must be nonnegative")
    if slot_offset is None:
        slot_offset = 0
    elif isinstance(slot_offset, bool) or not isinstance(slot_offset, Integral):
        raise TypeError("slot_offset must be an integer when configured")
    elif not 1 <= slot_offset <= 32:
        raise ValueError("configured slot_offset must be in [1, 32]")
    if available_slot_offsets is None:
        if selected_available_offset_index is not None:
            raise ValueError("an available offset index requires an available-slot-offset list")
        selected_offset = 0
    else:
        if (
            not isinstance(available_slot_offsets, Sequence)
            or isinstance(available_slot_offsets, (str, bytes))
            or not 1 <= len(available_slot_offsets) <= 4
        ):
            raise ValueError("available_slot_offsets must contain one to four values")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) or value < 0
            for value in available_slot_offsets
        ):
            raise ValueError("available slot offsets must be integers in [0, 7]")
        if any(value > 7 for value in available_slot_offsets):
            raise ValueError("available slot offsets must be in [0, 7]")
        if len(set(available_slot_offsets)) != len(available_slot_offsets):
            raise ValueError("available slot offsets must be distinct")
        if isinstance(selected_available_offset_index, bool) or not isinstance(
            selected_available_offset_index, Integral
        ):
            raise TypeError("selected_available_offset_index must be an integer")
        if not 0 <= selected_available_offset_index < len(available_slot_offsets):
            raise ValueError("selected available offset index is outside the configured list")
        selected_offset = int(available_slot_offsets[int(selected_available_offset_index)])
    return int(triggering_dci_slot) + int(slot_offset) + selected_offset


def nr_srs_dci_0_3_scheduled_cell_ids(
    cell_set_configs: Sequence[Mapping[str, object]],
    *,
    scheduled_cell_set_indicator: int | None = None,
    scheduled_cells_indicator: int | None = None,
    default_set_id: int | None = None,
) -> tuple[str, ...]:
    """Resolve decoded DCI 0_3 cell-set and scheduled-cells indicators.

    Each normalized, already-decoded RRC configuration has ``set_id``,
    ``scheduled_cell_ids`` (the ordered ``scheduledCellListDCI-0-3``), and
    optional ``scheduled_cell_combos`` (the rows of
    ``scheduledCellComboListDCI-0-3`` as zero-based indexes into that list).
    A present cell-set indicator indexes configurations ordered by set ID. If
    that field is absent, a single configured set is selected implicitly;
    multiple configured sets require the caller to supply the configured
    ``default_set_id`` for the PDCCH cell/context. A scheduled-cells indicator
    indexes the configured combination rows; when no table exists, or it has
    one row, that indicator is absent and the implied row/list is used.

    The returned cell identifiers follow the RRC list order, which must
    already represent ascending serving-cell indexes. This function consumes
    decoded values; it does not parse DCI bits or ASN.1 PER.
    """
    if not isinstance(cell_set_configs, Sequence) or isinstance(cell_set_configs, (str, bytes)):
        raise TypeError("cell_set_configs must be an ordered sequence")
    if not 1 <= len(cell_set_configs) <= 4:
        raise ValueError("cell_set_configs must contain 1..4 configured sets")

    normalized: list[tuple[int, tuple[str, ...], object]] = []
    seen_set_ids: set[int] = set()
    seen_cells: set[str] = set()
    for config in cell_set_configs:
        if not isinstance(config, Mapping):
            raise TypeError("each cell-set config must be a mapping")
        set_id = config.get("set_id")
        if isinstance(set_id, bool) or not isinstance(set_id, Integral):
            raise TypeError("each set_id must be an integer")
        set_id = int(set_id)
        if not 0 <= set_id <= 3 or set_id in seen_set_ids:
            raise ValueError("set_id values must be unique integers in [0, 3]")
        seen_set_ids.add(set_id)

        raw_cells = config.get("scheduled_cell_ids")
        if not isinstance(raw_cells, Sequence) or isinstance(raw_cells, (str, bytes)):
            raise TypeError("scheduled_cell_ids must be an ordered sequence")
        cells = tuple(raw_cells)
        if not 2 <= len(cells) <= 4 or any(not isinstance(cell, str) or not cell for cell in cells):
            raise ValueError("scheduled_cell_ids must contain 2..4 nonempty IDs")
        if len(set(cells)) != len(cells):
            raise ValueError("scheduled_cell_ids must be unique within each set")
        if seen_cells.intersection(cells):
            raise ValueError("a scheduled cell cannot appear in more than one set")
        seen_cells.update(cells)

        combos = config.get("scheduled_cell_combos")
        normalized.append((set_id, cells, combos))

    normalized.sort(key=lambda item: item[0])
    if scheduled_cell_set_indicator is None:
        if default_set_id is None:
            if len(normalized) != 1:
                raise ValueError(
                    "multiple cell sets require scheduled_cell_set_indicator or default_set_id"
                )
            selected_set = normalized[0]
        else:
            if isinstance(default_set_id, bool) or not isinstance(default_set_id, Integral):
                raise TypeError("default_set_id must be an integer")
            matches = [item for item in normalized if item[0] == int(default_set_id)]
            if not matches:
                raise ValueError("default_set_id does not identify a configured set")
            selected_set = matches[0]
    else:
        if isinstance(scheduled_cell_set_indicator, bool) or not isinstance(
            scheduled_cell_set_indicator, Integral
        ):
            raise TypeError("scheduled_cell_set_indicator must be an integer")
        if default_set_id is not None:
            raise ValueError("default_set_id must be omitted when the set indicator is present")
        index = int(scheduled_cell_set_indicator)
        if not 0 <= index < len(normalized):
            raise ValueError("scheduled_cell_set_indicator selects no configured set")
        selected_set = normalized[index]

    _, cells, combos_obj = selected_set
    if combos_obj is None:
        if scheduled_cells_indicator is not None:
            raise ValueError(
                "scheduled_cells_indicator must be absent when no combo list is configured"
            )
        return cells
    if not isinstance(combos_obj, Sequence) or isinstance(combos_obj, (str, bytes)):
        raise TypeError("scheduled_cell_combos must be an ordered sequence")
    combos = tuple(combos_obj)
    if not 1 <= len(combos) <= 16:
        raise ValueError("scheduled_cell_combos must contain 1..16 rows")
    normalized_combos: list[tuple[int, ...]] = []
    for combo in combos:
        if not isinstance(combo, Sequence) or isinstance(combo, (str, bytes)):
            raise TypeError("each scheduled-cell combo must be an ordered index sequence")
        indices = tuple(combo)
        if not 1 <= len(indices) <= len(cells):
            raise ValueError("each scheduled-cell combo must select 1..N configured cells")
        if any(isinstance(i, bool) or not isinstance(i, Integral) for i in indices):
            raise TypeError("scheduled-cell combo indexes must be integers")
        index_values = tuple(int(i) for i in indices)
        if any(not 0 <= i < len(cells) for i in index_values):
            raise ValueError("scheduled-cell combo index is outside scheduled_cell_ids")
        if tuple(sorted(set(index_values))) != index_values:
            raise ValueError("scheduled-cell combo indexes must be unique and ascending")
        normalized_combos.append(index_values)
    if len(combos) == 1:
        if scheduled_cells_indicator is not None:
            raise ValueError("scheduled_cells_indicator is absent for a single combo row")
        combo_index = 0
    else:
        if isinstance(scheduled_cells_indicator, bool) or not isinstance(
            scheduled_cells_indicator, Integral
        ):
            raise TypeError("scheduled_cells_indicator must select a configured combo row")
        combo_index = int(scheduled_cells_indicator)
        if not 0 <= combo_index < len(combos):
            raise ValueError("scheduled_cells_indicator selects no configured combo row")

    return tuple(cells[i] for i in normalized_combos[combo_index])


def nr_srs_dci_0_3_request_list_index_from_bits(
    payload_bits: Sequence[int | bool],
    *,
    field_offset: int,
    request_list_entry_count: int,
) -> int | None:
    """Decode the DCI 0_3 SRS-request selector from an MSB-first payload.

    ``field_offset`` is the zero-based bit position in the already-decoded
    DCI information payload; determining the full DCI field layout remains
    the caller's responsibility. The selector width is ``ceil(log2(N))``
    for the configured ``srs-RequestListDCI-0-3`` length ``N``. With no list,
    the field is absent and this returns ``None``. A one-entry list has a
    zero-bit selector and resolves to entry zero. Reserved codepoints are
    rejected instead of silently aliasing a configured row.

    Bit positions follow the serialized payload order: index zero is the
    first (most-significant) bit. This function does not decode CRC, polar
    coding, or ASN.1 RRC configuration.
    """
    if not isinstance(payload_bits, Sequence) or isinstance(payload_bits, (str, bytes)):
        raise TypeError("payload_bits must be a sequence of decoded bits")
    if isinstance(field_offset, bool) or not isinstance(field_offset, Integral):
        raise TypeError("field_offset must be an integer")
    if field_offset < 0:
        raise ValueError("field_offset must be nonnegative")
    if isinstance(request_list_entry_count, bool) or not isinstance(
        request_list_entry_count, Integral
    ):
        raise TypeError("request_list_entry_count must be an integer")
    entry_count = int(request_list_entry_count)
    if not 0 <= entry_count <= 16:
        raise ValueError("request_list_entry_count must be in [0, 16]")
    for index, bit in enumerate(payload_bits):
        if isinstance(bit, (bool, np.bool_)):
            continue
        if isinstance(bit, Integral) and int(bit) in {0, 1}:
            continue
        raise ValueError(f"payload_bits[{index}] must be a Boolean or integer bit")
    if int(field_offset) > len(payload_bits):
        raise ValueError("field_offset lies beyond the DCI payload")

    if entry_count == 0:
        return None
    bit_width = (entry_count - 1).bit_length()
    if bit_width == 0:
        return 0
    end = int(field_offset) + bit_width
    if end > len(payload_bits):
        raise ValueError("DCI payload does not contain the complete SRS request selector")
    selector = 0
    for bit in payload_bits[int(field_offset) : end]:
        selector = (selector << 1) | int(bit)
    if selector >= entry_count:
        raise ValueError("SRS request selector uses an unconfigured reserved codepoint")
    return selector


def _nr_srs_dci_0_3_antenna_ports_width(
    *,
    mode: str,
    cell_ids: Sequence[str],
    transform_precoding_enabled_by_cell: Mapping[str, bool],
    dmrs_port_configuration_by_cell: Mapping[str, Sequence[Mapping[str, object]]],
    full_cell_roster: Sequence[str],
) -> int:
    """Derive DCI 0_3 antenna-port width from PUSCH DMRS configuration."""
    if not isinstance(mode, str) or mode not in {"type1a", "type2"}:
        raise ValueError("antenna_ports_mode must be 'type1a' or 'type2'")
    if not isinstance(transform_precoding_enabled_by_cell, Mapping) or not isinstance(
        dmrs_port_configuration_by_cell, Mapping
    ):
        raise TypeError("antenna-port configuration inputs must be mappings")
    if set(transform_precoding_enabled_by_cell) != set(full_cell_roster):
        raise ValueError(
            "transform_precoding_enabled_by_cell must cover the full scheduled cell list"
        )
    if set(dmrs_port_configuration_by_cell) != set(full_cell_roster):
        raise ValueError("dmrs_port_configuration_by_cell must cover the full scheduled cell list")
    widths: dict[str, int] = {}
    for cell in full_cell_roster:
        transform_precoding = transform_precoding_enabled_by_cell[cell]
        if type(transform_precoding) is not bool:
            raise TypeError("transform-precoding configuration values must be booleans")
        configs = dmrs_port_configuration_by_cell[cell]
        if not isinstance(configs, Sequence) or isinstance(configs, (str, bytes)):
            raise TypeError(f"DMRS port configurations for {cell!r} must be a sequence")
        if not 1 <= len(configs) <= 2:
            raise ValueError(
                f"DMRS port configurations for {cell!r} must contain one or two mapping types"
            )
        mapping_widths: list[int] = []
        mapping_types: set[str] = set()
        for config in configs:
            if not isinstance(config, Mapping):
                raise TypeError("each DMRS port configuration must be a mapping")
            if set(config) != {
                "mapping_type",
                "dmrs_type",
                "dmrs_type_enhanced",
                "max_length",
            }:
                raise ValueError("DMRS port configuration has an unsupported field roster")
            mapping_type = config["mapping_type"]
            if not isinstance(mapping_type, str) or mapping_type not in {"A", "B"}:
                raise ValueError("DMRS mapping_type must be 'A' or 'B'")
            if mapping_type in mapping_types:
                raise ValueError("DMRS mapping types must be unique per cell")
            mapping_types.add(mapping_type)
            dmrs_type = config["dmrs_type"]
            enhanced = config["dmrs_type_enhanced"]
            max_length = config["max_length"]
            if isinstance(dmrs_type, bool) or not isinstance(dmrs_type, Integral):
                raise TypeError("dmrs_type must be integer 1 or 2")
            if int(dmrs_type) not in {1, 2}:
                raise ValueError("dmrs_type must be integer 1 or 2")
            if type(enhanced) is not bool:
                raise TypeError("dmrs_type_enhanced must be a boolean")
            if isinstance(max_length, bool) or not isinstance(max_length, Integral):
                raise TypeError("max_length must be integer 1 or 2")
            if int(max_length) not in {1, 2}:
                raise ValueError("max_length must be integer 1 or 2")
            dmrs_type_value = int(dmrs_type)
            max_length_value = int(max_length)
            if transform_precoding:
                if dmrs_type_value != 1:
                    raise ValueError(
                        "transform precoding requires DMRS type 1 for antenna-port width"
                    )
                width = 2 * max_length_value
            else:
                width = 2 + max_length_value + int(dmrs_type_value == 2) + int(enhanced)
            mapping_widths.append(width)
        widths[cell] = max(mapping_widths)
    if mode == "type1a":
        return max((widths[cell] for cell in cell_ids), default=0)
    return sum(widths[cell] for cell in cell_ids)


def _nr_srs_dci_0_3_precoding_width(
    *,
    mode: str,
    cell_ids: Sequence[str],
    tx_config_by_cell: Mapping[str, str],
    transform_precoding_enabled_by_cell: Mapping[str, bool],
    codebook_srs_antenna_port_counts_by_cell: Mapping[str, Sequence[int]],
    codebook_subset_by_cell: Mapping[str, str | None],
    max_rank_for_codebook_by_cell: Mapping[str, int | None],
    ul_full_power_transmission_by_cell: Mapping[str, str | None],
    full_cell_roster: Sequence[str],
) -> int:
    """Derive and aggregate codebook TPMI widths from decoded RRC state."""
    if not isinstance(mode, str) or mode not in {"type1a", "type2"}:
        raise ValueError("tpmi_mode must be 'type1a' or 'type2'")
    configuration_maps = (
        tx_config_by_cell,
        transform_precoding_enabled_by_cell,
        codebook_srs_antenna_port_counts_by_cell,
        codebook_subset_by_cell,
        max_rank_for_codebook_by_cell,
        ul_full_power_transmission_by_cell,
    )
    if any(not isinstance(roster, Mapping) for roster in configuration_maps):
        raise TypeError("precoding configuration inputs must be mappings")
    for name, roster in zip(
        (
            "tx_config_by_cell",
            "transform_precoding_enabled_by_cell",
            "codebook_srs_antenna_port_counts_by_cell",
            "codebook_subset_by_cell",
            "max_rank_for_codebook_by_cell",
            "ul_full_power_transmission_by_cell",
        ),
        configuration_maps,
        strict=True,
    ):
        if set(roster) != set(full_cell_roster):
            raise ValueError(f"{name} must cover the full scheduled cell list")
    allowed_subsets = {
        "fullyAndPartialAndNonCoherent",
        "partialAndNonCoherent",
        "nonCoherent",
    }
    allowed_full_power = {None, "fullpower", "fullpowerMode1", "fullpowerMode2"}
    widths: dict[str, int] = {}
    for cell in full_cell_roster:
        tx_config = tx_config_by_cell[cell]
        if not isinstance(tx_config, str) or tx_config not in {
            "codebook",
            "nonCodebook",
        }:
            raise ValueError(f"tx_config_by_cell[{cell!r}] must be 'codebook' or 'nonCodebook'")
        transform_precoding = transform_precoding_enabled_by_cell[cell]
        if type(transform_precoding) is not bool:
            raise TypeError("transform-precoding configuration values must be booleans")
        raw_ports = codebook_srs_antenna_port_counts_by_cell[cell]
        if not isinstance(raw_ports, Sequence) or isinstance(raw_ports, (str, bytes)):
            raise TypeError(f"codebook SRS antenna-port counts for {cell!r} must be a sequence")
        if tx_config == "nonCodebook":
            if (
                len(raw_ports) != 0
                or codebook_subset_by_cell[cell] is not None
                or max_rank_for_codebook_by_cell[cell] is not None
                or ul_full_power_transmission_by_cell[cell] is not None
            ):
                raise ValueError(
                    f"non-codebook cell {cell!r} must not have codebook-only configuration"
                )
            widths[cell] = 0
            continue
        if not raw_ports:
            raise ValueError(f"codebook cell {cell!r} must configure an SRS resource")
        ports: list[int] = []
        for raw_port_count in raw_ports:
            if isinstance(raw_port_count, bool) or not isinstance(raw_port_count, Integral):
                raise TypeError("codebook SRS antenna-port counts must be integers")
            port_count = int(raw_port_count)
            if port_count not in {1, 2, 4}:
                raise ValueError("DCI 0_3 codebook SRS antenna-port counts must be 1, 2, or 4")
            ports.append(port_count)
        port_count = max(ports)
        if port_count == 1:
            if (
                codebook_subset_by_cell[cell] is not None
                or max_rank_for_codebook_by_cell[cell] is not None
                or ul_full_power_transmission_by_cell[cell] is not None
            ):
                raise ValueError(
                    f"single-port codebook cell {cell!r} must not supply rank/subset/power mode"
                )
            widths[cell] = 0
            continue
        subset = codebook_subset_by_cell[cell]
        if not isinstance(subset, str) or subset not in allowed_subsets:
            raise ValueError(
                f"codebook subset for {cell!r} must be one of the supported RRC choices"
            )
        raw_rank = max_rank_for_codebook_by_cell[cell]
        if isinstance(raw_rank, bool) or not isinstance(raw_rank, Integral):
            raise TypeError(f"maximum codebook rank for {cell!r} must be an integer")
        rank = int(raw_rank)
        if not 1 <= rank <= port_count:
            raise ValueError(f"maximum codebook rank for {cell!r} must be in [1, {port_count}]")
        full_power = ul_full_power_transmission_by_cell[cell]
        if full_power not in allowed_full_power:
            raise ValueError(f"ul-FullPowerTransmission value for {cell!r} is invalid")
        mode1 = full_power == "fullpowerMode1"
        if port_count == 2:
            if mode1:
                if rank == 2 and not transform_precoding:
                    if subset != "nonCoherent":
                        raise ValueError(
                            "2-port fullpowerMode1 rank-2 codebook requires nonCoherent subset"
                        )
                    widths[cell] = 2
                else:
                    if subset != "nonCoherent":
                        raise ValueError(
                            "2-port fullpowerMode1 rank-1 codebook requires nonCoherent subset"
                        )
                    widths[cell] = 2
            elif rank == 1 or transform_precoding:
                if subset not in {"fullyAndPartialAndNonCoherent", "nonCoherent"}:
                    raise ValueError("2-port rank-1 codebook does not support partial-only subset")
                widths[cell] = 3 if subset == "fullyAndPartialAndNonCoherent" else 1
            else:
                if subset not in {"fullyAndPartialAndNonCoherent", "nonCoherent"}:
                    raise ValueError("2-port rank-2 codebook does not support partial-only subset")
                widths[cell] = 4 if subset == "fullyAndPartialAndNonCoherent" else 2
        else:
            if mode1:
                if rank == 1 or transform_precoding:
                    if subset not in {"partialAndNonCoherent", "nonCoherent"}:
                        raise ValueError("4-port fullpowerMode1 rank-1 subset is unsupported")
                    widths[cell] = 4 if subset == "partialAndNonCoherent" else 3
                elif rank == 2:
                    if subset not in {"partialAndNonCoherent", "nonCoherent"}:
                        raise ValueError("4-port fullpowerMode1 rank-2 subset is unsupported")
                    widths[cell] = 5 if subset == "partialAndNonCoherent" else 4
                else:
                    if subset not in {"partialAndNonCoherent", "nonCoherent"}:
                        raise ValueError("4-port fullpowerMode1 rank-3/4 subset is unsupported")
                    widths[cell] = 6 if subset == "partialAndNonCoherent" else 4
            elif rank == 1 or transform_precoding:
                widths[cell] = {
                    "fullyAndPartialAndNonCoherent": 5,
                    "partialAndNonCoherent": 4,
                    "nonCoherent": 2,
                }[subset]
            else:
                widths[cell] = {
                    "fullyAndPartialAndNonCoherent": 6,
                    "partialAndNonCoherent": 5,
                    "nonCoherent": 4,
                }[subset]
    if mode == "type1a":
        return max((widths[cell] for cell in cell_ids), default=0)
    return sum(widths[cell] for cell in cell_ids)


def _nr_srs_dci_0_3_sri_width(
    *,
    mode: str,
    cell_ids: Sequence[str],
    tx_config_by_cell: Mapping[str, str],
    srs_resource_count_by_cell: Mapping[str, int],
    max_noncodebook_layers_by_cell: Mapping[str, int | None],
    ue_supports_max_mimo_layers_by_cell: Mapping[str, bool],
    ue_max_noncodebook_layers_by_cell: Mapping[str, int | None] | None,
    full_cell_roster: Sequence[str],
) -> int:
    """Derive DCI 0_3 SRI width from configured PUSCH/SRS resource state.

    If a serving-cell ``maxMIMO-Layers`` value is absent for non-codebook
    operation, the corresponding UE-supported maximum must be supplied.
    """
    if not isinstance(mode, str) or mode not in {"type1a", "type2"}:
        raise ValueError("sri_mode must be 'type1a' or 'type2'")
    if any(
        not isinstance(roster, Mapping)
        for roster in (
            tx_config_by_cell,
            srs_resource_count_by_cell,
            max_noncodebook_layers_by_cell,
            ue_supports_max_mimo_layers_by_cell,
        )
    ):
        raise TypeError("SRI configuration inputs must be mappings")
    if ue_max_noncodebook_layers_by_cell is not None and not isinstance(
        ue_max_noncodebook_layers_by_cell, Mapping
    ):
        raise TypeError("UE non-codebook layer capabilities must be a mapping")
    widths: dict[str, int] = {}
    for name, roster in (
        ("tx_config_by_cell", tx_config_by_cell),
        ("srs_resource_count_by_cell", srs_resource_count_by_cell),
        ("max_noncodebook_layers_by_cell", max_noncodebook_layers_by_cell),
        ("ue_supports_max_mimo_layers_by_cell", ue_supports_max_mimo_layers_by_cell),
    ):
        if set(roster) != set(full_cell_roster):
            raise ValueError(f"{name} must cover the full scheduled cell list")
    if ue_max_noncodebook_layers_by_cell is not None and set(
        ue_max_noncodebook_layers_by_cell
    ) != set(full_cell_roster):
        raise ValueError(
            "ue_max_noncodebook_layers_by_cell must cover the full scheduled cell list"
        )
    for cell in full_cell_roster:
        tx_config = tx_config_by_cell[cell]
        if not isinstance(tx_config, str) or tx_config not in {
            "codebook",
            "nonCodebook",
        }:
            raise ValueError(f"tx_config_by_cell[{cell!r}] must be 'codebook' or 'nonCodebook'")
        supports_configured_max = ue_supports_max_mimo_layers_by_cell[cell]
        if type(supports_configured_max) is not bool:
            raise TypeError("ue_supports_max_mimo_layers_by_cell values must be booleans")
        raw_count = srs_resource_count_by_cell[cell]
        if isinstance(raw_count, bool) or not isinstance(raw_count, Integral):
            raise TypeError(f"SRS resource count for {cell!r} must be an integer")
        resource_count = int(raw_count)
        if not 1 <= resource_count <= 64:
            raise ValueError("SRS resource counts must be in [1, 64]")
        if tx_config == "codebook":
            widths[cell] = (resource_count - 1).bit_length()
            continue
        raw_layers = max_noncodebook_layers_by_cell[cell]
        if raw_layers is None or not supports_configured_max:
            if ue_max_noncodebook_layers_by_cell is None:
                raise ValueError(
                    "UE non-codebook layer capability is required when "
                    "maxMIMO-Layers is not configured or supported for "
                    f"{cell!r}"
                )
            raw_layers = ue_max_noncodebook_layers_by_cell[cell]
        if isinstance(raw_layers, bool) or not isinstance(raw_layers, Integral):
            raise TypeError(f"maximum non-codebook PUSCH layers for {cell!r} must be an integer")
        max_layers = int(raw_layers)
        if not 1 <= max_layers <= 8:
            raise ValueError("maximum non-codebook PUSCH layers must be in [1, 8]")
        selectable_combinations = sum(
            comb(resource_count, layer_count)
            for layer_count in range(1, min(max_layers, resource_count) + 1)
        )
        widths[cell] = (selectable_combinations - 1).bit_length()
    if mode == "type1a":
        return max((widths[cell] for cell in cell_ids), default=0)
    return sum(widths[cell] for cell in cell_ids)


def nr_srs_dci_0_3_frequency_assignment_widths(
    scheduled_cell_ids: Sequence[str],
    configured_bwp_ids_by_cell: Mapping[str, Sequence[int]],
    active_bwp_id_by_cell: Mapping[str, int],
    bwp_frequency_config_by_cell: Mapping[str, Mapping[int, Mapping[str, object]]],
) -> dict[str, int]:
    """Derive per-cell DCI 0_3 frequency-assignment widths from active UL BWPs.

    Regular allocation supports type 0, type 1, and dynamic switching. The
    interlace allocation type is supported for 15/30 kHz numerology when the
    decoded active-BWP configuration supplies its RB-set count.
    """
    cells = tuple(scheduled_cell_ids)
    if not cells or len(set(cells)) != len(cells):
        raise ValueError("scheduled_cell_ids must be nonempty and unique")
    if not isinstance(configured_bwp_ids_by_cell, Mapping) or not configured_bwp_ids_by_cell:
        raise ValueError("configured_bwp_ids_by_cell must be a nonempty mapping")
    full_cell_roster = set(configured_bwp_ids_by_cell)
    if not set(cells).issubset(full_cell_roster):
        raise ValueError("scheduled cells must be covered by the full BWP configuration roster")
    for name, value in (
        ("active_bwp_id_by_cell", active_bwp_id_by_cell),
        ("bwp_frequency_config_by_cell", bwp_frequency_config_by_cell),
    ):
        if not isinstance(value, Mapping) or set(value) != full_cell_roster:
            raise ValueError(f"{name} must exactly cover the full scheduled cell roster")

    rbg_sizes = {
        1: ((36, 2), (72, 4), (144, 8), (275, 16)),
        2: ((36, 4), (72, 8), (144, 16), (275, 16)),
        3: ((36, 8), (72, 16), (144, 32), (275, 32)),
    }
    widths: dict[str, int] = {}
    for cell in cells:
        raw_bwp_ids = configured_bwp_ids_by_cell[cell]
        if not isinstance(raw_bwp_ids, Sequence) or isinstance(raw_bwp_ids, (str, bytes)):
            raise TypeError(f"configured BWP IDs for {cell!r} must be an ordered sequence")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value <= 4
            for value in raw_bwp_ids
        ):
            raise ValueError(f"configured BWP IDs for {cell!r} must be integers in [0, 4]")
        bwp_ids = tuple(int(value) for value in raw_bwp_ids)
        if len(set(bwp_ids)) != len(bwp_ids) or not bwp_ids:
            raise ValueError(f"configured BWP IDs for {cell!r} must be nonempty and unique")
        raw_configs = bwp_frequency_config_by_cell[cell]
        if (
            not isinstance(raw_configs, Mapping)
            or any(isinstance(key, bool) or not isinstance(key, Integral) for key in raw_configs)
            or set(raw_configs) != set(bwp_ids)
        ):
            raise ValueError(f"BWP frequency configurations for {cell!r} must cover every BWP")
        active_id = active_bwp_id_by_cell[cell]
        if isinstance(active_id, bool) or not isinstance(active_id, Integral):
            raise TypeError(f"active BWP ID for {cell!r} must be an integer")
        if int(active_id) not in raw_configs:
            raise ValueError(f"active BWP {active_id!r} for {cell!r} is not configured")
        config = raw_configs[int(active_id)]
        if not isinstance(config, Mapping):
            raise TypeError(f"active BWP frequency configuration for {cell!r} must be a mapping")

        size_raw = config.get("size_prb")
        start_raw = config.get("start_prb")
        for label, value in (("size_prb", size_raw), ("start_prb", start_raw)):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{label} for {cell!r} must be an integer")
        size_prb = int(cast(Integral, size_raw))
        start_prb = int(cast(Integral, start_raw))
        if not 1 <= size_prb <= 275 or not 0 <= start_prb <= 274 or start_prb + size_prb > 275:
            raise ValueError(f"active BWP PRB extent for {cell!r} is outside supported bounds")

        interlace = config.get("use_interlace", False)
        if type(interlace) is not bool:
            raise TypeError(f"use_interlace for {cell!r} must be a boolean")
        if interlace:
            spacing = config.get("subcarrier_spacing_khz")
            if isinstance(spacing, bool) or not isinstance(spacing, Integral):
                raise TypeError(f"subcarrier_spacing_khz for {cell!r} must be an integer")
            base_width = {15: 6, 30: 5}.get(int(spacing))
            if base_width is None:
                raise ValueError("interlace DCI 0_3 supports 15 or 30 kHz spacing")
            rb_set_count = config.get("rb_set_count")
            if isinstance(rb_set_count, bool) or not isinstance(rb_set_count, Integral):
                raise TypeError(f"rb_set_count for {cell!r} must be an integer")
            rb_set_count = int(rb_set_count)
            if not 1 <= rb_set_count <= 275:
                raise ValueError(f"rb_set_count for {cell!r} must be in [1, 275]")
            widths[cell] = base_width + (rb_set_count - 1).bit_length()
            continue

        allocation = config.get("resource_allocation")
        if not isinstance(allocation, str) or allocation not in {
            "type0",
            "type1",
            "dynamicSwitch",
        }:
            raise ValueError(f"unsupported resource allocation mode for {cell!r}")
        hopping = config.get("frequency_hopping", False)
        if type(hopping) is not bool:
            raise TypeError(f"frequency_hopping for {cell!r} must be a boolean")
        if hopping:
            offset_count = config.get("frequency_hopping_offset_count")
            if isinstance(offset_count, bool) or not isinstance(offset_count, Integral):
                raise TypeError(f"frequency_hopping_offset_count for {cell!r} must be an integer")
            offset_count = int(offset_count)
            if offset_count not in {2, 4}:
                raise ValueError("frequency hopping offset list must contain 2 or 4 values")
            hopping_width = (offset_count - 1).bit_length()
        else:
            hopping_width = 0

        rbg_config_raw = config.get("rbg_size_config", 1)
        if isinstance(rbg_config_raw, str) and rbg_config_raw.startswith("config"):
            rbg_config_raw = rbg_config_raw.removeprefix("config")
        if isinstance(rbg_config_raw, bool) or not isinstance(rbg_config_raw, (Integral, str)):
            raise TypeError(f"rbg_size_config for {cell!r} must be config1, config2, or config3")
        try:
            rbg_config = int(rbg_config_raw)
        except ValueError as exc:
            raise ValueError("rbg_size_config must be config1, config2, or config3") from exc
        if rbg_config not in rbg_sizes:
            raise ValueError("rbg_size_config must be config1, config2, or config3")
        type0_p = next(
            limit_size for limit, limit_size in rbg_sizes[rbg_config] if size_prb <= limit
        )
        type0_count = (size_prb + start_prb % type0_p + type0_p - 1) // type0_p
        type0_width = type0_count

        granularity_raw = config.get("type1_granularity_prb", 1)
        if isinstance(granularity_raw, bool) or not isinstance(granularity_raw, Integral):
            raise TypeError(f"type1_granularity_prb for {cell!r} must be an integer")
        granularity = int(granularity_raw)
        if granularity not in {1, 2, 4, 8}:
            raise ValueError("type1_granularity_prb must be one of 1, 2, 4, or 8")
        type1_count = (size_prb + start_prb % granularity + granularity - 1) // granularity
        type1_width = (type1_count * (type1_count + 1) // 2 - 1).bit_length()
        if allocation == "type0":
            widths[cell] = type0_width
        elif allocation == "type1":
            widths[cell] = type1_width + hopping_width
        else:
            widths[cell] = 1 + max(type0_width, type1_width + hopping_width)
    return widths


def _nr_srs_dci_0_3_pdcch_monitoring_adaptation_width(
    *,
    enabled: bool,
    skipping_duration_count: int | None,
    search_space_group_ids: Sequence[int] | None,
) -> int:
    """Derive the Release 19 DCI 0_3 PDCCH monitoring adaptation width."""
    if type(enabled) is not bool:
        raise TypeError("pdcch_monitoring_adaptation_enabled must be a boolean")
    if skipping_duration_count is not None:
        if isinstance(skipping_duration_count, bool) or not isinstance(
            skipping_duration_count, Integral
        ):
            raise TypeError("pdcch_skipping_duration_count must be an integer or None")
        skipping_duration_count = int(skipping_duration_count)
        if not 1 <= skipping_duration_count <= 64:
            raise ValueError("pdcch_skipping_duration_count must be in [1, 64]")
    if search_space_group_ids is not None:
        if not isinstance(search_space_group_ids, Sequence) or isinstance(
            search_space_group_ids, (str, bytes)
        ):
            raise TypeError("search_space_group_ids must be an ordered sequence or None")
        if any(
            isinstance(group_id, bool)
            or not isinstance(group_id, Integral)
            or int(group_id) not in {0, 1, 2}
            for group_id in search_space_group_ids
        ):
            raise ValueError("search_space_group_ids must contain only IDs 0, 1, and 2")
        group_ids = tuple(int(group_id) for group_id in search_space_group_ids)
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("search_space_group_ids must not contain duplicates")
        search_space_group_ids = group_ids
    if not enabled:
        return 0
    if skipping_duration_count is not None and search_space_group_ids is not None:
        return 2
    if skipping_duration_count is not None:
        return 1 if skipping_duration_count == 1 else 2
    if search_space_group_ids is not None:
        groups = set(search_space_group_ids)
        if groups == {0, 1}:
            return 1
        if groups == {0, 1, 2}:
            return 2
        raise ValueError(
            "enabled search-space group adaptation requires group IDs {0, 1} or {0, 1, 2}"
        )
    raise ValueError(
        "enabled PDCCH monitoring adaptation requires a skipping-duration list, "
        "a search-space group list, or both"
    )


def _nr_srs_dci_0_3_scell_dormancy_width(
    *,
    dormancy_configured: bool,
    dormancy_group_ids: Sequence[int] | None,
    carried_on_primary_cell_within_drx_active_time: bool,
    scell_has_multiple_dl_bwps: bool,
) -> int:
    """Derive the contextual SCell dormancy bitmap width for DCI 0_3."""
    for name, value in (
        ("dormancy_configured", dormancy_configured),
        (
            "carried_on_primary_cell_within_drx_active_time",
            carried_on_primary_cell_within_drx_active_time,
        ),
        ("scell_has_multiple_dl_bwps", scell_has_multiple_dl_bwps),
    ):
        if type(value) is not bool:
            raise TypeError(f"{name} must be a boolean")
    if dormancy_group_ids is None:
        if (
            dormancy_configured
            and carried_on_primary_cell_within_drx_active_time
            and scell_has_multiple_dl_bwps
        ):
            raise ValueError(
                "applicable SCell dormancy indication requires configured dormancy group IDs"
            )
        return 0
    if not isinstance(dormancy_group_ids, Sequence) or isinstance(dormancy_group_ids, (str, bytes)):
        raise TypeError("dormancy_group_ids must be an ordered sequence or None")
    if any(
        isinstance(group_id, bool) or not isinstance(group_id, Integral) or int(group_id) < 0
        for group_id in dormancy_group_ids
    ):
        raise ValueError("dormancy_group_ids must contain nonnegative integers")
    groups = tuple(int(group_id) for group_id in dormancy_group_ids)
    if len(set(groups)) != len(groups):
        raise ValueError("dormancy_group_ids must be unique")
    if not dormancy_configured:
        return 0
    if not carried_on_primary_cell_within_drx_active_time or not scell_has_multiple_dl_bwps:
        return 0
    if not 1 <= len(groups) <= 5:
        raise ValueError("applicable dormancy_group_ids must contain 1..5 groups")
    return len(groups)
