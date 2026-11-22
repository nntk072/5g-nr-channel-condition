"""NR SRS DCI 0_3 multicell aperiodic scheduling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any, cast

from .srs_bwp_schedules import (
    NRSRSBWPCellPeriodicResourceSetSchedules,
    NRSRSBWPCellResourceSetSchedules,
    NRSRSBWPPeriodicResourceSetSchedule,
    nr_srs_multicell_aperiodic_resource_set_schedules,
    nr_srs_periodic_resource_set_schedule,
)
from .srs_primitives import (
    NRSRSBWPFrequencyBounds,
    nr_srs_dci_0_3_request_list_index_from_bits,
    nr_srs_dci_0_3_scheduled_cell_ids,
    nr_srs_dci_0_3_ul_bwp_selection,
)
from .srs_resource_schedules import (
    NRSRSResourceSetAperiodicSchedule,
)


def nr_srs_dci_0_3_multicell_aperiodic_resource_set_schedules(
    request_list_index: int,
    request_list: Sequence[Sequence[int]],
    serving_cell_ids: Sequence[str],
    resource_sets_by_cell: Mapping[str, Sequence[Mapping[str, object]]],
    triggering_dci_slot_by_cell: Mapping[str, int],
    *,
    cell_set_configs: Sequence[Mapping[str, object]],
    scheduled_cell_set_indicator: int | None = None,
    scheduled_cells_indicator: int | None = None,
    default_set_id: int | None = None,
    offset_combo_index: int | None = None,
    offset_combos: Sequence[Sequence[int]] | None = None,
    offset_combo_cell_ids: Sequence[str] = (),
    supplementary_uplink_cells: Sequence[str] = (),
) -> tuple[tuple[str, tuple[NRSRSResourceSetAperiodicSchedule, ...]], ...]:
    """Resolve DCI 0_3 cell indicators and compose per-cell SRS schedules.

    The request-list row and resource-set mapping are scoped to the selected
    RRC cell set, as required by ``SRS-RequestCombo``. ``serving_cell_ids`` may
    be the wider serving-cell roster; its order is used to verify that the
    selected set's RRC cell list is in ascending serving-cell order.
    """
    scheduled_cells = nr_srs_dci_0_3_scheduled_cell_ids(
        cell_set_configs,
        scheduled_cell_set_indicator=scheduled_cell_set_indicator,
        scheduled_cells_indicator=scheduled_cells_indicator,
        default_set_id=default_set_id,
    )
    ordered_configs = sorted(cell_set_configs, key=lambda item: int(cast(Integral, item["set_id"])))
    if scheduled_cell_set_indicator is not None:
        set_index = int(scheduled_cell_set_indicator)
    elif default_set_id is not None:
        set_index = next(
            index
            for index, item in enumerate(ordered_configs)
            if int(cast(Integral, item["set_id"])) == int(default_set_id)
        )
    else:
        set_index = 0
    set_cells = tuple(cast(Sequence[str], ordered_configs[set_index]["scheduled_cell_ids"]))
    if not isinstance(serving_cell_ids, Sequence) or isinstance(serving_cell_ids, (str, bytes)):
        raise TypeError("serving_cell_ids must be an ordered sequence")
    global_cells = tuple(serving_cell_ids)
    if not global_cells or any(not isinstance(cell, str) or not cell for cell in global_cells):
        raise ValueError("serving_cell_ids must contain nonempty cell IDs")
    if len(set(global_cells)) != len(global_cells):
        raise ValueError("serving_cell_ids must be unique")
    if tuple(cell for cell in global_cells if cell in set(set_cells)) != set_cells:
        raise ValueError("selected set scheduled_cell_ids must follow serving_cell_ids order")
    if not isinstance(supplementary_uplink_cells, Sequence) or isinstance(
        supplementary_uplink_cells, (str, bytes)
    ):
        raise TypeError("supplementary_uplink_cells must be an ordered sequence")
    sul_cells = tuple(supplementary_uplink_cells)
    if any(not isinstance(cell, str) or not cell for cell in sul_cells):
        raise ValueError("supplementary_uplink_cells must contain nonempty cell IDs")
    if len(set(sul_cells)) != len(sul_cells) or not set(sul_cells).issubset(global_cells):
        raise ValueError("supplementary_uplink_cells must be a unique serving-cell subset")
    if not isinstance(resource_sets_by_cell, Mapping):
        raise TypeError("resource_sets_by_cell must be a mapping")
    if not set(set_cells).issubset(resource_sets_by_cell):
        raise ValueError("resource_sets_by_cell must include every cell in selected set")
    set_resource_sets = {cell: resource_sets_by_cell[cell] for cell in set_cells}
    return nr_srs_multicell_aperiodic_resource_set_schedules(
        request_list_index,
        request_list,
        set_cells,
        set_resource_sets,
        triggering_dci_slot_by_cell,
        scheduled_cell_ids=scheduled_cells,
        offset_combo_index=offset_combo_index,
        offset_combos=offset_combos,
        offset_combo_cell_ids=offset_combo_cell_ids,
        supplementary_uplink_cells=tuple(cell for cell in sul_cells if cell in set(set_cells)),
    )


def nr_srs_dci_0_3_multicell_aperiodic_resource_set_schedules_from_bits(
    payload_bits: Sequence[int | bool],
    *,
    field_offset: int,
    request_list: Sequence[Sequence[int]],
    serving_cell_ids: Sequence[str],
    resource_sets_by_cell: Mapping[str, Sequence[Mapping[str, object]]],
    triggering_dci_slot_by_cell: Mapping[str, int],
    cell_set_configs: Sequence[Mapping[str, object]],
    scheduled_cell_set_indicator: int | None = None,
    scheduled_cells_indicator: int | None = None,
    default_set_id: int | None = None,
    offset_combo_index: int | None = None,
    offset_combos: Sequence[Sequence[int]] | None = None,
    offset_combo_cell_ids: Sequence[str] = (),
    supplementary_uplink_cells: Sequence[str] = (),
) -> tuple[tuple[str, tuple[NRSRSResourceSetAperiodicSchedule, ...]], ...] | None:
    """Decode the DCI 0_3 request-list selector and compose its SRS schedules.

    ``field_offset`` is the caller-resolved bit offset in an already-decoded
    DCI payload. The complete DCI field layout and ASN.1 RRC configuration are
    outside this adapter. If no request-list field is configured, return
    ``None`` because there is no selector or aperiodic request to resolve.
    """
    if not isinstance(request_list, Sequence) or isinstance(request_list, (str, bytes)):
        raise TypeError("request_list must be a sequence of per-cell request entries")
    request_list_index = nr_srs_dci_0_3_request_list_index_from_bits(
        payload_bits,
        field_offset=field_offset,
        request_list_entry_count=len(request_list),
    )
    if request_list_index is None:
        return None
    return nr_srs_dci_0_3_multicell_aperiodic_resource_set_schedules(
        request_list_index,
        request_list,
        serving_cell_ids,
        resource_sets_by_cell,
        triggering_dci_slot_by_cell,
        cell_set_configs=cell_set_configs,
        scheduled_cell_set_indicator=scheduled_cell_set_indicator,
        scheduled_cells_indicator=scheduled_cells_indicator,
        default_set_id=default_set_id,
        offset_combo_index=offset_combo_index,
        offset_combos=offset_combos,
        offset_combo_cell_ids=offset_combo_cell_ids,
        supplementary_uplink_cells=supplementary_uplink_cells,
    )


def nr_srs_dci_0_3_multicell_aperiodic_resource_set_schedules_by_bwp(
    request_list_index: int,
    request_list: Sequence[Sequence[int]],
    serving_cell_ids: Sequence[str],
    resource_sets_by_cell_and_bwp: Mapping[str, Mapping[int, Sequence[Mapping[str, object]]]],
    triggering_dci_slot_by_cell: Mapping[str, int],
    *,
    cell_set_configs: Sequence[Mapping[str, object]],
    configured_ul_bwp_ids_by_cell: Mapping[str, Sequence[int]],
    active_ul_bwp_id_by_cell: Mapping[str, int],
    ul_bwp_frequency_bounds_by_cell_and_bwp: Mapping[str, Mapping[int, NRSRSBWPFrequencyBounds]],
    bwp_indicator: int | None,
    supports_active_bwp_change: bool = True,
    scheduled_cell_set_indicator: int | None = None,
    scheduled_cells_indicator: int | None = None,
    default_set_id: int | None = None,
    offset_combo_index: int | None = None,
    offset_combos: Sequence[Sequence[int]] | None = None,
    offset_combo_cell_ids: Sequence[str] = (),
    supplementary_uplink_cells: Sequence[str] = (),
) -> tuple[NRSRSBWPCellResourceSetSchedules, ...]:
    """Resolve DCI 0_3 cell/BWP choices and schedule SRS on those BWPs.

    Per-cell BWP resource-set tables cover every configured UL BWP in the
    selected cell set. Offset-combo cell membership is derived across those
    BWPs from aperiodic resource sets with multiple available slot offsets;
    a supplied roster, when nonempty, must match that derived RRC order.
    Caller-supplied frequency bounds are required for every configured UL BWP;
    each returned allocation must fit entirely inside its selected BWP in the
    same subcarrier coordinate system. Returned rows retain the selected BWP
    and its bounds even when no SRS set was triggered.
    """
    ordered_configs = sorted(cell_set_configs, key=lambda item: int(cast(Integral, item["set_id"])))
    if scheduled_cell_set_indicator is not None:
        set_index = int(scheduled_cell_set_indicator)
    elif default_set_id is not None:
        set_index = next(
            index
            for index, item in enumerate(ordered_configs)
            if int(cast(Integral, item["set_id"])) == int(default_set_id)
        )
    else:
        set_index = 0
    set_cells = tuple(cast(Sequence[str], ordered_configs[set_index]["scheduled_cell_ids"]))

    bwp_selection = nr_srs_dci_0_3_ul_bwp_selection(
        set_cells,
        configured_ul_bwp_ids_by_cell,
        active_ul_bwp_id_by_cell,
        bwp_indicator=bwp_indicator,
        supports_active_bwp_change=supports_active_bwp_change,
    )
    selected_bwp_by_cell = dict(bwp_selection.selected_bwp_id_by_cell)
    if not isinstance(ul_bwp_frequency_bounds_by_cell_and_bwp, Mapping):
        raise TypeError("ul_bwp_frequency_bounds_by_cell_and_bwp must be a mapping")
    if set(ul_bwp_frequency_bounds_by_cell_and_bwp) != set(set_cells):
        raise ValueError("UL BWP frequency-bound keys must exactly match the selected cell set")
    bounds_by_cell: dict[str, dict[int, NRSRSBWPFrequencyBounds]] = {}
    for cell in set_cells:
        raw_bounds = ul_bwp_frequency_bounds_by_cell_and_bwp[cell]
        if not isinstance(raw_bounds, Mapping):
            raise TypeError(f"UL BWP frequency bounds for {cell!r} must be a mapping")
        configured_bwps = tuple(
            sorted(int(bwp_id) for bwp_id in configured_ul_bwp_ids_by_cell[cell])
        )
        if any(
            isinstance(bwp_id, bool) or not isinstance(bwp_id, Integral) for bwp_id in raw_bounds
        ):
            raise TypeError(f"UL BWP frequency-bound IDs for {cell!r} must be integers")
        if set(raw_bounds) != set(configured_bwps):
            raise ValueError(
                f"UL BWP frequency-bound IDs for {cell!r} must match configured UL BWPs"
            )
        normalized_bounds: dict[int, NRSRSBWPFrequencyBounds] = {}
        for bwp_id, bounds in raw_bounds.items():
            if not isinstance(bounds, NRSRSBWPFrequencyBounds):
                raise TypeError(
                    f"UL BWP frequency bound for {cell!r}/BWP {bwp_id} has invalid type"
                )
            normalized_bounds[int(bwp_id)] = bounds
        bounds_by_cell[cell] = normalized_bounds
    if not isinstance(resource_sets_by_cell_and_bwp, Mapping):
        raise TypeError("resource_sets_by_cell_and_bwp must be a mapping")
    if set(resource_sets_by_cell_and_bwp) != set(set_cells):
        raise ValueError(
            "resource_sets_by_cell_and_bwp keys must exactly match the selected cell set"
        )

    selected_resource_sets: dict[str, Sequence[Mapping[str, object]]] = {}
    derived_offset_cells: list[str] = []
    for cell in set_cells:
        configured_bwps = tuple(
            sorted(int(bwp_id) for bwp_id in configured_ul_bwp_ids_by_cell[cell])
        )
        by_bwp = resource_sets_by_cell_and_bwp[cell]
        if not isinstance(by_bwp, Mapping):
            raise TypeError(f"per-BWP resource-set table for {cell!r} must be a mapping")
        if any(isinstance(bwp_id, bool) or not isinstance(bwp_id, Integral) for bwp_id in by_bwp):
            raise TypeError(f"per-BWP resource-set keys for {cell!r} must be integer IDs")
        if set(by_bwp) != set(configured_bwps):
            raise ValueError(
                f"per-BWP resource-set keys for {cell!r} must match configured UL BWPs"
            )
        for bwp_id in configured_bwps:
            records = by_bwp[bwp_id]
            if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
                raise TypeError(f"resource sets for {cell!r}/BWP {bwp_id} must be a sequence")
            for record in records:
                if not isinstance(record, Mapping):
                    raise TypeError("each per-BWP resource-set record must be a mapping")
                if record.get("resource_type") != "aperiodic":
                    continue
                allocation = record.get("allocation")
                if not isinstance(allocation, Mapping):
                    raise ValueError(
                        f"aperiodic resource set for {cell!r}/BWP {bwp_id} "
                        "requires an allocation mapping"
                    )
                available = allocation.get("available_slot_offsets")
                if available is None:
                    continue
                if not isinstance(available, Sequence) or isinstance(available, (str, bytes)):
                    raise TypeError("available_slot_offsets must be an ordered sequence")
                if not available:
                    raise ValueError("available_slot_offsets must not be empty")
                if len(available) > 1 and cell not in derived_offset_cells:
                    derived_offset_cells.append(cell)
        selected_resource_sets[cell] = by_bwp[selected_bwp_by_cell[cell]]

    derived_offset_roster = tuple(derived_offset_cells)
    if offset_combo_cell_ids and tuple(offset_combo_cell_ids) != derived_offset_roster:
        raise ValueError("offset_combo_cell_ids must match BWP-derived RRC offset eligibility")
    schedules = nr_srs_dci_0_3_multicell_aperiodic_resource_set_schedules(
        request_list_index,
        request_list,
        serving_cell_ids,
        selected_resource_sets,
        triggering_dci_slot_by_cell,
        cell_set_configs=cell_set_configs,
        scheduled_cell_set_indicator=scheduled_cell_set_indicator,
        scheduled_cells_indicator=scheduled_cells_indicator,
        default_set_id=default_set_id,
        offset_combo_index=offset_combo_index,
        offset_combos=offset_combos,
        offset_combo_cell_ids=derived_offset_roster,
        supplementary_uplink_cells=supplementary_uplink_cells,
    )
    return tuple(
        NRSRSBWPCellResourceSetSchedules(
            cell_id=cell,
            ul_bwp_id=selected_bwp_by_cell[cell],
            frequency_bounds=bounds_by_cell[cell][selected_bwp_by_cell[cell]],
            resource_set_schedules=cell_schedules,
        )
        for cell, cell_schedules in schedules
    )


def nr_srs_periodic_resource_set_schedule_by_bwp(
    absolute_slot_index: int,
    resource_set_id: int,
    resources: Sequence[Mapping[str, object]],
    *,
    cell_id: str,
    ul_bwp_id: int,
    frequency_bounds: NRSRSBWPFrequencyBounds,
    resource_type: str = "periodic",
    periodicity_slots: int | None = None,
    slot_offset: int | None = None,
    periodicity_and_offset: Mapping[str, object] | None = None,
    periodicity_and_offset_ext: Mapping[str, object] | None = None,
    positioning_resource: bool = False,
    numerology: int | None = None,
    hyper_frame_number: int | None = None,
    hyper_sfn_index: int | None = None,
) -> NRSRSBWPPeriodicResourceSetSchedule | None:
    """Resolve a periodic SRS set and enforce its selected UL BWP extent.

    The BWP is caller-selected from active higher-layer state. This function
    validates every active per-symbol frequency allocation against that
    BWP's absolute subcarrier bounds. An off-occasion set returns ``None``.
    """
    if not isinstance(frequency_bounds, NRSRSBWPFrequencyBounds):
        raise TypeError("frequency_bounds must be NRSRSBWPFrequencyBounds")
    schedule = nr_srs_periodic_resource_set_schedule(
        absolute_slot_index,
        resource_set_id,
        resources,
        resource_type=resource_type,
        periodicity_slots=periodicity_slots,
        slot_offset=slot_offset,
        periodicity_and_offset=periodicity_and_offset,
        periodicity_and_offset_ext=periodicity_and_offset_ext,
        positioning_resource=positioning_resource,
        numerology=numerology,
        hyper_frame_number=hyper_frame_number,
        hyper_sfn_index=hyper_sfn_index,
    )
    if schedule is None:
        return None
    return NRSRSBWPPeriodicResourceSetSchedule(
        cell_id=cell_id,
        ul_bwp_id=ul_bwp_id,
        frequency_bounds=frequency_bounds,
        resource_set_schedule=schedule,
    )


def nr_srs_multicell_periodic_resource_set_schedules_by_bwp(
    serving_cell_ids: Sequence[str],
    resource_sets_by_cell: Mapping[str, Sequence[Mapping[str, object]]],
    absolute_slot_index_by_cell: Mapping[str, int],
    active_ul_bwp_id_by_cell: Mapping[str, int],
    frequency_bounds_by_cell: Mapping[str, NRSRSBWPFrequencyBounds],
) -> tuple[NRSRSBWPCellPeriodicResourceSetSchedules, ...]:
    """Resolve periodic SRS sets for a frozen multi-cell roster.

    Each cell input contains the resource sets configured on its already
    selected active UL BWP. Set records have ``resource_set_id``,
    ``resource_type``, and ``resources`` fields, with optional common
    ``timing`` defaults. Per-resource timing can override those defaults.
    The output preserves serving-cell order and active-BWP identity, including
    cells with no SRS occasion in the supplied slot.
    """
    if not isinstance(serving_cell_ids, Sequence) or isinstance(serving_cell_ids, (str, bytes)):
        raise TypeError("serving_cell_ids must be an ordered sequence")
    cells = tuple(serving_cell_ids)
    if not cells or any(not isinstance(cell, str) or not cell.strip() for cell in cells):
        raise ValueError("serving_cell_ids must contain nonempty cell IDs")
    cells = tuple(cell.strip() for cell in cells)
    if len(set(cells)) != len(cells):
        raise ValueError("serving_cell_ids must be unique")
    input_maps = {
        "resource_sets_by_cell": resource_sets_by_cell,
        "absolute_slot_index_by_cell": absolute_slot_index_by_cell,
        "active_ul_bwp_id_by_cell": active_ul_bwp_id_by_cell,
        "frequency_bounds_by_cell": frequency_bounds_by_cell,
    }
    for name, values in input_maps.items():
        if not isinstance(values, Mapping):
            raise TypeError(f"{name} must be a mapping")
        if set(values) != set(cells):
            raise ValueError(f"{name} keys must exactly match serving_cell_ids")

    timing_fields = frozenset(
        {
            "periodicity_slots",
            "slot_offset",
            "periodicity_and_offset",
            "periodicity_and_offset_ext",
            "positioning_resource",
            "numerology",
            "hyper_frame_number",
            "hyper_sfn_index",
        }
    )
    output: list[NRSRSBWPCellPeriodicResourceSetSchedules] = []
    for cell in cells:
        raw_slot = absolute_slot_index_by_cell[cell]
        if isinstance(raw_slot, bool) or not isinstance(raw_slot, Integral):
            raise TypeError(f"absolute slot for {cell!r} must be an integer")
        if raw_slot < 0:
            raise ValueError(f"absolute slot for {cell!r} must be nonnegative")
        bwp_id = active_ul_bwp_id_by_cell[cell]
        if isinstance(bwp_id, bool) or not isinstance(bwp_id, Integral):
            raise TypeError(f"active UL BWP ID for {cell!r} must be an integer")
        if not 0 <= bwp_id <= 4:
            raise ValueError(f"active UL BWP ID for {cell!r} must be in [0, 4]")
        bounds = frequency_bounds_by_cell[cell]
        if not isinstance(bounds, NRSRSBWPFrequencyBounds):
            raise TypeError(f"frequency bounds for {cell!r} have an invalid type")
        raw_sets = resource_sets_by_cell[cell]
        if not isinstance(raw_sets, Sequence) or isinstance(raw_sets, (str, bytes)):
            raise TypeError(f"resource sets for {cell!r} must be an ordered sequence")
        seen_set_ids: set[int] = set()
        schedules: list[NRSRSBWPPeriodicResourceSetSchedule] = []
        for index, resource_set in enumerate(raw_sets):
            if not isinstance(resource_set, Mapping):
                raise TypeError(f"resource_sets_by_cell[{cell!r}][{index}] must be a mapping")
            if any(not isinstance(key, str) for key in resource_set):
                raise ValueError("SRS resource-set keys must be strings")
            allowed_set_fields = {"resource_set_id", "resource_type", "resources", "timing"}
            unsupported = set(resource_set) - allowed_set_fields
            missing = {"resource_set_id", "resource_type", "resources"} - set(resource_set)
            if unsupported or missing:
                raise ValueError(
                    f"SRS resource-set fields are invalid; missing={sorted(missing)}, "
                    f"unsupported={sorted(unsupported)}"
                )
            resource_set_id = resource_set["resource_set_id"]
            if isinstance(resource_set_id, bool) or not isinstance(resource_set_id, Integral):
                raise TypeError("resource_set_id must be an integer")
            if resource_set_id < 0 or int(resource_set_id) in seen_set_ids:
                raise ValueError("resource_set_ids must be unique nonnegative integers per cell")
            seen_set_ids.add(int(resource_set_id))
            timing = resource_set.get("timing", {})
            if not isinstance(timing, Mapping):
                raise TypeError("resource-set timing must be a mapping")
            if any(not isinstance(key, str) for key in timing):
                raise ValueError("resource-set timing keys must be strings")
            unsupported_timing = set(timing) - timing_fields
            if unsupported_timing:
                raise ValueError(
                    f"resource-set timing has unsupported fields: {sorted(unsupported_timing)}"
                )
            scheduled = nr_srs_periodic_resource_set_schedule_by_bwp(
                int(raw_slot),
                int(resource_set_id),
                cast(Sequence[Mapping[str, object]], resource_set["resources"]),
                cell_id=cell,
                ul_bwp_id=int(bwp_id),
                frequency_bounds=bounds,
                resource_type=cast(str, resource_set["resource_type"]),
                **cast(Any, timing),
            )
            if scheduled is not None:
                schedules.append(scheduled)
        output.append(
            NRSRSBWPCellPeriodicResourceSetSchedules(
                cell_id=cell,
                ul_bwp_id=int(bwp_id),
                frequency_bounds=bounds,
                resource_set_schedules=tuple(schedules),
            )
        )
    return tuple(output)
