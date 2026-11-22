"""NR SRS bandwidth-part and cell-level schedules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, cast

import numpy as np

from .srs_configuration import (
    nr_srs_multicell_request_resource_set_ids,
    nr_srs_port_cyclic_shift_indices,
)
from .srs_primitives import (
    NRSRSBWPFrequencyBounds,
    nr_srs_aperiodic_target_slot_index,
    nr_srs_triggered_resource_set_ids,
)
from .srs_resource_schedules import (
    NRSRSAperiodicResourceSchedule,
    NRSRSPeriodicResourceSetSchedule,
    NRSRSResourceSetAperiodicSchedule,
    nr_srs_port_sequences_type1,
    nr_srs_resource_frequency_allocations,
)


@dataclass(frozen=True, slots=True)
class NRSRSBWPCellResourceSetSchedules:
    """A cell's selected UL BWP, extent, and feasible aperiodic SRS schedules."""

    cell_id: str
    ul_bwp_id: int
    frequency_bounds: NRSRSBWPFrequencyBounds
    resource_set_schedules: tuple[NRSRSResourceSetAperiodicSchedule, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cell_id, str) or not self.cell_id:
            raise ValueError("cell_id must be a nonempty string")
        if (
            isinstance(self.ul_bwp_id, bool)
            or not isinstance(self.ul_bwp_id, Integral)
            or not 0 <= self.ul_bwp_id <= 4
        ):
            raise ValueError("ul_bwp_id must be in [0, 4]")
        if not isinstance(self.frequency_bounds, NRSRSBWPFrequencyBounds):
            raise TypeError("frequency_bounds must be NRSRSBWPFrequencyBounds")
        schedules = tuple(self.resource_set_schedules)
        if any(not isinstance(item, NRSRSResourceSetAperiodicSchedule) for item in schedules):
            raise ValueError("resource_set_schedules must contain SRS set schedules")
        for schedule in schedules:
            for allocation in schedule.allocations:
                allocation_end = (
                    allocation.start_subcarrier
                    + allocation.bandwidth.allocated_bandwidth_subcarriers
                )
                if (
                    allocation.start_subcarrier < self.frequency_bounds.start_subcarrier
                    or allocation_end > self.frequency_bounds.end_subcarrier_exclusive
                ):
                    raise ValueError(
                        f"SRS allocation for {self.cell_id!r} resource set "
                        f"{schedule.resource_set_id} lies outside selected UL BWP "
                        f"{self.ul_bwp_id}"
                    )
        object.__setattr__(self, "ul_bwp_id", int(self.ul_bwp_id))
        object.__setattr__(self, "resource_set_schedules", schedules)


@dataclass(frozen=True, slots=True)
class NRSRSBWPPeriodicResourceSetSchedule:
    """An active periodic SRS set proven to fit inside one selected UL BWP."""

    cell_id: str
    ul_bwp_id: int
    frequency_bounds: NRSRSBWPFrequencyBounds
    resource_set_schedule: NRSRSPeriodicResourceSetSchedule

    def __post_init__(self) -> None:
        if not isinstance(self.cell_id, str) or not self.cell_id.strip():
            raise ValueError("cell_id must be a nonempty string")
        if (
            isinstance(self.ul_bwp_id, bool)
            or not isinstance(self.ul_bwp_id, Integral)
            or not 0 <= self.ul_bwp_id <= 4
        ):
            raise ValueError("ul_bwp_id must be in [0, 4]")
        if not isinstance(self.frequency_bounds, NRSRSBWPFrequencyBounds):
            raise TypeError("frequency_bounds must be NRSRSBWPFrequencyBounds")
        if not isinstance(self.resource_set_schedule, NRSRSPeriodicResourceSetSchedule):
            raise TypeError("resource_set_schedule must be NRSRSPeriodicResourceSetSchedule")
        for resource_id, schedule in self.resource_set_schedule.resources:
            for allocation in schedule.allocations:
                allocation_end = (
                    allocation.start_subcarrier
                    + allocation.bandwidth.allocated_bandwidth_subcarriers
                )
                if (
                    allocation.start_subcarrier < self.frequency_bounds.start_subcarrier
                    or allocation_end > self.frequency_bounds.end_subcarrier_exclusive
                ):
                    raise ValueError(
                        f"SRS allocation for {self.cell_id!r} resource {resource_id} "
                        f"lies outside selected UL BWP {self.ul_bwp_id}"
                    )
        object.__setattr__(self, "cell_id", self.cell_id.strip())
        object.__setattr__(self, "ul_bwp_id", int(self.ul_bwp_id))


@dataclass(frozen=True, slots=True)
class NRSRSBWPCellPeriodicResourceSetSchedules:
    """A cell's active-BWP context and its active periodic SRS resource sets."""

    cell_id: str
    ul_bwp_id: int
    frequency_bounds: NRSRSBWPFrequencyBounds
    resource_set_schedules: tuple[NRSRSBWPPeriodicResourceSetSchedule, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cell_id, str) or not self.cell_id.strip():
            raise ValueError("cell_id must be a nonempty string")
        if (
            isinstance(self.ul_bwp_id, bool)
            or not isinstance(self.ul_bwp_id, Integral)
            or not 0 <= self.ul_bwp_id <= 4
        ):
            raise ValueError("ul_bwp_id must be in [0, 4]")
        if not isinstance(self.frequency_bounds, NRSRSBWPFrequencyBounds):
            raise TypeError("frequency_bounds must be NRSRSBWPFrequencyBounds")
        schedules = tuple(self.resource_set_schedules)
        if any(not isinstance(item, NRSRSBWPPeriodicResourceSetSchedule) for item in schedules):
            raise TypeError("resource_set_schedules must contain BWP-validated periodic sets")
        if any(
            item.cell_id != self.cell_id.strip()
            or item.ul_bwp_id != int(self.ul_bwp_id)
            or item.frequency_bounds != self.frequency_bounds
            for item in schedules
        ):
            raise ValueError("periodic resource sets must share the cell's active-BWP binding")
        resource_set_ids = tuple(item.resource_set_schedule.resource_set_id for item in schedules)
        if len(set(resource_set_ids)) != len(resource_set_ids):
            raise ValueError("periodic SRS resource-set IDs must be unique within a cell")
        object.__setattr__(self, "cell_id", self.cell_id.strip())
        object.__setattr__(self, "ul_bwp_id", int(self.ul_bwp_id))
        object.__setattr__(self, "resource_set_schedules", schedules)


def nr_srs_multicell_aperiodic_resource_set_schedules(
    request_list_index: int,
    request_list: Sequence[Sequence[int]],
    serving_cell_ids: Sequence[str],
    resource_sets_by_cell: Mapping[str, Sequence[Mapping[str, object]]],
    triggering_dci_slot_by_cell: Mapping[str, int],
    *,
    scheduled_cell_ids: Sequence[str],
    offset_combo_index: int | None = None,
    offset_combos: Sequence[Sequence[int]] | None = None,
    offset_combo_cell_ids: Sequence[str] = (),
    supplementary_uplink_cells: Sequence[str] = (),
) -> tuple[tuple[str, tuple[NRSRSResourceSetAperiodicSchedule, ...]], ...]:
    """Compose decoded multi-cell SRS requests, offset combos, and allocations.

    Request/offset table rows, the full configured serving-cell roster, and the
    DCI-selected scheduled-cell subset are already decoded from DCI/RRC.
    ``offset_combo_cell_ids`` is the ordered subset represented in each RRC
    ``SRS-OffsetCombo`` row; it can include configured cells that are not
    co-scheduled in this DCI. Each resource set's allocation carries its own
    available-slot-offset list and base ``slot_offset``. The DCI slot is
    supplied for each scheduled cell because cells may use different
    numerologies; this adapter does not convert time between numerologies.
    """
    selected = nr_srs_multicell_request_resource_set_ids(
        request_list_index,
        request_list,
        serving_cell_ids,
        resource_sets_by_cell,
        scheduled_cell_ids=scheduled_cell_ids,
        supplementary_uplink_cells=supplementary_uplink_cells,
    )
    cells = tuple(serving_cell_ids)
    if not isinstance(scheduled_cell_ids, Sequence) or isinstance(scheduled_cell_ids, (str, bytes)):
        raise TypeError("scheduled_cell_ids must be an ordered sequence")
    scheduled_cells = tuple(scheduled_cell_ids)
    if not scheduled_cells or any(
        not isinstance(cell, str) or not cell for cell in scheduled_cells
    ):
        raise ValueError("scheduled_cell_ids must contain nonempty cell IDs")
    if len(set(scheduled_cells)) != len(scheduled_cells) or not set(scheduled_cells).issubset(
        cells
    ):
        raise ValueError("scheduled_cell_ids must be a unique subset of serving_cell_ids")
    if tuple(cell for cell in cells if cell in set(scheduled_cells)) != scheduled_cells:
        raise ValueError("scheduled_cell_ids must follow serving-cell order")
    if not isinstance(triggering_dci_slot_by_cell, Mapping):
        raise TypeError("triggering_dci_slot_by_cell must be a mapping")
    if set(triggering_dci_slot_by_cell) != set(scheduled_cells):
        raise ValueError("triggering_dci_slot_by_cell keys must exactly match scheduled_cell_ids")
    for cell_id, slot in triggering_dci_slot_by_cell.items():
        if isinstance(slot, bool) or not isinstance(slot, Integral) or slot < 0:
            raise ValueError(f"triggering DCI slot for {cell_id!r} must be nonnegative")

    if not isinstance(offset_combo_cell_ids, Sequence) or isinstance(
        offset_combo_cell_ids, (str, bytes)
    ):
        raise TypeError("offset_combo_cell_ids must be an ordered sequence")
    combo_cells = tuple(offset_combo_cell_ids)
    if any(not isinstance(cell, str) or not cell for cell in combo_cells):
        raise ValueError("offset_combo_cell_ids must contain nonempty cell IDs")
    if len(set(combo_cells)) != len(combo_cells) or not set(combo_cells).issubset(cells):
        raise ValueError("offset_combo_cell_ids must be a unique subset of serving_cell_ids")
    if tuple(cell for cell in cells if cell in set(combo_cells)) != combo_cells:
        raise ValueError("offset_combo_cell_ids must follow serving-cell order")

    selected_offset_by_cell: dict[str, int] = {}
    if offset_combos is None:
        if offset_combo_index is not None or combo_cells:
            raise ValueError("offset combo index/cell roster requires offset_combos")
    else:
        if not isinstance(offset_combos, Sequence) or isinstance(offset_combos, (str, bytes)):
            raise TypeError("offset_combos must be an ordered sequence of decoded rows")
        if not 1 <= len(offset_combos) <= 8 or not combo_cells:
            raise ValueError("offset_combos needs 1..8 rows and a nonempty cell roster")
        if isinstance(offset_combo_index, bool) or not isinstance(offset_combo_index, Integral):
            raise TypeError("offset_combo_index must select a configured offset-combo row")
        offset_combo_index = int(offset_combo_index)
        if not 0 <= offset_combo_index < len(offset_combos):
            raise ValueError("offset_combo_index does not select a configured row")
        row = offset_combos[offset_combo_index]
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise TypeError("selected offset-combo row must be a sequence")
        if len(row) != len(combo_cells):
            raise ValueError("selected offset-combo row must match offset_combo_cell_ids")
        for cell_id, value in zip(combo_cells, row, strict=True):
            if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value <= 3:
                raise ValueError("SRS offset-combo values must be integer indices in [0, 3]")
            selected_offset_by_cell[cell_id] = int(value)

    schedules_by_cell = dict(selected)
    result: list[tuple[str, tuple[NRSRSResourceSetAperiodicSchedule, ...]]] = []
    for cell_id in scheduled_cells:
        resource_sets = resource_sets_by_cell[cell_id]
        selected_ids = schedules_by_cell[cell_id]
        sets_by_id: dict[int, Mapping[str, object]] = {}
        for record in resource_sets:
            if not isinstance(record, Mapping):
                continue
            resource_set_id_value = record.get("resource_set_id")
            if isinstance(resource_set_id_value, bool) or not isinstance(
                resource_set_id_value, Integral
            ):
                raise ValueError("SRS resource_set_id must be an integer")
            sets_by_id[int(resource_set_id_value)] = record
        cell_schedules: list[NRSRSResourceSetAperiodicSchedule] = []
        for resource_set_id in selected_ids:
            resource_set = sets_by_id[resource_set_id]
            allocation = resource_set.get("allocation")
            if not isinstance(allocation, Mapping):
                raise ValueError(
                    f"selected resource set {resource_set_id} for {cell_id!r} "
                    "requires an allocation mapping"
                )
            resolved_allocation = dict(allocation)
            available_offsets = resolved_allocation.get("available_slot_offsets")
            if available_offsets is not None:
                if not isinstance(available_offsets, Sequence) or isinstance(
                    available_offsets, (str, bytes)
                ):
                    raise TypeError("available_slot_offsets must be an ordered sequence")
                if not available_offsets:
                    raise ValueError("available_slot_offsets must not be empty")
                if len(available_offsets) == 1:
                    offset_index = 0
                else:
                    try:
                        offset_index = selected_offset_by_cell[cell_id]
                    except KeyError as exc:
                        raise ValueError(
                            f"cell {cell_id!r} needs an SRS offset-combo entry for "
                            f"resource set {resource_set_id}"
                        ) from exc
                if offset_index >= len(available_offsets):
                    raise ValueError(
                        f"SRS offset-combo index {offset_index} is outside the available "
                        f"offset list for {cell_id!r} resource set {resource_set_id}"
                    )
                resolved_allocation["selected_available_offset_index"] = offset_index
            elif cell_id in selected_offset_by_cell:
                # This cell may be represented in a combo because another set
                # or BWP has multiple offsets; no list means the base offset applies.
                resolved_allocation.pop("selected_available_offset_index", None)

            schedule = nr_srs_aperiodic_resource_schedule(
                int(triggering_dci_slot_by_cell[cell_id]), **cast(Any, resolved_allocation)
            )
            cell_schedules.append(
                NRSRSResourceSetAperiodicSchedule(
                    resource_set_id=resource_set_id,
                    target_slot_index=schedule.target_slot_index,
                    allocations=schedule.allocations,
                )
            )
        result.append((cell_id, tuple(cell_schedules)))
    return tuple(result)


def nr_srs_periodic_resource_schedule(
    absolute_slot_index: int,
    *,
    c_srs: int,
    transmission_comb: int,
    b_srs: int,
    b_hop: int,
    srs_symbol_count: int,
    resource_type: str,
    periodicity_slots: int | None = None,
    slot_offset: int | None = None,
    periodicity_and_offset: Mapping[str, object] | None = None,
    periodicity_and_offset_ext: Mapping[str, object] | None = None,
    frequency_domain_position: int = 0,
    frequency_domain_shift_rb: int = 0,
    comb_offset: int = 0,
    reference_subcarrier: int = 0,
    repetition_factor: int | None = None,
    ports8_tdm: bool = False,
    positioning_resource: bool = False,
    numerology: int | None = None,
    hyper_frame_number: int | None = None,
    hyper_sfn_index: int | None = None,
) -> NRSRSAperiodicResourceSchedule | None:
    """Resolve one periodic/semi-persistent resource at an absolute slot.

    Returns ``None`` for a non-occasion slot or a Hyper-SFN-gated positioning
    occasion. The caller still resolves resource-set selection, active-BWP
    feasibility, and collisions with other transmissions.
    """

    if isinstance(absolute_slot_index, bool) or not isinstance(absolute_slot_index, Integral):
        raise TypeError("absolute_slot_index must be an integer")
    if absolute_slot_index < 0:
        raise ValueError("absolute_slot_index must be nonnegative")
    if resource_type not in {"periodic", "semi-persistent"}:
        raise ValueError("periodic schedule requires periodic or semi-persistent resource_type")
    allocations = nr_srs_resource_frequency_allocations(
        c_srs,
        transmission_comb=transmission_comb,
        b_srs=b_srs,
        b_hop=b_hop,
        srs_symbol_count=srs_symbol_count,
        resource_type=resource_type,
        frequency_domain_position=frequency_domain_position,
        frequency_domain_shift_rb=frequency_domain_shift_rb,
        comb_offset=comb_offset,
        reference_subcarrier=reference_subcarrier,
        repetition_factor=repetition_factor,
        ports8_tdm=ports8_tdm,
        absolute_slot_index=int(absolute_slot_index),
        periodicity_slots=periodicity_slots,
        slot_offset=slot_offset,
        periodicity_and_offset=periodicity_and_offset,
        periodicity_and_offset_ext=periodicity_and_offset_ext,
        positioning_resource=positioning_resource,
        numerology=numerology,
        hyper_frame_number=hyper_frame_number,
        hyper_sfn_index=hyper_sfn_index,
    )
    if allocations is None:
        return None
    return NRSRSAperiodicResourceSchedule(int(absolute_slot_index), allocations)


def nr_srs_periodic_resource_set_schedule(
    absolute_slot_index: int,
    resource_set_id: int,
    resources: Sequence[Mapping[str, object]],
    *,
    resource_type: str = "periodic",
    periodicity_slots: int | None = None,
    slot_offset: int | None = None,
    periodicity_and_offset: Mapping[str, object] | None = None,
    periodicity_and_offset_ext: Mapping[str, object] | None = None,
    positioning_resource: bool = False,
    numerology: int | None = None,
    hyper_frame_number: int | None = None,
    hyper_sfn_index: int | None = None,
) -> NRSRSPeriodicResourceSetSchedule | None:
    """Resolve all resources in one periodic set against shared occasion timing.

    Each decoded resource row has a unique ``resource_id`` and an ``allocation``
    mapping containing ``c_srs``, ``transmission_comb``, ``b_srs``,
    ``b_hop``, and ``srs_symbol_count``, plus the per-resource frequency and
    port options accepted by :func:`nr_srs_periodic_resource_schedule`. An
    optional ``timing`` mapping can override the common occasion inputs for an
    individual resource, since periodicity and offset belong to the resource
    configuration. The returned set contains only resources active in this
    slot; ``None`` means none are active.
    """
    if isinstance(absolute_slot_index, bool) or not isinstance(absolute_slot_index, Integral):
        raise TypeError("absolute_slot_index must be an integer")
    if absolute_slot_index < 0:
        raise ValueError("absolute_slot_index must be nonnegative")
    if not isinstance(resources, Sequence) or isinstance(resources, (str, bytes)):
        raise TypeError("resources must be an ordered sequence of decoded SRS resources")
    if not resources:
        raise ValueError("resources must contain at least one decoded SRS resource")
    preflight_resource_ids: set[int] = set()
    for index, resource in enumerate(resources):
        if not isinstance(resource, Mapping):
            raise TypeError(f"resources[{index}] must be a mapping")
        resource_id = resource.get("resource_id")
        if isinstance(resource_id, bool) or not isinstance(resource_id, Integral):
            raise TypeError(f"resources[{index}].resource_id must be an integer")
        resource_id = int(resource_id)
        if resource_id < 0 or resource_id in preflight_resource_ids:
            raise ValueError("resource IDs must be unique nonnegative integers")
        preflight_resource_ids.add(resource_id)
    if isinstance(resource_set_id, bool) or not isinstance(resource_set_id, Integral):
        raise TypeError("resource_set_id must be an integer")
    if resource_set_id < 0:
        raise ValueError("resource_set_id must be nonnegative")
    if not isinstance(resource_type, str) or resource_type not in {
        "periodic",
        "semi-persistent",
    }:
        raise ValueError("resource_type must be periodic or semi-persistent")
    if not isinstance(positioning_resource, bool):
        raise TypeError("positioning_resource must be a boolean")

    required = {"c_srs", "transmission_comb", "b_srs", "b_hop", "srs_symbol_count"}
    optional = {
        "frequency_domain_position",
        "frequency_domain_shift_rb",
        "comb_offset",
        "reference_subcarrier",
        "repetition_factor",
        "ports8_tdm",
    }
    timing = {
        "periodicity_slots": periodicity_slots,
        "slot_offset": slot_offset,
        "periodicity_and_offset": periodicity_and_offset,
        "periodicity_and_offset_ext": periodicity_and_offset_ext,
        "positioning_resource": positioning_resource,
        "numerology": numerology,
        "hyper_frame_number": hyper_frame_number,
        "hyper_sfn_index": hyper_sfn_index,
    }
    timing_keys = frozenset(timing)
    seen_resource_ids: set[int] = set()
    schedules: list[tuple[int, NRSRSAperiodicResourceSchedule | None]] = []
    for index, resource in enumerate(resources):
        if not isinstance(resource, Mapping):
            raise TypeError(f"resources[{index}] must be a mapping")
        resource_id = resource.get("resource_id")
        if isinstance(resource_id, bool) or not isinstance(resource_id, Integral):
            raise TypeError(f"resources[{index}].resource_id must be an integer")
        resource_id = int(resource_id)
        if resource_id < 0 or resource_id in seen_resource_ids:
            raise ValueError("resource IDs must be unique nonnegative integers")
        seen_resource_ids.add(resource_id)
        unsupported_resource_fields = set(resource) - {"resource_id", "allocation", "timing"}
        if unsupported_resource_fields:
            raise ValueError(
                f"resources[{index}] has unsupported fields: "
                f"{sorted(map(str, unsupported_resource_fields))}"
            )
        allocation = resource.get("allocation")
        if not isinstance(allocation, Mapping):
            raise ValueError(f"resources[{index}].allocation must be a mapping")
        fields = set(allocation)
        if any(not isinstance(field, str) for field in fields):
            raise ValueError(f"resources[{index}].allocation keys must be strings")
        missing = required - fields
        unsupported = fields - required - optional
        if missing or unsupported:
            raise ValueError(
                f"resources[{index}].allocation fields are invalid; "
                f"missing={sorted(missing)}, unsupported={sorted(unsupported)}"
            )
        resource_timing = resource.get("timing", {})
        if not isinstance(resource_timing, Mapping):
            raise ValueError(f"resources[{index}].timing must be a mapping")
        if any(not isinstance(field, str) for field in resource_timing):
            raise ValueError(f"resources[{index}].timing keys must be strings")
        unsupported_timing = set(resource_timing) - timing_keys
        if unsupported_timing:
            raise ValueError(
                f"resources[{index}].timing has unsupported fields: {sorted(unsupported_timing)}"
            )
        schedule = nr_srs_periodic_resource_schedule(
            absolute_slot_index,
            resource_type=resource_type,
            **(timing | dict(resource_timing)),
            **cast(Any, allocation),
        )
        schedules.append((resource_id, schedule))

    active = [
        (resource_id, schedule) for resource_id, schedule in schedules if schedule is not None
    ]
    if not active:
        return None
    return NRSRSPeriodicResourceSetSchedule(
        resource_set_id=int(resource_set_id),
        target_slot_index=int(absolute_slot_index),
        resources=tuple(
            (resource_id, cast(NRSRSAperiodicResourceSchedule, schedule))
            for resource_id, schedule in active
        ),
    )


def nr_srs_aperiodic_resource_schedule(
    triggering_dci_slot: int,
    *,
    c_srs: int,
    transmission_comb: int,
    b_srs: int,
    b_hop: int,
    srs_symbol_count: int,
    slot_offset: int | None = None,
    available_slot_offsets: Sequence[int] | None = None,
    selected_available_offset_index: int | None = None,
    frequency_domain_position: int = 0,
    frequency_domain_shift_rb: int = 0,
    comb_offset: int = 0,
    reference_subcarrier: int = 0,
    repetition_factor: int | None = None,
    ports8_tdm: bool = False,
) -> NRSRSAperiodicResourceSchedule:
    """Resolve one caller-selected aperiodic SRS set to target slot and hops.

    The caller resolves DCI-to-resource-set selection and passes the selected
    Release 17 available-slot-offset index. This adapter then derives every
    resource-symbol frequency allocation. It does not parse DCI or ASN.1 RRC.
    """

    target_slot = nr_srs_aperiodic_target_slot_index(
        triggering_dci_slot,
        slot_offset=slot_offset,
        available_slot_offsets=available_slot_offsets,
        selected_available_offset_index=selected_available_offset_index,
    )
    allocations = nr_srs_resource_frequency_allocations(
        c_srs,
        transmission_comb=transmission_comb,
        b_srs=b_srs,
        b_hop=b_hop,
        srs_symbol_count=srs_symbol_count,
        resource_type="aperiodic",
        frequency_domain_position=frequency_domain_position,
        frequency_domain_shift_rb=frequency_domain_shift_rb,
        comb_offset=comb_offset,
        reference_subcarrier=reference_subcarrier,
        repetition_factor=repetition_factor,
        ports8_tdm=ports8_tdm,
    )
    if allocations is None:
        raise RuntimeError("aperiodic SRS allocation unexpectedly resolved inactive")
    return NRSRSAperiodicResourceSchedule(target_slot, allocations)


def nr_srs_aperiodic_resource_set_schedules(
    triggering_dci_slot: int,
    srs_request_codepoint: int,
    resource_sets: Sequence[Mapping[str, object]],
    *,
    dci_format: str,
    srs_tpc_pdcch_group: str = "typeB",
    srs_request_mode: int | None = None,
    supplementary_uplink: bool = False,
) -> tuple[NRSRSResourceSetAperiodicSchedule, ...]:
    """Resolve a supported DCI request across every matching configured set.

    Each selected aperiodic set has an ``allocation`` mapping containing
    ``c_srs``, ``transmission_comb``, ``b_srs``, ``b_hop``, and
    ``srs_symbol_count``, plus the optional fields accepted by
    :func:`nr_srs_aperiodic_resource_schedule`. The trigger resolver validates
    all set identities and trigger assignments; frequency allocation is
    required only for sets selected by this DCI request.
    """

    selected_ids = nr_srs_triggered_resource_set_ids(
        srs_request_codepoint,
        resource_sets,
        dci_format=dci_format,
        srs_tpc_pdcch_group=srs_tpc_pdcch_group,
        srs_request_mode=srs_request_mode,
        supplementary_uplink=supplementary_uplink,
    )
    records_by_id: dict[int, Mapping[str, object]] = {}
    for record in resource_sets:
        if isinstance(record, Mapping):
            records_by_id[cast(int, record["resource_set_id"])] = record
    required = {
        "c_srs",
        "transmission_comb",
        "b_srs",
        "b_hop",
        "srs_symbol_count",
    }
    optional = {
        "slot_offset",
        "available_slot_offsets",
        "selected_available_offset_index",
        "frequency_domain_position",
        "frequency_domain_shift_rb",
        "comb_offset",
        "reference_subcarrier",
        "repetition_factor",
        "ports8_tdm",
    }
    schedules: list[NRSRSResourceSetAperiodicSchedule] = []
    for resource_set_id in selected_ids:
        resource_set = records_by_id[resource_set_id]
        allocation = resource_set.get("allocation")
        if not isinstance(allocation, Mapping):
            raise ValueError(
                f"selected resource set {resource_set_id} requires an allocation mapping"
            )
        allocation_fields = set(allocation)
        missing = required - allocation_fields
        unsupported = allocation_fields - required - optional
        if missing or unsupported:
            raise ValueError(
                f"selected resource set {resource_set_id} allocation fields are invalid; "
                f"missing={sorted(missing)}, unsupported={sorted(unsupported)}"
            )
        schedule = nr_srs_aperiodic_resource_schedule(
            triggering_dci_slot,
            **cast(Any, allocation),
        )
        schedules.append(
            NRSRSResourceSetAperiodicSchedule(
                resource_set_id=resource_set_id,
                target_slot_index=schedule.target_slot_index,
                allocations=schedule.allocations,
            )
        )
    return tuple(schedules)


def nr_srs_scheduled_port_sequences_type1(
    sequence_length: int,
    *,
    antenna_ports: tuple[int, ...],
    base_cyclic_shift: int,
    srs_symbol_count: int,
    sequence_id: int,
    slot_index: int,
    start_symbol_index: int,
    hopping_mode: str,
    transmission_comb: int,
    ports8_tdm: bool = False,
    cyclic_shift_hopping_offsets: tuple[int, ...] | None = None,
    cyclic_shift_hopping_id: int | None = None,
    radio_frame_index: int | None = None,
    slots_per_frame: int | None = None,
    hopping_subset: tuple[bool, ...] | None = None,
    hopping_finer_granularity: bool = False,
    slot_symbol_count: int = 14,
) -> np.ndarray:
    """Resolve TS 38.211 shifts and generate configured ports.

    For eight-port TDM, also pass ``ports8_tdm=True`` to the multiport mapper
    so inactive port-symbol elements are omitted from its grid and mask.
    """

    shifts = nr_srs_port_cyclic_shift_indices(
        antenna_ports,
        transmission_comb=transmission_comb,
        base_cyclic_shift=base_cyclic_shift,
        ports8_tdm=ports8_tdm,
    )
    return nr_srs_port_sequences_type1(
        sequence_length,
        cyclic_shift_indices=shifts,
        srs_symbol_count=srs_symbol_count,
        sequence_id=sequence_id,
        slot_index=slot_index,
        start_symbol_index=start_symbol_index,
        hopping_mode=hopping_mode,
        transmission_comb=transmission_comb,
        cyclic_shift_hopping_offsets=cyclic_shift_hopping_offsets,
        cyclic_shift_hopping_id=cyclic_shift_hopping_id,
        radio_frame_index=radio_frame_index,
        slots_per_frame=slots_per_frame,
        hopping_subset=hopping_subset,
        hopping_finer_granularity=hopping_finer_granularity,
        slot_symbol_count=slot_symbol_count,
        allow_duplicate_cyclic_shifts=True,
    )
