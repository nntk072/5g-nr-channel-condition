"""NR SRS resource-set schedules and intercell resource audits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np

from .srs_configuration import (
    NRSRSFixedFrequencyAllocation,
    nr_srs_cyclic_shift_hopping_offsets,
    nr_srs_hopped_sequence_type1,
    nr_srs_type1_bandwidth_configuration,
)
from .srs_constants import (
    _MAX_SRS_INTERCELL_AUDIT_COORDINATES,
    _MAX_SRS_INTERCELL_AUDIT_RESOURCE_ELEMENTS,
    _SRS_BANDWIDTH_TABLE,
    _SRS_SYMBOL_COUNTS,
    NRSRSHoppedMultiportResourceGrid,
    NRSRSHoppedResourceGrid,
    _normalized_multiport_beta,
)
from .srs_lookups import (
    nr_srs_frequency_position_indices,
    nr_srs_periodicity_and_offset_slots,
    nr_srs_positioning_symbol_offset,
    nr_srs_transmission_counter,
    srs_cyclic_shift_count,
)


@dataclass(frozen=True, slots=True)
class NRSRSAperiodicResourceSchedule:
    """Resolved active slot and per-symbol allocations for one SRS resource."""

    target_slot_index: int
    allocations: tuple[NRSRSFixedFrequencyAllocation, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.target_slot_index, bool)
            or not isinstance(self.target_slot_index, Integral)
            or self.target_slot_index < 0
        ):
            raise ValueError("target_slot_index must be a nonnegative integer")
        allocations = tuple(self.allocations)
        if not allocations or any(
            not isinstance(item, NRSRSFixedFrequencyAllocation) for item in allocations
        ):
            raise ValueError("allocations must contain fixed SRS frequency allocations")
        object.__setattr__(self, "target_slot_index", int(self.target_slot_index))
        object.__setattr__(self, "allocations", allocations)


@dataclass(frozen=True, slots=True)
class NRSRSPeriodicResourceSetSchedule:
    """One active periodic SRS set with its ordered resource schedules."""

    resource_set_id: int
    target_slot_index: int
    resources: tuple[tuple[int, NRSRSAperiodicResourceSchedule], ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.resource_set_id, bool)
            or not isinstance(self.resource_set_id, Integral)
            or self.resource_set_id < 0
        ):
            raise ValueError("resource_set_id must be a nonnegative integer")
        if (
            isinstance(self.target_slot_index, bool)
            or not isinstance(self.target_slot_index, Integral)
            or self.target_slot_index < 0
        ):
            raise ValueError("target_slot_index must be a nonnegative integer")
        resources = tuple(self.resources)
        if not resources:
            raise ValueError("resources must contain at least one scheduled SRS resource")
        resource_ids: list[int] = []
        normalized_resources: list[tuple[int, NRSRSAperiodicResourceSchedule]] = []
        for item in resources:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise TypeError("each resource entry must be a (resource_id, schedule) pair")
            resource_id, schedule = item
            if (
                isinstance(resource_id, bool)
                or not isinstance(resource_id, Integral)
                or resource_id < 0
            ):
                raise ValueError("resource IDs must be nonnegative integers")
            if not isinstance(schedule, NRSRSAperiodicResourceSchedule):
                raise TypeError("each resource schedule must be a periodic SRS schedule")
            if schedule.target_slot_index != int(self.target_slot_index):
                raise ValueError("all resource schedules must target the resource-set slot")
            resource_ids.append(int(resource_id))
            normalized_resources.append((int(resource_id), schedule))
        if len(set(resource_ids)) != len(resource_ids):
            raise ValueError("resource IDs must be unique within the SRS resource set")
        object.__setattr__(self, "resource_set_id", int(self.resource_set_id))
        object.__setattr__(self, "target_slot_index", int(self.target_slot_index))
        object.__setattr__(
            self,
            "resources",
            tuple(normalized_resources),
        )


@dataclass(frozen=True, slots=True)
class NRSRSResourceSetAperiodicSchedule:
    """Resolved SRS resource-set identity, target slot, and symbol allocations."""

    resource_set_id: int
    target_slot_index: int
    allocations: tuple[NRSRSFixedFrequencyAllocation, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.resource_set_id, bool)
            or not isinstance(self.resource_set_id, Integral)
            or self.resource_set_id < 0
        ):
            raise ValueError("resource_set_id must be a nonnegative integer")
        schedule = NRSRSAperiodicResourceSchedule(self.target_slot_index, tuple(self.allocations))
        object.__setattr__(self, "resource_set_id", int(self.resource_set_id))
        object.__setattr__(self, "target_slot_index", schedule.target_slot_index)
        object.__setattr__(self, "allocations", schedule.allocations)


@dataclass(frozen=True, slots=True)
class NRSRSScheduledResourceOccurrence:
    """One resolved SRS resource on a shared, caller-defined OFDM grid.

    ``start_subcarrier`` values in allocations must use the same absolute
    subcarrier coordinate for every occurrence with the same ``grid_id``.
    ``target_slot_index`` is the linear slot index in that common numerology.
    One allocation may be broadcast over all SRS symbols, or a distinct
    allocation may be supplied per symbol.
    """

    cell_id: str
    resource_set_id: int
    resource_id: int
    grid_id: str
    carrier_hz: float
    numerology: int
    subcarrier_spacing_hz: float
    target_slot_index: int
    slot_symbol_count: int
    start_symbol_index: int
    srs_symbol_count: int
    allocations: tuple[NRSRSFixedFrequencyAllocation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cell_id, str) or not self.cell_id.strip():
            raise ValueError("cell_id must be a nonempty string")
        if not isinstance(self.grid_id, str) or not self.grid_id.strip():
            raise ValueError("grid_id must be a nonempty string")
        for name in (
            "resource_set_id",
            "resource_id",
            "numerology",
            "target_slot_index",
            "slot_symbol_count",
            "start_symbol_index",
            "srs_symbol_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
        if self.resource_set_id < 0 or self.resource_id < 0:
            raise ValueError("SRS resource and resource-set IDs must be nonnegative")
        if not 0 <= self.numerology <= 6:
            raise ValueError("numerology must be between 0 and 6")
        if self.target_slot_index < 0:
            raise ValueError("target_slot_index must be nonnegative")
        if self.slot_symbol_count not in {12, 14}:
            raise ValueError("slot_symbol_count must be 12 or 14")
        if (
            self.srs_symbol_count not in _SRS_SYMBOL_COUNTS
            or self.srs_symbol_count > self.slot_symbol_count
        ):
            raise ValueError("srs_symbol_count is unsupported for the slot")
        allocations = tuple(self.allocations)
        if not allocations or any(
            not isinstance(item, NRSRSFixedFrequencyAllocation) for item in allocations
        ):
            raise ValueError("allocations must contain fixed SRS frequency allocations")
        if len(allocations) not in {1, self.srs_symbol_count}:
            raise ValueError("allocations must be shared or provided once per SRS symbol")
        if not 0 <= self.start_symbol_index < self.slot_symbol_count:
            raise ValueError("start_symbol_index must lie inside the slot")
        if self.start_symbol_index + self.srs_symbol_count > self.slot_symbol_count:
            raise ValueError("SRS symbols extend past the slot")
        if isinstance(self.carrier_hz, bool) or not isinstance(self.carrier_hz, Real):
            raise TypeError("carrier_hz must be a finite positive number")
        carrier_hz = float(self.carrier_hz)
        if not np.isfinite(carrier_hz) or carrier_hz <= 0.0:
            raise ValueError("carrier_hz must be a finite positive number")
        if isinstance(self.subcarrier_spacing_hz, bool) or not isinstance(
            self.subcarrier_spacing_hz, Real
        ):
            raise TypeError("subcarrier_spacing_hz must be a finite positive number")
        spacing_hz = float(self.subcarrier_spacing_hz)
        expected_spacing_hz = 15_000.0 * (2 ** int(self.numerology))
        if not np.isfinite(spacing_hz) or spacing_hz != expected_spacing_hz:
            raise ValueError("subcarrier_spacing_hz disagrees with numerology")
        for name in (
            "resource_set_id",
            "resource_id",
            "numerology",
            "target_slot_index",
            "slot_symbol_count",
            "start_symbol_index",
            "srs_symbol_count",
        ):
            object.__setattr__(self, name, int(getattr(self, name)))
        object.__setattr__(self, "cell_id", self.cell_id.strip())
        object.__setattr__(self, "grid_id", self.grid_id.strip())
        object.__setattr__(self, "carrier_hz", carrier_hz)
        object.__setattr__(self, "subcarrier_spacing_hz", spacing_hz)
        object.__setattr__(self, "allocations", allocations)


def audit_nr_srs_intercell_re_overlap(
    occurrences: Sequence[NRSRSScheduledResourceOccurrence],
) -> dict[str, Any]:
    """Report shared SRS RE occupancy on one common caller-defined OFDM grid.

    This is a scheduling diagnostic only. It does not evaluate cyclic-shift
    orthogonality, power, propagation, spectral leakage, or receiver impact.
    """
    if isinstance(occurrences, (str, bytes)) or not isinstance(occurrences, Sequence):
        raise ValueError("occurrences must be a sequence of scheduled SRS resources")
    normalized = tuple(occurrences)
    if not normalized or any(
        not isinstance(item, NRSRSScheduledResourceOccurrence) for item in normalized
    ):
        raise ValueError("occurrences must be a nonempty sequence of typed SRS resources")
    contexts = {
        (item.grid_id, item.carrier_hz, item.numerology, item.subcarrier_spacing_hz)
        for item in normalized
    }
    if len(contexts) != 1:
        raise ValueError("one audit requires a single common grid, carrier, and numerology")
    identities = [
        (item.cell_id, item.resource_set_id, item.resource_id, item.target_slot_index)
        for item in normalized
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("scheduled SRS occurrence identities must be unique")
    expanded_count = sum(
        sum(
            item.allocations[0 if len(item.allocations) == 1 else symbol].bandwidth.sequence_length
            for symbol in range(item.srs_symbol_count)
        )
        for item in normalized
    )
    if expanded_count > _MAX_SRS_INTERCELL_AUDIT_RESOURCE_ELEMENTS:
        raise ValueError("SRS overlap audit exceeds its resource-element limit")

    occupancy: dict[tuple[int, int, int], list[int]] = {}
    for occurrence_index, item in enumerate(normalized):
        for symbol_offset in range(item.srs_symbol_count):
            allocation = item.allocations[0 if len(item.allocations) == 1 else symbol_offset]
            symbol_index = item.start_symbol_index + symbol_offset
            for sequence_index in range(allocation.bandwidth.sequence_length):
                subcarrier_index = (
                    allocation.start_subcarrier
                    + allocation.comb_offset
                    + sequence_index * allocation.bandwidth.transmission_comb
                )
                coordinate = (item.target_slot_index, symbol_index, subcarrier_index)
                occupancy.setdefault(coordinate, []).append(occurrence_index)

    collision_coordinates: list[dict[str, Any]] = []
    pair_counts: dict[tuple[int, int], int] = {}
    overlap_count = 0
    for (slot, symbol, subcarrier), occurrence_indices in sorted(occupancy.items()):
        if len({normalized[index].cell_id for index in occurrence_indices}) < 2:
            continue
        overlap_count += 1
        participants = sorted(
            set(occurrence_indices),
            key=lambda index: (
                normalized[index].cell_id,
                normalized[index].resource_set_id,
                normalized[index].resource_id,
            ),
        )
        for left_position, left in enumerate(participants):
            for right in participants[left_position + 1 :]:
                if normalized[left].cell_id == normalized[right].cell_id:
                    continue
                pair = (min(left, right), max(left, right))
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
        if len(collision_coordinates) < _MAX_SRS_INTERCELL_AUDIT_COORDINATES:
            collision_coordinates.append(
                {
                    "slot_index": slot,
                    "symbol_index": symbol,
                    "subcarrier_index": subcarrier,
                    "participants": [
                        {
                            "cell_id": normalized[index].cell_id,
                            "resource_set_id": normalized[index].resource_set_id,
                            "resource_id": normalized[index].resource_id,
                        }
                        for index in participants
                    ],
                }
            )

    grid_id, carrier_hz, numerology, spacing_hz = next(iter(contexts))
    pairwise = [
        {
            "left": {
                "cell_id": normalized[left].cell_id,
                "resource_set_id": normalized[left].resource_set_id,
                "resource_id": normalized[left].resource_id,
            },
            "right": {
                "cell_id": normalized[right].cell_id,
                "resource_set_id": normalized[right].resource_set_id,
                "resource_id": normalized[right].resource_id,
            },
            "shared_resource_element_count": count,
        }
        for (left, right), count in sorted(
            pair_counts.items(),
            key=lambda entry: (
                normalized[entry[0][0]].cell_id,
                normalized[entry[0][1]].cell_id,
                normalized[entry[0][0]].resource_set_id,
                normalized[entry[0][0]].resource_id,
                normalized[entry[0][1]].resource_set_id,
                normalized[entry[0][1]].resource_id,
            ),
        )
    ]
    return {
        "schema": "loslab.nr_srs_intercell_re_overlap.v1",
        "status": "INTERCELL_RE_OVERLAP_DETECTED" if overlap_count else "NO_INTERCELL_RE_OVERLAP",
        "admission_possible": False,
        "grid": {
            "grid_id": grid_id,
            "carrier_hz": carrier_hz,
            "numerology": numerology,
            "subcarrier_spacing_hz": spacing_hz,
        },
        "scheduled_occurrence_count": len(normalized),
        "expanded_resource_element_count": expanded_count,
        "shared_intercell_resource_element_count": overlap_count,
        "pairwise_resource_overlaps": pairwise,
        "collision_coordinates": collision_coordinates,
        "collision_coordinates_truncated": overlap_count > len(collision_coordinates),
        "limitations": [
            "Shared RE occupancy is not proof of interference; cyclic-shift and port orthogonality are not evaluated.",
            "Power, propagation, synchronization error, spectral leakage, and receiver impact are not evaluated.",
            "Grid identity and absolute subcarrier coordinates are caller supplied and must be common across cells.",
            "Linear slot indices must share one time origin across all scheduled cells.",
        ],
    }


def nr_srs_fixed_frequency_allocation(
    c_srs: int,
    *,
    transmission_comb: int,
    b_srs: int = 0,
    b_hop: int = 0,
    frequency_domain_position: int = 0,
    frequency_domain_shift_rb: int = 0,
    comb_offset: int = 0,
    reference_subcarrier: int = 0,
    n_srs: int | None = None,
) -> NRSRSFixedFrequencyAllocation:
    """Resolve the type-1 SRS frequency allocation for one transmission.

    For active hopping, ``n_srs`` is the TS 38.211 transmission counter.
    Deriving that counter from periodic or semi-persistent scheduling is
    outside this resolver. ``reference_subcarrier`` selects the caller's
    CRB/BWP grid origin. SRS-PosResource symbol offsets remain a separate
    mapping step via :func:`nr_srs_positioning_symbol_offset`.
    """
    configuration = nr_srs_type1_bandwidth_configuration(
        c_srs, transmission_comb=transmission_comb, b_srs=b_srs
    )
    for name, value in (
        ("b_hop", b_hop),
        ("frequency_domain_position", frequency_domain_position),
        ("frequency_domain_shift_rb", frequency_domain_shift_rb),
        ("comb_offset", comb_offset),
        ("reference_subcarrier", reference_subcarrier),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")
    b_hop = int(b_hop)
    if b_hop > 3:
        raise ValueError("b_hop must be in [0, 3]")
    comb = configuration.transmission_comb
    if int(comb_offset) >= comb:
        raise ValueError("comb_offset must be smaller than transmission_comb")

    position_indices = nr_srs_frequency_position_indices(
        c_srs,
        b_srs=configuration.b_srs,
        b_hop=b_hop,
        frequency_domain_position=int(frequency_domain_position),
        n_srs=n_srs,
    )
    frequency_tree_offset = 0
    for level, position in enumerate(position_indices):
        bandwidth_rb = _SRS_BANDWIDTH_TABLE[configuration.c_srs][level][0]
        frequency_tree_offset += 12 * bandwidth_rb * position

    start_subcarrier = (
        int(reference_subcarrier) + 12 * int(frequency_domain_shift_rb) + frequency_tree_offset
    )
    return NRSRSFixedFrequencyAllocation(
        bandwidth=configuration,
        position_indices=position_indices,
        frequency_domain_position=int(frequency_domain_position),
        reference_subcarrier=int(reference_subcarrier),
        frequency_domain_shift_rb=int(frequency_domain_shift_rb),
        comb_offset=int(comb_offset),
        start_subcarrier=start_subcarrier,
        b_hop=b_hop,
        n_srs=n_srs,
    )


def nr_srs_resource_frequency_allocations(
    c_srs: int,
    *,
    transmission_comb: int,
    b_srs: int,
    b_hop: int,
    srs_symbol_count: int,
    resource_type: str,
    frequency_domain_position: int = 0,
    frequency_domain_shift_rb: int = 0,
    comb_offset: int = 0,
    reference_subcarrier: int = 0,
    repetition_factor: int | None = None,
    ports8_tdm: bool = False,
    absolute_slot_index: int | None = None,
    periodicity_slots: int | None = None,
    slot_offset: int | None = None,
    periodicity_and_offset: Mapping[str, object] | None = None,
    periodicity_and_offset_ext: Mapping[str, object] | None = None,
    positioning_resource: bool = False,
    numerology: int | None = None,
    hyper_frame_number: int | None = None,
    hyper_sfn_index: int | None = None,
) -> tuple[NRSRSFixedFrequencyAllocation, ...] | None:
    """Resolve each resource-symbol allocation from slot timing or RRC CHOICE.

    ``None`` means the periodic or semi-persistent resource is not active in
    the supplied slot. Periodicity may be supplied as resolved slot values or
    as the RRC ``periodicityAndOffset`` CHOICE and optional Release 16
    extension. For periodic positioning resources, the optional Hyper-SFN
    configuration is checked in the transmission counter. This does not parse
    complete ASN.1 RRC messages or resolve resource-set triggering.
    """
    if periodicity_and_offset is not None or periodicity_and_offset_ext is not None:
        if periodicity_slots is not None or slot_offset is not None:
            raise ValueError("supply resolved periodicity or the RRC CHOICE, not both")
        if resource_type == "aperiodic":
            raise ValueError("aperiodic SRS does not use periodicityAndOffset")
        periodicity_slots, slot_offset = nr_srs_periodicity_and_offset_slots(
            periodicity_and_offset,
            periodicity_and_offset_ext=periodicity_and_offset_ext,
        )

    if isinstance(srs_symbol_count, bool) or not isinstance(srs_symbol_count, Integral):
        raise TypeError("srs_symbol_count must be an integer")
    srs_symbol_count = int(srs_symbol_count)
    if srs_symbol_count not in _SRS_SYMBOL_COUNTS:
        raise ValueError("srs_symbol_count is unsupported")
    allocations: list[NRSRSFixedFrequencyAllocation] = []
    for resource_symbol_index in range(srs_symbol_count):
        counter = nr_srs_transmission_counter(
            resource_type=resource_type,
            srs_symbol_count=srs_symbol_count,
            resource_symbol_index=resource_symbol_index,
            repetition_factor=repetition_factor,
            ports8_tdm=ports8_tdm,
            absolute_slot_index=absolute_slot_index,
            periodicity_slots=periodicity_slots,
            slot_offset=slot_offset,
            positioning_resource=positioning_resource,
            numerology=numerology,
            hyper_frame_number=hyper_frame_number,
            hyper_sfn_index=hyper_sfn_index,
        )
        if counter is None:
            return None
        allocations.append(
            nr_srs_fixed_frequency_allocation(
                c_srs,
                transmission_comb=transmission_comb,
                b_srs=b_srs,
                b_hop=b_hop,
                frequency_domain_position=frequency_domain_position,
                frequency_domain_shift_rb=frequency_domain_shift_rb,
                comb_offset=comb_offset,
                reference_subcarrier=reference_subcarrier,
                n_srs=counter,
            )
        )
    return tuple(allocations)


def nr_srs_resource_sequences_type1(
    sequence_length: int,
    *,
    srs_symbol_count: int,
    sequence_id: int,
    slot_index: int,
    start_symbol_index: int,
    hopping_mode: str,
    transmission_comb: int,
    cyclic_shift_index: int = 0,
    cyclic_shift_hopping_offsets: tuple[int, ...] | None = None,
    cyclic_shift_hopping_id: int | None = None,
    radio_frame_index: int | None = None,
    slots_per_frame: int | None = None,
    hopping_subset: tuple[bool, ...] | None = None,
    hopping_finer_granularity: bool = False,
    slot_symbol_count: int = 14,
) -> np.ndarray:
    """Generate one type-1 sequence row for each symbol in an SRS resource.

    ``start_symbol_index`` is ``l0`` within the slot and ``slot_index`` is
    local to the 10-ms radio frame. Frequency allocation is intentionally
    excluded; the returned ``[symbols, occupied subcarriers]`` matrix can be
    passed directly to :func:`map_nr_srs_resource`. To derive cyclic-shift
    hopping, provide its identity, radio-frame index, and slots per frame;
    this derives one ``f_csh`` value per resource symbol. Derived offsets and
    caller-supplied explicit offsets are mutually exclusive.
    """
    for name, value in (
        ("sequence_length", sequence_length),
        ("srs_symbol_count", srs_symbol_count),
        ("slot_symbol_count", slot_symbol_count),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    sequence_length = int(sequence_length)
    srs_symbol_count = int(srs_symbol_count)
    slot_symbol_count = int(slot_symbol_count)
    if srs_symbol_count not in _SRS_SYMBOL_COUNTS or srs_symbol_count > slot_symbol_count:
        raise ValueError("unsupported SRS symbol count for the declared slot")
    if slot_symbol_count not in {12, 14}:
        raise ValueError("slot_symbol_count must be 12 or 14")
    if not isinstance(hopping_finer_granularity, bool):
        raise TypeError("hopping_finer_granularity must be a boolean")
    if cyclic_shift_hopping_id is None:
        if (
            radio_frame_index is not None
            or slots_per_frame is not None
            or hopping_subset is not None
        ):
            raise ValueError("cyclic-shift hopping coordinates require cyclic_shift_hopping_id")
    else:
        if cyclic_shift_hopping_offsets is not None:
            raise ValueError("do not combine derived and explicit cyclic-shift hopping offsets")
        if radio_frame_index is None or slots_per_frame is None:
            raise ValueError("cyclic-shift hopping requires radio_frame_index and slots_per_frame")
        cyclic_shift_hopping_offsets = nr_srs_cyclic_shift_hopping_offsets(
            transmission_comb=transmission_comb,
            cyclic_shift_hopping_id=cyclic_shift_hopping_id,
            radio_frame_index=radio_frame_index,
            slot_index=slot_index,
            slots_per_frame=slots_per_frame,
            slot_symbol_count=slot_symbol_count,
            start_symbol_index=start_symbol_index,
            srs_symbol_count=srs_symbol_count,
            hopping_subset=hopping_subset,
            hopping_finer_granularity=hopping_finer_granularity,
        )
    if cyclic_shift_hopping_offsets is None:
        hopping_offsets = (0,) * srs_symbol_count
    else:
        if not isinstance(cyclic_shift_hopping_offsets, tuple):
            raise TypeError("cyclic_shift_hopping_offsets must be a tuple of integers")
        if len(cyclic_shift_hopping_offsets) != srs_symbol_count:
            raise ValueError("cyclic-shift hopping offsets must match the SRS symbol count")
        if any(
            isinstance(offset, bool) or not isinstance(offset, Integral)
            for offset in cyclic_shift_hopping_offsets
        ):
            raise TypeError("cyclic_shift_hopping_offsets must contain integers")
        hopping_offsets = tuple(int(offset) for offset in cyclic_shift_hopping_offsets)
        maximum_hopping_offset = srs_cyclic_shift_count(transmission_comb) * (
            2 if hopping_finer_granularity else 1
        )
        if any(not 0 <= offset < maximum_hopping_offset for offset in hopping_offsets):
            raise ValueError("cyclic-shift hopping offsets exceed the configured granularity")
    rows = [
        nr_srs_hopped_sequence_type1(
            sequence_length,
            sequence_id=sequence_id,
            slot_index=slot_index,
            start_symbol_index=start_symbol_index,
            resource_symbol_index=resource_symbol_index,
            hopping_mode=hopping_mode,
            transmission_comb=transmission_comb,
            cyclic_shift_index=cyclic_shift_index,
            cyclic_shift_hopping_offset=hopping_offsets[resource_symbol_index],
            hopping_finer_granularity=hopping_finer_granularity,
            slot_symbol_count=slot_symbol_count,
        )
        for resource_symbol_index in range(srs_symbol_count)
    ]
    return np.stack(rows, axis=0)


def nr_srs_port_sequences_type1(
    sequence_length: int,
    *,
    cyclic_shift_indices: tuple[int, ...],
    srs_symbol_count: int,
    sequence_id: int,
    slot_index: int,
    start_symbol_index: int,
    hopping_mode: str,
    transmission_comb: int,
    cyclic_shift_hopping_offsets: tuple[int, ...] | None = None,
    cyclic_shift_hopping_id: int | None = None,
    radio_frame_index: int | None = None,
    slots_per_frame: int | None = None,
    hopping_subset: tuple[bool, ...] | None = None,
    hopping_finer_granularity: bool = False,
    slot_symbol_count: int = 14,
    allow_duplicate_cyclic_shifts: bool = False,
) -> np.ndarray:
    """Generate caller-scheduled SRS sequences for multiple transmit ports.

    The returned array has shape ``[ports, SRS symbols, occupied subcarriers]``.
    ``cyclic_shift_indices`` are explicit zero-based indices in the comb-specific
    cyclic-shift set. This helper applies the same group/sequence hopping state
    to every port; derivation of valid port-specific shifts and resource
    scheduling from higher-layer configuration remain caller-owned. Optional
    cyclic-shift hopping coordinates are forwarded unchanged to each port so
    all ports share the same per-symbol ``f_csh`` sequence.
    """
    if not isinstance(cyclic_shift_indices, tuple) or not cyclic_shift_indices:
        raise TypeError("cyclic_shift_indices must be a nonempty tuple of integers")
    shift_count = srs_cyclic_shift_count(transmission_comb)
    for index in cyclic_shift_indices:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("cyclic_shift_indices must contain integers")
        if not 0 <= index < shift_count:
            raise ValueError(f"cyclic-shift indices must be in [0, {shift_count}) for this comb")
    if not isinstance(allow_duplicate_cyclic_shifts, bool):
        raise TypeError("allow_duplicate_cyclic_shifts must be a boolean")
    if not allow_duplicate_cyclic_shifts and len(set(cyclic_shift_indices)) != len(
        cyclic_shift_indices
    ):
        raise ValueError("cyclic_shift_indices must be unique across ports")
    return np.stack(
        [
            nr_srs_resource_sequences_type1(
                sequence_length,
                srs_symbol_count=srs_symbol_count,
                sequence_id=sequence_id,
                slot_index=slot_index,
                start_symbol_index=start_symbol_index,
                hopping_mode=hopping_mode,
                transmission_comb=transmission_comb,
                cyclic_shift_index=int(cyclic_shift_index),
                cyclic_shift_hopping_offsets=cyclic_shift_hopping_offsets,
                cyclic_shift_hopping_id=cyclic_shift_hopping_id,
                radio_frame_index=radio_frame_index,
                slots_per_frame=slots_per_frame,
                hopping_subset=hopping_subset,
                hopping_finer_granularity=hopping_finer_granularity,
                slot_symbol_count=slot_symbol_count,
            )
            for cyclic_shift_index in cyclic_shift_indices
        ],
        axis=0,
    )


def map_nr_srs_hopped_resource(
    sequence: np.ndarray,
    *,
    frequency_allocations: tuple[NRSRSFixedFrequencyAllocation, ...]
    | list[NRSRSFixedFrequencyAllocation],
    slot_symbol_count: int,
    subcarrier_count: int,
    start_position_offset: int,
    srs_symbol_count: int,
    beta_srs: float = 1.0,
    positioning_resource: bool = False,
) -> NRSRSHoppedResourceGrid:
    """Map an SRS resource whose frequency allocation can vary by symbol.

    Supply one allocation to reuse a fixed position, or one allocation per
    resource symbol for repetition-group hopping. Each allocation's ``n_srs``
    field is retained in the result. Positioning-resource comb offsets are
    applied per symbol when ``positioning_resource`` is true.
    """
    if not isinstance(positioning_resource, bool):
        raise TypeError("positioning_resource must be a boolean")
    for name, value in (
        ("slot_symbol_count", slot_symbol_count),
        ("subcarrier_count", subcarrier_count),
        ("start_position_offset", start_position_offset),
        ("srs_symbol_count", srs_symbol_count),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    slot_symbol_count = int(slot_symbol_count)
    subcarrier_count = int(subcarrier_count)
    start_position_offset = int(start_position_offset)
    srs_symbol_count = int(srs_symbol_count)
    allocations = tuple(frequency_allocations)
    if not allocations or any(
        not isinstance(item, NRSRSFixedFrequencyAllocation) for item in allocations
    ):
        raise TypeError("frequency_allocations must contain resolved SRS allocation objects")
    if len(allocations) not in {1, srs_symbol_count}:
        raise ValueError("frequency_allocations must contain one or one-per-symbol allocation")
    if slot_symbol_count not in {12, 14}:
        raise ValueError("slot_symbol_count must be 12 or 14")
    if subcarrier_count < 1:
        raise ValueError("subcarrier_count must be positive")
    if srs_symbol_count not in _SRS_SYMBOL_COUNTS or srs_symbol_count > slot_symbol_count:
        raise ValueError("unsupported SRS symbol count for the declared slot")
    if start_position_offset < srs_symbol_count - 1 or start_position_offset >= slot_symbol_count:
        raise ValueError("start_position_offset must fit the SRS symbols in the slot")
    if (
        isinstance(beta_srs, bool)
        or not isinstance(beta_srs, Real)
        or not np.isfinite(float(beta_srs))
        or beta_srs <= 0
    ):
        raise ValueError("beta_srs must be a finite positive real number")

    first = allocations[0]
    if any(
        allocation.bandwidth != first.bandwidth
        or allocation.comb_offset != first.comb_offset
        or allocation.frequency_domain_position != first.frequency_domain_position
        or allocation.frequency_domain_shift_rb != first.frequency_domain_shift_rb
        or allocation.reference_subcarrier != first.reference_subcarrier
        or allocation.b_hop != first.b_hop
        for allocation in allocations[1:]
    ):
        raise ValueError("per-symbol frequency allocations must share one SRS configuration")
    rows_allocations = allocations * srs_symbol_count if len(allocations) == 1 else allocations
    width = first.bandwidth.allocated_bandwidth_subcarriers
    comb_size = first.bandwidth.transmission_comb
    offsets = (
        tuple(
            nr_srs_positioning_symbol_offset(
                comb_size,
                srs_symbol_count=srs_symbol_count,
                resource_symbol_index=index,
            )
            for index in range(srs_symbol_count)
        )
        if positioning_resource
        else (0,) * srs_symbol_count
    )
    values = np.asarray(sequence)
    if not np.iscomplexobj(values):
        raise TypeError("sequence must be complex-valued")
    if values.ndim not in {1, 2}:
        raise ValueError("sequence must have shape [occupied REs] or [SRS symbols, occupied REs]")
    if values.shape[-1] != first.bandwidth.sequence_length:
        raise ValueError("sequence length does not match the configured SRS bandwidth and comb")
    if values.ndim == 2 and values.shape[0] != srs_symbol_count:
        raise ValueError("per-symbol SRS sequence must include exactly srs_symbol_count rows")
    if not np.isfinite(values).all():
        raise ValueError("sequence must contain only finite values")

    first_symbol = slot_symbol_count - 1 - start_position_offset
    symbol_indices = tuple(range(first_symbol, first_symbol + srs_symbol_count))
    rows = (
        values
        if values.ndim == 2
        else np.broadcast_to(values, (srs_symbol_count, first.bandwidth.sequence_length))
    )
    grid = np.zeros((slot_symbol_count, subcarrier_count), dtype=np.complex128)
    mask = np.zeros((slot_symbol_count, subcarrier_count), dtype=np.bool_)
    subcarriers_by_symbol: list[tuple[int, ...]] = []
    for row_index, (symbol, allocation, positioning_offset) in enumerate(
        zip(symbol_indices, rows_allocations, offsets, strict=True)
    ):
        if allocation.start_subcarrier + width > subcarrier_count:
            raise ValueError("a hopped SRS allocation does not fit inside the subcarrier grid")
        effective_comb_offset = (allocation.comb_offset + positioning_offset) % comb_size
        subcarriers = tuple(
            allocation.start_subcarrier + effective_comb_offset + comb_size * index
            for index in range(first.bandwidth.sequence_length)
        )
        # The modulo above bounds the offset below one comb. The resolved
        # sequence length then ends at or before the allocated bandwidth edge.
        subcarriers_by_symbol.append(subcarriers)
        grid[symbol, list(subcarriers)] = beta_srs * rows[row_index]
        mask[symbol, list(subcarriers)] = True

    return NRSRSHoppedResourceGrid(
        grid=grid,
        mask=mask,
        symbol_indices=symbol_indices,
        subcarrier_indices_by_symbol=tuple(subcarriers_by_symbol),
        position_indices_by_symbol=tuple(item.position_indices for item in rows_allocations),
        n_srs_by_symbol=tuple(item.n_srs for item in rows_allocations),
        positioning_offsets=offsets,
    )


def map_nr_srs_hopped_multiport_resource(
    sequences: np.ndarray,
    *,
    frequency_allocations: tuple[NRSRSFixedFrequencyAllocation, ...]
    | list[NRSRSFixedFrequencyAllocation],
    slot_symbol_count: int,
    subcarrier_count: int,
    start_position_offset: int,
    srs_symbol_count: int,
    beta_srs: float = 1.0,
    positioning_resource: bool = False,
    four_port_srs_3tx: bool = False,
) -> NRSRSHoppedMultiportResourceGrid:
    """Map caller-scheduled multiport sequences over resolved SRS hops.

    Applies the TS 38.211 per-port ``1/sqrt(N_ap)`` amplitude factor; the
    four-port three-transmit configuration uses an effective count of three.
    """
    values = np.asarray(sequences)
    if not np.iscomplexobj(values):
        raise TypeError("sequences must be complex-valued")
    if values.ndim != 3 or values.shape[0] < 1:
        raise ValueError("sequences must have shape [ports, SRS symbols, occupied REs]")
    if values.shape[1] != srs_symbol_count:
        raise ValueError("each port must have exactly srs_symbol_count rows")
    normalized_beta = _normalized_multiport_beta(beta_srs, values.shape[0], four_port_srs_3tx)
    mapped = tuple(
        map_nr_srs_hopped_resource(
            port_sequences,
            frequency_allocations=frequency_allocations,
            slot_symbol_count=slot_symbol_count,
            subcarrier_count=subcarrier_count,
            start_position_offset=start_position_offset,
            srs_symbol_count=srs_symbol_count,
            beta_srs=normalized_beta,
            positioning_resource=positioning_resource,
        )
        for port_sequences in values
    )
    first = mapped[0]
    return NRSRSHoppedMultiportResourceGrid(
        grid=np.stack([item.grid for item in mapped], axis=0),
        mask=np.stack([item.mask for item in mapped], axis=0),
        symbol_indices=first.symbol_indices,
        subcarrier_indices_by_symbol=first.subcarrier_indices_by_symbol,
        position_indices_by_symbol=first.position_indices_by_symbol,
        n_srs_by_symbol=first.n_srs_by_symbol,
        positioning_offsets=first.positioning_offsets,
    )
