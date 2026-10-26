"""NR downlink positioning reference-signal sequence, mapping, and timing view.

The sequence and resource mapping follow 3GPP TS 38.211 V19.2.0,
clauses 5.2.1 and 7.4.1.7. This module consumes already-resolved higher-layer
configuration; it does not decode RRC or implement PRS occasion scheduling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
_PRS_SYMBOL_COMB_PAIRS = frozenset(
    {
        (1, 2),
        (2, 2),
        (4, 2),
        (6, 2),
        (12, 2),
        (1, 4),
        (4, 4),
        (12, 4),
        (1, 6),
        (6, 6),
        (12, 6),
        (1, 12),
        (12, 12),
    }
)
_PRS_SYMBOL_OFFSETS = {
    2: (0, 1),
    4: (0, 2, 1, 3),
    6: (0, 3, 1, 4, 2, 5),
    12: (0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11),
}
_PRS_PERIODICITY_BASE_SLOTS = (
    4,
    5,
    8,
    10,
    16,
    20,
    32,
    40,
    64,
    80,
    160,
    320,
    640,
    1280,
    2560,
    5120,
    10240,
)
_PRS_REPETITION_FACTORS = frozenset({1, 2, 4, 6, 8, 16, 32})
_PRS_TIME_GAPS = frozenset({1, 2, 4, 8, 16, 32})
_PRS_OPTION1_BITMAP_LENGTHS = frozenset({2, 4, 6, 8, 16, 32})
_PRS_MUTING_REPETITION_FACTORS = frozenset({1, 2, 4, 8})


def _integer(value: object, *, label: str, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{label} must be {bound}")
    return value


def _gold_bits(c_init: int, count: int) -> NDArray[np.uint8]:
    """Generate the TS 38.211 length-31 Gold sequence after Nc=1600."""
    _integer(c_init, label="c_init", minimum=0, maximum=(1 << 31) - 1)
    _integer(count, label="count", minimum=0)
    end = 1600 + count
    x1 = np.zeros(end + 31, dtype=np.uint8)
    x2 = np.zeros(end + 31, dtype=np.uint8)
    x1[0] = 1
    for index in range(31):
        x2[index] = (c_init >> index) & 1
    for index in range(end):
        x1[index + 31] = x1[index + 3] ^ x1[index]
        x2[index + 31] = x2[index + 3] ^ x2[index + 2] ^ x2[index + 1] ^ x2[index]
    return x1[1600:end] ^ x2[1600:end]


def nr_dl_prs_c_init(
    *,
    prs_sequence_id: int,
    slot_index: int,
    numerology: int,
    symbol_index: int,
    symbols_per_slot: int = 14,
) -> int:
    """Return the downlink PRS Gold-sequence seed for one slot symbol."""
    prs_sequence_id = _integer(
        prs_sequence_id, label="prs_sequence_id", minimum=0, maximum=4095
    )
    numerology = _integer(numerology, label="numerology", minimum=0, maximum=6)
    slots_per_frame = 10 * (1 << numerology)
    slot_index = _integer(
        slot_index, label="slot_index", minimum=0, maximum=slots_per_frame - 1
    )
    symbols_per_slot = _integer(
        symbols_per_slot, label="symbols_per_slot", minimum=12, maximum=14
    )
    if symbols_per_slot not in (12, 14):
        raise ValueError("symbols_per_slot must be 12 or 14")
    symbol_index = _integer(
        symbol_index, label="symbol_index", minimum=0, maximum=symbols_per_slot - 1
    )
    low_id = prs_sequence_id % 1024
    return (
        (1 << 22) * (prs_sequence_id // 1024)
        + (1 << 10) * (symbols_per_slot * slot_index + symbol_index + 1) * (2 * low_id + 1)
        + low_id
    ) % (1 << 31)


def nr_dl_prs_sequence(
    *,
    prs_sequence_id: int,
    slot_index: int,
    numerology: int,
    symbol_index: int,
    length: int,
    symbols_per_slot: int = 14,
) -> ComplexArray:
    """Generate the unit-power QPSK PRS sequence for one OFDM symbol."""
    length = _integer(length, label="length", minimum=1)
    c_init = nr_dl_prs_c_init(
        prs_sequence_id=prs_sequence_id,
        slot_index=slot_index,
        numerology=numerology,
        symbol_index=symbol_index,
        symbols_per_slot=symbols_per_slot,
    )
    bits = _gold_bits(c_init, 2 * length).reshape(length, 2)
    sequence = ((1.0 - 2.0 * bits[:, 0]) + 1j * (1.0 - 2.0 * bits[:, 1])) / math.sqrt(2.0)
    return np.asarray(sequence, dtype=np.complex128)


@dataclass(frozen=True, slots=True, eq=False)
class NRPRSResourceGrid:
    """Mapped one-port PRS resource on one slot's local resource grid."""

    grid: ComplexArray
    resource_mask: BoolArray
    symbol_indices: NDArray[np.int64]
    subcarrier_indices: tuple[NDArray[np.int64], ...]
    sequence_id: int
    slot_index: int
    numerology: int
    subcarrier_spacing_hz: float
    beta_prs: float
    comb_size: int = 1


@dataclass(frozen=True, slots=True)
class NRPRSResourceConfiguration:
    """Resolved one-port PRS fields for one resource in a set and one slot."""

    resource_id: str
    resource_start_subcarrier: int
    resource_bandwidth_rbs: int
    prs_sequence_id: int
    comb_size: int
    resource_element_offset: int
    start_symbol: int
    num_symbols: int
    beta_prs: float = 1.0


@dataclass(frozen=True, slots=True, eq=False)
class NRPRSResourceSetGrid:
    """Composed single-slot PRS grid with per-resource masks and collision map."""

    grid: ComplexArray
    resource_ids: tuple[str, ...]
    resource_masks: tuple[BoolArray, ...]
    collision_mask: BoolArray
    colliding_resource_element_count: int
    numerology: int
    slot_index: int


def map_nr_dl_prs_resource(
    *,
    slot_symbol_count: int,
    subcarrier_count: int,
    resource_start_subcarrier: int,
    resource_bandwidth_rbs: int,
    prs_sequence_id: int,
    slot_index: int,
    numerology: int,
    subcarrier_spacing_hz: float,
    comb_size: int,
    resource_element_offset: int,
    start_symbol: int,
    num_symbols: int,
    beta_prs: float = 1.0,
    reserved_symbol_mask: BoolArray | None = None,
    reserved_resource_mask: BoolArray | None = None,
) -> NRPRSResourceGrid:
    """Generate and map one configured downlink PRS resource into an empty grid.

    `resource_start_subcarrier` is the PRS resource's first subcarrier relative
    to this grid's point-A reference. Reserved symbols (for example, symbols
    occupied by a configured SS/PBCH block) are supplied by the caller and
    cause a fail-closed error if they overlap this PRS resource.
    """
    slot_symbol_count = _integer(
        slot_symbol_count, label="slot_symbol_count", minimum=12, maximum=14
    )
    if slot_symbol_count not in (12, 14):
        raise ValueError("slot_symbol_count must be 12 or 14")
    subcarrier_count = _integer(subcarrier_count, label="subcarrier_count", minimum=1)
    resource_start_subcarrier = _integer(
        resource_start_subcarrier, label="resource_start_subcarrier", minimum=0
    )
    resource_bandwidth_rbs = _integer(
        resource_bandwidth_rbs, label="resource_bandwidth_rbs", minimum=1
    )
    prs_sequence_id = _integer(
        prs_sequence_id, label="prs_sequence_id", minimum=0, maximum=4095
    )
    numerology = _integer(numerology, label="numerology", minimum=0, maximum=6)
    slots_per_frame = 10 * (1 << numerology)
    slot_index = _integer(
        slot_index, label="slot_index", minimum=0, maximum=slots_per_frame - 1
    )
    comb_size = _integer(comb_size, label="comb_size", minimum=1, maximum=12)
    resource_element_offset = _integer(
        resource_element_offset,
        label="resource_element_offset",
        minimum=0,
        maximum=comb_size - 1,
    )
    start_symbol = _integer(start_symbol, label="start_symbol", minimum=0)
    num_symbols = _integer(num_symbols, label="num_symbols", minimum=1)
    if (num_symbols, comb_size) not in _PRS_SYMBOL_COMB_PAIRS:
        raise ValueError("num_symbols and comb_size are not a supported TS 38.211 PRS pair")
    if start_symbol + num_symbols > slot_symbol_count:
        raise ValueError("PRS symbols must fit inside the configured slot")
    if resource_start_subcarrier + 12 * resource_bandwidth_rbs > subcarrier_count:
        raise ValueError("PRS resource bandwidth must fit inside the subcarrier grid")
    if isinstance(subcarrier_spacing_hz, bool) or not isinstance(subcarrier_spacing_hz, Real):
        raise TypeError("subcarrier_spacing_hz must be a real scalar")
    spacing = float(subcarrier_spacing_hz)
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("subcarrier_spacing_hz must be finite and positive")
    if isinstance(beta_prs, bool) or not isinstance(beta_prs, Real):
        raise TypeError("beta_prs must be a real scalar")
    beta = float(beta_prs)
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta_prs must be finite and positive")
    if reserved_symbol_mask is None:
        reserved = np.zeros(slot_symbol_count, dtype=np.bool_)
    else:
        reserved = np.asarray(reserved_symbol_mask)
        if reserved.shape != (slot_symbol_count,) or reserved.dtype.kind != "b":
            raise ValueError("reserved_symbol_mask must be boolean with one entry per slot symbol")
    if reserved_resource_mask is None:
        reserved_re = np.zeros((slot_symbol_count, subcarrier_count), dtype=np.bool_)
    else:
        reserved_re = np.asarray(reserved_resource_mask)
        if reserved_re.shape != (slot_symbol_count, subcarrier_count) or reserved_re.dtype.kind != "b":
            raise ValueError(
                "reserved_resource_mask must be boolean with shape "
                "[slot_symbol_count, subcarrier_count]"
            )
    symbols = np.arange(start_symbol, start_symbol + num_symbols, dtype=np.int64)
    if np.any(reserved[symbols]):
        raise ValueError("configured PRS resource overlaps a reserved slot symbol")

    grid = np.zeros((slot_symbol_count, subcarrier_count), dtype=np.complex128)
    mask = np.zeros(grid.shape, dtype=np.bool_)
    indices_by_symbol: list[NDArray[np.int64]] = []
    for symbol in symbols:
        symbol_offset = int(symbol - start_symbol)
        k_prime = _PRS_SYMBOL_OFFSETS[comb_size][symbol_offset]
        residue = (resource_element_offset + k_prime) % comb_size
        local_indices = np.arange(residue, 12 * resource_bandwidth_rbs, comb_size)
        subcarriers = np.asarray(resource_start_subcarrier + local_indices, dtype=np.int64)
        if np.any(reserved_re[symbol, subcarriers]):
            raise ValueError("configured PRS resource overlaps a reserved resource element")
        sequence = nr_dl_prs_sequence(
            prs_sequence_id=prs_sequence_id,
            slot_index=slot_index,
            numerology=numerology,
            symbol_index=int(symbol),
            length=len(subcarriers),
            symbols_per_slot=slot_symbol_count,
        )
        grid[symbol, subcarriers] = beta * sequence
        mask[symbol, subcarriers] = True
        indices_by_symbol.append(subcarriers)
    grid.setflags(write=False)
    mask.setflags(write=False)
    symbols.setflags(write=False)
    for subcarriers in indices_by_symbol:
        subcarriers.setflags(write=False)
    return NRPRSResourceGrid(
        grid=grid,
        resource_mask=mask,
        symbol_indices=symbols,
        subcarrier_indices=tuple(indices_by_symbol),
        sequence_id=prs_sequence_id,
        slot_index=slot_index,
        numerology=numerology,
        subcarrier_spacing_hz=spacing,
        beta_prs=beta,
        comb_size=comb_size,
    )


def map_nr_dl_prs_resource_set(
    *,
    resources: Sequence[NRPRSResourceConfiguration],
    slot_symbol_count: int,
    subcarrier_count: int,
    slot_index: int,
    numerology: int,
    subcarrier_spacing_hz: float,
    reserved_symbol_mask: BoolArray | None = None,
    reserved_resource_mask: BoolArray | None = None,
    max_resource_count: int = 64,
    max_total_resource_cells: int = 1_048_576,
) -> NRPRSResourceSetGrid:
    """Map and compose a bounded set of resolved PRS resources in one slot.

    The returned ``grid`` sums resource contributions on shared REs, matching
    linear waveform superposition. ``collision_mask`` and each member mask keep
    those overlaps observable. This API does not decode RRC, choose beams,
    schedule a resource set, or combine antenna ports; timing/muting admission
    remains the caller's responsibility.
    """
    if isinstance(resources, (str, bytes)) or not isinstance(resources, Sequence):
        raise TypeError("resources must be a sequence of NRPRSResourceConfiguration values")
    configurations = tuple(resources)
    if not configurations:
        raise ValueError("resources must be nonempty")
    if any(not isinstance(item, NRPRSResourceConfiguration) for item in configurations):
        raise TypeError("resources must contain NRPRSResourceConfiguration values")
    if any(
        not isinstance(item.resource_id, str)
        or not item.resource_id
        or item.resource_id != item.resource_id.strip()
        for item in configurations
    ):
        raise ValueError("every PRS resource needs a nonempty canonical resource_id")
    resource_ids = tuple(item.resource_id for item in configurations)
    if len(set(resource_ids)) != len(resource_ids):
        raise ValueError("PRS resource IDs must be unique within a resource set")
    max_resource_count = _integer(
        max_resource_count, label="max_resource_count", minimum=1
    )
    max_total_resource_cells = _integer(
        max_total_resource_cells, label="max_total_resource_cells", minimum=1
    )
    if len(configurations) > max_resource_count:
        raise ValueError("PRS resource count exceeds max_resource_count")
    slot_symbol_count = _integer(
        slot_symbol_count, label="slot_symbol_count", minimum=12, maximum=14
    )
    if slot_symbol_count not in (12, 14):
        raise ValueError("slot_symbol_count must be 12 or 14")
    subcarrier_count = _integer(subcarrier_count, label="subcarrier_count", minimum=1)
    numerology = _integer(numerology, label="numerology", minimum=0, maximum=6)
    slot_index = _integer(
        slot_index,
        label="slot_index",
        minimum=0,
        maximum=10 * (1 << numerology) - 1,
    )
    total_cells = len(configurations) * slot_symbol_count * subcarrier_count
    if total_cells > max_total_resource_cells:
        raise ValueError("PRS resource set exceeds max_total_resource_cells")

    member_grids = tuple(
        map_nr_dl_prs_resource(
            slot_symbol_count=slot_symbol_count,
            subcarrier_count=subcarrier_count,
            resource_start_subcarrier=configuration.resource_start_subcarrier,
            resource_bandwidth_rbs=configuration.resource_bandwidth_rbs,
            prs_sequence_id=configuration.prs_sequence_id,
            slot_index=slot_index,
            numerology=numerology,
            subcarrier_spacing_hz=subcarrier_spacing_hz,
            comb_size=configuration.comb_size,
            resource_element_offset=configuration.resource_element_offset,
            start_symbol=configuration.start_symbol,
            num_symbols=configuration.num_symbols,
            beta_prs=configuration.beta_prs,
            reserved_symbol_mask=reserved_symbol_mask,
            reserved_resource_mask=reserved_resource_mask,
        )
        for configuration in configurations
    )
    masks = tuple(member.resource_mask for member in member_grids)
    occupancy = np.sum(np.stack(masks, axis=0), axis=0, dtype=np.int64)
    collision_mask = occupancy > 1
    composite = np.sum(
        np.stack(tuple(member.grid for member in member_grids), axis=0),
        axis=0,
        dtype=np.complex128,
    )
    if not np.isfinite(composite).all():
        raise ValueError("composed PRS resource set contains non-finite samples")
    composite.setflags(write=False)
    collision_mask.setflags(write=False)
    return NRPRSResourceSetGrid(
        grid=composite,
        resource_ids=resource_ids,
        resource_masks=masks,
        collision_mask=collision_mask,
        colliding_resource_element_count=int(np.count_nonzero(collision_mask)),
        numerology=numerology,
        slot_index=slot_index,
    )


@dataclass(frozen=True, slots=True)
class NRPRSResourceSetTiming:
    """Decoded timing and muting fields for one downlink PRS resource set.

    Muting bitmaps are supplied in logical increasing-index order. The caller
    is responsible for mapping ASN.1 BIT STRING bit order into this sequence.
    """

    numerology: int
    periodicity_slots: int
    periodicity_offset_slots: int
    resource_slot_offset: int = 0
    repetition_factor: int = 1
    time_gap_slots: int | None = None
    option1_bitmap: tuple[bool, ...] | None = None
    option2_bitmap: tuple[bool, ...] | None = None
    muting_bit_repetition_factor: int = 1

    def __post_init__(self) -> None:
        numerology = _integer(self.numerology, label="numerology", minimum=0, maximum=3)
        period_bases = {base * (1 << numerology) for base in _PRS_PERIODICITY_BASE_SLOTS}
        period = _integer(self.periodicity_slots, label="periodicity_slots", minimum=1)
        if period not in period_bases:
            raise ValueError("periodicity_slots is not supported for the configured numerology")
        _integer(
            self.periodicity_offset_slots,
            label="periodicity_offset_slots",
            minimum=0,
            maximum=period - 1,
        )
        resource_offset = _integer(
            self.resource_slot_offset, label="resource_slot_offset", minimum=0
        )
        repetitions = _integer(self.repetition_factor, label="repetition_factor", minimum=1)
        if repetitions not in _PRS_REPETITION_FACTORS:
            raise ValueError("repetition_factor is not a supported TS 38.214 value")
        if self.time_gap_slots is None:
            if repetitions > 1:
                raise ValueError("time_gap_slots is required when repetition_factor exceeds one")
            gap = 1
        else:
            gap = _integer(self.time_gap_slots, label="time_gap_slots", minimum=1)
        if gap not in _PRS_TIME_GAPS:
            raise ValueError("time_gap_slots is not a supported TS 38.214 value")
        if resource_offset + (repetitions - 1) * gap >= period:
            raise ValueError("the PRS resource repetitions extend beyond the set periodicity")
        muting_repeat = _integer(
            self.muting_bit_repetition_factor,
            label="muting_bit_repetition_factor",
            minimum=1,
        )
        if muting_repeat not in _PRS_MUTING_REPETITION_FACTORS:
            raise ValueError("muting_bit_repetition_factor must be one of 1, 2, 4, or 8")
        bitmap1 = self._validate_bitmap(self.option1_bitmap, "option1_bitmap")
        bitmap2 = self._validate_bitmap(self.option2_bitmap, "option2_bitmap")
        if bitmap1 is not None:
            if len(bitmap1) not in _PRS_OPTION1_BITMAP_LENGTHS:
                raise ValueError("option1_bitmap length must be 2, 4, 6, 8, 16, or 32")
            if period * muting_repeat * len(bitmap1) > (1 << numerology) * 10240:
                raise ValueError("option1 muting cycle exceeds the 10-second SFN duration")
        if bitmap2 is not None and len(bitmap2) != repetitions:
            raise ValueError("option2_bitmap length must equal repetition_factor")
        object.__setattr__(self, "option1_bitmap", bitmap1)
        object.__setattr__(self, "option2_bitmap", bitmap2)
        object.__setattr__(self, "time_gap_slots", gap)

    @staticmethod
    def _validate_bitmap(
        bitmap: Sequence[bool] | None, label: str
    ) -> tuple[bool, ...] | None:
        if bitmap is None:
            return None
        if isinstance(bitmap, (str, bytes)) or not isinstance(bitmap, Sequence) or not bitmap:
            raise TypeError(f"{label} must be a nonempty sequence of Boolean values")
        if any(not isinstance(bit, (bool, np.bool_)) for bit in bitmap):
            raise TypeError(f"{label} must contain only Boolean values")
        return tuple(bool(bit) for bit in bitmap)


@dataclass(frozen=True, slots=True)
class NRPRSResourceOccasion:
    """Whether one absolute slot carries a specified PRS resource repetition."""

    system_frame_number: int
    slot_in_frame: int
    absolute_slot: int
    is_resource_occasion: bool
    transmitted: bool
    set_instance_index: int | None
    repetition_index: int | None
    option1_bit_index: int | None
    option2_bit_index: int | None
    status: str


def nr_dl_prs_resource_occasion(
    *,
    system_frame_number: int,
    slot_in_frame: int,
    timing: NRPRSResourceSetTiming,
) -> NRPRSResourceOccasion:
    """Resolve PRS periodicity, resource repetitions, and Option-1/2 muting.

    The frame number follows the NR 10-bit SFN cycle. Resource-set offsets are
    interpreted relative to SFN 0 slot 0, as in the decoded TS 38.214 fields.
    """
    if not isinstance(timing, NRPRSResourceSetTiming):
        raise TypeError("timing must be NRPRSResourceSetTiming")
    numerology = timing.numerology
    slots_per_frame = 10 * (1 << numerology)
    sfn = _integer(
        system_frame_number, label="system_frame_number", minimum=0, maximum=1023
    )
    slot = _integer(slot_in_frame, label="slot_in_frame", minimum=0, maximum=slots_per_frame - 1)
    absolute_slot = sfn * slots_per_frame + slot
    base_slot = timing.periodicity_offset_slots + timing.resource_slot_offset
    relative_slot = absolute_slot - base_slot
    periodicity = timing.periodicity_slots
    set_instance_index = relative_slot // periodicity
    slot_in_period = relative_slot % periodicity
    gap = timing.time_gap_slots if timing.time_gap_slots is not None else 1
    repetition_index = slot_in_period // gap
    is_occasion = (
        slot_in_period % gap == 0 and repetition_index < timing.repetition_factor
    )
    if not is_occasion:
        return NRPRSResourceOccasion(
            system_frame_number=sfn,
            slot_in_frame=slot,
            absolute_slot=absolute_slot,
            is_resource_occasion=False,
            transmitted=False,
            set_instance_index=set_instance_index,
            repetition_index=None,
            option1_bit_index=None,
            option2_bit_index=None,
            status="NOT_A_RESOURCE_OCCASION",
        )

    option1_index = None
    option2_index = None
    option1_active = True
    option2_active = True
    if timing.option1_bitmap is not None:
        option1_index = (
            relative_slot
            // (timing.periodicity_slots * timing.muting_bit_repetition_factor)
        ) % len(timing.option1_bitmap)
        option1_active = timing.option1_bitmap[option1_index]
    if timing.option2_bitmap is not None:
        option2_index = repetition_index % len(timing.option2_bitmap)
        option2_active = timing.option2_bitmap[option2_index]
    transmitted = option1_active and option2_active
    if transmitted:
        status = "TRANSMIT"
    elif not option1_active and not option2_active:
        status = "MUTED_BY_OPTION_1_AND_2"
    elif not option1_active:
        status = "MUTED_BY_OPTION_1"
    else:
        status = "MUTED_BY_OPTION_2"
    return NRPRSResourceOccasion(
        system_frame_number=sfn,
        slot_in_frame=slot,
        absolute_slot=absolute_slot,
        is_resource_occasion=True,
        transmitted=transmitted,
        set_instance_index=set_instance_index,
        repetition_index=repetition_index,
        option1_bit_index=option1_index,
        option2_bit_index=option2_index,
        status=status,
    )


@dataclass(frozen=True, slots=True, eq=False)
class NRPRSTimingProfile:
    """Noncoherent matched-filter delay profile from a mapped PRS resource."""

    delay_grid_s: FloatArray
    profile_power: FloatArray
    peak_delay_s: float
    peak_power: float
    peak_to_median_ratio: float
    comb_ambiguity_period_s: float
    semantics: str = "strongest_component_delay_on_caller_grid"


@dataclass(frozen=True, slots=True)
class NRPRSDLRSTDMeasurement:
    """DL-RSTD from externally estimated received subframe-start times."""

    target_tp_id: str
    reference_tp_id: str
    target_subframe_start_time_s: float
    selected_reference_subframe_start_time_s: float
    selected_reference_index: int
    rstd_s: float
    frequency_range: int
    measurement_reference_point: str
    semantics: str = "target_minus_closest_reference_subframe_start"


_NR_TIMING_QUALITY_RESOLUTIONS_M = (0.1, 1.0, 10.0, 30.0)


@dataclass(frozen=True, slots=True)
class NRPRSTimingQuality:
    """NR-TimingQuality value and its declared metre resolution.

    The standard carries an integer in 0..31 and one of four metre
    resolutions. ``from_uncertainty_m`` rounds upward so the reported bin does
    not claim less uncertainty than the supplied estimate. The estimate itself
    remains caller-provided measurement evidence.
    """

    value: int
    resolution_m: float

    def __post_init__(self) -> None:
        _integer(self.value, label="timing quality value", minimum=0, maximum=31)
        if (
            isinstance(self.resolution_m, bool)
            or not isinstance(self.resolution_m, Real)
            or self.resolution_m not in _NR_TIMING_QUALITY_RESOLUTIONS_M
        ):
            raise ValueError("timing quality resolution must be 0.1, 1, 10, or 30 metres")

    @classmethod
    def from_uncertainty_m(
        cls, uncertainty_m: Real, *, resolution_m: float
    ) -> NRPRSTimingQuality:
        try:
            finite_uncertainty = math.isfinite(float(uncertainty_m))
        except (OverflowError, TypeError, ValueError):
            finite_uncertainty = False
        if isinstance(uncertainty_m, bool) or not isinstance(uncertainty_m, Real) or not finite_uncertainty or uncertainty_m < 0:
            raise ValueError("timing uncertainty must be finite, real, and nonnegative")
        if (
            isinstance(resolution_m, bool)
            or not isinstance(resolution_m, Real)
            or resolution_m not in _NR_TIMING_QUALITY_RESOLUTIONS_M
        ):
            raise ValueError("timing quality resolution must be 0.1, 1, 10, or 30 metres")
        value = math.ceil(float(uncertainty_m) / float(resolution_m))
        if value > 31:
            raise ValueError("timing uncertainty exceeds the selected NR-TimingQuality range")
        return cls(value=value, resolution_m=float(resolution_m))

    @property
    def represented_uncertainty_m(self) -> float:
        return self.value * self.resolution_m

    def to_dict(self) -> dict[str, int | float]:
        return {"value": self.value, "resolution_m": self.resolution_m}


@dataclass(frozen=True, slots=True)
class NRPRSTDReportElement:
    """One TP's DL-RSTD semantic report row, before LPP ASN.1 encoding."""

    tp_id: str
    measurement_subframe_start_time_s: float
    selected_reference_subframe_start_time_s: float
    rstd_s: float
    timing_quality: NRPRSTimingQuality
    is_reference: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "tp_id": self.tp_id,
            "measurement_subframe_start_time_s": self.measurement_subframe_start_time_s,
            "selected_reference_subframe_start_time_s": (
                self.selected_reference_subframe_start_time_s
            ),
            "rstd_s": self.rstd_s,
            "timing_quality": self.timing_quality.to_dict(),
            "is_reference": self.is_reference,
        }


@dataclass(frozen=True, slots=True)
class NRPRSTDReport:
    """Input-scoped DL-TDOA semantic report with the required zero reference row.

    This is not an LPP encoder: timestamps are retained as supplied seconds,
    and resource/cell identifiers and ASN.1 extension fields are not inferred.
    """

    reference_tp_id: str
    reference_time_candidates_s: tuple[float, ...]
    frequency_range: int
    measurement_reference_point: str
    elements: tuple[NRPRSTDReportElement, ...]
    schema: str = "umcd.nr_dl_tdoa_measurement_report.v1"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "reference_tp_id": self.reference_tp_id,
            "reference_time_candidates_s": list(self.reference_time_candidates_s),
            "frequency_range": self.frequency_range,
            "measurement_reference_point": self.measurement_reference_point,
            "elements": [element.to_dict() for element in self.elements],
            "semantics": "pre_lpp_asn1_semantic_report",
        }


def build_nr_dl_tdoa_measurement_report(
    *,
    reference_tp_id: str,
    reference_subframe_start_times_s: Sequence[float],
    target_subframe_start_times_s: Mapping[str, Real],
    timing_quality_by_tp: Mapping[str, NRPRSTimingQuality],
    frequency_range: int,
) -> NRPRSTDReport:
    """Build bounded DL-RSTD rows from already-estimated TP subframe starts.

    The reference TP is included with zero RSTD as required by the NR DL-TDOA
    report semantics. Each neighbour uses the closest supplied reference
    subframe-start estimate. No PRS waveform is processed and no timestamp,
    resource ID, or uncertainty estimate is inferred by this composition API.
    """
    if not isinstance(reference_tp_id, str) or not reference_tp_id or reference_tp_id != reference_tp_id.strip():
        raise ValueError("reference_tp_id must be a nonempty canonical string")
    if not isinstance(target_subframe_start_times_s, Mapping):
        raise TypeError("target_subframe_start_times_s must be a TP-to-time mapping")
    if not isinstance(timing_quality_by_tp, Mapping):
        raise TypeError("timing_quality_by_tp must be a TP-to-quality mapping")
    target_ids = tuple(target_subframe_start_times_s)
    if any(
        not isinstance(tp_id, str) or not tp_id or tp_id != tp_id.strip()
        for tp_id in target_ids
    ):
        raise ValueError("target TP IDs must be nonempty canonical strings")
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("target TP IDs must be unique")
    if reference_tp_id in target_ids:
        raise ValueError("reference TP must not also appear in target measurements")
    expected_quality_ids = {reference_tp_id, *target_ids}
    if set(timing_quality_by_tp) != expected_quality_ids:
        raise ValueError("timing_quality_by_tp must cover exactly the reference and target TPs")
    for tp_id, quality in timing_quality_by_tp.items():
        if not isinstance(quality, NRPRSTimingQuality):
            raise TypeError(f"timing quality for {tp_id!r} must be NRPRSTimingQuality")
    frequency_range = _integer(
        frequency_range, label="frequency_range", minimum=1, maximum=2
    )
    if not isinstance(reference_subframe_start_times_s, Sequence) or isinstance(
        reference_subframe_start_times_s, (str, bytes)
    ):
        raise TypeError("reference_subframe_start_times_s must be a nonempty sequence")
    if not reference_subframe_start_times_s:
        raise ValueError("reference_subframe_start_times_s must be nonempty")
    reference_times: list[float] = []
    for index, value in enumerate(reference_subframe_start_times_s):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"reference subframe start {index} must be finite and real")
        reference_times.append(float(value))

    elements = [
        NRPRSTDReportElement(
            tp_id=reference_tp_id,
            measurement_subframe_start_time_s=reference_times[0],
            selected_reference_subframe_start_time_s=reference_times[0],
            rstd_s=0.0,
            timing_quality=timing_quality_by_tp[reference_tp_id],
            is_reference=True,
        )
    ]
    for tp_id in sorted(target_ids):
        measurement = measure_nr_dl_rstd(
            target_tp_id=tp_id,
            reference_tp_id=reference_tp_id,
            target_subframe_start_time_s=target_subframe_start_times_s[tp_id],
            reference_subframe_start_times_s=reference_times,
            frequency_range=frequency_range,
        )
        elements.append(
            NRPRSTDReportElement(
                tp_id=tp_id,
                measurement_subframe_start_time_s=measurement.target_subframe_start_time_s,
                selected_reference_subframe_start_time_s=(
                    measurement.selected_reference_subframe_start_time_s
                ),
                rstd_s=measurement.rstd_s,
                timing_quality=timing_quality_by_tp[tp_id],
                is_reference=False,
            )
        )
    return NRPRSTDReport(
        reference_tp_id=reference_tp_id,
        reference_time_candidates_s=tuple(reference_times),
        frequency_range=frequency_range,
        measurement_reference_point=(
            "UE_antenna_connector" if frequency_range == 1 else "UE_antenna"
        ),
        elements=tuple(elements),
    )


def measure_nr_dl_rstd(
    *,
    target_tp_id: str,
    reference_tp_id: str,
    target_subframe_start_time_s: float,
    reference_subframe_start_times_s: Sequence[float],
    frequency_range: int,
) -> NRPRSDLRSTDMeasurement:
    """Compute target-minus-nearest-reference NR DL-RSTD from start estimates.

    The received subframe-start estimates are inputs; this function does not
    derive them from PRS samples. It applies the TS 38.215 nearest-reference
    rule and records the required UE measurement reference point by frequency
    range. An exactly equidistant reference pair is rejected as ambiguous.
    """

    for value, label in (
        (target_tp_id, "target_tp_id"),
        (reference_tp_id, "reference_tp_id"),
    ):
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"{label} must be a nonempty canonical string")
    if target_tp_id == reference_tp_id:
        raise ValueError("target and reference TP IDs must differ")
    if (
        isinstance(target_subframe_start_time_s, bool)
        or not isinstance(target_subframe_start_time_s, Real)
        or not math.isfinite(float(target_subframe_start_time_s))
    ):
        raise ValueError("target_subframe_start_time_s must be finite and real")
    if not isinstance(reference_subframe_start_times_s, Sequence) or isinstance(
        reference_subframe_start_times_s, (str, bytes)
    ):
        raise TypeError("reference_subframe_start_times_s must be a nonempty sequence")
    if not reference_subframe_start_times_s:
        raise ValueError("reference_subframe_start_times_s must be nonempty")
    reference_times: list[float] = []
    for index, value in enumerate(reference_subframe_start_times_s):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"reference subframe start {index} must be finite and real")
        reference_times.append(float(value))
    frequency_range = _integer(
        frequency_range, label="frequency_range", minimum=1, maximum=2
    )
    target_time = float(target_subframe_start_time_s)
    offsets = [target_time - value for value in reference_times]
    if any(not math.isfinite(value) for value in offsets):
        raise ValueError("subframe-start differences must remain finite")
    nearest_distance = min(abs(value) for value in offsets)
    nearest = [
        index
        for index, value in enumerate(offsets)
        if math.isclose(abs(value), nearest_distance, rel_tol=1e-12, abs_tol=1e-15)
    ]
    nearest_times = {reference_times[index] for index in nearest}
    if len(nearest_times) != 1:
        raise ValueError("target subframe start is equidistant from multiple reference starts")
    selected_index = nearest[0]
    reference_time = reference_times[selected_index]
    return NRPRSDLRSTDMeasurement(
        target_tp_id=target_tp_id,
        reference_tp_id=reference_tp_id,
        target_subframe_start_time_s=target_time,
        selected_reference_subframe_start_time_s=reference_time,
        selected_reference_index=selected_index,
        rstd_s=target_time - reference_time,
        frequency_range=frequency_range,
        measurement_reference_point=(
            "UE_antenna_connector" if frequency_range == 1 else "UE_antenna"
        ),
    )


def nr_dl_prs_matched_delay_profile(
    received_grid: ComplexArray,
    resource: NRPRSResourceGrid,
    *,
    delay_grid_s: FloatArray,
) -> NRPRSTimingProfile:
    """Compute a noncoherent PRS delay profile over an explicit delay grid.

    This is a receiver-observable strongest-component timing baseline. It does
    not claim first-arrival ToA; PRS comb periodicity, bandwidth, multipath,
    synchronization error, and the caller's delay grid limit interpretation.
    ``comb_ambiguity_period_s`` reports the delay period implied by the
    configured comb spacing; it does not resolve that ambiguity.
    """
    received = np.asarray(received_grid)
    if (
        received.shape != resource.grid.shape
        or not np.iscomplexobj(received)
        or not np.isfinite(received).all()
    ):
        raise ValueError("received_grid must be finite complex and match the mapped PRS grid")
    comb_size = _integer(resource.comb_size, label="resource.comb_size", minimum=1)
    spacing = resource.subcarrier_spacing_hz
    if (
        isinstance(spacing, bool)
        or not isinstance(spacing, Real)
        or not math.isfinite(float(spacing))
        or spacing <= 0.0
    ):
        raise ValueError("resource.subcarrier_spacing_hz must be finite and positive")
    raw_delays = np.asarray(delay_grid_s)
    if raw_delays.dtype.kind not in "iuf":
        raise TypeError("delay_grid_s must contain real numeric values")
    delays = np.asarray(raw_delays, dtype=np.float64)
    if (
        delays.ndim != 1
        or delays.size < 2
        or not np.isfinite(delays).all()
        or np.any(delays < 0)
        or np.any(np.diff(delays) <= 0)
    ):
        raise ValueError("delay_grid_s must be finite, nonnegative, and strictly increasing")
    profile = np.zeros(delays.shape, dtype=np.float64)
    for symbol, subcarriers in zip(resource.symbol_indices, resource.subcarrier_indices, strict=True):
        transmit = resource.grid[symbol, subcarriers]
        observed = received[symbol, subcarriers]
        channel = observed * np.conjugate(transmit) / (np.abs(transmit) ** 2)
        frequencies = subcarriers.astype(np.float64) * resource.subcarrier_spacing_hz
        steering = np.exp(2j * np.pi * frequencies[:, None] * delays[None, :])
        response = np.mean(channel[:, None] * steering, axis=0)
        profile += np.abs(response) ** 2
    if not np.isfinite(profile).all():
        raise ValueError("PRS matched-filter profile overflowed")
    peak_index = int(np.argmax(profile))
    peak_power = float(profile[peak_index])
    positive = profile[profile > 0]
    median = float(np.median(positive)) if positive.size else 0.0
    ratio = peak_power / median if median > 0 else (math.inf if peak_power > 0 else 0.0)
    if not math.isfinite(ratio):
        ratio = float(np.finfo(np.float64).max)
    ambiguity_period = 1.0 / (comb_size * float(spacing))
    delays = np.array(delays, copy=True)
    profile.setflags(write=False)
    delays.setflags(write=False)
    return NRPRSTimingProfile(
        delay_grid_s=delays,
        profile_power=profile,
        peak_delay_s=float(delays[peak_index]),
        peak_power=peak_power,
        peak_to_median_ratio=ratio,
        comb_ambiguity_period_s=ambiguity_period,
    )
