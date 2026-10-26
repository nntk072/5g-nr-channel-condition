"""NR BCCH-BCH MIB and empty message-class alternative bit parsing."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

_BIT_COUNT = 24


@dataclass(frozen=True)
class PBCHMIB:
    """Decoded Release-19 MIB fields carried by the PBCH BCH message."""

    system_frame_number_msb: int
    subcarrier_spacing_common: Literal["scs15or60", "scs30or120"]
    ssb_subcarrier_offset: int
    dmrs_type_a_position: Literal["pos2", "pos3"]
    control_resource_set_zero: int
    search_space_zero: int
    cell_barred: Literal["barred", "notBarred"]
    intra_freq_reselection: Literal["allowed", "notAllowed"]
    spare: int


@dataclass(frozen=True, slots=True)
class PBCHMessageClassExtension:
    """Empty BCCH-BCH alternative with preserved uninterpreted trailing bits.

    TS 38.331 defines this CHOICE alternative as an empty SEQUENCE. The
    24-bit PBCH-side representation can still contain trailing bits; this
    object preserves them without assigning ASN.1 field semantics.
    """

    trailing_bits: tuple[int, ...] = (0,) * (_BIT_COUNT - 1)

    def __post_init__(self) -> None:
        values = tuple(self.trailing_bits)
        if len(values) != _BIT_COUNT - 1:
            raise ValueError("trailing_bits must contain 23 bits")
        if any(
            isinstance(bit, (bool, np.bool_)) or not isinstance(bit, Integral) or int(bit) not in (0, 1)
            for bit in values
        ):
            raise ValueError("trailing_bits must contain only integer binary values")
        object.__setattr__(self, "trailing_bits", tuple(int(bit) for bit in values))


PBCHBCCHMessage: TypeAlias = PBCHMIB | PBCHMessageClassExtension


def nr_bcch_bch_message_parse(bits: NDArray[np.generic]) -> PBCHBCCHMessage:
    """Parse the 24-bit BCCH-BCH CHOICE and retain its selected alternative."""
    array = _binary_vector(bits)
    if int(array[0]) == 0:
        return nr_pbch_mib_parse(array)
    return PBCHMessageClassExtension(tuple(int(bit) for bit in array[1:]))


def nr_bcch_bch_message_serialize(message: PBCHBCCHMessage) -> NDArray[np.uint8]:
    """Serialize either BCCH-BCH alternative into its 24-bit representation."""
    if isinstance(message, PBCHMIB):
        return nr_pbch_mib_serialize(message)
    if not isinstance(message, PBCHMessageClassExtension):
        raise TypeError("message must be PBCHMIB or PBCHMessageClassExtension")
    bits = np.asarray((1, *message.trailing_bits), dtype=np.uint8)
    bits.setflags(write=False)
    return bits


def nr_pbch_mib_parse(bcch_bch_bits: NDArray[np.generic]) -> PBCHMIB:
    """Parse a 24-bit BCCH-BCH message containing the MIB alternative.

    The first bit is the one-bit CHOICE selector for ``mib`` versus
    ``messageClassExtension``. The remaining 23 bits are the fixed MIB
    fields, in their TS 38.331 ASN.1 sequence order. The separate physical
    k_SSB MSB and SFN least-significant bits are not part of this message.
    """
    bits = _binary_vector(bcch_bch_bits)
    if int(bits[0]) != 0:
        raise ValueError("BCCH-BCH message selects messageClassExtension, not mib")

    return PBCHMIB(
        system_frame_number_msb=_read_uint(bits, 1, 6),
        subcarrier_spacing_common=("scs15or60", "scs30or120")[int(bits[7])],
        ssb_subcarrier_offset=_read_uint(bits, 8, 4),
        dmrs_type_a_position=("pos2", "pos3")[int(bits[12])],
        control_resource_set_zero=_read_uint(bits, 13, 4),
        search_space_zero=_read_uint(bits, 17, 4),
        cell_barred=("barred", "notBarred")[int(bits[21])],
        intra_freq_reselection=("allowed", "notAllowed")[int(bits[22])],
        spare=int(bits[23]),
    )


def nr_pbch_mib_serialize(mib: PBCHMIB) -> NDArray[np.uint8]:
    """Serialize a fixed-schema MIB alternative as a 24-bit BCCH-BCH value."""
    if not isinstance(mib, PBCHMIB):
        raise TypeError("mib must be a PBCHMIB")

    bits = np.zeros(_BIT_COUNT, dtype=np.uint8)
    bits[1:7] = _uint_bits(mib.system_frame_number_msb, 6, "system_frame_number_msb")
    bits[7] = _enum_index(
        mib.subcarrier_spacing_common,
        ("scs15or60", "scs30or120"),
        "subcarrier_spacing_common",
    )
    bits[8:12] = _uint_bits(mib.ssb_subcarrier_offset, 4, "ssb_subcarrier_offset")
    bits[12] = _enum_index(mib.dmrs_type_a_position, ("pos2", "pos3"), "dmrs_type_a_position")
    bits[13:17] = _uint_bits(mib.control_resource_set_zero, 4, "control_resource_set_zero")
    bits[17:21] = _uint_bits(mib.search_space_zero, 4, "search_space_zero")
    bits[21] = _enum_index(mib.cell_barred, ("barred", "notBarred"), "cell_barred")
    bits[22] = _enum_index(
        mib.intra_freq_reselection,
        ("allowed", "notAllowed"),
        "intra_freq_reselection",
    )
    bits[23] = _uint_bits(mib.spare, 1, "spare")[0]
    bits.setflags(write=False)
    return bits


def _binary_vector(bits: NDArray[np.generic]) -> NDArray[np.uint8]:
    array = np.asarray(bits)
    if array.shape != (_BIT_COUNT,):
        raise ValueError("bcch_bch_bits must have shape (24,)")
    if array.dtype.kind not in "biu" or np.any(array < 0) or np.any(array > 1):
        raise ValueError("bcch_bch_bits must contain only binary values")
    return np.asarray(array, dtype=np.uint8)


def _read_uint(bits: NDArray[np.uint8], start: int, width: int) -> int:
    value = 0
    for bit in bits[start : start + width]:
        value = (value << 1) | int(bit)
    return value


def _uint_bits(value: object, width: int, name: str) -> NDArray[np.uint8]:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    integer = int(value)
    if not 0 <= integer < 1 << width:
        raise ValueError(f"{name} must be in [0, {(1 << width) - 1}]")
    return np.asarray(
        [(integer >> shift) & 1 for shift in range(width - 1, -1, -1)], dtype=np.uint8
    )


def _enum_index(value: object, choices: tuple[str, str], name: str) -> np.uint8:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be one of {choices}")
    try:
        return np.uint8(choices.index(value))
    except ValueError as error:
        raise ValueError(f"{name} must be one of {choices}") from error
