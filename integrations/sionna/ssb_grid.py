"""Resource-grid assembly for one known NR SS/PBCH-block occasion."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from .nr_sync import nr_pss_sequence, nr_sss_sequence
from .pbch_dmrs import PBCHDMRSResourceMap, nr_pbch_dmrs_ssb

_SSB_SUBCARRIER_COUNT = 240
_SSB_SYMBOL_COUNT = 4
_PBCH_DATA_RE_COUNT = 432


@dataclass(frozen=True)
class NRSSBResourceGrid:
    """A single 4×240 SS/PBCH-block grid, before OFDM modulation or scaling."""

    nid1: int
    nid2: int
    physical_cell_id: int
    grid: NDArray[np.complex128]
    pbch_dmrs: PBCHDMRSResourceMap


def _pbch_data_coordinates(
    dmrs: PBCHDMRSResourceMap,
) -> tuple[NDArray[np.int16], NDArray[np.int8]]:
    """Return PBCH data REs in increasing subcarrier then symbol order."""
    dmrs_reserved = set(zip(dmrs.subcarriers.tolist(), dmrs.ofdm_symbols.tolist(), strict=True))
    coordinates = [
        (subcarrier, symbol)
        for subcarrier in range(_SSB_SUBCARRIER_COUNT)
        for symbol in (1, 2, 3)
        if (symbol in (1, 3) or (subcarrier < 48 or subcarrier >= 192))
        and (subcarrier, symbol) not in dmrs_reserved
    ]
    if len(coordinates) != _PBCH_DATA_RE_COUNT:
        raise RuntimeError("internal PBCH data resource count is inconsistent")
    return (
        np.fromiter((item[0] for item in coordinates), dtype=np.int16),
        np.fromiter((item[1] for item in coordinates), dtype=np.int8),
    )


def map_nr_ssb_resource_grid(
    nid1: int,
    nid2: int,
    pbch_symbols: NDArray[np.number],
    *,
    ssb_index_in_half_frame: int,
    half_frame_index: int,
    max_candidate_blocks_per_half_frame: int,
) -> NRSSBResourceGrid:
    """Map PSS, SSS, PBCH-DM-RS, and supplied PBCH symbols into one SSB grid.

    ``pbch_symbols`` must already contain the 432 modulation symbols in the
    order expected by PBCH mapping. This function does not encode or scramble
    a PBCH payload, perform BCH coding, scale signal power, or OFDM-modulate.
    The returned grid contains zeros at all resource elements marked unused
    by the SS/PBCH-block table.
    """
    if isinstance(nid1, (bool, np.bool_)) or not isinstance(nid1, (int, np.integer)):
        raise TypeError("nid1 must be an integer")
    if isinstance(nid2, (bool, np.bool_)) or not isinstance(nid2, (int, np.integer)):
        raise TypeError("nid2 must be an integer")
    nid1_int = int(nid1)
    nid2_int = int(nid2)
    if not 0 <= nid1_int <= 335:
        raise ValueError("nid1 must be in [0, 335]")
    if not 0 <= nid2_int <= 2:
        raise ValueError("nid2 must be in [0, 2]")

    raw_pbch = np.asarray(pbch_symbols)
    if raw_pbch.dtype == np.dtype(bool) or not np.issubdtype(raw_pbch.dtype, np.number):
        raise TypeError("pbch_symbols must be a real or complex numeric array")
    data = np.asarray(raw_pbch, dtype=np.complex128)
    if data.shape != (_PBCH_DATA_RE_COUNT,):
        raise ValueError("pbch_symbols must have shape (432,)")
    if not np.isfinite(data.real).all() or not np.isfinite(data.imag).all():
        raise ValueError("pbch_symbols must be finite")

    cell_id = 3 * nid1_int + nid2_int
    dmrs = nr_pbch_dmrs_ssb(
        cell_id,
        ssb_index_in_half_frame,
        half_frame_index=half_frame_index,
        max_candidate_blocks_per_half_frame=max_candidate_blocks_per_half_frame,
    )
    grid = np.zeros((_SSB_SYMBOL_COUNT, _SSB_SUBCARRIER_COUNT), dtype=np.complex128)
    grid[0, 56:183] = nr_pss_sequence(nid2_int)
    grid[2, 56:183] = nr_sss_sequence(nid1_int, nid2_int)
    grid[dmrs.ofdm_symbols, dmrs.subcarriers] = dmrs.sequence

    data_subcarriers, data_symbols = _pbch_data_coordinates(dmrs)
    grid[data_symbols, data_subcarriers] = data
    grid.setflags(write=False)
    return NRSSBResourceGrid(
        nid1=nid1_int,
        nid2=nid2_int,
        physical_cell_id=cell_id,
        grid=grid,
        pbch_dmrs=dmrs,
    )


def scale_nr_ssb_resource_grid_epre(
    ssb_grid: NDArray[np.generic], *, pss_to_sss_epre_db: float
) -> NDArray[np.complex128]:
    """Apply the TS 38.213 assumed PSS-to-SSS EPRE ratio of 0 or 3 dB.

    SSS, PBCH data, and PBCH DM-RS remain at the same EPRE. The input is
    treated as a relative resource grid; this function does not convert
    ``ss-PBCH-BlockPower`` in dBm to absolute watts or waveform voltage.
    """
    if isinstance(pss_to_sss_epre_db, (bool, np.bool_)) or not isinstance(pss_to_sss_epre_db, Real):
        raise TypeError("pss_to_sss_epre_db must be a real number")
    ratio_db = float(pss_to_sss_epre_db)
    if ratio_db not in (0.0, 3.0):
        raise ValueError("pss_to_sss_epre_db must be 0 or 3 dB")

    source = np.asarray(ssb_grid)
    if source.shape != (_SSB_SYMBOL_COUNT, _SSB_SUBCARRIER_COUNT):
        raise ValueError("ssb_grid must have shape (4, 240)")
    if source.dtype.kind not in "iufc" or source.dtype.kind == "b":
        raise TypeError("ssb_grid must be a real or complex numeric array")
    result = np.asarray(source, dtype=np.complex128).copy()
    if not np.isfinite(result.real).all() or not np.isfinite(result.imag).all():
        raise ValueError("ssb_grid must be finite")

    result[0, 56:183] *= 10.0 ** (ratio_db / 20.0)
    result.setflags(write=False)
    return result
