"""SRS resource-grid types for single, hopped, and positioning allocations."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True, slots=True, eq=False)
class NRSRSResourceGrid:
    """One mapped SRS resource on a slot-local symbol/subcarrier grid."""

    grid: np.ndarray
    mask: np.ndarray
    symbol_indices: tuple[int, ...]
    subcarrier_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        grid = np.asarray(self.grid)
        mask = np.asarray(self.mask)
        if grid.ndim != 2 or not np.iscomplexobj(grid):
            raise ValueError("SRS grid must be complex [slot symbols, subcarriers]")
        if mask.shape != grid.shape or mask.dtype != np.bool_:
            raise ValueError("SRS mask must be boolean and match the grid")
        if not np.isfinite(grid).all():
            raise ValueError("SRS grid must be finite")
        symbols = tuple(self.symbol_indices)
        subcarriers = tuple(self.subcarrier_indices)
        if not symbols or not subcarriers:
            raise ValueError("SRS resource indices must be nonempty")
        if any(
            isinstance(index, bool) or not isinstance(index, Integral)
            for index in (*symbols, *subcarriers)
        ):
            raise TypeError("SRS resource indices must be integers")
        if (
            len(set(symbols)) != len(symbols)
            or len(set(subcarriers)) != len(subcarriers)
            or any(index < 0 or index >= grid.shape[0] for index in symbols)
            or any(index < 0 or index >= grid.shape[1] for index in subcarriers)
        ):
            raise ValueError("SRS resource indices must be unique and inside the grid")
        expected_mask = np.zeros(grid.shape, dtype=np.bool_)
        expected_mask[np.ix_(symbols, subcarriers)] = True
        if not np.array_equal(mask, expected_mask):
            raise ValueError("SRS mask disagrees with its declared indices")
        grid_view = grid.view()
        mask_view = mask.view()
        grid_view.setflags(write=False)
        mask_view.setflags(write=False)
        object.__setattr__(self, "grid", grid_view)
        object.__setattr__(self, "mask", mask_view)
        object.__setattr__(self, "symbol_indices", tuple(int(index) for index in symbols))
        object.__setattr__(self, "subcarrier_indices", tuple(int(index) for index in subcarriers))


@dataclass(frozen=True, slots=True, eq=False)
class NRSRSHoppedResourceGrid:
    """Mapped SRS with per-symbol frequency-hop and resource-element rosters."""

    grid: np.ndarray
    mask: np.ndarray
    symbol_indices: tuple[int, ...]
    subcarrier_indices_by_symbol: tuple[tuple[int, ...], ...]
    position_indices_by_symbol: tuple[tuple[int, ...], ...]
    n_srs_by_symbol: tuple[int | None, ...]
    positioning_offsets: tuple[int, ...]

    def __post_init__(self) -> None:
        grid = np.asarray(self.grid)
        mask = np.asarray(self.mask)
        symbols = tuple(self.symbol_indices)
        subcarriers = tuple(tuple(row) for row in self.subcarrier_indices_by_symbol)
        positions = tuple(tuple(row) for row in self.position_indices_by_symbol)
        counters = tuple(self.n_srs_by_symbol)
        offsets = tuple(self.positioning_offsets)
        if grid.ndim != 2 or not np.iscomplexobj(grid):
            raise ValueError("hopped SRS grid must be complex [slot symbols, subcarriers]")
        if mask.shape != grid.shape or mask.dtype != np.bool_:
            raise ValueError("hopped SRS mask must be boolean and match the grid")
        if not np.isfinite(grid).all():
            raise ValueError("hopped SRS grid must be finite")
        rosters = (subcarriers, positions, counters, offsets)
        if not symbols or any(len(roster) != len(symbols) for roster in rosters):
            raise ValueError("hopped SRS symbol and per-symbol rosters must align")
        if any(not row for row in (*subcarriers, *positions)):
            raise ValueError("hopped SRS per-symbol rosters must be nonempty")
        indices = (
            *symbols,
            *offsets,
            *(index for roster in (*subcarriers, *positions) for index in roster),
        )
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in indices):
            raise TypeError("hopped SRS indices and counters must be integers")
        if any(
            value is not None and (isinstance(value, bool) or not isinstance(value, Integral))
            for value in counters
        ):
            raise TypeError("hopped SRS counters must be integers or None")
        if (
            len(set(symbols)) != len(symbols)
            or any(symbol < 0 or symbol >= grid.shape[0] for symbol in symbols)
            or any(
                len(set(row)) != len(row)
                or any(index < 0 or index >= grid.shape[1] for index in row)
                for row in subcarriers
            )
            or any(value < 0 for value in offsets)
            or any(value is not None and value < 0 for value in counters)
            or any(value < 0 for row in positions for value in row)
        ):
            raise ValueError("hopped SRS indices and counters must be nonnegative and in-grid")
        expected_mask = np.zeros(grid.shape, dtype=np.bool_)
        for symbol, row in zip(symbols, subcarriers, strict=True):
            expected_mask[symbol, np.asarray(row, dtype=np.intp)] = True
        if not np.array_equal(mask, expected_mask):
            raise ValueError("hopped SRS mask disagrees with its per-symbol RE roster")
        grid_view = grid.view()
        mask_view = mask.view()
        grid_view.setflags(write=False)
        mask_view.setflags(write=False)
        object.__setattr__(self, "grid", grid_view)
        object.__setattr__(self, "mask", mask_view)
        object.__setattr__(self, "symbol_indices", tuple(int(value) for value in symbols))
        object.__setattr__(
            self,
            "subcarrier_indices_by_symbol",
            tuple(tuple(int(value) for value in row) for row in subcarriers),
        )
        object.__setattr__(
            self,
            "position_indices_by_symbol",
            tuple(tuple(int(value) for value in row) for row in positions),
        )
        object.__setattr__(
            self,
            "n_srs_by_symbol",
            tuple(None if value is None else int(value) for value in counters),
        )
        object.__setattr__(self, "positioning_offsets", tuple(int(value) for value in offsets))


@dataclass(frozen=True, slots=True, eq=False)
class NRSRSHoppedMultiportResourceGrid:
    """Per-port grids for a hopped SRS resource with shared RE rosters."""

    grid: np.ndarray
    mask: np.ndarray
    symbol_indices: tuple[int, ...]
    subcarrier_indices_by_symbol: tuple[tuple[int, ...], ...]
    position_indices_by_symbol: tuple[tuple[int, ...], ...]
    n_srs_by_symbol: tuple[int | None, ...]
    positioning_offsets: tuple[int, ...]

    def __post_init__(self) -> None:
        grid = np.asarray(self.grid)
        mask = np.asarray(self.mask)
        symbols = tuple(self.symbol_indices)
        subcarriers = tuple(tuple(row) for row in self.subcarrier_indices_by_symbol)
        positions = tuple(tuple(row) for row in self.position_indices_by_symbol)
        counters = tuple(self.n_srs_by_symbol)
        offsets = tuple(self.positioning_offsets)
        if grid.ndim != 3 or not np.iscomplexobj(grid):
            raise ValueError(
                "hopped multiport grid must be complex [ports, slot symbols, subcarriers]"
            )
        if grid.shape[0] < 1 or not np.isfinite(grid).all():
            raise ValueError("hopped multiport grid must have ports and finite values")
        if mask.shape != grid.shape or mask.dtype != np.bool_:
            raise ValueError("hopped multiport mask must be boolean and match the grid")
        rosters = (subcarriers, positions, counters, offsets)
        if not symbols or any(len(roster) != len(symbols) for roster in rosters):
            raise ValueError("hopped multiport symbol and per-symbol rosters must align")
        if any(not row for row in (*subcarriers, *positions)):
            raise ValueError("hopped multiport per-symbol rosters must be nonempty")
        indices = (
            *symbols,
            *offsets,
            *(index for roster in (*subcarriers, *positions) for index in roster),
        )
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in indices):
            raise TypeError("hopped multiport indices must be integers")
        if any(
            value is not None and (isinstance(value, bool) or not isinstance(value, Integral))
            for value in counters
        ):
            raise TypeError("hopped multiport counters must be integers or None")
        if (
            len(set(symbols)) != len(symbols)
            or any(symbol < 0 or symbol >= grid.shape[1] for symbol in symbols)
            or any(
                len(set(row)) != len(row)
                or any(index < 0 or index >= grid.shape[2] for index in row)
                for row in subcarriers
            )
            or any(value < 0 for value in offsets)
            or any(value is not None and value < 0 for value in counters)
            or any(value < 0 for row in positions for value in row)
        ):
            raise ValueError("hopped multiport indices must be nonnegative and in-grid")
        expected_mask = np.zeros(grid.shape, dtype=np.bool_)
        for symbol, row in zip(symbols, subcarriers, strict=True):
            expected_mask[:, symbol, np.asarray(row, dtype=np.intp)] = True
        if not np.array_equal(mask, expected_mask):
            raise ValueError("hopped multiport mask disagrees with its per-symbol RE roster")
        grid_view = grid.view()
        mask_view = mask.view()
        grid_view.setflags(write=False)
        mask_view.setflags(write=False)
        object.__setattr__(self, "grid", grid_view)
        object.__setattr__(self, "mask", mask_view)
        object.__setattr__(self, "symbol_indices", tuple(int(value) for value in symbols))
        object.__setattr__(
            self,
            "subcarrier_indices_by_symbol",
            tuple(tuple(int(value) for value in row) for row in subcarriers),
        )
        object.__setattr__(
            self,
            "position_indices_by_symbol",
            tuple(tuple(int(value) for value in row) for row in positions),
        )
        object.__setattr__(
            self,
            "n_srs_by_symbol",
            tuple(None if value is None else int(value) for value in counters),
        )
        object.__setattr__(self, "positioning_offsets", tuple(int(value) for value in offsets))


@dataclass(frozen=True, slots=True, eq=False)
class NRSRSPositioningResourceGrid:
    """Mapped positioning SRS with its per-symbol comb locations recorded."""

    grid: np.ndarray
    mask: np.ndarray
    symbol_indices: tuple[int, ...]
    subcarrier_indices_by_symbol: tuple[tuple[int, ...], ...]
    positioning_offsets: tuple[int, ...]

    def __post_init__(self) -> None:
        grid = np.asarray(self.grid)
        mask = np.asarray(self.mask)
        symbols = tuple(self.symbol_indices)
        subcarriers = tuple(tuple(row) for row in self.subcarrier_indices_by_symbol)
        offsets = tuple(self.positioning_offsets)
        if grid.ndim != 2 or not np.iscomplexobj(grid):
            raise ValueError("positioning SRS grid must be complex [slot symbols, subcarriers]")
        if mask.shape != grid.shape or mask.dtype != np.bool_:
            raise ValueError("positioning SRS mask must be boolean and match the grid")
        if not np.isfinite(grid).all():
            raise ValueError("positioning SRS grid must be finite")
        if not symbols or len(symbols) != len(subcarriers) or len(symbols) != len(offsets):
            raise ValueError("symbol, subcarrier, and positioning-offset rosters must align")
        if any(not row for row in subcarriers):
            raise ValueError("each positioning SRS symbol must occupy at least one subcarrier")
        values = (*symbols, *offsets, *(index for row in subcarriers for index in row))
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
            raise TypeError("positioning SRS resource indices must be integers")
        if (
            len(set(symbols)) != len(symbols)
            or any(symbol < 0 or symbol >= grid.shape[0] for symbol in symbols)
            or any(
                len(set(row)) != len(row)
                or any(index < 0 or index >= grid.shape[1] for index in row)
                for row in subcarriers
            )
            or any(offset < 0 for offset in offsets)
        ):
            raise ValueError("positioning SRS resource indices must be unique and inside the grid")
        expected_mask = np.zeros(grid.shape, dtype=np.bool_)
        for symbol, row in zip(symbols, subcarriers, strict=True):
            expected_mask[symbol, np.asarray(row, dtype=np.intp)] = True
        if not np.array_equal(mask, expected_mask):
            raise ValueError("positioning SRS mask disagrees with its declared indices")
        grid_view = grid.view()
        mask_view = mask.view()
        grid_view.setflags(write=False)
        mask_view.setflags(write=False)
        object.__setattr__(self, "grid", grid_view)
        object.__setattr__(self, "mask", mask_view)
        object.__setattr__(self, "symbol_indices", tuple(int(value) for value in symbols))
        object.__setattr__(
            self,
            "subcarrier_indices_by_symbol",
            tuple(tuple(int(value) for value in row) for row in subcarriers),
        )
        object.__setattr__(self, "positioning_offsets", tuple(int(value) for value in offsets))


@dataclass(frozen=True, slots=True, eq=False)
class NRSRSPositioningMultiportResourceGrid:
    """Mapped positioning SRS with per-port grids and per-symbol RE rosters."""

    grid: np.ndarray
    mask: np.ndarray
    symbol_indices: tuple[int, ...]
    subcarrier_indices_by_symbol: tuple[tuple[int, ...], ...]
    positioning_offsets: tuple[int, ...]
    active_port_symbols: tuple[tuple[bool, ...], ...] | None = None

    def __post_init__(self) -> None:
        grid = np.asarray(self.grid)
        mask = np.asarray(self.mask)
        symbols = tuple(self.symbol_indices)
        subcarriers = tuple(tuple(row) for row in self.subcarrier_indices_by_symbol)
        offsets = tuple(self.positioning_offsets)
        if grid.ndim != 3 or not np.iscomplexobj(grid):
            raise ValueError(
                "positioning multiport grid must be complex [ports, slot symbols, subcarriers]"
            )
        if grid.shape[0] < 1 or not np.isfinite(grid).all():
            raise ValueError("positioning multiport grid must have ports and finite values")
        if mask.shape != grid.shape or mask.dtype != np.bool_:
            raise ValueError("positioning multiport mask must be boolean and match the grid")
        if not symbols or len(symbols) != len(subcarriers) or len(symbols) != len(offsets):
            raise ValueError("symbol, subcarrier, and positioning-offset rosters must align")
        if any(not row for row in subcarriers):
            raise ValueError("each positioning SRS symbol must occupy at least one subcarrier")
        values = (*symbols, *offsets, *(index for row in subcarriers for index in row))
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
            raise TypeError("positioning multiport resource indices must be integers")
        if (
            len(set(symbols)) != len(symbols)
            or any(symbol < 0 or symbol >= grid.shape[1] for symbol in symbols)
            or any(
                len(set(row)) != len(row)
                or any(index < 0 or index >= grid.shape[2] for index in row)
                for row in subcarriers
            )
            or any(offset < 0 for offset in offsets)
        ):
            raise ValueError("positioning multiport indices must be unique and inside the grid")
        active = self.active_port_symbols
        if active is None:
            active = tuple((True,) * len(symbols) for _ in range(grid.shape[0]))
        else:
            active = tuple(tuple(row) for row in active)
            if len(active) != grid.shape[0] or any(
                len(row) != len(symbols)
                or any(not isinstance(value, (bool, np.bool_)) for value in row)
                for row in active
            ):
                raise ValueError("active_port_symbols must be boolean [ports, resource symbols]")
            active = tuple(tuple(bool(value) for value in row) for row in active)
        expected_mask = np.zeros(grid.shape, dtype=np.bool_)
        for port_index, port_activity in enumerate(active):
            for symbol, row, is_active in zip(symbols, subcarriers, port_activity, strict=True):
                if is_active:
                    expected_mask[port_index, symbol, np.asarray(row, dtype=np.intp)] = True
        if not np.array_equal(mask, expected_mask):
            raise ValueError("positioning multiport mask disagrees with its declared indices")
        grid_view = grid.view()
        mask_view = mask.view()
        grid_view.setflags(write=False)
        mask_view.setflags(write=False)
        object.__setattr__(self, "grid", grid_view)
        object.__setattr__(self, "mask", mask_view)
        object.__setattr__(self, "symbol_indices", tuple(int(value) for value in symbols))
        object.__setattr__(
            self,
            "subcarrier_indices_by_symbol",
            tuple(tuple(int(value) for value in row) for row in subcarriers),
        )
        object.__setattr__(self, "positioning_offsets", tuple(int(value) for value in offsets))
        object.__setattr__(self, "active_port_symbols", active)


@dataclass(frozen=True, slots=True, eq=False)
class NRSRSMultiportResourceGrid:
    """SRS resources mapped onto independent transmit-port slot grids."""

    grid: np.ndarray
    mask: np.ndarray
    symbol_indices: tuple[int, ...]
    subcarrier_indices: tuple[int, ...]
    active_port_symbols: tuple[tuple[bool, ...], ...] | None = None

    def __post_init__(self) -> None:
        grid = np.asarray(self.grid)
        mask = np.asarray(self.mask)
        if grid.ndim != 3 or not np.iscomplexobj(grid):
            raise ValueError("multiport SRS grid must be complex [ports, symbols, subcarriers]")
        if grid.shape[0] < 1 or not np.isfinite(grid).all():
            raise ValueError("multiport SRS grid must have ports and finite values")
        if mask.shape != grid.shape or mask.dtype != np.bool_:
            raise ValueError("multiport SRS mask must be boolean and match the grid")
        symbols = tuple(self.symbol_indices)
        subcarriers = tuple(self.subcarrier_indices)
        if not symbols or not subcarriers:
            raise ValueError("SRS resource indices must be nonempty")
        if any(
            isinstance(index, bool) or not isinstance(index, Integral)
            for index in (*symbols, *subcarriers)
        ):
            raise TypeError("SRS resource indices must be integers")
        if (
            len(set(symbols)) != len(symbols)
            or len(set(subcarriers)) != len(subcarriers)
            or any(index < 0 or index >= grid.shape[1] for index in symbols)
            or any(index < 0 or index >= grid.shape[2] for index in subcarriers)
        ):
            raise ValueError("SRS resource indices must be unique and inside the grid")
        active = self.active_port_symbols
        if active is None:
            active = tuple((True,) * len(symbols) for _ in range(grid.shape[0]))
        else:
            active = tuple(tuple(row) for row in active)
            if len(active) != grid.shape[0] or any(
                len(row) != len(symbols)
                or any(not isinstance(value, (bool, np.bool_)) for value in row)
                for row in active
            ):
                raise ValueError("active_port_symbols must be boolean [ports, resource symbols]")
            active = tuple(tuple(bool(value) for value in row) for row in active)
        expected_mask = np.zeros(grid.shape, dtype=np.bool_)
        for port_index, active_symbols in enumerate(active):
            for symbol_index, is_active in zip(symbols, active_symbols, strict=True):
                if is_active:
                    expected_mask[port_index, symbol_index, np.asarray(subcarriers)] = True
        if not np.array_equal(mask, expected_mask):
            raise ValueError("multiport SRS mask disagrees with its declared indices")
        grid_view = grid.view()
        mask_view = mask.view()
        grid_view.setflags(write=False)
        mask_view.setflags(write=False)
        object.__setattr__(self, "grid", grid_view)
        object.__setattr__(self, "mask", mask_view)
        object.__setattr__(self, "symbol_indices", tuple(int(index) for index in symbols))
        object.__setattr__(self, "subcarrier_indices", tuple(int(index) for index in subcarriers))
        object.__setattr__(self, "active_port_symbols", active)
