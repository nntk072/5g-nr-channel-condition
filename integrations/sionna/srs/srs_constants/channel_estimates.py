"""SRS channel-estimate records for single and multiport measurements."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True, slots=True, eq=False)
class NRSRSSinglePortChannelEstimate:
    """Per-resource-element LS channel estimates from one mapped SRS port.

    Entries outside ``mask`` are zero. No averaging or interpolation is
    applied, so the result preserves the observed time-frequency locations.
    """

    channel_by_re: np.ndarray
    mask: np.ndarray
    symbol_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        channel = np.asarray(self.channel_by_re)
        mask = np.asarray(self.mask)
        symbols = tuple(self.symbol_indices)
        if channel.ndim != 2 or not np.iscomplexobj(channel):
            raise ValueError("channel_by_re must be complex [slot symbols, subcarriers]")
        if mask.shape != channel.shape or mask.dtype != np.bool_:
            raise ValueError("channel estimate mask must be boolean and match channel_by_re")
        if not np.isfinite(channel).all():
            raise ValueError("channel_by_re must contain only finite values")
        if np.any(channel[~mask] != 0):
            raise ValueError("channel_by_re must be zero outside the SRS resource mask")
        if (
            not symbols
            or any(isinstance(i, bool) or not isinstance(i, Integral) for i in symbols)
            or len(set(symbols)) != len(symbols)
            or any(i < 0 or i >= channel.shape[0] for i in symbols)
        ):
            raise ValueError("symbol_indices must be unique valid slot-symbol indices")
        channel_view = channel.view()
        mask_view = mask.view()
        channel_view.setflags(write=False)
        mask_view.setflags(write=False)
        object.__setattr__(self, "channel_by_re", channel_view)
        object.__setattr__(self, "mask", mask_view)
        object.__setattr__(self, "symbol_indices", tuple(int(i) for i in symbols))


@dataclass(frozen=True, slots=True, eq=False)
class NRSRSMultiportChannelEstimate:
    """Multiport LS estimates and conditional noise covariance by subcarrier.

    ``channel_by_rx_port_subcarrier`` has shape ``[rx antennas, TX ports,
    subcarriers]``. ``noise_covariance_by_rx_subcarrier`` has shape
    ``[rx antennas, subcarriers, TX ports, TX ports]``. Unavailable
    subcarriers are zero-filled and identified by a false validity mask plus a
    reason code; no frequency interpolation is performed. The LS model assumes
    the channel is constant over this SRS resource's symbols. Noise may use one
    variance per RX antenna or a per-RX/per-symbol variance profile; the latter
    drives weighted LS and is retained in the result. In profile mode,
    ``noise_variance_by_rx_ant`` is its per-RX slot-symbol mean. Synchronization
    errors and channel evolution remain unmodeled.
    """

    channel_by_rx_port_subcarrier: np.ndarray
    noise_covariance_by_rx_subcarrier: np.ndarray
    valid_subcarrier_mask: np.ndarray
    rank_by_subcarrier: np.ndarray
    condition_number_by_subcarrier: tuple[float | None, ...]
    reason_codes: tuple[str | None, ...]
    noise_variance_by_rx_ant: np.ndarray
    symbol_indices: tuple[int, ...]
    noise_variance_by_rx_symbol: np.ndarray | None = None

    def __post_init__(self) -> None:
        channel = np.asarray(self.channel_by_rx_port_subcarrier)
        covariance = np.asarray(self.noise_covariance_by_rx_subcarrier)
        valid = np.asarray(self.valid_subcarrier_mask)
        rank = np.asarray(self.rank_by_subcarrier)
        noise = np.asarray(self.noise_variance_by_rx_ant)
        symbol_noise = (
            None
            if self.noise_variance_by_rx_symbol is None
            else np.asarray(self.noise_variance_by_rx_symbol)
        )
        conditions = tuple(self.condition_number_by_subcarrier)
        reasons = tuple(self.reason_codes)
        symbols = tuple(self.symbol_indices)
        if channel.ndim != 3 or not np.iscomplexobj(channel):
            raise ValueError(
                "channel estimate must be complex [rx antennas, TX ports, subcarriers]"
            )
        rx_count, port_count, subcarrier_count = channel.shape
        if min(rx_count, port_count, subcarrier_count) < 1:
            raise ValueError("channel estimate dimensions must be nonempty")
        if covariance.shape != (rx_count, subcarrier_count, port_count, port_count):
            raise ValueError(
                "noise covariance shape must be [rx antennas, subcarriers, ports, ports]"
            )
        if not np.iscomplexobj(covariance):
            raise TypeError("noise covariance must be complex-valued")
        if valid.shape != (subcarrier_count,) or valid.dtype != np.bool_:
            raise ValueError("valid_subcarrier_mask must be boolean [subcarriers]")
        if rank.shape != (subcarrier_count,) or rank.dtype.kind not in "iu":
            raise ValueError("rank_by_subcarrier must be an integer [subcarriers] vector")
        if len(conditions) != subcarrier_count or len(reasons) != subcarrier_count:
            raise ValueError("condition and reason rosters must match subcarrier count")
        if noise.shape != (rx_count,) or noise.dtype.kind not in "iuf":
            raise ValueError("noise_variance_by_rx_ant must be real [rx antennas]")
        if (
            not symbols
            or any(
                isinstance(value, bool) or not isinstance(value, Integral) or value < 0
                for value in symbols
            )
            or len(set(symbols)) != len(symbols)
        ):
            raise ValueError("symbol_indices must be unique nonnegative integers")
        if symbol_noise is not None and (
            symbol_noise.ndim != 2
            or symbol_noise.shape[0] != rx_count
            or symbol_noise.shape[1] <= max(symbols, default=-1)
            or symbol_noise.dtype.kind not in "iuf"
            or not np.isfinite(symbol_noise).all()
            or np.any(symbol_noise <= 0)
        ):
            raise ValueError("noise_variance_by_rx_symbol must be positive real [rx, symbols]")
        if (
            not np.isfinite(channel).all()
            or not np.isfinite(covariance).all()
            or not np.isfinite(noise).all()
            or np.any(noise <= 0)
            or np.any(rank < 0)
            or np.any(rank > port_count)
        ):
            raise ValueError("estimates, covariance, noise, or rank contain invalid values")
        for index, (is_valid, condition, reason) in enumerate(
            zip(valid, conditions, reasons, strict=True)
        ):
            if is_valid:
                if rank[index] != port_count or condition is None or not np.isfinite(condition):
                    raise ValueError("valid subcarriers require full rank and finite conditioning")
                if reason is not None:
                    raise ValueError("valid subcarriers cannot have an unavailable reason")
            elif not isinstance(reason, str) or not reason:
                raise ValueError("unavailable subcarriers require a reason code")
            if not is_valid and (
                np.any(channel[:, :, index] != 0) or np.any(covariance[:, index] != 0)
            ):
                raise ValueError("unavailable subcarriers must have zero estimates and covariance")
        channel_view = channel.view()
        covariance_view = covariance.view()
        valid_view = valid.view()
        rank_view = rank.view()
        noise_view = noise.view()
        symbol_noise_view = None if symbol_noise is None else symbol_noise.view()
        arrays_to_freeze = [channel_view, covariance_view, valid_view, rank_view, noise_view]
        if symbol_noise_view is not None:
            arrays_to_freeze.append(symbol_noise_view)
        for value in arrays_to_freeze:
            value.setflags(write=False)
        object.__setattr__(self, "channel_by_rx_port_subcarrier", channel_view)
        object.__setattr__(self, "noise_covariance_by_rx_subcarrier", covariance_view)
        object.__setattr__(self, "valid_subcarrier_mask", valid_view)
        object.__setattr__(self, "rank_by_subcarrier", rank_view)
        object.__setattr__(self, "condition_number_by_subcarrier", conditions)
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "noise_variance_by_rx_ant", noise_view)
        object.__setattr__(self, "symbol_indices", tuple(int(value) for value in symbols))
        object.__setattr__(self, "noise_variance_by_rx_symbol", symbol_noise_view)
