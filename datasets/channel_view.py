"""Named-axis physical channel view for one heterogeneous directed link."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from .network import NetworkEpisode
from .time_axis import canonical_time_axis


class ChannelDelayReference(StrEnum):
    """Time origin encoded by the channel's complex phase slope."""

    ABSOLUTE_PROPAGATION = "absolute_propagation_delay"
    RELATIVE = "relative_delay"


class ChannelPowerScale(StrEnum):
    """Whether channel amplitude preserves the declared link budget."""

    ABSOLUTE = "absolute_power_gain"
    NORMALIZED = "normalized_power"


@dataclass(frozen=True, slots=True, eq=False)
class PhysicalChannelView:
    """Complex CFR for one link, without padding heterogeneous network axes.

    Axes are fixed by contract to ``[frame, rx_port, tx_port, subcarrier]``.
    Subcarrier coordinates are baseband offsets in Hz; ``carrier_hz`` is
    separately retained. A normalized channel must name its exact transform.
    """

    view_id: str
    link_id: str
    channel: np.ndarray
    frame_times_s: tuple[float, ...]
    rx_port_ids: tuple[str, ...]
    tx_port_ids: tuple[str, ...]
    carrier_hz: float
    subcarrier_offsets_hz: np.ndarray
    delay_reference: ChannelDelayReference
    power_scale: ChannelPowerScale
    normalization_recipe: str | None = None
    valid_mask: np.ndarray | None = None
    axes: tuple[str, ...] = ("frame", "rx_port", "tx_port", "subcarrier")

    def __post_init__(self) -> None:
        for name in ("view_id", "link_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        values = np.array(self.channel, copy=True)
        if values.ndim != 4 or not np.iscomplexobj(values) or not np.isfinite(values).all():
            raise ValueError("channel must be finite complex [frame, rx_port, tx_port, subcarrier]")
        times = canonical_time_axis(self.frame_times_s, name="frame_times_s")
        rx_ids, tx_ids = tuple(self.rx_port_ids), tuple(self.tx_port_ids)
        if not rx_ids or not tx_ids or any(not item.strip() for item in (*rx_ids, *tx_ids)):
            raise ValueError("rx_port_ids and tx_port_ids must be non-empty strings")
        if len(set(rx_ids)) != len(rx_ids) or len(set(tx_ids)) != len(tx_ids):
            raise ValueError("port IDs must be unique within each link endpoint")
        if values.shape[:3] != (len(times), len(rx_ids), len(tx_ids)):
            raise ValueError("channel leading axes disagree with frame and port metadata")
        raw_valid = (
            np.ones(values.shape, dtype=np.bool_)
            if self.valid_mask is None
            else np.asarray(self.valid_mask)
        )
        if raw_valid.dtype.kind != "b" or raw_valid.shape != values.shape:
            raise ValueError("valid_mask must be boolean and match every channel element")
        valid = np.array(raw_valid, dtype=np.bool_, copy=True)
        if np.any(values[~valid] != 0.0):
            raise ValueError("masked channel elements must use zero storage")
        frequencies = np.array(self.subcarrier_offsets_hz, dtype=np.float64, copy=True)
        if frequencies.shape != (values.shape[-1],) or not np.isfinite(frequencies).all():
            raise ValueError("subcarrier_offsets_hz must map the complete final channel axis")
        if (
            len(frequencies) < 1
            or np.any(np.diff(frequencies) <= 0)
            or not np.any(frequencies == 0)
        ):
            raise ValueError("baseband offsets must be strictly increasing and include DC")
        if not math.isfinite(self.carrier_hz) or self.carrier_hz <= 0:
            raise ValueError("carrier_hz must be finite and positive")
        if not isinstance(self.delay_reference, ChannelDelayReference):
            raise TypeError("delay_reference must use ChannelDelayReference")
        if not isinstance(self.power_scale, ChannelPowerScale):
            raise TypeError("power_scale must use ChannelPowerScale")
        recipe = self.normalization_recipe
        if self.power_scale is ChannelPowerScale.NORMALIZED or (
            self.delay_reference is ChannelDelayReference.RELATIVE
        ):
            if not isinstance(recipe, str) or not recipe.strip():
                raise ValueError("normalized channels require an explicit normalization_recipe")
        if tuple(self.axes) != ("frame", "rx_port", "tx_port", "subcarrier"):
            raise ValueError("channel axes must use the canonical named-axis order")
        values.setflags(write=False)
        valid.setflags(write=False)
        frequencies.setflags(write=False)
        object.__setattr__(self, "channel", values)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "frame_times_s", times)
        object.__setattr__(self, "rx_port_ids", rx_ids)
        object.__setattr__(self, "tx_port_ids", tx_ids)
        object.__setattr__(self, "subcarrier_offsets_hz", frequencies)
        object.__setattr__(self, "axes", tuple(self.axes))

    def validate_for(self, episode: NetworkEpisode) -> None:
        """Check this per-link channel axes against its parent episode contract."""
        link = episode.link(self.link_id)
        if self.frame_times_s != episode.frame_times_s:
            raise ValueError("channel frame times must match the parent episode time axis")
        if self.carrier_hz != episode.carrier_hz:
            raise ValueError("channel carrier must match the parent episode carrier")
        if not np.array_equal(self.subcarrier_offsets_hz, episode.subcarrier_offsets_hz):
            raise ValueError("channel frequency offsets must match the parent episode grid")
        rx_node = episode.node(link.rx_node_id)
        tx_node = episode.node(link.tx_node_id)
        rx_array = episode.array(rx_node.array_id)
        tx_array = episode.array(tx_node.array_id)
        expected_rx_ports = tuple(rx_array.port_ids[index] for index in link.rx_port_indices)
        expected_tx_ports = tuple(tx_array.port_ids[index] for index in link.tx_port_indices)
        if self.rx_port_ids != expected_rx_ports:
            raise ValueError("channel RX port IDs do not match the directed-link port map")
        if self.tx_port_ids != expected_tx_ports:
            raise ValueError("channel TX port IDs do not match the directed-link port map")
