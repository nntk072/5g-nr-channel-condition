"""Named-axis physical channel view for one heterogeneous directed link."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real

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


class PhaseComparisonDisposition(StrEnum):
    """Why two complex views can or cannot share a coherent phase comparison."""

    ELIGIBLE = "eligible_same_declared_gauge"
    REFERENCE_MISSING = "reference_missing"
    GAUGE_MISMATCH = "phase_gauge_mismatch"
    AXIS_MISMATCH = "channel_axes_mismatch"
    FREQUENCY_MISMATCH = "frequency_grid_mismatch"
    DELAY_REFERENCE_MISMATCH = "delay_reference_mismatch"
    POWER_REFERENCE_MISMATCH = "power_reference_mismatch"


@dataclass(frozen=True, slots=True, eq=False)
class PhysicalChannelView:
    """Complex CFR for one link, without padding heterogeneous network axes.

    Axes are fixed by contract to ``[frame, rx_port, tx_port, subcarrier]``.
    Subcarrier coordinates are baseband offsets in Hz; ``carrier_hz`` is
    separately retained. A normalized channel must name its exact transform.
    ``phase_gauge_id`` is an optional producer declaration; it does not prove
    calibration, reference-plane validity, or coherent phase accuracy.
    ``source_time_alignment_sha256`` binds generated channel frames to the
    episode's replayable named-source timing report when one is available.
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
    phase_gauge_id: str | None = None
    source_time_alignment_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in ("view_id", "link_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValueError(f"{name} must be a non-empty canonical string")
        values = np.array(self.channel, copy=True)
        if values.ndim != 4 or not np.iscomplexobj(values) or not np.isfinite(values).all():
            raise ValueError("channel must be finite complex [frame, rx_port, tx_port, subcarrier]")
        times = canonical_time_axis(self.frame_times_s, name="frame_times_s")
        rx_ids, tx_ids = tuple(self.rx_port_ids), tuple(self.tx_port_ids)
        if (
            not rx_ids
            or not tx_ids
            or any(
                not isinstance(item, str) or not item.strip() or item != item.strip()
                for item in (*rx_ids, *tx_ids)
            )
        ):
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
        raw_frequencies = np.asarray(self.subcarrier_offsets_hz)
        if raw_frequencies.dtype.kind not in "iuf":
            raise TypeError("subcarrier_offsets_hz must use a real numeric dtype")
        frequencies = np.array(raw_frequencies, dtype=np.float64, copy=True)
        if frequencies.shape != (values.shape[-1],) or not np.isfinite(frequencies).all():
            raise ValueError("subcarrier_offsets_hz must map the complete final channel axis")
        if (
            len(frequencies) < 1
            or np.any(np.diff(frequencies) <= 0)
            or not np.any(frequencies == 0)
        ):
            raise ValueError("baseband offsets must be strictly increasing and include DC")
        if isinstance(self.carrier_hz, bool) or not isinstance(self.carrier_hz, Real):
            raise TypeError("carrier_hz must be a real numeric scalar")
        carrier_hz = float(self.carrier_hz)
        if not math.isfinite(carrier_hz) or carrier_hz <= 0:
            raise ValueError("carrier_hz must be finite and positive")
        if not isinstance(self.delay_reference, ChannelDelayReference):
            raise TypeError("delay_reference must use ChannelDelayReference")
        if not isinstance(self.power_scale, ChannelPowerScale):
            raise TypeError("power_scale must use ChannelPowerScale")
        gauge_id = self.phase_gauge_id
        if gauge_id is not None and (
            not isinstance(gauge_id, str) or not gauge_id.strip() or gauge_id != gauge_id.strip()
        ):
            raise ValueError("phase_gauge_id must be null or a non-empty canonical string")
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
        object.__setattr__(self, "carrier_hz", carrier_hz)
        object.__setattr__(self, "phase_gauge_id", gauge_id)
        alignment_digest = self.source_time_alignment_sha256
        if alignment_digest is not None and (
            not isinstance(alignment_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", alignment_digest) is None
        ):
            raise ValueError("source_time_alignment_sha256 must be null or lowercase SHA-256")
        object.__setattr__(self, "source_time_alignment_sha256", alignment_digest)
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
        source_alignment = episode.source_time_alignment
        if source_alignment is None:
            if self.source_time_alignment_sha256 is not None:
                raise ValueError(
                    "channel source-time digest requires episode source-time alignment"
                )
        elif self.source_time_alignment_sha256 != source_alignment.sha256:
            raise ValueError("channel source-time digest must match the episode alignment report")


def phase_comparison_disposition(
    left: PhysicalChannelView,
    right: PhysicalChannelView,
) -> PhaseComparisonDisposition:
    """Return whether two views may be compared coherently without gauge fitting.

    A shared declared gauge is necessary, not proof of calibration or physical
    correctness. This check deliberately refuses automatic phase alignment,
    delay shifting, gain normalization, and cross-frequency comparison.
    """
    if not isinstance(left, PhysicalChannelView) or not isinstance(right, PhysicalChannelView):
        raise TypeError("phase comparison requires two PhysicalChannelView values")
    if left.phase_gauge_id is None or right.phase_gauge_id is None:
        return PhaseComparisonDisposition.REFERENCE_MISSING
    if left.phase_gauge_id != right.phase_gauge_id:
        return PhaseComparisonDisposition.GAUGE_MISMATCH
    if (
        left.link_id != right.link_id
        or left.frame_times_s != right.frame_times_s
        or left.rx_port_ids != right.rx_port_ids
        or left.tx_port_ids != right.tx_port_ids
        or left.axes != right.axes
    ):
        return PhaseComparisonDisposition.AXIS_MISMATCH
    if left.carrier_hz != right.carrier_hz or not np.array_equal(
        left.subcarrier_offsets_hz, right.subcarrier_offsets_hz
    ):
        return PhaseComparisonDisposition.FREQUENCY_MISMATCH
    if left.delay_reference is not right.delay_reference:
        return PhaseComparisonDisposition.DELAY_REFERENCE_MISMATCH
    if (
        left.power_scale is not right.power_scale
        or left.normalization_recipe != right.normalization_recipe
    ):
        return PhaseComparisonDisposition.POWER_REFERENCE_MISMATCH
    return PhaseComparisonDisposition.ELIGIBLE
