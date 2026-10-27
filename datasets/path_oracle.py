"""Ragged physical path truth with absolute delays and explicit solve outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from .network import NetworkEpisode


class PathSolveStatus(StrEnum):
    """Outcome for a link/frame/antenna-pair path solve."""

    SOLVED = "solved"
    NO_PATH = "no_path"
    SOLVER_FAILURE = "solver_failure"
    UNCONVERGED = "unconverged"


def _freeze_vector(value: np.ndarray, *, name: str, dtype: np.dtype) -> np.ndarray:
    original = np.asarray(value)
    if original.dtype.kind not in "iufc":
        raise TypeError(f"{name} must use a real or complex numeric dtype")
    if dtype.kind in "iu" and original.dtype.kind not in "iu":
        raise TypeError(f"{name} must use an integer dtype")
    if dtype.kind in "iu":
        limits = np.iinfo(dtype)
        outside_range = (
            np.any(original > limits.max)
            if original.dtype.kind == "u"
            else np.any(original < limits.min) or np.any(original > limits.max)
        )
        if outside_range:
            raise ValueError(f"{name} values must fit {dtype.name}")
    if dtype.kind == "f" and original.dtype.kind == "c":
        raise TypeError(f"{name} must use a real numeric dtype")
    array = np.array(original, dtype=dtype, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if array.dtype.kind in "fc" and not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True, eq=False)
class PathOracleTable:
    """CSR-style path table grouped by link, frame, RX port and TX port.

    ``path_offsets[g]:path_offsets[g+1]`` indexes each ragged group. Delays
    are absolute propagation delays in seconds; path gains use the convention
    recorded by the pinned simulator. Ports in group keys are zero-based,
    link-local port positions, not global antenna indices. A group with no
    paths remains present and carries its distinct solver outcome.

    Optional physical fields are either populated for every path in this solve
    or absent for the whole table. Partial per-path availability should be
    represented by a future field-validity extension, not sentinel zeros.
    Interaction rows use ``interaction_offsets[path]:[path+1]`` and retain
    Sionna's depth order. Object and primitive IDs are scene-local; for sensing
    interactions, Sionna's object field contains a sensing-target index.
    """

    link_ids: tuple[str, ...]
    frame_indices: np.ndarray
    rx_port_indices: np.ndarray
    tx_port_indices: np.ndarray
    path_offsets: np.ndarray
    solve_status: tuple[PathSolveStatus, ...]
    delay_s: np.ndarray
    complex_gain: np.ndarray
    interaction_order: np.ndarray
    doppler_hz: np.ndarray | None = None
    aoa_azimuth_elevation_rad: np.ndarray | None = None
    aod_azimuth_elevation_rad: np.ndarray | None = None
    path_kind: tuple[str, ...] | None = None
    interaction_offsets: np.ndarray | None = None
    interaction_types: np.ndarray | None = None
    object_ids: np.ndarray | None = None
    primitive_ids: np.ndarray | None = None
    interaction_vertices_m: np.ndarray | None = None
    interaction_vertex_frame: str | None = None
    coefficient_convention: str = "sionna_rt_path_coefficient_v1"

    def __post_init__(self) -> None:
        links = tuple(self.link_ids)
        status = tuple(self.solve_status)
        if not links or any(
            not isinstance(link, str) or not link.strip() or link != link.strip()
            for link in links
        ):
            raise ValueError("path groups need non-empty canonical link IDs")
        group_count = len(links)
        frames = _freeze_vector(self.frame_indices, name="frame_indices", dtype=np.dtype(np.int64))
        rx_ports = _freeze_vector(
            self.rx_port_indices, name="rx_port_indices", dtype=np.dtype(np.int64)
        )
        tx_ports = _freeze_vector(
            self.tx_port_indices, name="tx_port_indices", dtype=np.dtype(np.int64)
        )
        offsets_input = np.asarray(self.path_offsets)
        if offsets_input.dtype.kind not in "iu":
            raise TypeError("path_offsets must use an integer dtype")
        offsets = _freeze_vector(offsets_input, name="path_offsets", dtype=np.dtype(np.int64))
        if any(len(values) != group_count for values in (frames, rx_ports, tx_ports, status)):
            raise ValueError("each path group needs one frame, RX/TX port, and solve status")
        if len(offsets) != group_count + 1 or offsets[0] != 0:
            raise ValueError("path_offsets must have one leading zero and one end per group")
        if np.any(np.diff(offsets) < 0):
            raise ValueError("path_offsets must be nondecreasing")
        path_count = int(offsets[-1])
        if np.any(frames < 0) or np.any(rx_ports < 0) or np.any(tx_ports < 0):
            raise ValueError("frame and link-local port indices must be nonnegative")
        if (
            len(set(zip(links, frames.tolist(), rx_ports.tolist(), tx_ports.tolist())))
            != group_count
        ):
            raise ValueError("path group keys must be unique")
        if any(not isinstance(value, PathSolveStatus) for value in status):
            raise TypeError("solve_status entries must be PathSolveStatus values")

        delays = _freeze_vector(self.delay_s, name="delay_s", dtype=np.dtype(np.float64))
        gains = _freeze_vector(
            self.complex_gain, name="complex_gain", dtype=np.dtype(np.complex128)
        )
        orders = _freeze_vector(
            self.interaction_order, name="interaction_order", dtype=np.dtype(np.int64)
        )
        if len(delays) != path_count or len(gains) != path_count or len(orders) != path_count:
            raise ValueError("path arrays must match the ragged path_offsets total")
        if np.any(delays < 0) or np.any(orders < 0):
            raise ValueError("absolute path delays and interaction orders must be nonnegative")
        if not np.isfinite(gains.real).all() or not np.isfinite(gains.imag).all():
            raise ValueError("complex_gain must contain finite complex coefficients")

        doppler = self._optional_vector(self.doppler_hz, name="doppler_hz", count=path_count)
        aoa = self._optional_angles(
            self.aoa_azimuth_elevation_rad, name="aoa_azimuth_elevation_rad", count=path_count
        )
        aod = self._optional_angles(
            self.aod_azimuth_elevation_rad, name="aod_azimuth_elevation_rad", count=path_count
        )
        kinds = None if self.path_kind is None else tuple(self.path_kind)
        if kinds is not None and (
            len(kinds) != path_count
            or any(
                not isinstance(kind, str) or not kind.strip() or kind != kind.strip()
                for kind in kinds
            )
        ):
            raise ValueError("path_kind must provide one non-empty canonical value per path")

        interaction_values = (
            self.interaction_offsets,
            self.interaction_types,
            self.object_ids,
            self.primitive_ids,
            self.interaction_vertices_m,
        )
        interaction_data = None
        if any(value is not None for value in interaction_values):
            if any(value is None for value in interaction_values):
                raise ValueError(
                    "interaction offsets, types, IDs, and vertices must be supplied together"
                )
            raw_offsets = np.asarray(self.interaction_offsets)
            if raw_offsets.dtype.kind not in "iu":
                raise TypeError("interaction_offsets must use an integer dtype")
            interaction_offsets = _freeze_vector(
                raw_offsets, name="interaction_offsets", dtype=np.dtype(np.int64)
            )
            if len(interaction_offsets) != path_count + 1 or interaction_offsets[0] != 0:
                raise ValueError("interaction_offsets must contain one leading zero per path table")
            if np.any(np.diff(interaction_offsets) < 0):
                raise ValueError("interaction_offsets must be nondecreasing")
            interaction_count = int(interaction_offsets[-1])
            if not np.array_equal(np.diff(interaction_offsets), orders):
                raise ValueError("interaction offsets must agree with per-path interaction_order")
            interaction_types = _freeze_vector(
                np.asarray(self.interaction_types),
                name="interaction_types",
                dtype=np.dtype(np.int64),
            )
            object_ids = _freeze_vector(
                np.asarray(self.object_ids), name="object_ids", dtype=np.dtype(np.int64)
            )
            primitive_ids = _freeze_vector(
                np.asarray(self.primitive_ids), name="primitive_ids", dtype=np.dtype(np.int64)
            )
            raw_vertices = np.asarray(self.interaction_vertices_m)
            if raw_vertices.dtype.kind not in "iuf":
                raise TypeError("interaction_vertices_m must use a real numeric dtype")
            vertices = np.array(raw_vertices, dtype=np.float64, copy=True)
            if any(
                len(values) != interaction_count
                for values in (interaction_types, object_ids, primitive_ids)
            ):
                raise ValueError("interaction arrays must match interaction_offsets total")
            if (
                np.any(interaction_types <= 0)
                or np.any(object_ids < 0)
                or np.any(primitive_ids < 0)
            ):
                raise ValueError("interaction types and IDs must be nonnegative valid values")
            if vertices.shape != (interaction_count, 3) or not np.isfinite(vertices).all():
                raise ValueError("interaction_vertices_m must have finite shape [interaction, xyz]")
            if (
                not isinstance(self.interaction_vertex_frame, str)
                or not self.interaction_vertex_frame.strip()
                or self.interaction_vertex_frame != self.interaction_vertex_frame.strip()
            ):
                raise ValueError("interaction vertices require a canonical named coordinate frame")
            vertices.setflags(write=False)
            interaction_data = (
                interaction_offsets,
                interaction_types,
                object_ids,
                primitive_ids,
                vertices,
            )
        elif self.interaction_vertex_frame is not None:
            raise ValueError("interaction_vertex_frame requires interaction geometry arrays")
        if (
            not isinstance(self.coefficient_convention, str)
            or not self.coefficient_convention.strip()
            or self.coefficient_convention != self.coefficient_convention.strip()
        ):
            raise ValueError("coefficient_convention must be a non-empty canonical string")
        for index, outcome in enumerate(status):
            path_number = int(offsets[index + 1] - offsets[index])
            if outcome is PathSolveStatus.SOLVED and path_number == 0:
                raise ValueError("a solved path group must contain at least one path")
            if outcome in (PathSolveStatus.NO_PATH, PathSolveStatus.SOLVER_FAILURE) and path_number:
                raise ValueError(f"{outcome.value} groups must not contain fabricated path rows")

        object.__setattr__(self, "link_ids", links)
        object.__setattr__(self, "frame_indices", frames)
        object.__setattr__(self, "rx_port_indices", rx_ports)
        object.__setattr__(self, "tx_port_indices", tx_ports)
        object.__setattr__(self, "path_offsets", offsets)
        object.__setattr__(self, "solve_status", status)
        object.__setattr__(self, "delay_s", delays)
        object.__setattr__(self, "complex_gain", gains)
        object.__setattr__(self, "interaction_order", orders)
        object.__setattr__(self, "doppler_hz", doppler)
        object.__setattr__(self, "aoa_azimuth_elevation_rad", aoa)
        object.__setattr__(self, "aod_azimuth_elevation_rad", aod)
        object.__setattr__(self, "path_kind", kinds)
        if interaction_data is not None:
            object.__setattr__(self, "interaction_offsets", interaction_data[0])
            object.__setattr__(self, "interaction_types", interaction_data[1])
            object.__setattr__(self, "object_ids", interaction_data[2])
            object.__setattr__(self, "primitive_ids", interaction_data[3])
            object.__setattr__(self, "interaction_vertices_m", interaction_data[4])
            object.__setattr__(self, "interaction_vertex_frame", self.interaction_vertex_frame)

    @staticmethod
    def _optional_vector(value: np.ndarray | None, *, name: str, count: int) -> np.ndarray | None:
        if value is None:
            return None
        array = _freeze_vector(value, name=name, dtype=np.dtype(np.float64))
        if len(array) != count:
            raise ValueError(f"{name} must match the ragged path count")
        return array

    @staticmethod
    def _optional_angles(value: np.ndarray | None, *, name: str, count: int) -> np.ndarray | None:
        if value is None:
            return None
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise TypeError(f"{name} must use a real numeric dtype")
        array = np.array(raw, dtype=np.float64, copy=True)
        if array.shape != (count, 2) or not np.isfinite(array).all():
            raise ValueError(f"{name} must have finite shape [path, azimuth/elevation]")
        if np.any(array[:, 0] < -np.pi) or np.any(array[:, 0] > np.pi):
            raise ValueError(f"{name} azimuth must be in [-pi, pi] radians")
        if np.any(array[:, 1] < -np.pi / 2) or np.any(array[:, 1] > np.pi / 2):
            raise ValueError(f"{name} elevation must be in [-pi/2, pi/2] radians")
        array.setflags(write=False)
        return array

    @property
    def group_count(self) -> int:
        """Number of represented link/frame/port solves, including empty ones."""
        return len(self.link_ids)

    @property
    def path_count(self) -> int:
        """Number of unpadded physical path rows."""
        return len(self.delay_s)

    def group_slice(self, group_index: int) -> slice:
        """Return the ragged path slice for one group."""
        if isinstance(group_index, bool) or not isinstance(group_index, int):
            raise TypeError("group_index must be an integer")
        if group_index < 0 or group_index >= self.group_count:
            raise IndexError(group_index)
        return slice(int(self.path_offsets[group_index]), int(self.path_offsets[group_index + 1]))

    def validate_for(self, episode: NetworkEpisode) -> None:
        """Reject path groups that do not resolve to a valid episode link/frame/port."""
        links = {link.link_id: link for link in episode.links}
        for index, link_id in enumerate(self.link_ids):
            if link_id not in links:
                raise ValueError(f"path group references missing episode link {link_id!r}")
            if self.frame_indices[index] >= len(episode.frame_times_s):
                raise ValueError("path group frame index is outside the episode time axis")
            link = links[link_id]
            if self.rx_port_indices[index] >= len(link.rx_port_indices):
                raise ValueError("path group RX port index is outside the link port map")
            if self.tx_port_indices[index] >= len(link.tx_port_indices):
                raise ValueError("path group TX port index is outside the link port map")
