"""Conversion from native Sionna RT Paths to the unified path-truth table.

The adapter intentionally imports Sionna only when called. Device indices are
selected explicitly, while output antenna indices are link-local array ports.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from datasets.path_oracle import PathOracleTable, PathSolveStatus


def _numpy(value: Any) -> np.ndarray:
    """Materialize a Dr.Jit value (or array-like test double) as NumPy."""
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _port_shape(value: np.ndarray, target: tuple[int, ...], *, name: str) -> np.ndarray:
    """Broadcast a Sionna path field to [rx, rx_port, tx, tx_port, path]."""
    if value.ndim == 3:
        value = value[:, None, :, None, :]
    if value.ndim != 5:
        raise ValueError(f"Sionna {name} must have device-pair or per-port path axes")
    try:
        return np.broadcast_to(value, target)
    except ValueError as error:
        raise ValueError(
            f"Sionna {name} shape {value.shape} cannot broadcast to path coefficient shape {target}"
        ) from error


def _interaction_shape(
    value: np.ndarray,
    target: tuple[int, ...],
    *,
    name: str,
    xyz_axis: bool = False,
) -> np.ndarray:
    """Broadcast depth-indexed Sionna data to every antenna-port pair."""
    if xyz_axis:
        if value.ndim == 5:
            value = value[:, :, None, :, None, :, :]
        if value.ndim != 7 or value.shape[-1] != 3:
            raise ValueError(f"Sionna {name} must have depth, path, and XYZ axes")
        expected = (value.shape[0],) + target + (3,)
    else:
        if value.ndim == 4:
            value = value[:, :, None, :, None, :]
        if value.ndim != 6:
            raise ValueError(f"Sionna {name} must have depth plus path axes")
        expected = (value.shape[0],) + target
    try:
        return np.broadcast_to(value, expected)
    except ValueError as error:
        raise ValueError(f"Sionna {name} do not match the path coefficient axes") from error


def sionna_paths_to_oracle_table(
    paths: object,
    *,
    link_id: str,
    frame_index: int = 0,
    rx_device_index: int = 0,
    tx_device_index: int = 0,
    solve_status: PathSolveStatus | None = None,
) -> PathOracleTable:
    """Convert one device pair from a pinned Sionna RT ``Paths`` result.

    Sionna's path coefficients have axes ``[rx, rx_ant, tx, tx_ant, path]``.
    With synthetic arrays, delays, angles, validity and Doppler instead use
    ``[rx, tx, path]``; those fields are broadcast to the coefficient axes.
    Interaction order counts non-``NONE`` interaction slots. Sionna's zenith
    angle ``theta`` is converted to elevation ``pi/2 - theta`` and output angle
    pairs are ``[azimuth, elevation]`` in radians.

    ``Paths`` cannot indicate a failed or unconverged solver run. Callers that
    know the solve outcome may pass ``solve_status`` to apply that status to
    each represented port pair. By default, non-empty pairs are solved and
    empty pairs have no path. The output includes every RX/TX antenna-port
    pair, including explicit empty groups.
    """
    if not isinstance(link_id, str) or not link_id.strip():
        raise ValueError("link_id must be a non-empty string")
    for name, value in (
        ("frame_index", frame_index),
        ("rx_device_index", rx_device_index),
        ("tx_device_index", tx_device_index),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if solve_status is not None and not isinstance(solve_status, PathSolveStatus):
        raise TypeError("solve_status must be a PathSolveStatus value or None")

    valid = _numpy(getattr(paths, "valid")).astype(bool, copy=False)
    gain_components = getattr(paths, "a")
    if not isinstance(gain_components, (tuple, list)) or len(gain_components) != 2:
        raise ValueError("Sionna Paths.a must contain real and imaginary components")
    gain_real = _numpy(gain_components[0])
    gain_imag = _numpy(gain_components[1])
    if gain_real.shape != gain_imag.shape or gain_real.ndim != 5:
        raise ValueError("Sionna path coefficients must have matching five-axis shapes")
    target_shape = gain_real.shape
    valid = _port_shape(valid, target_shape, name="valid")
    delays = _port_shape(_numpy(getattr(paths, "tau")), target_shape, name="tau")
    doppler = _port_shape(_numpy(getattr(paths, "doppler")), target_shape, name="doppler")
    theta_t = _port_shape(_numpy(getattr(paths, "theta_t")), target_shape, name="theta_t")
    phi_t = _port_shape(_numpy(getattr(paths, "phi_t")), target_shape, name="phi_t")
    theta_r = _port_shape(_numpy(getattr(paths, "theta_r")), target_shape, name="theta_r")
    phi_r = _port_shape(_numpy(getattr(paths, "phi_r")), target_shape, name="phi_r")

    try:
        from sionna.rt.constants import InteractionType

        interaction_slots = _interaction_shape(
            _numpy(getattr(paths, "interactions")), target_shape, name="interactions"
        )
        object_slots = _interaction_shape(
            _numpy(getattr(paths, "objects")), target_shape, name="objects"
        )
        primitive_slots = _interaction_shape(
            _numpy(getattr(paths, "primitives")), target_shape, name="primitives"
        )
        vertex_slots = _interaction_shape(
            _numpy(getattr(paths, "vertices")), target_shape, name="vertices", xyz_axis=True
        )
        interaction_order = np.count_nonzero(interaction_slots != int(InteractionType.NONE), axis=0)
    except ImportError as error:
        raise RuntimeError("Sionna RT is required to interpret path interactions") from error

    rx_count, rx_port_count, tx_count, tx_port_count, _ = target_shape
    if rx_device_index >= rx_count or tx_device_index >= tx_count:
        raise IndexError("selected TX/RX device index is outside Sionna Paths axes")

    group_links: list[str] = []
    group_frames: list[int] = []
    group_rx_ports: list[int] = []
    group_tx_ports: list[int] = []
    group_status: list[PathSolveStatus] = []
    offsets = [0]
    flat_delays: list[np.ndarray] = []
    flat_gains: list[np.ndarray] = []
    flat_orders: list[np.ndarray] = []
    flat_doppler: list[np.ndarray] = []
    flat_aoa: list[np.ndarray] = []
    flat_aod: list[np.ndarray] = []
    flat_kinds: list[str] = []
    interaction_offsets = [0]
    interaction_type_parts: list[np.ndarray] = []
    object_id_parts: list[np.ndarray] = []
    primitive_id_parts: list[np.ndarray] = []
    vertex_parts: list[np.ndarray] = []
    interaction_labels = (
        (int(InteractionType.SPECULAR), "specular_reflection"),
        (int(InteractionType.DIFFUSE), "diffuse_reflection"),
        (int(InteractionType.REFRACTION), "refraction"),
        (int(InteractionType.DIFFRACTION), "diffraction"),
        (int(InteractionType.SENSING), "sensing"),
    )

    for rx_port in range(rx_port_count):
        for tx_port in range(tx_port_count):
            selection = (
                rx_device_index,
                rx_port,
                tx_device_index,
                tx_port,
                slice(None),
            )
            mask = valid[selection]
            path_delays = delays[selection][mask]
            path_gains = gain_real[selection][mask] + 1j * gain_imag[selection][mask]
            path_orders = interaction_order[selection][mask]
            path_interactions = interaction_slots[(slice(None),) + selection][..., mask]
            path_objects = object_slots[(slice(None),) + selection][..., mask]
            path_primitives = primitive_slots[(slice(None),) + selection][..., mask]
            path_vertices = vertex_slots[(slice(None),) + selection + (slice(None),)][..., mask, :]
            path_doppler = doppler[selection][mask]
            phi_aoa = np.asarray(phi_r[selection][mask], dtype=np.float64)
            phi_aod = np.asarray(phi_t[selection][mask], dtype=np.float64)
            aoa = np.stack(
                (
                    (phi_aoa + np.pi) % (2.0 * np.pi) - np.pi,
                    np.pi / 2.0 - theta_r[selection][mask],
                ),
                axis=-1,
            )
            aod = np.stack(
                (
                    (phi_aod + np.pi) % (2.0 * np.pi) - np.pi,
                    np.pi / 2.0 - theta_t[selection][mask],
                ),
                axis=-1,
            )
            group_links.append(link_id)
            group_frames.append(frame_index)
            group_rx_ports.append(rx_port)
            group_tx_ports.append(tx_port)
            inferred_status = PathSolveStatus.SOLVED if mask.any() else PathSolveStatus.NO_PATH
            group_status.append(solve_status or inferred_status)
            flat_delays.append(path_delays)
            flat_gains.append(path_gains)
            flat_orders.append(path_orders)
            flat_doppler.append(path_doppler)
            flat_aoa.append(aoa)
            flat_aod.append(aod)
            valid_interactions = path_interactions.T != int(InteractionType.NONE)
            per_path_interactions = np.count_nonzero(valid_interactions, axis=1)
            interaction_offsets.extend(
                (np.cumsum(per_path_interactions) + interaction_offsets[-1]).tolist()
            )
            interaction_type_parts.append(path_interactions.T[valid_interactions])
            object_id_parts.append(path_objects.T[valid_interactions])
            primitive_id_parts.append(path_primitives.T[valid_interactions])
            vertex_parts.append(path_vertices.transpose(1, 0, 2)[valid_interactions])
            for path_column in range(path_interactions.shape[-1]):
                labels = []
                for raw_type in path_interactions[:, path_column]:
                    interaction_type = int(raw_type)
                    if interaction_type == int(InteractionType.NONE):
                        continue
                    matched = False
                    for bit, label in interaction_labels:
                        if interaction_type & bit:
                            labels.append(label)
                            matched = True
                    if not matched:
                        labels.append(f"interaction_mask_{interaction_type}")
                flat_kinds.append("direct" if not labels else "+".join(labels))
            offsets.append(offsets[-1] + int(mask.sum()))

    def concatenate(
        parts: list[np.ndarray], *, shape: tuple[int, ...], dtype: np.dtype
    ) -> np.ndarray:
        nonempty = [part for part in parts if len(part)]
        if nonempty:
            return np.concatenate(nonempty, axis=0).astype(dtype, copy=False)
        return np.empty(shape, dtype=dtype)

    return PathOracleTable(
        link_ids=tuple(group_links),
        frame_indices=np.asarray(group_frames, dtype=np.int64),
        rx_port_indices=np.asarray(group_rx_ports, dtype=np.int64),
        tx_port_indices=np.asarray(group_tx_ports, dtype=np.int64),
        path_offsets=np.asarray(offsets, dtype=np.int64),
        solve_status=tuple(group_status),
        delay_s=concatenate(flat_delays, shape=(0,), dtype=np.dtype(np.float64)),
        complex_gain=concatenate(flat_gains, shape=(0,), dtype=np.dtype(np.complex128)),
        interaction_order=concatenate(flat_orders, shape=(0,), dtype=np.dtype(np.int64)),
        doppler_hz=concatenate(flat_doppler, shape=(0,), dtype=np.dtype(np.float64)),
        aoa_azimuth_elevation_rad=concatenate(flat_aoa, shape=(0, 2), dtype=np.dtype(np.float64)),
        aod_azimuth_elevation_rad=concatenate(flat_aod, shape=(0, 2), dtype=np.dtype(np.float64)),
        path_kind=tuple(flat_kinds),
        interaction_offsets=np.asarray(interaction_offsets, dtype=np.int64),
        interaction_types=concatenate(interaction_type_parts, shape=(0,), dtype=np.dtype(np.int64)),
        object_ids=concatenate(object_id_parts, shape=(0,), dtype=np.dtype(np.int64)),
        primitive_ids=concatenate(primitive_id_parts, shape=(0,), dtype=np.dtype(np.int64)),
        interaction_vertices_m=concatenate(vertex_parts, shape=(0, 3), dtype=np.dtype(np.float64)),
        interaction_vertex_frame="sionna_rt_scene_xyz_m",
    )
