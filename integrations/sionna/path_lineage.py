"""Opt-in, read-only stage snapshots for a pinned Sionna RT PathSolver call."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, cast

import numpy as np


def _numpy(value: Any) -> np.ndarray:
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class _CandidateGeneratorProxy:
    def __init__(self, delegate: Any, observer: PathLineageObserver) -> None:
        self._delegate = delegate
        self._observer = observer

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result = self._delegate(*args, **kwargs)
        self._observer.capture_buffer("candidate_generator_return", result)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class PathLineageObserver:
    """Capture per-stage counts and each zero-order record without solver writes."""

    def __init__(self) -> None:
        self.context: dict[str, Any] | None = None
        self.records: list[dict[str, Any]] = []
        self._case_start = 0
        self._previous_zero_indices: set[int] = set()
        self._writer_stage: str | None = None
        self._patched = False

    def begin_case(self, *, link_id: str, seed: int, depth: int, samples: int) -> None:
        self.context = {
            "link_id": link_id,
            "seed": seed,
            "max_depth": depth,
            "samples_per_src": samples,
        }
        self._case_start = len(self.records)
        self._previous_zero_indices = set()

    def finish_case(self) -> list[dict[str, Any]]:
        case_records = self.records[self._case_start :]
        self.context = None
        return case_records

    def _array(self, buffer: Any, name: str, count: int) -> np.ndarray | None:
        try:
            value = getattr(buffer, name)
            if value is None:
                return None
            return _numpy(value).reshape(-1)[:count]
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return None

    @staticmethod
    def _vector_at(axes: tuple[np.ndarray | None, ...], index: int) -> list[float] | None:
        if any(axis is None for axis in axes):
            return None
        return [float(cast(np.ndarray, axis)[index]) for axis in axes]

    def capture_buffer(self, stage: str, buffer: Any) -> None:
        if self.context is None:
            return
        try:
            import drjit as dr

            counter = buffer.paths_counter
            dr.eval(counter)
            counter_values = _numpy(counter).astype(np.int64).reshape(-1)
            count = int(np.max(counter_values, initial=0))
            depth_size = int(buffer.depth_dim_size)
            dr.eval(
                buffer.valid,
                buffer.source_indices,
                buffer.target_indices,
                buffer.interaction_types.array,
                buffer.shapes,
                buffer.primitives,
                buffer.k_tx,
                buffer.k_rx,
            )
            lineage_arrays = [
                getattr(buffer, name)
                for name in (
                    "_lineage_stage",
                    "_lineage_depth",
                    "_lineage_valid",
                    "_lineage_written",
                )
                if hasattr(buffer, name)
            ]
            if lineage_arrays:
                dr.eval(*lineage_arrays)
            interactions = _numpy(buffer.interaction_types.array).reshape(-1, depth_size)[:count]
            zero_indices = set(np.flatnonzero(np.all(interactions == 0, axis=1)).tolist())
            valid = self._array(buffer, "valid", count)
            sources = self._array(buffer, "source_indices", count)
            targets = self._array(buffer, "target_indices", count)
            shapes = _numpy(buffer.shapes).reshape(-1, depth_size)[:count]
            primitives = _numpy(buffer.primitives).reshape(-1, depth_size)[:count]
            vertices = np.stack(
                [
                    _numpy(buffer.vertices_x).reshape(-1, depth_size)[:count],
                    _numpy(buffer.vertices_y).reshape(-1, depth_size)[:count],
                    _numpy(buffer.vertices_z).reshape(-1, depth_size)[:count],
                ],
                axis=-1,
            )
            k_tx = tuple(
                self._array(buffer, axis, count) for axis in ("_k_tx_x", "_k_tx_y", "_k_tx_z")
            )
            k_rx = tuple(
                self._array(buffer, axis, count) for axis in ("_k_rx_x", "_k_rx_y", "_k_rx_z")
            )
            lineage_depth = self._array(buffer, "_lineage_depth", count)
            lineage_stage = self._array(buffer, "_lineage_stage", count)
            lineage_valid = self._array(buffer, "_lineage_valid", count)
            lineage_written = self._array(buffer, "_lineage_written", count)
            order_counts = Counter(int(np.count_nonzero(row)) for row in interactions)
            zero_rows = []
            for index in sorted(zero_indices):
                candidate_depth = int(lineage_depth[index]) if lineage_depth is not None else None
                has_writer_lineage = (
                    lineage_stage is not None
                    and lineage_written is not None
                    and bool(lineage_written[index])
                )
                writer_stage = None
                if has_writer_lineage and lineage_stage is not None:
                    writer_stage = {1: "direct_los", 2: "shoot_and_bounce"}.get(
                        int(lineage_stage[index])
                    )
                zero_rows.append(
                    {
                        "buffer_index": int(index),
                        "new_zero_order_index_since_previous_candidate_stage": (
                            int(index) not in self._previous_zero_indices
                        ),
                        "source_index": int(sources[index]) if sources is not None else None,
                        "target_index": int(targets[index]) if targets is not None else None,
                        "valid": bool(valid[index]) if valid is not None else None,
                        "interaction_types": interactions[index].astype(int).tolist(),
                        "shape_ids": shapes[index].astype(int).tolist(),
                        "primitive_ids": primitives[index].astype(int).tolist(),
                        "vertices_m": vertices[index].astype(float).tolist(),
                        "k_tx": self._vector_at(k_tx, index),
                        "k_rx": self._vector_at(k_rx, index),
                        "writer_stage": writer_stage,
                        "candidate_depth": candidate_depth if has_writer_lineage else None,
                        "candidate_valid_before_image_method": (
                            bool(lineage_valid[index])
                            if has_writer_lineage and lineage_valid is not None
                            else None
                        ),
                        "buffer_write_active": bool(lineage_written[index])
                        if lineage_written is not None
                        else None,
                    }
                )
            record = {
                **self.context,
                "stage": stage,
                "buffer_count": count,
                "zero_interaction_count": len(zero_indices),
                "interaction_order_counts": {
                    str(order): count for order, count in sorted(order_counts.items())
                },
                "zero_interaction_records": zero_rows,
            }
            if "candidate_generator" in stage or stage in {"after_los", "after_shoot_and_bounce"}:
                self._previous_zero_indices = zero_indices
            self.records.append(_json_safe(record))
        except Exception as error:
            raise RuntimeError(f"path-lineage snapshot failed at {stage}: {error}") from error

    def capture_paths(self, paths: Any) -> None:
        if self.context is None:
            return
        try:
            valid = _numpy(paths.valid).astype(bool).reshape(-1)
            interactions = _numpy(paths.interactions.array).astype(np.uint32)
            interactions = interactions.reshape(paths._paths_buffer.max_depth, -1, valid.size)[
                :, 0, :
            ]
            zero_indices = np.flatnonzero(np.all(interactions == 0, axis=0))
            source = _numpy(paths._paths_buffer.source_indices).astype(np.int64).reshape(-1)
            target = _numpy(paths._paths_buffer.target_indices).astype(np.int64).reshape(-1)
            lineage_stage = self._array(paths._paths_buffer, "_lineage_stage", valid.size)
            lineage_depth = self._array(paths._paths_buffer, "_lineage_depth", valid.size)
            lineage_valid = self._array(paths._paths_buffer, "_lineage_valid", valid.size)
            lineage_written = self._array(paths._paths_buffer, "_lineage_written", valid.size)
            tau = _numpy(paths.tau).reshape(-1)
            theta_t = _numpy(paths.theta_t).reshape(-1)
            phi_t = _numpy(paths.phi_t).reshape(-1)
            a_real, a_imag = paths.a
            gain = _numpy(a_real).reshape(-1) + 1j * _numpy(a_imag).reshape(-1)
            rows = [
                {
                    "path_index": int(index),
                    "path_ind": int(_numpy(paths._path_ind).reshape(-1)[index]),
                    "source_index": int(source[index]),
                    "target_index": int(target[index]),
                    "valid": bool(valid[index]),
                    "interaction_types": interactions[:, index].astype(int).tolist(),
                    "delay_s": float(tau[index]),
                    "coefficient_real": float(gain[index].real),
                    "coefficient_imag": float(gain[index].imag),
                    "theta_t_rad": float(theta_t[index]),
                    "phi_t_rad": float(phi_t[index]),
                    "writer_stage": (
                        {1: "direct_los", 2: "shoot_and_bounce"}.get(int(lineage_stage[index]))
                        if lineage_stage is not None
                        and lineage_written is not None
                        and bool(lineage_written[index])
                        else None
                    ),
                    "candidate_depth": (
                        int(lineage_depth[index])
                        if lineage_depth is not None
                        and lineage_written is not None
                        and bool(lineage_written[index])
                        else None
                    ),
                    "candidate_valid_before_image_method": (
                        bool(lineage_valid[index])
                        if lineage_valid is not None
                        and lineage_written is not None
                        and bool(lineage_written[index])
                        else None
                    ),
                }
                for index in zero_indices
            ]
            self.records.append(
                _json_safe(
                    {
                        **self.context,
                        "stage": "final_paths_tensor",
                        "buffer_count": int(valid.size),
                        "zero_interaction_count": int(len(zero_indices)),
                        "zero_interaction_records": rows,
                    }
                )
            )
        except Exception as error:
            raise RuntimeError(f"path-lineage snapshot failed at final Paths: {error}") from error

    def attach(self, solver: Any) -> None:
        """Install temporary instance/class wrappers; run in a dedicated process."""
        if self._patched:
            raise RuntimeError("observer is already attached")
        import drjit as dr
        import mitsuba as mi
        from sionna.rt.path_solvers.paths import Paths
        from sionna.rt.path_solvers.paths_buffer import PathsBuffer

        counter = solver._candidate_generator
        generator = counter._delegate
        for method_name, stage in (
            ("_los", "after_los"),
            ("_shoot_and_bounce", "after_shoot_and_bounce"),
        ):
            original = getattr(generator, method_name)

            def wrapped(
                *args: Any, _original: Any = original, _stage: str = stage, **kwargs: Any
            ) -> Any:
                previous_stage = self._writer_stage
                self._writer_stage = _stage
                try:
                    result = _original(*args, **kwargs)
                finally:
                    self._writer_stage = previous_stage
                self.capture_buffer(_stage, args[3])
                return result

            setattr(generator, method_name, wrapped)

        original_add_paths = PathsBuffer.add_paths

        def wrapped_add_paths(
            buffer: Any,
            depth: Any,
            indices: Any,
            sample_data: Any,
            valid: Any,
            tgt_index: Any,
            k_tx: Any,
            k_rx: Any,
            active: Any,
        ) -> Any:
            if not hasattr(buffer, "_lineage_stage"):
                buffer._lineage_stage = dr.zeros(mi.UInt, buffer.buffer_size)
                buffer._lineage_depth = dr.zeros(mi.UInt, buffer.buffer_size)
                buffer._lineage_valid = dr.zeros(mi.Bool, buffer.buffer_size)
                buffer._lineage_written = dr.zeros(mi.Bool, buffer.buffer_size)
            lane_count = dr.width(active)
            depth_values = (
                dr.full(mi.UInt, depth, lane_count)
                if dr.width(depth) == 1 and lane_count > 1
                else depth
            )
            stage_code = {"after_los": 1, "after_shoot_and_bounce": 2}.get(
                self._writer_stage or "", 0
            )
            dr.scatter(
                buffer._lineage_stage,
                dr.full(mi.UInt, stage_code, lane_count),
                indices,
                active=active,
            )
            dr.scatter(buffer._lineage_depth, depth_values, indices, active=active)
            dr.scatter(buffer._lineage_valid, valid, indices, active=active)
            dr.scatter(
                buffer._lineage_written,
                dr.full(mi.Bool, True, lane_count),
                indices,
                active=active,
            )
            result = original_add_paths(
                buffer, depth, indices, sample_data, valid, tgt_index, k_tx, k_rx, active
            )
            dr.schedule(
                buffer._lineage_stage,
                buffer._lineage_depth,
                buffer._lineage_valid,
                buffer._lineage_written,
            )
            return result

        PathsBuffer.add_paths = wrapped_add_paths

        counter._delegate = _CandidateGeneratorProxy(generator, self)
        for attribute, stage in (
            ("_image_method", "after_image_method"),
            ("_field_calculator", "after_field_calculator"),
        ):
            original = getattr(solver, attribute)

            def wrapped_component(
                *args: Any, _original: Any = original, _stage: str = stage, **kwargs: Any
            ) -> Any:
                result = _original(*args, **kwargs)
                self.capture_buffer(_stage, result)
                return result

            setattr(solver, attribute, wrapped_component)

        for method_name in ("shrink", "detach_geometry", "discard_invalid"):
            original = getattr(PathsBuffer, method_name)
            stage_before = f"before_{method_name}"
            stage_after = f"after_{method_name}"

            def wrapped_buffer_method(
                buffer: Any,
                *args: Any,
                _original: Any = original,
                _before: str = stage_before,
                _after: str = stage_after,
                **kwargs: Any,
            ) -> Any:
                self.capture_buffer(_before, buffer)
                lineage_arrays: tuple[Any, ...] | None = None
                valid_indices: Any = None
                if _before == "before_discard_invalid" and hasattr(buffer, "_lineage_stage"):
                    lineage_arrays = (
                        buffer._lineage_stage,
                        buffer._lineage_depth,
                        buffer._lineage_valid,
                        buffer._lineage_written,
                    )
                    valid_indices = dr.compress(buffer.valid)
                    dr.eval(*lineage_arrays, valid_indices)
                result = _original(buffer, *args, **kwargs)
                if lineage_arrays is not None and valid_indices is not None:
                    (
                        buffer._lineage_stage,
                        buffer._lineage_depth,
                        buffer._lineage_valid,
                        buffer._lineage_written,
                    ) = (
                        dr.gather(mi.UInt, array, valid_indices)
                        if index in {0, 1}
                        else dr.gather(mi.Bool, array, valid_indices)
                        for index, array in enumerate(lineage_arrays)
                    )
                self.capture_buffer(_after, buffer)
                return result

            setattr(PathsBuffer, method_name, wrapped_buffer_method)

        original_init = Paths.__init__

        def wrapped_paths_init(paths: Any, *args: Any, **kwargs: Any) -> None:
            original_init(paths, *args, **kwargs)
            self.capture_paths(paths)

        Paths.__init__ = wrapped_paths_init
        self._patched = True
