"""Candidate-buffer accounting for the pinned Sionna RT PathSolver."""

from __future__ import annotations

from typing import Any

import numpy as np


class CandidatePathCounter:
    """Capture stored candidates before invalid paths are discarded.

    Sionna's buffer counter is global across transmitters. The source-index
    array lets us recover the per-source counts that its path cap actually
    limits, including sources that stored no candidates.
    """

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.last_candidate_path_count: int | None = None
        self.last_candidate_path_counts_per_source: tuple[int, ...] | None = None
        self.last_total_candidate_path_count: int | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        import drjit as dr

        paths_buffer = self._delegate(*args, **kwargs)
        counter = paths_buffer.paths_counter
        source_indices = getattr(
            paths_buffer, "source_indices", getattr(paths_buffer, "src_indices", None)
        )
        dr.eval(counter, source_indices) if source_indices is not None else dr.eval(counter)
        values = counter.numpy() if hasattr(counter, "numpy") else counter
        counts = np.asarray(values, dtype=np.int64).reshape(-1)
        if counts.size == 0 or np.any(counts < 0):
            raise RuntimeError("Sionna returned an invalid candidate path counter")
        total_count = int(np.max(counts))
        if source_indices is None:
            # Keep compatibility with minimal test doubles and older APIs.
            per_source = tuple(int(count) for count in counts)
            if len(per_source) == 1:
                per_source = (total_count,)
        else:
            raw_indices = (
                source_indices.numpy() if hasattr(source_indices, "numpy") else source_indices
            )
            indices = np.asarray(raw_indices, dtype=np.int64).reshape(-1)
            if total_count > indices.size:
                raise RuntimeError(
                    "Sionna candidate counter exceeds its stored source-index buffer"
                )
            src_positions = kwargs.get("src_positions")
            if src_positions is None:
                src_positions = next(
                    (arg for arg in args if getattr(arg, "shape", None) is not None), None
                )
            if src_positions is not None:
                try:
                    source_count = int(dr.shape(src_positions)[1])
                except (IndexError, TypeError, ValueError) as error:
                    raise RuntimeError(
                        "Cannot determine the number of Sionna transmitter sources"
                    ) from error
            else:
                source_count = int(indices[:total_count].max()) + 1 if total_count else 1
            used_indices = indices[:total_count]
            if np.any(used_indices < 0) or np.any(used_indices >= source_count):
                raise RuntimeError("Sionna returned an out-of-range candidate source index")
            per_source = tuple(
                int(count) for count in np.bincount(used_indices, minlength=source_count)
            )
            if sum(per_source) != total_count:
                raise RuntimeError("Per-source candidate counts do not match Sionna's buffer")
        self.last_candidate_path_counts_per_source = per_source
        self.last_candidate_path_count = max(per_source, default=0)
        self.last_total_candidate_path_count = total_count
        return paths_buffer


def instrument_path_solver_candidate_counter(solver: Any) -> CandidatePathCounter:
    """Wrap a PathSolver candidate generator and return its per-call counter."""
    candidate_generator = getattr(solver, "_candidate_generator", None)
    if candidate_generator is None:
        raise TypeError("PathSolver does not expose its candidate generator")
    counter = CandidatePathCounter(candidate_generator)
    solver._candidate_generator = counter
    return counter
