"""Candidate-buffer accounting for the pinned Sionna RT PathSolver."""

from __future__ import annotations

from typing import Any

import numpy as np


class CandidatePathCounter:
    """Capture Sionna's stored candidates before invalid paths are discarded."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.last_candidate_path_count: int | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        import drjit as dr

        paths_buffer = self._delegate(*args, **kwargs)
        dr.eval(paths_buffer.paths_counter)
        counter = paths_buffer.paths_counter
        values = counter.numpy() if hasattr(counter, "numpy") else counter
        counts = np.asarray(values, dtype=np.int64).reshape(-1)
        if counts.size == 0 or np.any(counts < 0):
            raise RuntimeError("Sionna returned an invalid candidate path counter")
        self.last_candidate_path_count = int(np.max(counts))
        return paths_buffer


def instrument_path_solver_candidate_counter(solver: Any) -> CandidatePathCounter:
    """Wrap a PathSolver candidate generator and return its per-call counter."""
    candidate_generator = getattr(solver, "_candidate_generator", None)
    if candidate_generator is None:
        raise TypeError("PathSolver does not expose its candidate generator")
    counter = CandidatePathCounter(candidate_generator)
    solver._candidate_generator = counter
    return counter
