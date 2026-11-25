import sys
import types

import numpy as np
import pytest
from integrations.sionna.candidate_audit import (
    CandidatePathCounter,
    instrument_path_solver_candidate_counter,
)


class TensorValue:
    def __init__(self, value):
        self.value = np.asarray(value)

    def numpy(self):
        return self.value


class PathsBuffer:
    def __init__(self, counts, source_indices=None, *, alias=False):
        self.paths_counter = counts
        if source_indices is not None:
            name = "src_indices" if alias else "source_indices"
            setattr(self, name, source_indices)


def install_drjit(monkeypatch, *, shape=None):
    calls = []
    dr = types.ModuleType("drjit")
    dr.eval = lambda *values: calls.append(values)
    dr.shape = shape or (lambda value: value.shape)
    monkeypatch.setitem(sys.modules, "drjit", dr)
    return calls


def test_candidate_counter_preserves_legacy_counter_without_source_indices(monkeypatch):
    eval_calls = install_drjit(monkeypatch)
    buffer = PathsBuffer(TensorValue([4]))
    calls = []
    counter = CandidatePathCounter(lambda *args, **kwargs: calls.append((args, kwargs)) or buffer)

    result = counter("scene", quality=3)

    assert result is buffer
    assert calls == [(("scene",), {"quality": 3})]
    assert len(eval_calls) == 1 and len(eval_calls[0]) == 1
    assert counter.last_candidate_path_counts_per_source == (4,)
    assert counter.last_candidate_path_count == 4
    assert counter.last_total_candidate_path_count == 4


def test_candidate_counter_keeps_legacy_multiple_counter_values(monkeypatch):
    install_drjit(monkeypatch)
    counter = CandidatePathCounter(lambda: PathsBuffer([2, 3]))

    counter()

    assert counter.last_candidate_path_counts_per_source == (2, 3)
    assert counter.last_candidate_path_count == 3
    assert counter.last_total_candidate_path_count == 3


def test_candidate_counter_counts_each_source_and_keeps_empty_sources(monkeypatch):
    eval_calls = install_drjit(monkeypatch)
    positions = types.SimpleNamespace(shape=(3, 4, 3))
    buffer = PathsBuffer(TensorValue([5]), TensorValue([2, 0, 2, 0, 2]))
    counter = CandidatePathCounter(lambda **kwargs: buffer)

    counter(src_positions=positions)

    assert eval_calls[0][1] is buffer.source_indices
    assert counter.last_candidate_path_counts_per_source == (2, 0, 3, 0)
    assert counter.last_candidate_path_count == 3
    assert counter.last_total_candidate_path_count == 5


def test_candidate_counter_finds_positions_in_positional_arguments_and_source_alias(monkeypatch):
    install_drjit(monkeypatch)
    positions = types.SimpleNamespace(shape=(3, 3, 3))
    buffer = PathsBuffer([2], [1, 2], alias=True)
    counter = CandidatePathCounter(lambda *_args: buffer)

    counter(positions)

    assert counter.last_candidate_path_counts_per_source == (0, 1, 1)


@pytest.mark.parametrize("counts", [[], [-1]])
def test_candidate_counter_rejects_empty_or_negative_counts(monkeypatch, counts):
    install_drjit(monkeypatch)
    counter = CandidatePathCounter(lambda: PathsBuffer(counts))

    with pytest.raises(RuntimeError, match="invalid candidate path counter"):
        counter()


def test_candidate_counter_rejects_counter_larger_than_source_buffer(monkeypatch):
    install_drjit(monkeypatch)
    counter = CandidatePathCounter(lambda: PathsBuffer([3], [0, 1]))

    with pytest.raises(RuntimeError, match="exceeds its stored source-index buffer"):
        counter()


@pytest.mark.parametrize("indices", [[-1], [2]])
def test_candidate_counter_rejects_indices_outside_transmitter_roster(monkeypatch, indices):
    install_drjit(monkeypatch)
    positions = types.SimpleNamespace(shape=(3, 2, 3))
    counter = CandidatePathCounter(lambda **_kwargs: PathsBuffer([1], indices))

    with pytest.raises(RuntimeError, match="out-of-range candidate source index"):
        counter(src_positions=positions)


@pytest.mark.parametrize("failure", [IndexError, TypeError, ValueError])
def test_candidate_counter_wraps_unreadable_transmitter_shape(monkeypatch, failure):
    def broken_shape(_value):
        raise failure("shape unavailable")

    install_drjit(monkeypatch, shape=broken_shape)
    positions = types.SimpleNamespace(shape=(3, 2, 3))
    counter = CandidatePathCounter(lambda **_kwargs: PathsBuffer([1], [0]))

    with pytest.raises(RuntimeError, match="Cannot determine the number"):
        counter(src_positions=positions)


def test_candidate_counter_infers_roster_and_handles_no_stored_paths(monkeypatch):
    install_drjit(monkeypatch)
    inferred = CandidatePathCounter(lambda: PathsBuffer([2], [1, 0]))
    inferred()
    assert inferred.last_candidate_path_counts_per_source == (1, 1)

    empty = CandidatePathCounter(lambda: PathsBuffer([0], []))
    empty()
    assert empty.last_candidate_path_counts_per_source == (0,)


def test_candidate_counter_instruments_solver_and_rejects_missing_generator():
    solver = types.SimpleNamespace(_candidate_generator=lambda: "paths")

    counter = instrument_path_solver_candidate_counter(solver)

    assert solver._candidate_generator is counter
    assert counter._delegate() == "paths"
    with pytest.raises(TypeError, match="does not expose its candidate generator"):
        instrument_path_solver_candidate_counter(types.SimpleNamespace())
