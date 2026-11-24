"""Path-lineage observer: stage snapshots of PathSolver buffers, without solver writes."""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from integrations.sionna.path_lineage import (
    PathLineageObserver,
    _CandidateGeneratorProxy,
    _json_safe,
    _numpy,
)


class FakeArray:
    """Stands in for a Dr.Jit array: exposes its values through ``numpy()``."""

    def __init__(self, values) -> None:
        self._values = np.asarray(values)

    def numpy(self) -> np.ndarray:
        return self._values


@pytest.fixture
def fake_drjit(monkeypatch):
    module = types.ModuleType("drjit")
    module.eval = lambda *arrays: None
    monkeypatch.setitem(sys.modules, "drjit", module)
    return module


def _buffer(*, depth: int = 2, n: int = 4, lineage: bool = True, zero_rows=(0, 2)):
    interactions = np.ones((n, depth), dtype=np.int64)
    for row in zero_rows:
        interactions[row] = 0
    buffer = types.SimpleNamespace(
        paths_counter=FakeArray([n]),
        depth_dim_size=depth,
        valid=FakeArray(np.arange(n) % 2 == 0),
        source_indices=FakeArray(np.arange(n)),
        target_indices=FakeArray(np.arange(n) + 10),
        interaction_types=types.SimpleNamespace(array=FakeArray(interactions.reshape(-1))),
        shapes=FakeArray(np.arange(n * depth)),
        primitives=FakeArray(np.arange(n * depth) + 100),
        k_tx=FakeArray(np.zeros(n)),
        k_rx=FakeArray(np.zeros(n)),
        vertices_x=FakeArray(np.linspace(0, 1, n * depth)),
        vertices_y=FakeArray(np.linspace(1, 2, n * depth)),
        vertices_z=FakeArray(np.linspace(2, 3, n * depth)),
        _k_tx_x=FakeArray(np.full(n, 0.1)),
        _k_tx_y=FakeArray(np.full(n, 0.2)),
        _k_tx_z=FakeArray(np.full(n, 0.3)),
        _k_rx_x=FakeArray(np.full(n, 0.4)),
        _k_rx_y=FakeArray(np.full(n, 0.5)),
        _k_rx_z=FakeArray(np.full(n, 0.6)),
    )
    if lineage:
        buffer._lineage_stage = FakeArray(np.array([1, 2, 0, 2]))
        buffer._lineage_depth = FakeArray(np.array([1, 2, 0, 1]))
        buffer._lineage_valid = FakeArray(np.array([True, False, True, True]))
        buffer._lineage_written = FakeArray(np.array([True, True, False, True]))
    return buffer


def test_numpy_accepts_arrays_and_array_like_objects() -> None:
    assert _numpy(FakeArray([1, 2])).tolist() == [1, 2]
    assert _numpy([3, 4]).tolist() == [3, 4]


def test_json_safe_converts_numpy_scalars_and_nonfinite_floats() -> None:
    value = {
        "a": np.float64(1.5),
        "b": [np.int64(2), float("nan"), float("inf"), (np.bool_(True),)],
        1: "kept",
    }
    safe = _json_safe(value)
    assert safe == {"a": 1.5, "b": [2, None, None, [True]], "1": "kept"}
    json.dumps(safe, allow_nan=False)


def test_candidate_proxy_records_the_delegate_result_and_forwards_attributes() -> None:
    class Delegate:
        width = 7

        def __call__(self, value):
            return value * 2

    observer = PathLineageObserver()
    observer.context = {"link_id": "l", "seed": 1, "max_depth": 2, "samples_per_src": 3}
    captured = []
    observer.capture_buffer = lambda stage, buffer: captured.append((stage, buffer))
    proxy = _CandidateGeneratorProxy(Delegate(), observer)
    assert proxy(5) == 10
    assert captured == [("candidate_generator_return", 10)]
    assert proxy.width == 7


def test_case_lifecycle_isolates_records_per_case(fake_drjit) -> None:
    observer = PathLineageObserver()
    observer.capture_buffer("ignored", _buffer())
    assert observer.records == []
    observer.begin_case(link_id="first", seed=1, depth=2, samples=4)
    observer.capture_buffer("candidate_generator_return", _buffer())
    first = observer.finish_case()
    assert [record["link_id"] for record in first] == ["first"]
    assert observer.context is None
    observer.begin_case(link_id="second", seed=2, depth=2, samples=4)
    observer.capture_buffer("after_shoot_and_bounce", _buffer())
    second = observer.finish_case()
    assert [record["link_id"] for record in second] == ["second"]
    assert len(observer.records) == 2


def test_buffer_snapshot_maps_writer_lineage_and_tracks_new_zero_order_paths(fake_drjit) -> None:
    observer = PathLineageObserver()
    observer.begin_case(link_id="link", seed=3, depth=2, samples=4)
    observer.capture_buffer("candidate_generator_return", _buffer(zero_rows=(0, 2)))
    observer.capture_buffer("after_shoot_and_bounce", _buffer(zero_rows=(0,)))
    first, second = observer.finish_case()

    assert first["buffer_count"] == 4
    assert first["zero_interaction_count"] == 2
    assert first["interaction_order_counts"] == {"0": 2, "2": 2}
    rows = {row["buffer_index"]: row for row in first["zero_interaction_records"]}
    assert set(rows) == {0, 2}
    # Stage codes 1 and 2 map to the named writer stages; index 2 has stage 0, so none.
    assert rows[0]["writer_stage"] == "direct_los"
    assert rows[0]["candidate_depth"] == 1
    assert rows[0]["candidate_valid_before_image_method"] is True
    assert rows[0]["buffer_write_active"] is True
    assert rows[2]["writer_stage"] is None
    assert rows[2]["buffer_write_active"] is False
    assert rows[0]["k_tx"] == pytest.approx([0.1, 0.2, 0.3])
    assert rows[0]["vertices_m"][0] == pytest.approx([0.0, 1.0, 2.0])
    assert all(row["new_zero_order_index_since_previous_candidate_stage"] for row in first["zero_interaction_records"])

    # The second snapshot only flags zero-order paths that were not present in the earlier stage.
    new_flags = {
        row["buffer_index"]: row["new_zero_order_index_since_previous_candidate_stage"]
        for row in second["zero_interaction_records"]
    }
    assert new_flags == {0: False}
    assert second["stage"] == "after_shoot_and_bounce"


def test_snapshot_without_writer_lineage_reports_missing_writer_fields(fake_drjit) -> None:
    observer = PathLineageObserver()
    observer.begin_case(link_id="bare", seed=0, depth=2, samples=1)
    observer.capture_buffer("before_detach_geometry", _buffer(lineage=False, zero_rows=(1,)))
    (record,) = observer.finish_case()
    row = record["zero_interaction_records"][0]
    assert row["writer_stage"] is None
    assert row["candidate_depth"] is None
    assert row["candidate_valid_before_image_method"] is None
    assert row["buffer_write_active"] is None


def test_snapshot_handles_missing_optional_direction_arrays(fake_drjit) -> None:
    buffer = _buffer(zero_rows=(0,))
    buffer._k_tx_x = None
    observer = PathLineageObserver()
    observer.begin_case(link_id="nodir", seed=0, depth=2, samples=1)
    observer.capture_buffer("after_discard_invalid", buffer)
    (record,) = observer.finish_case()
    assert record["zero_interaction_records"][0]["k_tx"] is None


def test_buffer_snapshot_outside_a_case_is_a_noop(fake_drjit) -> None:
    observer = PathLineageObserver()
    observer.capture_buffer("anything", object())
    assert observer.records == []


def test_buffer_snapshot_failures_name_the_stage(fake_drjit) -> None:
    observer = PathLineageObserver()
    observer.begin_case(link_id="broken", seed=0, depth=2, samples=1)
    with pytest.raises(RuntimeError, match="snapshot failed at after_los"):
        observer.capture_buffer("after_los", object())


def test_buffer_snapshot_without_optional_lineage_arrays_still_reports(fake_drjit) -> None:
    buffer = _buffer(lineage=False, zero_rows=())
    observer = PathLineageObserver()
    observer.begin_case(link_id="none-zero", seed=0, depth=2, samples=1)
    observer.capture_buffer("after_image_method", buffer)
    (record,) = observer.finish_case()
    assert record["zero_interaction_count"] == 0
    assert record["zero_interaction_records"] == []


def _paths(*, n: int = 3, depth: int = 2, zero_column: int = 1):
    interactions = np.ones((depth, n), dtype=np.uint32)
    interactions[:, zero_column] = 0
    buffer = types.SimpleNamespace(
        max_depth=depth,
        source_indices=FakeArray(np.arange(n)),
        target_indices=FakeArray(np.arange(n) + 5),
        _lineage_stage=FakeArray(np.array([1, 2, 0])),
        _lineage_depth=FakeArray(np.array([1, 2, 0])),
        _lineage_valid=FakeArray(np.array([True, True, False])),
        _lineage_written=FakeArray(np.array([True, True, True])),
    )
    return types.SimpleNamespace(
        valid=FakeArray(np.array([True, True, False])[:n]),
        interactions=types.SimpleNamespace(array=FakeArray(interactions.reshape(-1))),
        _paths_buffer=buffer,
        _path_ind=FakeArray(np.arange(n) * 10),
        tau=FakeArray(np.linspace(1e-9, 3e-9, n)),
        theta_t=FakeArray(np.full(n, 0.5)),
        phi_t=FakeArray(np.full(n, 0.25)),
        a=(FakeArray(np.arange(n, dtype=float)), FakeArray(np.arange(n, dtype=float) * -1)),
    )


def test_final_paths_snapshot_records_zero_order_paths_and_writer_lineage() -> None:
    observer = PathLineageObserver()
    observer.begin_case(link_id="final", seed=9, depth=2, samples=3)
    observer.capture_paths(_paths(zero_column=1))
    (record,) = observer.finish_case()
    assert record["stage"] == "final_paths_tensor"
    assert record["buffer_count"] == 3
    assert record["zero_interaction_count"] == 1
    (row,) = record["zero_interaction_records"]
    assert row["path_index"] == 1
    assert row["path_ind"] == 10
    assert row["valid"] is True
    assert row["writer_stage"] == "shoot_and_bounce"
    assert row["candidate_depth"] == 2
    assert row["coefficient_real"] == pytest.approx(1.0)
    assert row["coefficient_imag"] == pytest.approx(-1.0)
    assert row["delay_s"] == pytest.approx(2e-9)


def test_final_paths_snapshot_without_lineage_or_outside_a_case() -> None:
    paths = _paths(zero_column=0)
    delattr_targets = [paths._paths_buffer]
    for target in delattr_targets:
        target._lineage_stage = None
        target._lineage_written = None
    observer = PathLineageObserver()
    observer.capture_paths(paths)
    assert observer.records == []
    observer.begin_case(link_id="nolineage", seed=1, depth=2, samples=3)
    observer.capture_paths(paths)
    (record,) = observer.finish_case()
    (row,) = record["zero_interaction_records"]
    assert row["writer_stage"] is None
    assert row["candidate_depth"] is None
    assert row["candidate_valid_before_image_method"] is None


def test_final_paths_failures_are_wrapped_with_context() -> None:
    observer = PathLineageObserver()
    observer.begin_case(link_id="broken", seed=1, depth=2, samples=3)
    with pytest.raises(RuntimeError, match="final Paths"):
        observer.capture_paths(object())
