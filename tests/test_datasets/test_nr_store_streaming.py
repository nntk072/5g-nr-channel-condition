from __future__ import annotations

import hashlib
import io
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest
from datasets import iter_nr_record_shards
from datasets.nr import NRChannelRecord, NRObservation
from datasets.nr_store import NRStoredRecord, read_nr_record, write_nr_record
from receivers.nr import NRChannelEstimate, NRPUSCHConfig


def _record(sample_id: str) -> NRChannelRecord:
    estimate = NRChannelEstimate(
        h_hat=np.zeros((1, 1, 1, 1, 1, 14, 24), dtype=np.complex64),
        error_variance=np.zeros((1, 1, 1, 1, 1, 14, 24), dtype=np.float32),
        pilot_mask=np.zeros((1, 1, 14, 24), dtype=np.bool_),
        metadata={"receiver": "streaming-fixture"},
    )
    observation = NRObservation(
        sample_id=sample_id,
        received_grid=np.zeros((1, 1, 1, 14, 24), dtype=np.complex64),
        channel_estimate=estimate,
        fft_bin_indices=tuple(range(24)),
        receiver_config=NRPUSCHConfig(n_size_bwp=2, subcarrier_spacing_khz=30),
    )
    return NRChannelRecord(
        observation=observation,
        targets={"profile_proxy": np.asarray(1, dtype=np.int8)},
        target_masks={},
        provenance={"source": "streaming-test"},
    )


def _estimated_budget(root, stored: NRStoredRecord) -> int:
    path = root / stored.relative_path
    with zipfile.ZipFile(path) as archive:
        uncompressed_bytes = sum(member.file_size for member in archive.infolist())
    return 2 * path.stat().st_size + 2 * uncompressed_bytes


def test_shards_are_bounded_lazy_and_ordered_by_sample_id(tmp_path, monkeypatch):
    stored_records = [
        write_nr_record(tmp_path, _record(sample_id)) for sample_id in ("c", "a", "b")
    ]
    budget = max(_estimated_budget(tmp_path, stored) for stored in stored_records)

    import datasets.nr_store as nr_store

    real_reader = nr_store.read_nr_record
    loaded_ids: list[str] = []

    def tracked_reader(root, manifest):
        loaded_ids.append(manifest.sample_id)
        return real_reader(root, manifest)

    monkeypatch.setattr(nr_store, "read_nr_record", tracked_reader)
    shards = iter_nr_record_shards(tmp_path, max_shard_bytes=budget)

    first = next(shards)
    assert tuple(record.observation.sample_id for record in first) == ("a",)
    assert loaded_ids == ["a"]
    remaining = tuple(shards)
    assert tuple(record.observation.sample_id for shard in remaining for record in shard) == (
        "b",
        "c",
    )
    assert loaded_ids == ["a", "b", "c"]


def test_shard_order_is_repeatable_independent_of_write_order(tmp_path):
    for sample_id in ("zeta", "alpha", "mu"):
        write_nr_record(tmp_path, _record(sample_id))

    def ordered_ids():
        return tuple(
            record.observation.sample_id
            for shard in iter_nr_record_shards(tmp_path, max_shard_bytes=10_000_000)
            for record in shard
        )

    assert ordered_ids() == ("alpha", "mu", "zeta")
    assert ordered_ids() == ("alpha", "mu", "zeta")


def test_records_are_grouped_until_the_declared_memory_budget(tmp_path):
    stored_records = [
        write_nr_record(tmp_path, _record(sample_id)) for sample_id in ("c", "a", "b")
    ]
    budget = sum(_estimated_budget(tmp_path, stored) for stored in stored_records)

    shards = tuple(iter_nr_record_shards(tmp_path, max_shard_bytes=budget))

    assert len(shards) == 1
    assert tuple(record.observation.sample_id for record in shards[0]) == ("a", "b", "c")


def test_closing_shard_generator_removes_temporary_inventory(tmp_path, monkeypatch):
    stored_records = [write_nr_record(tmp_path, _record(sample_id)) for sample_id in ("a", "b")]
    budget = max(_estimated_budget(tmp_path, stored) for stored in stored_records)

    import datasets.nr_store as nr_store

    temporary_dirs = []
    real_temporary_directory = nr_store.tempfile.TemporaryDirectory

    def tracked_temporary_directory(*args, **kwargs):
        temporary_dir = real_temporary_directory(*args, **kwargs)
        temporary_dirs.append(Path(temporary_dir.name))
        return temporary_dir

    monkeypatch.setattr(nr_store.tempfile, "TemporaryDirectory", tracked_temporary_directory)
    shards = iter_nr_record_shards(tmp_path, max_shard_bytes=budget)

    assert tuple(record.observation.sample_id for record in next(shards)) == ("a",)
    assert len(temporary_dirs) == 1
    assert temporary_dirs[0].is_dir()

    shards.close()

    assert not temporary_dirs[0].exists()


def test_reader_exception_removes_temporary_inventory_and_propagates(tmp_path, monkeypatch):
    write_nr_record(tmp_path, _record("reader-error"))

    import datasets.nr_store as nr_store

    temporary_dirs = []
    real_temporary_directory = nr_store.tempfile.TemporaryDirectory

    def tracked_temporary_directory(*args, **kwargs):
        temporary_dir = real_temporary_directory(*args, **kwargs)
        temporary_dirs.append(Path(temporary_dir.name))
        return temporary_dir

    monkeypatch.setattr(nr_store.tempfile, "TemporaryDirectory", tracked_temporary_directory)
    error = RuntimeError("record reader failed")

    def failing_reader(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(nr_store, "read_nr_record", failing_reader)

    with pytest.raises(RuntimeError, match="record reader failed") as raised:
        next(iter_nr_record_shards(tmp_path, max_shard_bytes=10_000_000))

    assert raised.value is error
    assert len(temporary_dirs) == 1
    assert not temporary_dirs[0].exists()


def test_oversized_record_fails_before_arrays_are_loaded(tmp_path, monkeypatch):
    write_nr_record(tmp_path, _record("large"))

    import datasets.nr_store as nr_store

    monkeypatch.setattr(
        nr_store,
        "read_nr_record",
        lambda *_args, **_kwargs: pytest.fail("oversized payload must not be decoded"),
    )
    with pytest.raises(ValueError, match="exceeds max_shard_bytes"):
        next(iter_nr_record_shards(tmp_path, max_shard_bytes=1))


def test_malformed_npy_shape_is_rejected_before_array_decode(tmp_path):
    stored = write_nr_record(tmp_path, _record("malformed-header"))
    record_path = tmp_path / stored.relative_path
    buffer = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        buffer,
        {"descr": np.dtype("float64").str, "fortran_order": False, "shape": (10**9,)},
    )
    buffer.write(b"\0")
    with zipfile.ZipFile(record_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("oversized.npy", buffer.getvalue())

    with pytest.raises(ValueError, match="header size does not match"):
        next(iter_nr_record_shards(tmp_path, max_shard_bytes=10_000_000))


def test_record_reader_rejects_symlink_escape(tmp_path):
    stored = write_nr_record(tmp_path, _record("symlink-escape"))
    object_path = tmp_path / stored.relative_path
    outside_path = tmp_path.parent / f"{tmp_path.name}-outside.npz"
    shutil.copyfile(object_path, outside_path)
    object_path.unlink()
    try:
        object_path.symlink_to(outside_path)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(ValueError, match="resolves outside the store"):
        read_nr_record(tmp_path, stored)


def test_record_reader_rejects_fifo_without_blocking(tmp_path):
    stored = write_nr_record(tmp_path, _record("fifo-record"))
    object_path = tmp_path / stored.relative_path
    object_path.unlink()
    os.mkfifo(object_path)

    with pytest.raises(ValueError, match="regular file"):
        read_nr_record(tmp_path, stored)


@pytest.mark.parametrize("max_shard_bytes", [True, 0, -1, 1.5])
def test_invalid_shard_byte_budgets_fail_closed(tmp_path, max_shard_bytes):
    with pytest.raises((TypeError, ValueError), match="max_shard_bytes"):
        next(iter_nr_record_shards(tmp_path, max_shard_bytes=max_shard_bytes))


def test_misnamed_manifest_fails_closed(tmp_path):
    stored = write_nr_record(tmp_path, _record("duplicate"))
    manifest = tmp_path / "manifests" / f"{hashlib.sha256(b'other-name').hexdigest()}.json"
    shutil.copyfile(
        tmp_path / "manifests" / f"{hashlib.sha256(b'duplicate').hexdigest()}.json", manifest
    )

    with pytest.raises(ValueError, match="filename does not match its sample ID"):
        next(iter_nr_record_shards(tmp_path, max_shard_bytes=10_000_000))
    assert stored.sample_id == "duplicate"


def test_duplicate_manifest_json_keys_are_rejected(tmp_path):
    stored = write_nr_record(tmp_path, _record("ambiguous"))
    manifest_key = hashlib.sha256(stored.sample_id.encode()).hexdigest()
    manifest_path = tmp_path / "manifests" / f"{manifest_key}.json"
    manifest_path.write_text(
        '{"sample_id":"ambiguous","sample_id":"ambiguous",'
        f'"content_sha256":"{stored.content_sha256}",'
        f'"relative_path":"{stored.relative_path}","schema_version":1}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON object key: sample_id"):
        next(iter_nr_record_shards(tmp_path, max_shard_bytes=10_000_000))


def test_duplicate_sample_id_manifests_are_rejected(tmp_path, monkeypatch):
    stored = write_nr_record(tmp_path, _record("duplicate-sample"))
    manifest_key = hashlib.sha256(stored.sample_id.encode()).hexdigest()
    manifest_dir = tmp_path / "manifests"
    original = manifest_dir / f"{manifest_key}.json"
    alias = manifest_dir / "alias.json"
    try:
        alias.symlink_to(original)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    import datasets.nr_store as nr_store

    monkeypatch.setattr(nr_store, "_nr_record_memory_estimate", lambda *_args, **_kwargs: 1)
    with pytest.raises(ValueError, match="duplicate NR record manifest"):
        next(iter_nr_record_shards(tmp_path, max_shard_bytes=10_000_000))


def test_empty_store_yields_no_shards(tmp_path):
    assert tuple(iter_nr_record_shards(tmp_path, max_shard_bytes=10_000_000)) == ()
