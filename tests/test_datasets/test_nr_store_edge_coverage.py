"""Focused boundary checks for the pickle-free NR store."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import sqlite3
from dataclasses import replace
from pathlib import Path

import datasets.nr_store as nr_store
import numpy as np
import pytest
from datasets.nr import NRChannelRecord, NRObservation, NRTrajectory, TargetDefinition
from datasets.nr_store import (
    iter_nr_record_shards,
    read_nr_record,
    read_nr_trajectory,
    write_nr_record,
    write_nr_trajectory,
)
from receivers.nr import NRChannelEstimate, NRPUSCHConfig


def _record(sample_id: str) -> NRChannelRecord:
    estimate = NRChannelEstimate(
        h_hat=np.zeros((1, 1, 1, 1, 1, 14, 24), dtype=np.complex64),
        error_variance=np.zeros((1, 1, 1, 1, 1, 14, 24), dtype=np.float32),
        pilot_mask=np.zeros((1, 1, 14, 24), dtype=np.bool_),
        metadata={"receiver": "edge-coverage-fixture"},
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
        provenance={"source": "edge-coverage-test"},
        target_definitions={
            "profile_proxy": TargetDefinition.versioned(
                semantic="edge coverage fixture",
                units="class index",
                source="edge coverage test",
                reference="fixed synthetic test value",
                aggregation_rule="one scalar per record",
                definition_version="tests.nr-store-edge.profile-proxy.v1",
            )
        },
    )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.asarray([object()], dtype=object), "may not contain Python objects"),
        (np.asarray(["text"]), "must use a numeric, boolean, or complex dtype"),
    ],
)
def test_numpy_boundary_rejects_non_numeric_values(value, message):
    with pytest.raises(TypeError, match=message):
        nr_store._as_numpy(value, field="fixture")


def test_shard_inventory_rejects_manifest_directory_escape(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "manifests").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="manifest directory resolves outside"):
        next(iter_nr_record_shards(root, max_shard_bytes=1024))


def test_shard_inventory_rejects_manifest_file_escape(tmp_path):
    stored = write_nr_record(tmp_path, _record("manifest-file-escape"))
    manifest_key = hashlib.sha256(stored.sample_id.encode()).hexdigest()
    manifest = tmp_path / "manifests" / f"{manifest_key}.json"
    outside = tmp_path.parent / f"{tmp_path.name}-manifest.json"
    shutil.copyfile(manifest, outside)
    manifest.unlink()
    manifest.symlink_to(outside)

    with pytest.raises(ValueError, match="manifest path resolves outside"):
        next(iter_nr_record_shards(tmp_path, max_shard_bytes=10_000_000))


def test_record_reader_rejects_manifest_path_outside_store(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.json"
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest must remain inside its store"):
        read_nr_record(tmp_path, outside)


def test_store_record_open_uses_portable_open_fallback(tmp_path, monkeypatch):
    stored = write_nr_record(tmp_path, _record("portable-open"))
    monkeypatch.setattr(nr_store.os, "name", "nt")

    with nr_store._open_store_record(tmp_path, stored.relative_path) as stream:
        assert stream.read(4)


def test_stable_read_rejects_data_that_grows_past_limit(tmp_path, monkeypatch):
    path = tmp_path / "payload"
    path.write_bytes(b"x")
    real_open = nr_store._open_store_record

    class GrowingStream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.stream.close()

        def fileno(self):
            return self.stream.fileno()

        def read(self, _size):
            return b"xxx"

    monkeypatch.setattr(
        nr_store,
        "_open_store_record",
        lambda base, relative: GrowingStream(real_open(base, relative)),
    )

    with pytest.raises(ValueError, match="payload exceeds the 2-byte limit"):
        nr_store._stable_store_record_bytes(tmp_path, "payload", maximum_bytes=2, label="payload")


def test_stable_read_rejects_file_that_changes_during_read(tmp_path, monkeypatch):
    path = tmp_path / "payload"
    path.write_bytes(b"stable")
    real_identity = nr_store._file_identity
    calls = 0

    def changed_identity(info):
        nonlocal calls
        calls += 1
        identity = real_identity(info)
        return identity if calls == 1 else (*identity[:-1], identity[-1] + 1)

    monkeypatch.setattr(nr_store, "_file_identity", changed_identity)

    with pytest.raises(ValueError, match="payload changed while it was being read"):
        nr_store._stable_store_record_bytes(tmp_path, "payload", maximum_bytes=20, label="payload")


def test_shard_inventory_ignores_non_json_manifest_entries(tmp_path):
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "notes.txt").write_text("not a manifest", encoding="utf-8")

    assert tuple(iter_nr_record_shards(tmp_path, max_shard_bytes=1024)) == ()


def test_shard_inventory_propagates_unrelated_sqlite_integrity_errors(tmp_path, monkeypatch):
    write_nr_record(tmp_path, _record("sqlite-error"))
    real_connect = sqlite3.connect

    class FailingConnection:
        def __init__(self, connection):
            self.connection = connection

        def execute(self, statement, *args):
            if statement.startswith("INSERT INTO inventory"):
                raise sqlite3.IntegrityError("injected non-unique integrity error")
            return self.connection.execute(statement, *args)

        def close(self):
            self.connection.close()

    monkeypatch.setattr(
        nr_store.sqlite3,
        "connect",
        lambda *args, **kwargs: FailingConnection(real_connect(*args, **kwargs)),
    )

    with pytest.raises(sqlite3.IntegrityError, match="non-unique integrity error"):
        next(iter_nr_record_shards(tmp_path, max_shard_bytes=10_000_000))


def test_immutable_publish_cleans_up_when_temp_creation_fails(tmp_path, monkeypatch):
    def fail_temp_file(*_args, **_kwargs):
        raise OSError("injected temporary-file failure")

    monkeypatch.setattr(nr_store.tempfile, "NamedTemporaryFile", fail_temp_file)

    with pytest.raises(OSError, match="injected temporary-file failure"):
        nr_store._publish_immutable(tmp_path / "record.bin", b"payload")

    assert not list(tmp_path.rglob(".*"))


def test_record_reader_rejects_manifest_with_non_object_json(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest must be a JSON object"):
        read_nr_record(tmp_path, manifest)


def test_record_reader_rejects_payload_manifest_schema_mismatch(tmp_path):
    stored = write_nr_record(tmp_path, _record("schema-mismatch"))
    manifest_key = hashlib.sha256(stored.sample_id.encode()).hexdigest()
    manifest = tmp_path / "manifests" / f"{manifest_key}.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["schema_version"] = 4
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="payload schema version does not match"):
        read_nr_record(tmp_path, manifest)


def test_record_reader_rejects_unsupported_schema_pointer(tmp_path):
    stored = write_nr_record(tmp_path, _record("unsupported-schema"))

    with pytest.raises(ValueError, match="unsupported NR record schema version: 99"):
        read_nr_record(tmp_path, replace(stored, schema_version=99))


def test_record_reader_accepts_relative_manifest_path(tmp_path):
    stored = write_nr_record(tmp_path, _record("relative-manifest"))
    manifest_key = hashlib.sha256(stored.sample_id.encode()).hexdigest()

    loaded = read_nr_record(tmp_path, Path("manifests") / f"{manifest_key}.json")

    assert loaded.observation.sample_id == stored.sample_id


def _read_modified_record(root, stored, mutation, message: str):
    object_path = root / stored.relative_path
    with np.load(object_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    metadata = json.loads(arrays["metadata_json"].tobytes().decode("utf-8"))
    mutation(metadata, arrays)
    arrays["metadata_json"] = np.frombuffer(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        dtype=np.uint8,
    )
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    object_path.write_bytes(payload)
    forged_pointer = replace(stored, content_sha256=hashlib.sha256(payload).hexdigest())

    with pytest.raises(ValueError, match=message):
        read_nr_record(root, forged_pointer)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda metadata, _arrays: metadata.update(reference_transport_block_bits_present="yes"),
            "presence flag must be Boolean",
        ),
        (
            lambda metadata, _arrays: metadata.update(reference_transport_block_bits_present=True),
            "reference-payload flag differs from the stored arrays",
        ),
        (
            lambda metadata, _arrays: metadata.update(target_reason_codes={}),
            "reason codes for every target",
        ),
        (
            lambda metadata, _arrays: metadata.update(target_reason_codebook=[]),
            "reason-code dictionary is unsupported",
        ),
        (
            lambda _metadata, arrays: arrays.update(
                target_reason_code_0000=np.asarray([255], dtype=np.uint8)
            ),
            "invalid stored target reason codes",
        ),
    ],
)
def test_record_reader_rejects_malformed_schema_five_metadata(tmp_path, mutation, message):
    stored = write_nr_record(tmp_path, _record("bad-schema-five"))

    _read_modified_record(tmp_path, stored, mutation, message)


def test_trajectory_reader_rejects_manifest_path_outside_store(tmp_path):
    write_nr_trajectory(
        tmp_path, NRTrajectory("outside-manifest", (0.0,), (_record("trajectory-sample"),))
    )
    outside = tmp_path.parent / f"{tmp_path.name}-trajectory.json"
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="trajectory manifest must remain inside"):
        read_nr_trajectory(tmp_path, outside)
