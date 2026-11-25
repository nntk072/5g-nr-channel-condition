from __future__ import annotations

import hashlib
import io
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest
from artifacts import build_dag
from artifacts.build_dag import BuildNodeSpec, resolve_build_artifact, run_build_node


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spec(input_path, *, dependencies=None, node_id="fixture.materialize") -> BuildNodeSpec:
    return BuildNodeSpec(
        node_id=node_id,
        node_version="1.0.0",
        configuration={"seed": 19},
        configuration_schema={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": ["seed"],
            "properties": {"seed": {"type": "integer", "minimum": 0}},
            "additionalProperties": False,
        },
        source_revision="test-revision",
        source_bundle_sha256="a" * 64,
        input_files={str(input_path): _digest(input_path)},
        dependencies={} if dependencies is None else dependencies,
        outputs=("result/report.json",),
    )


def test_build_node_publishes_content_once_and_reuses_verified_result(tmp_path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"input")
    spec = _spec(source)
    store = tmp_path / "store"
    calls = 0

    def build(output):
        nonlocal calls
        calls += 1
        target = output / "result/report.json"
        target.parent.mkdir()
        target.write_text(json.dumps({"source": _digest(source)}), encoding="utf-8")

    first = run_build_node(store, spec, build)
    assert first.reused is False
    assert calls == 1
    artifact = resolve_build_artifact(store, first, "result/report.json")
    assert json.loads(artifact.read_text(encoding="utf-8")) == {"source": _digest(source)}

    repeated = run_build_node(
        store,
        spec,
        lambda _: pytest.fail("a verified build node should be reused without rebuilding"),
    )
    assert repeated.reused is True
    assert repeated.node_key == first.node_key
    assert calls == 1


def test_build_node_failure_publishes_no_manifest_and_retry_is_safe(tmp_path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"input")
    store = tmp_path / "store"
    spec = _spec(source)

    def interrupted(output):
        target = output / "result/report.json"
        target.parent.mkdir()
        target.write_text("partial", encoding="utf-8")
        raise RuntimeError("simulated stage failure")

    with pytest.raises(RuntimeError, match="simulated stage failure"):
        run_build_node(store, spec, interrupted)
    assert not (store / "nodes" / f"{spec.node_key}.json").exists()
    if (store / "objects").exists():
        assert list((store / "objects").glob("*/*")) == []

    def retry(output):
        target = output / "result/report.json"
        target.parent.mkdir()
        target.write_bytes(b"")

    result = run_build_node(
        store,
        spec,
        retry,
    )
    assert result.reused is False
    assert resolve_build_artifact(store, result, "result/report.json").read_bytes() == b""


@pytest.mark.parametrize("emit_extra", [False, True])
def test_build_node_rejects_output_roster_drift(tmp_path, emit_extra: bool) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"input")
    spec = _spec(source)

    def build(output):
        if emit_extra:
            (output / "unexpected.txt").write_text("extra", encoding="utf-8")

    with pytest.raises(ValueError, match="output roster differs"):
        run_build_node(tmp_path / "store", spec, build)


def test_build_node_rejects_changed_input_and_corrupt_cached_object(tmp_path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"input")
    spec = _spec(source)
    store = tmp_path / "store"

    def build(output):
        target = output / "result/report.json"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"output")

    result = run_build_node(
        store,
        spec,
        build,
    )
    object_path = resolve_build_artifact(store, result, "result/report.json")
    object_path.chmod(0o644)
    object_path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity verification"):
        run_build_node(store, spec, lambda _: pytest.fail("corrupt cache cannot be reused"))

    changed_spec = _spec(source)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="input changed"):
        run_build_node(store, changed_spec, lambda _: pytest.fail("stale input cannot build"))


def test_build_node_schema_rejects_invalid_configuration_before_build(tmp_path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"input")
    spec = _spec(source)
    with pytest.raises(jsonschema.ValidationError):
        BuildNodeSpec(
            node_id=spec.node_id,
            node_version=spec.node_version,
            configuration={"seed": "not-an-integer"},
            configuration_schema=spec.configuration_schema,
            source_revision=spec.source_revision,
            source_bundle_sha256=spec.source_bundle_sha256,
            input_files=spec.input_files,
            dependencies=spec.dependencies,
            outputs=spec.outputs,
        )


def test_build_node_requires_verified_dependency_manifests(tmp_path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"input")
    store = tmp_path / "store"

    def build(output):
        target = output / "result/report.json"
        target.parent.mkdir(parents=True)
        target.write_text("ok", encoding="utf-8")

    parent = run_build_node(
        store,
        _spec(source, node_id="fixture.parent"),
        build,
    )
    child = _spec(
        source,
        node_id="fixture.child",
        dependencies={"parent": parent.node_key},
    )
    result = run_build_node(
        store,
        child,
        build,
    )
    assert result.reused is False

    child_store = tmp_path / "empty-store"
    with pytest.raises(ValueError, match="dependency.*not present"):
        run_build_node(
            child_store,
            child,
            lambda _: pytest.fail("missing dependency must block node execution"),
        )


def test_file_digest_rejects_nonregular_files_and_open_identity_drift(tmp_path, monkeypatch):
    regular = tmp_path / "regular.bin"
    regular.write_bytes(b"payload")
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        build_dag._file_digest(directory)

    symlink = tmp_path / "symlink.bin"
    symlink.symlink_to(regular)
    with pytest.raises(ValueError, match="regular file"):
        build_dag._file_digest(symlink)

    hardlink = tmp_path / "hardlink.bin"
    os.link(regular, hardlink)
    with pytest.raises(ValueError, match="forbidden links"):
        build_dag._file_digest(hardlink, require_single_link=True)

    original_fstat = os.fstat

    def altered_stat(descriptor, *, mode=None, device_delta=0):
        result = original_fstat(descriptor)
        return SimpleNamespace(
            st_dev=result.st_dev + device_delta,
            st_ino=result.st_ino,
            st_mode=result.st_mode if mode is None else mode,
            st_nlink=result.st_nlink,
            st_size=result.st_size,
            st_mtime_ns=result.st_mtime_ns,
            st_ctime_ns=result.st_ctime_ns,
        )

    with monkeypatch.context() as patched:
        patched.setattr(os, "fstat", lambda descriptor: altered_stat(descriptor, mode=stat.S_IFIFO))
        with pytest.raises(ValueError, match="changed type while opening"):
            build_dag._file_digest(regular)

    with monkeypatch.context() as patched:
        patched.setattr(os, "fstat", lambda descriptor: altered_stat(descriptor, device_delta=1))
        with pytest.raises(ValueError, match="changed while opening"):
            build_dag._file_digest(regular)


def test_file_digest_detects_changes_while_hashing(tmp_path, monkeypatch):
    source = tmp_path / "source.bin"
    source.write_bytes(b"original bytes")
    original_read = os.read
    changed = False

    def mutate_after_read(descriptor, size):
        nonlocal changed
        chunk = original_read(descriptor, size)
        if not changed:
            changed = True
            source.write_bytes(b"changed bytes")
        return chunk

    monkeypatch.setattr(os, "read", mutate_after_read)
    with pytest.raises(ValueError, match="changed while hashing"):
        build_dag._file_digest(source)


def test_stable_bytes_rejects_links_oversize_and_read_drift(tmp_path, monkeypatch):
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ValueError, match="single-link regular file"):
        build_dag._stable_bytes(directory, maximum_bytes=32)

    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    hardlink = tmp_path / "hardlink.bin"
    os.link(source, hardlink)
    with pytest.raises(ValueError, match="single-link regular file"):
        build_dag._stable_bytes(hardlink, maximum_bytes=32)
    hardlink.unlink()
    with pytest.raises(ValueError, match="bounded file contract"):
        build_dag._stable_bytes(source, maximum_bytes=2)

    monkeypatch.setattr(os, "read", lambda _descriptor, _size: b"x" * 8)
    with pytest.raises(ValueError, match="changed during bounded read"):
        build_dag._stable_bytes(source, maximum_bytes=7)

    monkeypatch.undo()
    original_read = os.read
    changed = False

    def mutate_after_read(descriptor, size):
        nonlocal changed
        chunk = original_read(descriptor, size)
        if not changed:
            changed = True
            source.write_bytes(b"modified")
        return chunk

    monkeypatch.setattr(os, "read", mutate_after_read)
    with pytest.raises(ValueError, match="changed during bounded read"):
        build_dag._stable_bytes(source, maximum_bytes=32)


def test_read_node_manifest_rejects_bad_key_and_invalid_json(tmp_path):
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        build_dag._read_node_manifest(tmp_path, "invalid")

    key = "a" * 64
    manifest_path = tmp_path / "nodes" / f"{key}.json"
    manifest_path.parent.mkdir()
    manifest_path.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="not valid JSON"):
        build_dag._read_node_manifest(tmp_path, key)


def test_read_node_manifest_rejects_content_address_and_identity_drift(tmp_path, monkeypatch):
    import jsonschema

    monkeypatch.setattr(jsonschema, "validate", lambda *_args, **_kwargs: None)
    root = tmp_path / "store"
    nodes = root / "nodes"
    nodes.mkdir(parents=True)
    key = "a" * 64
    path = nodes / f"{key}.json"
    path.write_text(json.dumps({"schema": "wrong", "node_key": key}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match its content address"):
        build_dag._read_node_manifest(root, key)

    source = tmp_path / "input.bin"
    source.write_bytes(b"input")
    spec = _spec(source)
    path = nodes / f"{spec.node_key}.json"
    identity = {**spec.identity_payload, "node_version": "changed"}
    path.write_text(
        json.dumps(
            {
                "schema": build_dag._NODE_SCHEMA,
                "node_key": spec.node_key,
                "identity": identity,
                "artifacts": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="identity differs from its content address"):
        build_dag._read_node_manifest(root, spec.node_key)


@pytest.mark.parametrize(
    ("artifact", "message"),
    [
        ({"path": "missing-fields"}, "artifact entry is malformed"),
        (
            {"path": 1, "sha256": "a" * 64, "size_bytes": 0},
            "artifact path is malformed",
        ),
        (
            {"path": "result/report.json", "sha256": "bad", "size_bytes": 0},
            "artifact digest is malformed",
        ),
        (
            {"path": "result/report.json", "sha256": "a" * 64, "size_bytes": True},
            "artifact size is malformed",
        ),
    ],
)
def test_read_node_manifest_rejects_malformed_artifacts(tmp_path, monkeypatch, artifact, message):
    import jsonschema

    monkeypatch.setattr(jsonschema, "validate", lambda *_args, **_kwargs: None)
    source = tmp_path / "input.bin"
    source.write_bytes(b"input")
    spec = _spec(source)
    root = tmp_path / "store"
    nodes = root / "nodes"
    nodes.mkdir(parents=True)
    manifest = {
        "schema": build_dag._NODE_SCHEMA,
        "node_key": spec.node_key,
        "identity": spec.identity_payload,
        "artifacts": [artifact],
    }
    (nodes / f"{spec.node_key}.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        build_dag._read_node_manifest(root, spec.node_key)


def test_read_node_manifest_rejects_output_roster_drift(tmp_path, monkeypatch):
    import jsonschema

    monkeypatch.setattr(jsonschema, "validate", lambda *_args, **_kwargs: None)
    source = tmp_path / "input.bin"
    source.write_bytes(b"input")
    spec = _spec(source)
    root = tmp_path / "store"
    nodes = root / "nodes"
    nodes.mkdir(parents=True)
    payload = b"output"
    digest = hashlib.sha256(payload).hexdigest()
    object_path = root / "objects" / digest[:2] / digest
    object_path.parent.mkdir(parents=True)
    object_path.write_bytes(payload)
    manifest = {
        "schema": build_dag._NODE_SCHEMA,
        "node_key": spec.node_key,
        "identity": spec.identity_payload,
        "artifacts": [{"path": "different.json", "sha256": digest, "size_bytes": len(payload)}],
    }
    (nodes / f"{spec.node_key}.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="output roster differs"):
        build_dag._read_node_manifest(root, spec.node_key)


def test_publish_object_rejects_copy_drift_and_corrupt_collision(tmp_path, monkeypatch):
    source = tmp_path / "source.bin"
    source.write_bytes(b"original")
    original_open = Path.open

    def altered_open(path, mode="r", *args, **kwargs):
        if path == source and mode == "rb":
            return io.BytesIO(b"changed")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", altered_open)
    with pytest.raises(ValueError, match="changed while publishing"):
        build_dag._publish_object(tmp_path / "store", source)
    monkeypatch.undo()

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = tmp_path / "store" / "objects" / digest[:2] / digest
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"corrupt")
    monkeypatch.setattr(
        os, "link", lambda *_args, **_kwargs: (_ for _ in ()).throw(FileExistsError())
    )
    with pytest.raises(ValueError, match="collision or corruption"):
        build_dag._publish_object(tmp_path / "store", source)


def test_publish_manifest_rejects_conflicting_immutable_key(tmp_path, monkeypatch):
    source = tmp_path / "input.bin"
    source.write_bytes(b"input")
    spec = _spec(source)
    node_directory = tmp_path / "store" / "nodes"
    node_directory.mkdir(parents=True)
    (node_directory / f"{spec.node_key}.json").write_bytes(b"different")
    monkeypatch.setattr(
        os, "link", lambda *_args, **_kwargs: (_ for _ in ()).throw(FileExistsError())
    )

    with pytest.raises(ValueError, match="different manifest"):
        build_dag._publish_manifest(tmp_path / "store", spec, [])


def test_publish_manifest_reuses_identical_immutable_key(tmp_path, monkeypatch):
    source = tmp_path / "input.bin"
    source.write_bytes(b"input")
    spec = _spec(source)
    store = tmp_path / "store"
    first = build_dag._publish_manifest(store, spec, [])
    monkeypatch.setattr(
        os, "link", lambda *_args, **_kwargs: (_ for _ in ()).throw(FileExistsError())
    )

    assert build_dag._publish_manifest(store, spec, []) == first


def test_build_node_rejects_symlink_output_and_failed_post_publish_verification(
    tmp_path, monkeypatch
):
    source = tmp_path / "input.bin"
    source.write_bytes(b"input")
    spec = _spec(source)

    def symlink_output(output):
        (output / "result").mkdir()
        (output / "result" / "report.json").symlink_to(source)

    with pytest.raises(ValueError, match="cannot contain symlinks"):
        run_build_node(tmp_path / "symlink-store", spec, symlink_output)

    monkeypatch.setattr(build_dag, "_load_existing_node", lambda *_args, **_kwargs: None)

    def valid_output(output):
        target = output / "result/report.json"
        target.parent.mkdir()
        target.write_bytes(b"ok")

    with pytest.raises(RuntimeError, match="could not be verified"):
        run_build_node(tmp_path / "verification-store", spec, valid_output)


def test_resolve_build_artifact_rejects_corrupt_object(tmp_path):
    payload = b"bad"
    digest = hashlib.sha256(b"expected").hexdigest()
    path = tmp_path / "objects" / digest[:2] / digest
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    result = build_dag.BuildNodeResult(
        "a" * 64,
        tmp_path / "manifest.json",
        ({"path": "report.json", "sha256": digest, "size_bytes": len(payload)},),
        True,
    )

    with pytest.raises(ValueError, match="integrity verification"):
        resolve_build_artifact(tmp_path, result, "report.json")
