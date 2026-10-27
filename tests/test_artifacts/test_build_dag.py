from __future__ import annotations

import hashlib
import json

import jsonschema
import pytest
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
