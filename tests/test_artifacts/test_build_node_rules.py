"""Build-node identity, manifest and publication rules for the workflow DAG."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from artifacts.build_dag import BuildNodeSpec, resolve_build_artifact, run_build_node
from tests.test_artifacts.test_build_dag import _spec

H = "b" * 64


def _base_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "node_id": "fixture.node",
        "node_version": "1.0.0",
        "configuration": {"seed": 1},
        "configuration_schema": {"type": "object"},
        "source_revision": "rev",
        "source_bundle_sha256": "a" * 64,
        "input_files": {},
        "dependencies": {},
        "outputs": ("out.bin",),
    }
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param(
            {"node_id": " padded"}, "node_id must be a canonical non-empty string", id="padded-id"
        ),
        pytest.param({"node_version": ""}, "node_version must be a canonical", id="empty-version"),
        pytest.param(
            {"source_revision": 7}, "source_revision must be a canonical", id="revision-type"
        ),
        pytest.param({"source_bundle_sha256": "A" * 64}, "lowercase SHA-256 hex", id="bundle-case"),
        pytest.param({"outputs": ()}, "non-empty and unique", id="no-outputs"),
        pytest.param({"outputs": ("a", "a")}, "non-empty and unique", id="duplicate-outputs"),
        pytest.param(
            {"outputs": ("b.bin", "a.bin")}, "sorted canonical order", id="unsorted-outputs"
        ),
        pytest.param({"outputs": ("../escape",)}, "safe relative POSIX paths", id="escape-output"),
        pytest.param({"input_files": {" padded": H}}, "must be canonical", id="input-name"),
        pytest.param(
            {"input_files": {"file": "B" * 64}}, "must be lowercase SHA-256", id="input-digest"
        ),
        pytest.param(
            {"dependencies": {"dep": "not-a-digest"}}, "must be lowercase SHA-256", id="dep-digest"
        ),
        pytest.param(
            {"configuration": {"value": float("nan")}},
            "finite canonical JSON",
            id="configuration-nan",
        ),
    ],
)
def test_build_node_spec_rejects_each_identity_defect(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        BuildNodeSpec(**_base_kwargs(**overrides))


def test_build_node_spec_accepts_the_canonical_identity() -> None:
    spec = BuildNodeSpec(**_base_kwargs(input_files={"file": H}, dependencies={"dep": H}))
    assert spec.outputs == ("out.bin",)


def test_run_build_node_requires_a_spec_and_callable_build(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="spec must be a BuildNodeSpec"):
        run_build_node(tmp_path, "not-a-spec", lambda directory: None)
    source = tmp_path / "source.bin"
    source.write_bytes(b"input")
    with pytest.raises(TypeError, match="build must be callable"):
        run_build_node(tmp_path, _spec(source), "not-callable")


def test_input_files_must_match_their_declared_digest(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"input")
    spec = dataclasses.replace(_spec(source), input_files={str(source): "0" * 64})
    with pytest.raises(ValueError):
        run_build_node(tmp_path / "store", spec, lambda directory: None)


def test_failed_builds_publish_no_node(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"input")

    def failing(directory: Path) -> None:
        raise RuntimeError("builder failed")

    with pytest.raises(RuntimeError, match="builder failed"):
        run_build_node(tmp_path / "store", _spec(source), failing)


def test_builds_that_write_the_wrong_roster_are_refused(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"input")

    def wrong(directory: Path) -> None:
        (directory / "unexpected.txt").write_text("x", encoding="utf-8")

    with pytest.raises(ValueError):
        run_build_node(tmp_path / "store", _spec(source), wrong)


def test_resolving_an_undeclared_output_is_a_key_error(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"input")

    def build(directory: Path) -> None:
        (directory / "result").mkdir(parents=True)
        (directory / "result" / "report.json").write_text(
            json.dumps({"ok": True}), encoding="utf-8"
        )

    result = run_build_node(tmp_path / "store", _spec(source), build)
    with pytest.raises(KeyError, match="no declared output"):
        resolve_build_artifact(tmp_path / "store", result, "missing.json")
    resolved = resolve_build_artifact(tmp_path / "store", result, "result/report.json")
    assert json.loads(resolved.read_bytes()) == {"ok": True}
