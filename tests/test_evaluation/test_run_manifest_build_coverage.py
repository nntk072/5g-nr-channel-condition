from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import evaluation.run_manifest.build as builder
import pytest


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    package = tmp_path / "package"
    package.mkdir()
    input_path = package / "input.json"
    input_path.write_text("{}", encoding="utf-8")
    artifact_path = package / "artifact.json"
    artifact_path.write_text("{}", encoding="utf-8")
    source_root = Path(__file__).resolve().parents[2]
    return input_path, artifact_path, source_root


def _stub_rooted_identity(monkeypatch: pytest.MonkeyPatch, *, missing: Path | None = None) -> None:
    def identify(_fd: int, package: Path, path: Path, **_kwargs: object):
        relative = path.relative_to(package).as_posix()
        payload = None if missing == path else b"{}"
        return relative, payload, 2, "a" * 64

    monkeypatch.setattr(builder, "_rooted_file_identity", identify)
    monkeypatch.setattr(
        builder, "_read_inventory_file", lambda *_args, **_kwargs: (None, 1, "b" * 64)
    )
    monkeypatch.setattr(builder, "_source_bundle_sha256", lambda _sources: "bundle")


def test_builder_rejects_runtime_inventory_without_captured_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path, artifact_path, source_root = _inputs(tmp_path)
    inventory = input_path.parent / "inventory.json"
    inventory.write_text("{}", encoding="utf-8")
    _stub_rooted_identity(monkeypatch, missing=inventory)

    with pytest.raises(ValueError, match="runtime inventory bytes were not captured"):
        builder.build_joint_evidence_run_manifest(
            "fixture-run",
            input_path,
            artifact_path,
            source_root=source_root,
            runtime_inventory_path=inventory,
        )


def test_builder_rejects_input_without_captured_bytes_for_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path, artifact_path, source_root = _inputs(tmp_path)
    _stub_rooted_identity(monkeypatch, missing=input_path)

    with pytest.raises(ValueError, match="joint evaluator input was not captured"):
        builder.build_joint_evidence_run_manifest(
            "fixture-run",
            input_path,
            artifact_path,
            source_root=source_root,
            qualification_contract={},
        )


def test_builder_rejects_same_input_and_artifact_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path, _artifact_path, source_root = _inputs(tmp_path)
    _stub_rooted_identity(monkeypatch)

    with pytest.raises(ValueError, match="input and artifact paths must be distinct"):
        builder.build_joint_evidence_run_manifest(
            "fixture-run", input_path, input_path, source_root=source_root
        )


def test_builder_rejects_source_roster_over_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path, artifact_path, source_root = _inputs(tmp_path)
    _stub_rooted_identity(monkeypatch)
    monkeypatch.setattr(builder, "_MAX_SOURCE_BUNDLE_BYTES", 0)

    with pytest.raises(ValueError, match="source roster exceeds its total byte limit"):
        builder.build_joint_evidence_run_manifest(
            "fixture-run", input_path, artifact_path, source_root=source_root
        )


def test_builder_rejects_non_dict_qualification_contract_after_source_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path, artifact_path, source_root = _inputs(tmp_path)
    _stub_rooted_identity(monkeypatch)
    monkeypatch.setattr(
        builder, "_validate_joint_evaluator_input_schema", lambda *_args, **_kwargs: "v1"
    )
    monkeypatch.setattr(builder, "_verify_qualification_contract_bindings", lambda *_args: None)

    class ContractLike:
        def __getitem__(self, key: str) -> dict[str, str]:
            assert key == "source_identity"
            return {"source_bundle_sha256": "bundle"}

    with pytest.raises(TypeError, match="qualification_contract must be an object"):
        builder.build_joint_evidence_run_manifest(
            "fixture-run",
            input_path,
            artifact_path,
            source_root=source_root,
            qualification_contract=ContractLike(),  # type: ignore[arg-type]
        )


def test_builder_rejects_runtime_inventory_that_fails_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import jsonschema

    input_path, artifact_path, source_root = _inputs(tmp_path)
    inventory = input_path.parent / "inventory.json"
    inventory.write_text("{}", encoding="utf-8")
    _stub_rooted_identity(monkeypatch)

    class InvalidInventoryValidator:
        def __init__(self, _schema: object) -> None:
            pass

        def iter_errors(self, _payload: object) -> list[SimpleNamespace]:
            return [SimpleNamespace(message="missing required identity")]

    monkeypatch.setattr(jsonschema, "Draft202012Validator", InvalidInventoryValidator)
    with pytest.raises(
        ValueError, match="runtime inventory violates v3 schema: missing required identity"
    ):
        builder.build_joint_evidence_run_manifest(
            "fixture-run",
            input_path,
            artifact_path,
            source_root=source_root,
            runtime_inventory_path=inventory,
        )


def test_builder_rejects_runtime_inventory_path_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import jsonschema

    input_path, artifact_path, source_root = _inputs(tmp_path)

    class ValidInventoryValidator:
        def __init__(self, _schema: object) -> None:
            pass

        def iter_errors(self, _payload: object) -> list[object]:
            return []

    monkeypatch.setattr(jsonschema, "Draft202012Validator", ValidInventoryValidator)
    with pytest.raises(
        ValueError, match="runtime inventory, input, and artifact paths must be distinct"
    ):
        builder.build_joint_evidence_run_manifest(
            "fixture-run",
            input_path,
            artifact_path,
            source_root=source_root,
            runtime_inventory_path=input_path,
        )
