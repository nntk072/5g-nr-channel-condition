"""Contract tests for the fail-closed D1 approval manifest helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from integrations.sionna.d1_generation_gate import (
    _APPROVAL_SCHEMA,
    _PUBLIC_RELEASE_GATES,
    _REQUIRED_GATES,
    _hash_relative_file,
    _parse_json_object,
    _read_stable_file,
    _validate_approval,
    authorize_d1_generation,
    canonical_sha256,
    require_d1_authorization,
    require_pilot_destination,
    rt_scene_asset_identity,
)


def _manifest(root: Path, request: dict[str, Any]) -> dict[str, Any]:
    plan = root / "plan.json"
    report = root / "report.json"
    plan.write_text('{"plan":"frozen"}\n', encoding="utf-8")
    report.write_text('{"report":"complete"}\n', encoding="utf-8")
    return {
        "schema": _APPROVAL_SCHEMA,
        "decision": "approved",
        "approval_id": "approval-test-001",
        "approved_by": "reviewer",
        "approved_at": "2026-10-23T12:00:00Z",
        "scope_request_sha256": canonical_sha256(request),
        "gate_statuses": dict.fromkeys(_REQUIRED_GATES, "passed"),
        "release_scope": "internal_research",
        "release_statuses": {},
        "qualification_artifacts": [
            {
                "role": "qualification_plan",
                "path": plan.name,
                "sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
            },
            {
                "role": "qualification_report",
                "path": report.name,
                "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            },
        ],
    }


def test_approval_manifest_accepts_exact_internal_research_contract(tmp_path: Path) -> None:
    request = {"operation": "rt_data", "seed": 7}
    manifest = _manifest(tmp_path, request)

    _validate_approval(manifest, tmp_path / "approval.json", canonical_sha256(request))


@pytest.mark.parametrize(
    ("change", "error", "message"),
    [
        (lambda m: m.update(extra=True), ValueError, "fields do not match"),
        (lambda m: m.update(schema="wrong"), PermissionError, "schema or decision"),
        (lambda m: m.update(decision="denied"), PermissionError, "schema or decision"),
        (lambda m: m.update(approval_id="  "), ValueError, "approval_id"),
        (lambda m: m.update(approved_by=""), ValueError, "approved_by"),
        (lambda m: m.update(approved_at=1), ValueError, "approved_at"),
        (lambda m: m.update(scope_request_sha256="0" * 64), PermissionError, "exact D1"),
        (lambda m: m.update(gate_statuses={}), PermissionError, "passed"),
        (lambda m: m.update(release_scope="unknown"), ValueError, "release_scope"),
        (lambda m: m.update(release_statuses={"claimed": "passed"}), ValueError, "must not claim"),
        (lambda m: m.update(qualification_artifacts=[]), ValueError, "plan and report"),
    ],
)
def test_approval_manifest_rejects_invalid_top_level_contract(
    tmp_path: Path, change, error: type[Exception], message: str
) -> None:
    request = {"operation": "rt_data"}
    manifest = _manifest(tmp_path, request)
    change(manifest)

    with pytest.raises(error, match=message):
        _validate_approval(manifest, tmp_path / "approval.json", canonical_sha256(request))


def test_public_release_manifest_requires_all_release_checks(tmp_path: Path) -> None:
    request = {"operation": "rt_data"}
    manifest = _manifest(tmp_path, request)
    manifest["release_scope"] = "public_release"
    manifest["release_statuses"] = dict.fromkeys(_PUBLIC_RELEASE_GATES, "passed")

    _validate_approval(manifest, tmp_path / "approval.json", canonical_sha256(request))
    manifest["release_statuses"].pop(next(iter(_PUBLIC_RELEASE_GATES)))

    with pytest.raises(PermissionError, match="public release approval"):
        _validate_approval(manifest, tmp_path / "approval.json", canonical_sha256(request))


@pytest.mark.parametrize(
    ("artifact_change", "error", "message"),
    [
        (lambda a: a.update(extra=True), ValueError, "exactly role, path, and sha256"),
        (lambda a: a.update(role="other"), ValueError, "one plan and one report"),
        (lambda a: a.update(path=""), ValueError, "non-empty relative path"),
        (lambda a: a.update(sha256="A" * 64), ValueError, "lowercase hexadecimal"),
        (lambda a: a.update(sha256="0" * 64), PermissionError, "hash mismatch"),
        (lambda a: a.update(path="../outside"), ValueError, "stay beneath"),
    ],
)
def test_approval_artifact_entries_are_validated_and_pinned(
    tmp_path: Path, artifact_change, error: type[Exception], message: str
) -> None:
    request = {"operation": "rt_data"}
    manifest = _manifest(tmp_path, request)
    artifact_change(manifest["qualification_artifacts"][0])

    with pytest.raises(error, match=message):
        _validate_approval(manifest, tmp_path / "approval.json", canonical_sha256(request))


def test_approval_artifact_path_rejects_symlink_components(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "evidence.json").write_text("evidence", encoding="utf-8")
    (tmp_path / "linked").symlink_to(actual, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink or non-directory"):
        _hash_relative_file(tmp_path, "linked/evidence.json")


def test_stable_approval_reader_bounds_and_checks_regular_files(tmp_path: Path) -> None:
    regular = tmp_path / "approval.json"
    regular.write_bytes(b"{}")

    assert _read_stable_file(regular, max_bytes=2) == b"{}"
    with pytest.raises(ValueError, match="too large"):
        _read_stable_file(regular, max_bytes=1)
    with pytest.raises(ValueError, match="regular file"):
        _read_stable_file(tmp_path, max_bytes=100)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"\xff", "valid UTF-8 JSON"),
        (b"{", "valid UTF-8 JSON"),
        (b"[]", "root must be a JSON object"),
        (b'{"a":1,"a":2}', "duplicate key"),
    ],
)
def test_manifest_parser_rejects_ambiguous_or_non_object_json(raw: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _parse_json_object(raw, label="test manifest")


def test_canonical_hash_rejects_non_finite_json_values() -> None:
    with pytest.raises(ValueError, match="finite JSON-compatible"):
        canonical_sha256({"measurement": float("nan")})


def _pilot_request() -> dict[str, Any]:
    return {
        "operation": "rt_data",
        "config": {
            "per_class": 1,
            "max_attempts": 2,
            "samples_per_src": 10,
            "max_depth": 1,
            "total_work_units": 20,
        },
        "destinations": ["/tmp/d1-gate-test"],
    }


def test_authorization_record_binds_request_operation_and_destinations() -> None:
    request = _pilot_request()
    authorization = authorize_d1_generation(request, pilot=True)

    assert authorization.mode == "pilot"
    assert authorization.request_sha256 == canonical_sha256(request)
    assert authorization.approval_manifest_sha256 is None
    assert authorization.destinations == ("/tmp/d1-gate-test",)
    assert authorization.operation == "rt_data"
    assert (
        require_d1_authorization(authorization, request, expected_operation="rt_data")
        is authorization
    )
    with pytest.raises(PermissionError, match="exact request"):
        require_d1_authorization(authorization, {**request, "seed": 2})
    with pytest.raises(PermissionError, match="cannot authorize"):
        require_d1_authorization(authorization, request, expected_operation="rt_cache")


def test_pilot_publisher_must_use_an_authorized_scratch_path() -> None:
    authorization = authorize_d1_generation(_pilot_request(), pilot=True)

    assert require_pilot_destination(
        authorization, "/tmp/d1-gate-test", publisher="rt_dataset"
    ) == Path("/tmp/d1-gate-test")
    with pytest.raises(PermissionError, match="not bound"):
        require_pilot_destination(authorization, "/tmp/other", publisher="rt_dataset")
    with pytest.raises(PermissionError, match="not for this publication"):
        require_pilot_destination(authorization, "/tmp/d1-gate-test", publisher="trajectory_store")
    with pytest.raises(PermissionError, match="fixed scratch root"):
        require_pilot_destination(authorization, "/var/tmp/outside", publisher="rt_dataset")


def test_scene_identity_requires_complete_local_asset_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene_path = tmp_path / "scene.xml"
    scene_path.write_text("<scene />", encoding="utf-8")
    sionna = ModuleType("sionna")
    rt = ModuleType("sionna.rt")
    rt.scene = ModuleType("sionna.rt.scene")
    rt.scene.fixture = str(scene_path)
    sionna.rt = rt
    monkeypatch.setitem(__import__("sys").modules, "sionna", sionna)
    monkeypatch.setitem(__import__("sys").modules, "sionna.rt", rt)

    scene_assets = ModuleType("integrations.sionna.scene_assets")
    scene_assets.scene_asset_manifest = lambda path: {
        "complete": True,
        "asset_set_sha256": "a" * 64,
        "asset_count": 3,
    }
    monkeypatch.setitem(__import__("sys").modules, "integrations.sionna.scene_assets", scene_assets)
    provenance = ModuleType("artifacts.provenance")
    provenance.versions = lambda: {"sionna_rt": "fixture"}
    monkeypatch.setitem(__import__("sys").modules, "artifacts.provenance", provenance)

    assert rt_scene_asset_identity("fixture") == {
        "scene_reference": str(scene_path),
        "asset_set_sha256": "a" * 64,
        "asset_count": 3,
        "runtime_versions": {"sionna_rt": "fixture"},
    }


def test_scene_identity_fails_closed_for_missing_runtime_asset_or_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    original_import = builtins.__import__

    def no_rt_runtime(name, *args, **kwargs):
        if name == "sionna.rt":
            raise ImportError("fixture without RT")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_rt_runtime)
    with pytest.raises(RuntimeError, match="configured Sionna RT runtime"):
        rt_scene_asset_identity("fixture")

    sionna = ModuleType("sionna")
    rt = ModuleType("sionna.rt")
    rt.scene = ModuleType("sionna.rt.scene")
    rt.scene.missing = str(tmp_path / "missing.xml")
    sionna.rt = rt
    monkeypatch.setitem(__import__("sys").modules, "sionna", sionna)
    monkeypatch.setitem(__import__("sys").modules, "sionna.rt", rt)
    monkeypatch.setattr(builtins, "__import__", original_import)
    with pytest.raises(ValueError, match="locally resolvable scene asset closure"):
        rt_scene_asset_identity("missing")

    scene_path = tmp_path / "incomplete.xml"
    scene_path.write_text("<scene />", encoding="utf-8")
    rt.scene.incomplete = str(scene_path)
    scene_assets = ModuleType("integrations.sionna.scene_assets")

    def incomplete_manifest(path: Path) -> dict[str, Any]:
        return {"complete": False}

    scene_assets.scene_asset_manifest = incomplete_manifest
    monkeypatch.setitem(__import__("sys").modules, "integrations.sionna.scene_assets", scene_assets)
    with pytest.raises(ValueError, match="complete scene asset provenance"):
        rt_scene_asset_identity("incomplete")


def test_authorization_helpers_reject_forged_seals_and_mutated_destinations() -> None:
    import integrations.sionna.d1_generation_gate as gate

    with pytest.raises(PermissionError, match="only be issued by the gate"):
        gate.D1GenerationAuthorization("pilot", "0" * 64, None, (), "rt_data", _seal=object())

    request = _pilot_request()
    authorization = authorize_d1_generation(request, pilot=True)
    authorization._seal = object()
    with pytest.raises(PermissionError, match="not issued by the gate"):
        require_d1_authorization(authorization, request)

    authorization = authorize_d1_generation(request, pilot=True)
    authorization._destinations = ("/tmp/altered",)
    with pytest.raises(PermissionError, match="destinations do not match"):
        require_d1_authorization(authorization, request)


@pytest.mark.parametrize(
    ("generation_request", "message"),
    [
        (
            {"operation": "rt_nr_trajectory", "config": "invalid", "destinations": ["/tmp/x"]},
            "full config",
        ),
        (
            {
                "operation": "rt_nr_trajectory",
                "config": {
                    "num_steps": 100,
                    "samples_per_src": 3_000,
                    "max_depth": 3,
                    "max_num_paths_per_src": 1_000,
                    "total_work_units": 300_000,
                },
                "destinations": ["/tmp/x"],
            },
            "total_work_units must be in",
        ),
        (
            {
                "operation": "rt_nr_trajectory",
                "config": {
                    "num_steps": 1,
                    "samples_per_src": 1,
                    "max_depth": 0,
                    "max_num_paths_per_src": 0,
                    "total_work_units": 1,
                },
                "destinations": ["/tmp/x"],
            },
            "max_num_paths_per_src",
        ),
    ],
)
def test_trajectory_pilot_work_and_path_bounds_are_enforced(
    generation_request: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        authorize_d1_generation(generation_request, pilot=True)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"sample_count": 0}, "sample_count"),
        ({"sample_count": 2, "total_work_units": 1}, "total_work_units must equal"),
        ({"input_artifact_count": 21}, "input_artifact_count"),
        ({"total_work_units": True}, "total_work_units must be an integer"),
    ],
)
def test_dmrs_pilot_request_bounds_are_enforced(change: dict[str, Any], message: str) -> None:
    request = {
        "operation": "rt_dmrs_observations",
        "sample_count": 1,
        "input_artifact_count": 0,
        "total_work_units": 1,
        "destinations": ["/tmp/dmrs"],
    }
    request.update(change)

    with pytest.raises(ValueError, match=message):
        authorize_d1_generation(request, pilot=True)


def test_json_helpers_accept_valid_objects_and_nested_artifact_paths(tmp_path: Path) -> None:
    assert _parse_json_object(b'{"decision":"approved"}', label="manifest") == {
        "decision": "approved"
    }
    evidence = tmp_path / "nested"
    evidence.mkdir()
    target = evidence / "evidence.json"
    target.write_text("frozen", encoding="utf-8")
    assert (
        _hash_relative_file(tmp_path, "nested/evidence.json")
        == hashlib.sha256(b"frozen").hexdigest()
    )
