from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import pytest
from scripts.build_result_promotion_manifest import (
    _load_policy,
    build_result_promotion_manifest,
    main,
)


def _valid_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "command": "python run.py --seed 1",
        "generated_at_utc": "2026-11-10T01:02:03Z",
        "source_sha256": "a" * 64,
        "config_sha256": "b" * 64,
    }
    record.update(overrides)
    return record


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _policy(tmp_path: Path, requested: list[str]) -> Path:
    path = tmp_path / "promotion-policy.json"
    _write_json(
        path,
        {
            "schema": "loslab.research_result_promotion_policy.v1",
            "requested_paths": requested,
        },
    )
    return path


def test_complete_requested_result_is_promotion_ready_and_schema_valid(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _write_json(root / "run.json", _valid_record())
    policy = _policy(tmp_path, ["run.json"])

    manifest = build_result_promotion_manifest(root, policy_path=policy)

    assert manifest["status"] == "PROMOTION_READY"
    assert manifest["qualification_status"] == "NOT_QUALIFIED"
    assert manifest["promoted_file_count"] == 1
    assert manifest["rejected_file_count"] == 0
    assert manifest["requested"][0]["promotion_status"] == "eligible"
    schema = json.loads(
        Path("datasets/schema/research-result-promotion-manifest.v1.schema.json").read_text()
    )
    jsonschema.validate(manifest, schema)


def test_requested_incomplete_result_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _write_json(root / "legacy.json", {"result": 3})
    policy = _policy(tmp_path, ["legacy.json"])

    manifest = build_result_promotion_manifest(root, policy_path=policy)

    assert manifest["status"] == "PROMOTION_REJECTED"
    assert manifest["rejected_file_count"] == 1
    assert manifest["requested"][0]["promotion_status"] == "rejected"
    assert manifest["requested"][0]["issue_count"] > 0


def test_nonrequested_incomplete_result_is_retained_as_diagnostic_only(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _write_json(root / "requested.json", _valid_record())
    _write_json(root / "legacy.json", {"old": True})
    policy = _policy(tmp_path, ["requested.json"])

    manifest = build_result_promotion_manifest(root, policy_path=policy)

    assert manifest["status"] == "PROMOTION_READY"
    assert [row["path"] for row in manifest["diagnostic_legacy"]] == ["legacy.json"]
    assert manifest["diagnostic_legacy"][0]["retention_status"] == "diagnostic_only"


def test_missing_requested_path_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir()
    policy = _policy(tmp_path, ["missing.json"])

    with pytest.raises(ValueError, match="references absent results"):
        build_result_promotion_manifest(root, policy_path=policy)


@pytest.mark.parametrize(
    "unsafe",
    ["../run.json", "/run.json", "./run.json", "nested\\run.json", "run.txt"],
)
def test_policy_rejects_unsafe_or_noncanonical_paths(tmp_path: Path, unsafe: str) -> None:
    policy = _policy(tmp_path, [unsafe])

    with pytest.raises(ValueError, match="canonical relative POSIX paths|JSON or JSON.GZ"):
        _load_policy(policy)


def test_policy_roster_must_be_sorted_and_unique(tmp_path: Path) -> None:
    unsorted_policy = _policy(tmp_path, ["z.json", "a.json"])
    with pytest.raises(ValueError, match="sorted and unique"):
        _load_policy(unsorted_policy)

    duplicate = tmp_path / "duplicate-policy.json"
    _write_json(
        duplicate,
        {
            "schema": "loslab.research_result_promotion_policy.v1",
            "requested_paths": ["a.json", "a.json"],
        },
    )
    with pytest.raises(jsonschema.ValidationError):
        _load_policy(duplicate)


def test_referenced_artifact_hash_mismatch_rejects_requested_result(tmp_path: Path) -> None:
    root = tmp_path / "results"
    artifact = root / "capture.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"actual bytes")
    _write_json(
        root / "run.json",
        _valid_record(inputs={"dataset_path": "capture.bin", "dataset_sha256": "0" * 64}),
    )
    policy = _policy(tmp_path, ["run.json"])

    manifest = build_result_promotion_manifest(root, policy_path=policy)

    assert manifest["status"] == "PROMOTION_REJECTED"
    codes = {issue["code"] for issue in manifest["requested"][0]["issues"]}
    assert "reference_hash_mismatch" in codes
    assert (
        manifest["requested"][0]["sha256"]
        == hashlib.sha256((root / "run.json").read_bytes()).hexdigest()
    )


def test_cli_refuses_to_overwrite_manifest(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _write_json(root / "run.json", _valid_record())
    policy = _policy(tmp_path, ["run.json"])
    output = tmp_path / "manifest.json"

    assert main([str(root), "--policy", str(policy), "--output", str(output)]) == 0
    first = output.read_bytes()
    with pytest.raises(SystemExit) as exc:
        main([str(root), "--policy", str(policy), "--output", str(output)])
    assert exc.value.code == 2
    assert output.read_bytes() == first
