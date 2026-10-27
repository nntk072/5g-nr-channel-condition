from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "audit_research_result_provenance.py"
_SPEC = importlib.util.spec_from_file_location("audit_research_result_provenance", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
audit_tree = _MODULE.audit_tree
main = _MODULE.main


def _valid_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "command": "python run.py --seed 1",
        "generated_at_utc": "2026-10-23T12:34:56Z",
        "source_sha256": "a" * 64,
        "config_sha256": "b" * 64,
    }
    record.update(overrides)
    return record


def _write_json(path: Path, record: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")


def test_complete_record_and_existing_relative_references_are_clean(tmp_path: Path) -> None:
    root = tmp_path / "results"
    source = root / "raw" / "capture.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fixture")
    _write_json(
        root / "run.json",
        _valid_record(inputs=["raw/capture.bin"], outputs=[{"path": "run.json"}]),
    )

    report = audit_tree(root)

    assert report["scanned_files"] == 1
    assert report["valid_json_files"] == 1
    assert report["files_with_issues"] == 0


def test_legacy_record_reports_missing_fields_without_failing_non_strict_cli(
    tmp_path: Path, capsys: object
) -> None:
    root = tmp_path / "results"
    _write_json(root / "legacy-20990101.json", {"result": 3})

    assert main([str(root)]) == 0
    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    codes = {issue["code"] for issue in output["files"][0]["issues"]}
    assert codes == {
        "missing_command",
        "missing_utc_timestamp",
        "missing_source_hash",
        "missing_config_hash",
    }


def test_strict_mode_exits_nonzero_for_incomplete_records(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _write_json(root / "incomplete.json", {})

    assert main([str(root), "--strict"]) == 1


def test_invalid_fields_and_unresolved_input_output_artifacts_are_reported(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _write_json(
        root / "bad.json",
        _valid_record(
            command=[],
            generated_at_utc="2026-10-23T12:34:56",
            source_sha256="xyz",
            config_sha256="invalid",
            inputs=["missing-input.npz"],
            outputs=[{"file": "missing-output.csv"}],
        ),
    )

    issues = audit_tree(root)["files"][0]["issues"]
    codes = {issue["code"] for issue in issues}
    assert codes == {
        "invalid_command",
        "invalid_utc_timestamp",
        "invalid_source_hash",
        "invalid_config_hash",
        "unresolved_input_reference",
        "unresolved_output_reference",
    }


def test_malformed_json_and_gzip_are_counted_and_inputs_unchanged(tmp_path: Path) -> None:
    root = tmp_path / "results"
    bad_json = root / "broken.json"
    bad_gzip = root / "broken.json.gz"
    bad_json.parent.mkdir(parents=True)
    bad_json.write_bytes(b"{bad")
    bad_gzip.write_bytes(b"not gzip")
    before = (bad_json.read_bytes(), bad_gzip.read_bytes())

    report = audit_tree(root)

    assert report["malformed_files"] == 2
    assert report["issue_counts"] == {"malformed_gzip": 1, "malformed_json": 1}
    assert (bad_json.read_bytes(), bad_gzip.read_bytes()) == before


def test_valid_gzip_is_scanned_and_remote_reference_is_not_fetched(tmp_path: Path) -> None:
    root = tmp_path / "results"
    path = root / "compressed.json.gz"
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(_valid_record(inputs=["https://example.invalid/data.npz"]), handle)

    report = audit_tree(root)

    assert report["valid_json_files"] == 1
    assert report["files_with_issues"] == 0


def test_repository_relative_declared_paths_resolve_from_checkout_root(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    root = repo / "docs" / "research" / "results"
    artifact = repo / "inputs" / "capture.bin"
    manifest = repo / "inputs" / "manifest.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"fixture")
    manifest.write_text("{}", encoding="utf-8")
    _write_json(
        root / "run.json",
        _valid_record(
            inputs={
                "dataset_path": "inputs/capture.bin",
                "dataset_sha256": hashlib.sha256(b"fixture").hexdigest(),
                "source_manifest_path": "inputs/manifest.json",
            }
        ),
    )

    report = audit_tree(root)

    assert report["files_with_issues"] == 0


def test_all_declared_path_fields_are_checked(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir()
    (root / "existing.bin").write_bytes(b"fixture")
    _write_json(
        root / "run.json",
        _valid_record(
            inputs={"dataset_path": "existing.bin", "source_manifest_path": "missing.json"}
        ),
    )

    issues = audit_tree(root)["files"][0]["issues"]

    assert [issue["code"] for issue in issues] == ["unresolved_input_reference"]
    assert "missing.json" in issues[0]["message"]


def test_declared_artifact_digest_is_compared_with_file_bytes(tmp_path: Path) -> None:
    root = tmp_path / "results"
    artifact = root / "capture.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"captured bytes")
    matching_digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    _write_json(
        root / "run.json",
        _valid_record(inputs={"dataset_path": "capture.bin", "dataset_sha256": matching_digest}),
    )

    report = audit_tree(root)

    assert report["files_with_issues"] == 0


def test_mismatched_and_malformed_artifact_digests_are_reported(tmp_path: Path) -> None:
    root = tmp_path / "results"
    artifact = root / "capture.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"captured bytes")
    _write_json(
        root / "run.json",
        _valid_record(
            inputs=[
                {"path": "capture.bin", "sha256": "a" * 64},
                {"path": "capture.bin", "sha256": "not-a-digest"},
                {"path": "capture.bin", "sha256": None},
            ]
        ),
    )

    issues = audit_tree(root)["files"][0]["issues"]

    assert [issue["code"] for issue in issues] == [
        "reference_hash_mismatch",
        "invalid_reference_hash",
        "invalid_reference_hash",
    ]


def test_descriptive_input_text_and_hashes_are_not_treated_as_file_paths(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _write_json(
        root / "run.json",
        _valid_record(
            inputs="target-domain current summary and summary nearest lookback_s in the past"
        ),
    )

    issues = audit_tree(root)["files"][0]["issues"]

    assert not any(issue["code"].startswith("unresolved_") for issue in issues)


def test_explicit_artifact_lists_check_bare_filenames(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _write_json(
        root / "run.json",
        _valid_record(input_files=["capture"], output_artifacts=["derived-result"]),
    )

    issues = audit_tree(root)["files"][0]["issues"]

    assert {issue["code"] for issue in issues} == {
        "unresolved_input_reference",
        "unresolved_output_reference",
    }


def test_local_file_uri_decodes_percent_escaped_filename(tmp_path: Path) -> None:
    root = tmp_path / "results"
    artifact = root / "capture 01.npz"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"fixture")
    _write_json(
        root / "run.json",
        _valid_record(input_artifacts=[{"uri": artifact.as_uri()}]),
    )

    assert audit_tree(root)["files_with_issues"] == 0


def test_legacy_provenance_aliases_are_recognized(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _write_json(
        root / "legacy-schema.json",
        {
            "argv": ["python", "run.py"],
            "created_utc": "2026-10-23T12:34:56+00:00",
            "package_source_sha256": "c" * 64,
            "configuration_sha256": "d" * 64,
        },
    )

    assert audit_tree(root)["files"][0]["issues"] == []


def test_per_source_hash_map_is_valid_provenance(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _write_json(
        root / "multi-source.json",
        _valid_record(
            source_sha256={
                "experiments/runner.py": "a" * 64,
                "integrations/adapter.py": "b" * 64,
            }
        ),
    )

    assert audit_tree(root)["files"][0]["issues"] == []


def test_malformed_per_source_hash_map_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _write_json(
        root / "bad-source-map.json",
        _valid_record(source_sha256={"runner.py": "not-a-hash"}),
    )

    codes = {issue["code"] for issue in audit_tree(root)["files"][0]["issues"]}

    assert codes == {"invalid_source_hash"}


def test_duplicate_source_hash_key_is_reported_instead_of_using_last_value(
    tmp_path: Path,
) -> None:
    root = tmp_path / "results"
    root.mkdir(parents=True)
    (root / "duplicate.json").write_text(
        '{"command":"python run.py","generated_at_utc":"2026-10-23T12:34:56Z",'
        '"source_sha256":"invalid","source_sha256":"'
        + "a" * 64
        + '","config_sha256":"'
        + "b" * 64
        + '"}',
        encoding="utf-8",
    )

    report = audit_tree(root)

    assert report["issue_counts"] == {"duplicate_json_key": 1}


def test_malformed_declared_reference_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _write_json(root / "bad-reference.json", _valid_record(inputs=[{"path": 12}]))

    issues = audit_tree(root)["files"][0]["issues"]

    assert {issue["code"] for issue in issues} == {"invalid_reference"}


def test_audit_does_not_infer_timestamp_from_filename(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _write_json(root / "result-2026-10-23T12-34-56Z.json", {})

    codes = {issue["code"] for issue in audit_tree(root)["files"][0]["issues"]}

    assert "missing_utc_timestamp" in codes
