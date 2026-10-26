from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest
from scripts import audit_seed_roster_collisions as seed_audit
from scripts.audit_seed_roster_collisions import audit_seed_roster_collisions, main


def test_seed_audit_finds_exact_seed_fields_in_json_and_gzip_json(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir()
    json_path = root / "result.json"
    json_path.write_text(
        json.dumps(
            {
                "config": {"channel_seeds": [100, 101], "bootstrap_seed": 202},
                "unrelated_measurement": 303,
            }
        ),
        encoding="utf-8",
    )
    compressed_path = root / "old-result.json.gz"
    compressed_path.write_bytes(gzip.compress(b'{"seed_roster":[202,404]}'))

    report = audit_seed_roster_collisions([root], [101, 202])

    assert report["status"] == "COLLISION_FOUND"
    assert report["global_exclusivity_established"] is False
    assert report["scanned_file_count"] == 2
    assert report["roots"] == [{"path": str(root.absolute()), "eligible_json_file_count": 2}]
    assert sorted(row["seed"] for match in report["collisions"] for row in match["matches"]) == [
        101,
        202,
        202,
    ]
    assert all(len(item["sha256"]) == 64 for item in report["scanned_files"])


def test_seed_audit_report_matches_its_versioned_schema(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    root = tmp_path / "results"
    root.mkdir()
    (root / "result.json").write_text('{"seed":101}', encoding="utf-8")
    report = audit_seed_roster_collisions([root], [101])
    schema_path = (
        Path(__file__).parents[2]
        / "datasets"
        / "schema"
        / "seed-roster-collision-audit.v2.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)

    jsonschema.Draft202012Validator(schema).validate(report)


def test_seed_audit_ignores_unstructured_numbers_strings_and_booleans(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir()
    (root / "report.json").write_text(
        '{"measurement":101,"empty_measurement":[],"label":"101",'
        '"channel_seed_enabled":true,"seed_offset":101,'
        '"transport_blocks_are_nested_within_channel_seed":true,'
        '"same_channel_truth_by_seed":true,"split_seed":"policy-id",'
        '"mean_test_auroc_over_7_seeds":0.5,"std_test_auroc_across_seeds":0.12}',
        encoding="utf-8",
    )

    report = audit_seed_roster_collisions([root], [101])

    assert report["status"] == "CLEAR_WITHIN_SCANNED_JSON_CORPUS"
    assert report["collisions"] == []
    assert report["malformed_seed_fields"] == []


def test_seed_audit_reads_seed_values_inside_per_seed_records(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir()
    (root / "report.json").write_text(
        '{"per_seed":[{"seed":101,"empty_paths":[]},{"seed":102,"empty_paths":[]}]}',
        encoding="utf-8",
    )

    report = audit_seed_roster_collisions([root], [101, 102])

    assert report["status"] == "COLLISION_FOUND"
    assert {
        match["seed"] for collision in report["collisions"] for match in collision["matches"]
    } == {
        101,
        102,
    }
    assert report["malformed_seed_fields"] == []


def test_seed_audit_reads_numeric_mapping_keys_and_allows_seed_identity_labels(
    tmp_path: Path,
) -> None:
    root = tmp_path / "results"
    root.mkdir()
    (root / "report.json").write_text(
        '{"seed":"rt-nr-scene-family-v1","split_hashes_by_seed":{"101":"a","102":"b"}}',
        encoding="utf-8",
    )

    report = audit_seed_roster_collisions([root], [101, 103])

    assert report["status"] == "COLLISION_FOUND"
    assert report["collisions"][0]["matches"] == [
        {"json_path": "split_hashes_by_seed::101", "seed": 101, "encoded_type": "key"}
    ]
    assert report["malformed_seed_fields"] == []


def test_seed_audit_does_not_treat_string_encoded_seed_as_clear(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir()
    (root / "report.json").write_text('{"channel_seed":"101"}', encoding="utf-8")

    report = audit_seed_roster_collisions([root], [101])

    assert report["status"] == "COLLISION_FOUND"
    assert report["collisions"][0]["matches"][0] == {
        "json_path": "channel_seed",
        "seed": 101,
        "encoded_type": "string",
    }


def test_seed_audit_checks_inclusive_ranges_and_seeded_result_keys(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir()
    (root / "report.json").write_text(
        '{"repetition_seed_range":[100,105],"seed1021_los_empty_calls_by_depth":[]}',
        encoding="utf-8",
    )

    report = audit_seed_roster_collisions([root], [101, 1021])
    matches = [row for collision in report["collisions"] for row in collision["matches"]]

    assert {(row["seed"], row["encoded_type"]) for row in matches} == {
        (101, "range"),
        (1021, "key"),
    }


def test_seed_audit_marks_ambiguous_ranges_unresolved(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir()
    (root / "report.json").write_text('{"repetition_seed_range":[5,"10"]}', encoding="utf-8")

    report = audit_seed_roster_collisions([root], [8])

    assert report["status"] == "MALFORMED_SEED_FIELDS"
    assert report["malformed_seed_fields"][0]["observed_type"] == "range"


def test_seed_audit_marks_malformed_seed_fields_unresolved(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir()
    (root / "report.json").write_text('{"channel_seed":true}', encoding="utf-8")

    report = audit_seed_roster_collisions([root], [101])

    assert report["status"] == "MALFORMED_SEED_FIELDS"
    assert report["malformed_seed_fields"] == [
        {
            "file_path": str(root / "report.json"),
            "json_path": "channel_seed",
            "observed_type": "boolean",
        }
    ]


@pytest.mark.parametrize(
    ("field_value", "observed_type"),
    [({}, "object"), ([], "empty_list")],
)
def test_seed_audit_marks_incomplete_structured_seed_values_unresolved(
    tmp_path: Path, field_value: Any, observed_type: str
) -> None:
    root = tmp_path / "results"
    root.mkdir()
    (root / "report.json").write_text(json.dumps({"channel_seeds": field_value}), encoding="utf-8")

    report = audit_seed_roster_collisions([root], [101])

    assert report["status"] == "MALFORMED_SEED_FIELDS"
    assert report["malformed_seed_fields"][0]["observed_type"] == observed_type


def test_seed_audit_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir()
    (root / "duplicate.json").write_text('{"seed":101,"seed":202}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        audit_seed_roster_collisions([root], [101])


def test_seed_audit_rejects_symlinked_result_files(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir()
    target = tmp_path / "target.json"
    target.write_text('{"seed":101}', encoding="utf-8")
    (root / "linked.json").symlink_to(target)

    with pytest.raises(ValueError, match="symlink result file"):
        audit_seed_roster_collisions([root], [101])


def test_seed_audit_rejects_hardlinked_result_files(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir()
    source = root / "source.json"
    source.write_text('{"seed":101}', encoding="utf-8")
    (root / "alias.json").hardlink_to(source)

    with pytest.raises(ValueError, match="single-link regular file"):
        audit_seed_roster_collisions([root], [101])


def test_seed_audit_requires_at_least_one_scannable_json_result(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir()

    with pytest.raises(ValueError, match="no .json or .json.gz"):
        audit_seed_roster_collisions([root], [101])


def test_seed_audit_reports_file_byte_limit_as_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "results"
    root.mkdir()
    (root / "result.json").write_text('{"seed":101}', encoding="utf-8")
    monkeypatch.setattr(seed_audit, "_MAX_FILE_BYTES", 4)

    report = audit_seed_roster_collisions([root], [101])

    assert report["status"] == "INCOMPLETE_SCAN"
    assert report["scanned_file_count"] == 0
    assert report["incomplete_files"] == [
        {
            "path": str(root / "result.json"),
            "reason": "corpus file exceeds compressed/input byte limit",
        }
    ]


def test_seed_audit_continues_after_expanded_json_limit_and_cli_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    root = tmp_path / "results"
    root.mkdir()
    (root / "oversized.json.gz").write_bytes(gzip.compress(b'{"seed":101,"padding":"123456789"}'))
    (root / "scanned.json").write_text('{"seed":202}', encoding="utf-8")
    monkeypatch.setattr(seed_audit, "_MAX_JSON_BYTES", 12)

    status = main(["--root", str(root), "--seed", "101", "--seed", "202"])

    report = json.loads(capsys.readouterr().out)
    assert status == 2
    assert report["status"] == "INCOMPLETE_SCAN"
    assert report["scanned_file_count"] == 1
    assert report["collisions"][0]["matches"][0]["seed"] == 202
    assert report["incomplete_files"] == [
        {
            "path": str(root / "oversized.json.gz"),
            "reason": "expanded JSON exceeds configured size limit",
        }
    ]


@pytest.mark.parametrize(
    ("seeds", "message"),
    [
        ([], "nonempty"),
        ([1, 1], "duplicates"),
        ([True], "nonnegative integers"),
        ([-1], "nonnegative"),
    ],
)
def test_seed_audit_rejects_invalid_rosters(tmp_path: Path, seeds: list[Any], message: str) -> None:
    root = tmp_path / "results"
    root.mkdir()

    with pytest.raises(ValueError, match=message):
        audit_seed_roster_collisions([root], seeds)  # type: ignore[arg-type]


def test_cli_returns_nonzero_for_a_collision_and_emits_the_audit(tmp_path: Path, capsys) -> None:
    root = tmp_path / "results"
    root.mkdir()
    (root / "result.json").write_text('{"channel_seed":8}', encoding="utf-8")

    status = main(["--root", str(root), "--seed", "8"])

    assert status == 1
    assert json.loads(capsys.readouterr().out)["status"] == "COLLISION_FOUND"


def test_cli_expands_half_open_seed_ranges(tmp_path: Path, capsys) -> None:
    root = tmp_path / "results"
    root.mkdir()
    (root / "result.json").write_text("{}", encoding="utf-8")

    status = main(["--root", str(root), "--seed-range", "10", "13"])

    report = json.loads(capsys.readouterr().out)
    assert status == 0
    assert report["reserved_seeds"] == [10, 11, 12]
    assert report["status"] == "CLEAR_WITHIN_SCANNED_JSON_CORPUS"
