from __future__ import annotations

import hashlib
import json

import jsonschema
import pytest
from datasets.split_policy import (
    LeakageRelation,
    SplitParentType,
    SplitPolicy,
    SplitSubject,
    freeze_split_policy_registry,
    make_split_policy_registry,
    read_split_policy_registry,
)
from scripts.bind_split_registry_children import _load_roster, bind_registry_children, main


def _registry(tmp_path):
    subjects = tuple(
        SplitSubject(
            subject_id=f"parent-{index}",
            parent_type=SplitParentType.NETWORK_EPISODE,
            source_family="sionna_rt",
            relations=(LeakageRelation("asset_family", f"asset-{index}"),),
        )
        for index in range(8)
    )
    registry = make_split_policy_registry(
        subjects, policy=SplitPolicy.release_v2(seed="child-binding-cli")
    )
    path = tmp_path / "parent-registry.json"
    freeze_split_policy_registry(path, registry)
    return path, registry


def _roster(path, subjects):
    payload = {
        "schema": "umcd.channel_dataset_child_roster.v1",
        "child_kind": "receiver_view",
        "bindings": [
            {
                "child_id": f"{subject.subject_id}:receiver",
                "parent_subject_id": subject.subject_id,
                "child_kind": "receiver_view",
            }
            for subject in subjects
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_child_binding_cli_publishes_new_lock_without_mutating_parent(tmp_path, capsys):
    parent_path, registry = _registry(tmp_path)
    parent_bytes = parent_path.read_bytes()
    roster = _roster(tmp_path / "children.json", tuple(registry.subjects.values()))
    output = tmp_path / "bound-candidate"

    assert (
        main(
            [
                "--registry",
                str(parent_path),
                "--child-roster",
                str(roster),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    cli_result = json.loads(capsys.readouterr().out)
    bound = read_split_policy_registry(output / "split-registry.json")
    report = json.loads((output / "binding-report.json").read_text(encoding="utf-8"))

    assert parent_path.read_bytes() == parent_bytes
    assert cli_result["qualification_status"] == report["qualification_status"] == "NOT_QUALIFIED"
    assert cli_result["child_binding_count"] == 8
    assert len(bound.child_bindings) == 8
    assert report["parent_registry_sha256"] == hashlib.sha256(parent_bytes).hexdigest()
    assert (
        report["bound_registry_file_sha256"]
        == hashlib.sha256((output / "split-registry.json").read_bytes()).hexdigest()
    )
    for binding in bound.child_bindings.values():
        assert bound.split_for_child(binding.child_id) == bound.split_for_parent(
            binding.parent_subject_id
        )


def test_child_binding_rejects_unknown_parent_without_publishing(tmp_path):
    parent_path, _registry_value = _registry(tmp_path)
    roster = tmp_path / "orphan.json"
    roster.write_text(
        json.dumps(
            {
                "schema": "umcd.channel_dataset_child_roster.v1",
                "child_kind": "receiver_view",
                "bindings": [
                    {
                        "child_id": "orphan:receiver",
                        "parent_subject_id": "missing-parent",
                        "child_kind": "receiver_view",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "rejected"

    with pytest.raises(ValueError, match="unknown split parent"):
        bind_registry_children(parent_path, (roster,), output_dir=output)

    assert not output.exists()


def test_child_binding_rejects_duplicate_child_ids_across_rosters(tmp_path):
    parent_path, registry = _registry(tmp_path)
    first = _roster(tmp_path / "first.json", (next(iter(registry.subjects.values())),))
    second = _roster(tmp_path / "second.json", (next(iter(registry.subjects.values())),))
    output = tmp_path / "duplicate"

    with pytest.raises(ValueError, match="duplicate child IDs"):
        bind_registry_children(parent_path, (first, second), output_dir=output)

    assert not output.exists()


def test_child_roster_loader_rejects_duplicate_json_keys_and_invalid_json(tmp_path):
    roster = tmp_path / "children.json"
    for payload, message in (
        (b'{"schema": 1, "schema": 2}', "duplicate JSON object key"),
        (b"{", "not valid bounded UTF-8 JSON"),
        (b"\xff", "not valid bounded UTF-8 JSON"),
    ):
        roster.write_bytes(payload)
        with pytest.raises(ValueError, match=message):
            _load_roster(roster)


def test_child_roster_loader_enforces_schema_and_unique_local_children(tmp_path):
    roster = tmp_path / "children.json"
    invalid = {
        "schema": "wrong-schema",
        "child_kind": "receiver_view",
        "bindings": [
            {
                "child_id": "child",
                "parent_subject_id": "parent",
                "child_kind": "receiver_view",
            }
        ],
    }
    roster.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(jsonschema.ValidationError):
        _load_roster(roster)

    duplicate = {
        "schema": "umcd.channel_dataset_child_roster.v1",
        "child_kind": "receiver_view",
        "bindings": [
            {
                "child_id": "same-child",
                "parent_subject_id": "parent-a",
                "child_kind": "receiver_view",
            },
            {
                "child_id": "same-child",
                "parent_subject_id": "parent-b",
                "child_kind": "receiver_view",
            },
        ],
    }
    roster.write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate child IDs"):
        _load_roster(roster)

    duplicate["bindings"][1]["child_id"] = "other-child"
    duplicate["bindings"][1]["child_kind"] = "other-kind"
    roster.write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(ValueError, match="disagree with its child_kind"):
        _load_roster(roster)


def test_child_roster_loader_rejects_hard_links_and_symlinks(tmp_path):
    roster = tmp_path / "roster.json"
    roster.write_text(
        json.dumps(
            {
                "schema": "umcd.channel_dataset_child_roster.v1",
                "child_kind": "receiver_view",
                "bindings": [
                    {
                        "child_id": "child",
                        "parent_subject_id": "parent",
                        "child_kind": "receiver_view",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    hardlink = tmp_path / "hardlink.json"
    hardlink.hardlink_to(roster)
    with pytest.raises(ValueError, match="single-link regular file"):
        _load_roster(hardlink)

    hardlink.unlink()
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(roster)
    with pytest.raises(ValueError, match="single-link regular file"):
        _load_roster(symlink)


def test_child_roster_reader_enforces_byte_limit_and_detects_concurrent_growth(
    tmp_path, monkeypatch
):
    import os

    import scripts.bind_split_registry_children as binder

    roster = tmp_path / "children.json"
    roster.write_bytes(b"{}")
    monkeypatch.setattr(binder, "_MAX_INPUT_BYTES", 1)
    with pytest.raises(ValueError, match="exceeds the 1-byte limit"):
        _load_roster(roster)

    actual_fstat = os.fstat

    def changed_fstat(descriptor):
        current = actual_fstat(descriptor)
        return type(
            "ChangedStat",
            (),
            {
                "st_dev": current.st_dev,
                "st_ino": current.st_ino,
                "st_size": current.st_size + 1,
                "st_mtime_ns": current.st_mtime_ns,
            },
        )()

    monkeypatch.setattr(binder.os, "fstat", changed_fstat)
    with pytest.raises(ValueError, match="changed while opening"):
        binder._read_regular_bytes(roster, label="split child roster")
    monkeypatch.setattr(binder.os, "fstat", actual_fstat)

    roster.write_bytes(b"x")
    actual_read = os.read

    def grow_before_read(descriptor, count):
        roster.write_bytes(b"expanded")
        monkeypatch.setattr(binder.os, "read", actual_read)
        return actual_read(descriptor, count)

    monkeypatch.setattr(binder.os, "read", grow_before_read)
    with pytest.raises(ValueError, match="exceeds the 1-byte limit"):
        binder._read_regular_bytes(roster, label="split child roster")


def test_child_roster_reader_detects_path_replacement_after_read(tmp_path, monkeypatch):
    import os

    import scripts.bind_split_registry_children as binder

    roster = tmp_path / "children.json"
    replacement = tmp_path / "replacement.json"
    roster.write_bytes(b"{}")
    replacement.write_bytes(b"replacement")
    actual_read = os.read
    replaced = False

    def replace_during_read(descriptor, count):
        nonlocal replaced
        chunk = actual_read(descriptor, count)
        if chunk and not replaced:
            replaced = True
            os.replace(replacement, roster)
        return chunk

    monkeypatch.setattr(binder.os, "read", replace_during_read)
    with pytest.raises(ValueError, match="changed while reading"):
        binder._read_regular_bytes(roster, label="split child roster")


def test_child_binding_rejects_empty_rosters_duplicate_roster_paths_and_existing_output(
    tmp_path,
):
    parent_path, registry = _registry(tmp_path)
    roster = _roster(tmp_path / "children.json", (next(iter(registry.subjects.values())),))
    output = tmp_path / "bound"
    with pytest.raises(ValueError, match="at least one --child-roster"):
        bind_registry_children(parent_path, (), output_dir=output)

    with pytest.raises(ValueError, match="roster paths must be unique"):
        bind_registry_children(parent_path, (roster, roster), output_dir=output)
    assert not output.exists()

    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        bind_registry_children(parent_path, (roster,), output_dir=output)


def test_child_binding_cleans_staging_when_frozen_registry_digest_disagrees(tmp_path, monkeypatch):
    import scripts.bind_split_registry_children as binder

    parent_path, registry = _registry(tmp_path)
    roster = _roster(tmp_path / "children.json", (next(iter(registry.subjects.values())),))
    output = tmp_path / "candidate"
    monkeypatch.setattr(binder, "freeze_split_policy_registry", lambda *_args: "wrong-digest")

    with pytest.raises(RuntimeError, match="bound registry bytes differ"):
        bind_registry_children(parent_path, (roster,), output_dir=output)

    assert not output.exists()
    assert not list(tmp_path.glob(".candidate.staging-*"))


def test_child_binding_detects_parent_registry_change_after_loading(tmp_path, monkeypatch):
    import scripts.bind_split_registry_children as binder

    parent_path, registry = _registry(tmp_path)
    roster = _roster(tmp_path / "children.json", (next(iter(registry.subjects.values())),))
    actual_read = binder._read_regular_bytes
    reads = 0

    def changed_second_read(path, *, label):
        nonlocal reads
        reads += 1
        payload = actual_read(path, label=label)
        return payload if reads == 1 else payload + b" "

    monkeypatch.setattr(binder, "_read_regular_bytes", changed_second_read)
    with pytest.raises(ValueError, match="changed while loading"):
        bind_registry_children(parent_path, (roster,), output_dir=tmp_path / "changed")


def test_child_binding_cli_reports_input_errors(tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                "--registry",
                str(tmp_path / "missing-registry.json"),
                "--child-roster",
                str(tmp_path / "children.json"),
                "--output-dir",
                str(tmp_path / "candidate"),
            ]
        )

    assert exit_info.value.code == 2
    assert "error:" in capsys.readouterr().err
