from __future__ import annotations

import hashlib
import json

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
from scripts.bind_split_registry_children import bind_registry_children, main


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
