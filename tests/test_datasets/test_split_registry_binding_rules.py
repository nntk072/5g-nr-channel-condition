"""Split registry bindings: frozen children, parent and child rosters, holdout, spatial audits and locks."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import MappingProxyType

import pytest
from datasets.split_policy import (
    LeakageNamespace,
    SplitChildBinding,
    SplitPolicy,
)
from datasets.split_policy import registry as registry_module
from datasets.split_policy.registry import (
    SplitPolicyRegistry,
    audit_split_policy_child_roster,
    audit_split_policy_parent_roster,
    audit_split_policy_registry,
    bind_spatiotemporal_audit_to_registry,
    bind_split_children,
    freeze_split_policy_registry,
    make_split_policy_registry,
    read_split_policy_registry,
    verify_registry_spatiotemporal_audit,
    verify_signed_split_policy_registry_attestation,
)
from tests.test_datasets.test_split_policy_v2 import (
    _four_way_registry,
    _subject,
    _unlinked_subjects,
)


def _registry(count: int = 12):
    return _four_way_registry(_unlinked_subjects(count))


def _binding(
    child_id: str, parent_id: str, child_kind: str = "channel_dataset"
) -> SplitChildBinding:
    return SplitChildBinding(
        child_id=child_id,
        parent_subject_id=parent_id,
        child_kind=child_kind,
    )


def test_child_bindings_are_idempotent_but_immutable() -> None:
    _, registry = _registry()
    parent, other_parent = list(registry.subjects)[:2]
    binding = _binding("child-1", parent)
    bound = bind_split_children(registry, [binding, binding])
    assert bound.child_bindings["child-1"] == binding
    assert bound.split_for_child("child-1") == registry.split_for_parent(parent)

    with pytest.raises(ValueError, match="conflicting parent or kind"):
        bind_split_children(registry, [binding, _binding("child-1", other_parent)])
    with pytest.raises(ValueError, match="frozen child binding cannot change"):
        bind_split_children(bound, [_binding("child-1", other_parent)])
    with pytest.raises(TypeError, match="bindings must contain SplitChildBinding"):
        bind_split_children(registry, ["not-a-binding"])


def test_registry_child_map_keys_and_collisions_are_checked() -> None:
    _, registry = _registry()
    parent = next(iter(registry.subjects))
    with pytest.raises(ValueError, match="child mapping keys must equal child IDs"):
        dataclasses.replace(registry, child_bindings={"wrong-key": _binding("child-1", parent)})
    with pytest.raises(TypeError, match="child_bindings must contain SplitChildBinding"):
        dataclasses.replace(registry, child_bindings={"child-1": "not-a-binding"})
    with pytest.raises(ValueError, match="collides with a parent subject ID"):
        dataclasses.replace(registry, child_bindings={parent: _binding(parent, parent)})


def test_lookups_report_absent_parents_and_children() -> None:
    _, registry = _registry()
    with pytest.raises(KeyError, match="parent subject is absent from frozen split"):
        registry.split_for_parent("never-registered")
    with pytest.raises(KeyError, match="child is absent from frozen split registry"):
        registry.split_for_child("never-bound")


def test_parent_roster_audit_detects_duplicates_drift_and_gaps() -> None:
    _, registry = _registry()
    subjects = list(registry.subjects.values())
    with pytest.raises(ValueError, match="duplicate split subject in audit"):
        audit_split_policy_parent_roster(registry, [subjects[0], subjects[0]])
    with pytest.raises(TypeError, match="subjects must contain SplitSubject values"):
        audit_split_policy_parent_roster(registry, ["not-a-subject"])
    with pytest.raises(TypeError, match="registry must be a SplitPolicyRegistry"):
        audit_split_policy_parent_roster("registry", subjects)
    drifted = dataclasses.replace(subjects[0], source_family="changed-family")
    with pytest.raises(ValueError, match="frozen split lineage changed"):
        audit_split_policy_parent_roster(registry, [drifted, *subjects[1:]])
    with pytest.raises(ValueError, match="do not cover exactly the frozen parent set"):
        audit_split_policy_parent_roster(registry, subjects[:-1])
    report = audit_split_policy_parent_roster(registry, subjects)
    assert report["status"] == "PASS_INPUT_SCOPED"
    assert report["parent_count"] == len(subjects)
    assert len(report["audit_sha256"]) == 64


def test_child_roster_audit_checks_types_duplicates_and_parent_binding() -> None:
    _, registry = _registry()
    parent, other_parent = list(registry.subjects)[:2]
    bound = bind_split_children(registry, [_binding("child-1", parent)])
    with pytest.raises(TypeError, match="registry must be a SplitPolicyRegistry"):
        audit_split_policy_child_roster("registry", [], child_kind="channel_dataset")
    with pytest.raises(TypeError, match="child_bindings must contain SplitChildBinding"):
        audit_split_policy_child_roster(bound, ["not-a-binding"], child_kind="channel_dataset")
    with pytest.raises(ValueError, match="duplicate child binding"):
        audit_split_policy_child_roster(
            bound,
            [_binding("child-1", parent), _binding("child-1", parent)],
            child_kind="channel_dataset",
        )
    with pytest.raises(ValueError, match="frozen parent binding changed"):
        audit_split_policy_child_roster(
            bound, [_binding("child-1", other_parent)], child_kind="channel_dataset"
        )
    report = audit_split_policy_child_roster(
        bound, [_binding("child-1", parent)], child_kind="channel_dataset"
    )
    assert report["status"] == "PASS_INPUT_SCOPED"


def test_spatiotemporal_audit_cannot_be_replaced_once_frozen() -> None:
    _, registry = _registry()
    subjects = list(registry.subjects.values())
    with pytest.raises(ValueError, match="no frozen spatiotemporal footprint audit"):
        verify_registry_spatiotemporal_audit(registry, [], spatial_buffer_m=0.0)
    with pytest.raises(TypeError, match="registry must be a SplitPolicyRegistry"):
        bind_spatiotemporal_audit_to_registry("registry", [], spatial_buffer_m=0.0)

    from datasets.split_policy import SpatiotemporalFootprint

    footprints = tuple(SpatiotemporalFootprint(subject.subject_id) for subject in subjects)
    bound = bind_spatiotemporal_audit_to_registry(registry, footprints, spatial_buffer_m=0.0)
    assert verify_registry_spatiotemporal_audit(bound, footprints, spatial_buffer_m=0.0) == dict(
        bound.spatiotemporal_audit
    )
    changed = (
        *footprints[:-1],
        SpatiotemporalFootprint(footprints[-1].subject_id, ((1.0, 2.0, 3.0),), "map"),
    )
    with pytest.raises(ValueError, match="cannot be replaced"):
        bind_spatiotemporal_audit_to_registry(bound, changed, spatial_buffer_m=0.0)
    with pytest.raises(ValueError, match="inventory differs"):
        verify_registry_spatiotemporal_audit(bound, changed, spatial_buffer_m=0.0)
    with pytest.raises(ValueError, match="assign every registered parent"):
        dataclasses.replace(registry, subject_to_component={})


def test_registry_audit_checks_exact_parent_and_child_inventories() -> None:
    _, registry = _registry()
    subjects = list(registry.subjects.values())
    with pytest.raises(ValueError, match="duplicate split subject in audit"):
        audit_split_policy_registry(registry, [subjects[0], subjects[0]])
    with pytest.raises(ValueError, match="absent from frozen split registry"):
        audit_split_policy_registry(registry, [_subject("unknown")])
    with pytest.raises(ValueError, match="do not cover exactly"):
        audit_split_policy_registry(registry, subjects[:-1])

    parent = subjects[0].subject_id
    bound = bind_split_children(registry, [_binding("child-a", parent)])
    with pytest.raises(ValueError, match="duplicate child binding"):
        audit_split_policy_registry(
            bound, subjects, [_binding("child-a", parent), _binding("child-a", parent)]
        )
    with pytest.raises(ValueError, match="child is absent from frozen split registry"):
        audit_split_policy_registry(bound, subjects, [_binding("unknown-child", parent)])
    with pytest.raises(ValueError, match="do not cover exactly the frozen child set"):
        audit_split_policy_registry(bound, subjects)
    audit_split_policy_registry(bound, subjects, [_binding("child-a", parent)])


def test_registry_audit_rejects_cross_component_shared_relations() -> None:
    from datasets.split_policy import LeakageRelation

    _, registry = _registry()
    original_subjects = list(registry.subjects.values())
    relation = LeakageRelation(LeakageNamespace.SITE, "shared-site")
    first, second = original_subjects[:2]
    altered_subjects = dict(registry.subjects)
    altered_subjects[first.subject_id] = dataclasses.replace(first, relations=(relation,))
    altered_subjects[second.subject_id] = dataclasses.replace(second, relations=(relation,))
    object.__setattr__(registry, "subjects", MappingProxyType(altered_subjects))
    components = dict(registry.subject_to_component)
    components[first.subject_id] = "forged-component-a"
    components[second.subject_id] = "forged-component-b"
    object.__setattr__(registry, "subject_to_component", MappingProxyType(components))

    with pytest.raises(ValueError, match="shared leakage relation crosses components"):
        audit_split_policy_registry(registry, list(altered_subjects.values()))


def test_registry_constructor_rejects_corrupt_frozen_spatiotemporal_audit() -> None:
    _, registry = _registry()
    subjects = list(registry.subjects.values())
    from datasets.split_policy import SpatiotemporalFootprint

    footprints = tuple(SpatiotemporalFootprint(subject.subject_id) for subject in subjects)
    bound = bind_spatiotemporal_audit_to_registry(registry, footprints, spatial_buffer_m=0.0)
    invalid_audit = {**bound.spatiotemporal_audit, "parent_count": -1}
    with pytest.raises(ValueError, match="audit is invalid or roster-mismatched"):
        SplitPolicyRegistry(
            bound.policy,
            bound.subject_to_component,
            bound.component_to_split,
            bound.subjects,
            bound.child_bindings,
            invalid_audit,
        )


def test_registry_lock_round_trips_and_is_never_overwritten(tmp_path: Path) -> None:
    _, registry = _registry()
    lock = tmp_path / "registry.json"
    digest = freeze_split_policy_registry(lock, registry)
    assert len(digest) == 64
    restored = read_split_policy_registry(lock)
    assert restored.to_dict()["registry_sha256"] == registry.to_dict()["registry_sha256"]
    with pytest.raises(TypeError, match="registry must be a SplitPolicyRegistry"):
        freeze_split_policy_registry(tmp_path / "bad.json", "registry")


def test_spatial_relations_require_a_bound_footprint_audit(tmp_path: Path) -> None:
    spatial = make_split_policy_registry(
        [_subject("a", (LeakageNamespace.SPATIAL_BUFFER, "site-1")), *_unlinked_subjects(10)],
        policy=SplitPolicy.release_v2(seed="spatial"),
    )
    with pytest.raises(ValueError, match="without a bound footprint audit"):
        freeze_split_policy_registry(tmp_path / "spatial.json", spatial)


def test_read_rejects_tampered_or_mislabelled_locks(tmp_path: Path) -> None:
    _, registry = _registry()
    lock = tmp_path / "registry.json"
    freeze_split_policy_registry(lock, registry)
    payload = json.loads(lock.read_text(encoding="utf-8"))
    payload["report"] = {"tampered": True}
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="hash is invalid"):
        read_split_policy_registry(tampered)

    bad_schema = tmp_path / "schema.json"
    bad_schema.write_text(
        json.dumps({"schema": "umcd.split_registry.v9", "schema_version": 9}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unsupported UMCD split registry schema"):
        read_split_policy_registry(bad_schema)
    list_root = tmp_path / "list.json"
    list_root.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be a JSON object"):
        read_split_policy_registry(list_root)


def test_read_rejects_non_object_spatiotemporal_audit_for_v3(tmp_path: Path) -> None:
    _, registry = _registry()
    payload = registry.to_dict()
    payload["schema"] = "umcd.split_registry.v3"
    payload["schema_version"] = 3
    payload["spatiotemporal_audit"] = None
    path = tmp_path / "bad-v3-audit.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="spatiotemporal_audit must be a JSON object"):
        read_split_policy_registry(path)


@pytest.mark.parametrize("identical", [True, False])
def test_registry_freeze_handles_concurrent_destination_creation(
    tmp_path: Path, monkeypatch, identical: bool
) -> None:
    _, registry = _registry()
    destination = tmp_path / "raced-registry.json"

    def racing_link(source, target):
        destination.write_bytes(Path(source).read_bytes() if identical else b"another lock")
        raise FileExistsError(target)

    monkeypatch.setattr(registry_module.os, "link", racing_link)
    if identical:
        assert freeze_split_policy_registry(destination, registry)
    else:
        with pytest.raises(FileExistsError, match="refusing to modify existing"):
            freeze_split_policy_registry(destination, registry)


def test_registry_freeze_propagates_temporary_file_creation_failure(
    tmp_path: Path, monkeypatch
) -> None:
    _, registry = _registry()

    def fail_temporary_file(*_args, **_kwargs):
        raise OSError("temporary storage unavailable")

    monkeypatch.setattr(registry_module.tempfile, "NamedTemporaryFile", fail_temporary_file)
    with pytest.raises(OSError, match="temporary storage unavailable"):
        freeze_split_policy_registry(tmp_path / "not-created.json", registry)


def test_split_registry_signer_id_must_be_single_line() -> None:
    with pytest.raises(ValueError, match="single-line principal"):
        verify_signed_split_policy_registry_attestation(
            "unused-lock",
            "unused-signature",
            trusted_allowed_signers_path="unused-trust",
            signer_id="reviewer\nforged",
        )
