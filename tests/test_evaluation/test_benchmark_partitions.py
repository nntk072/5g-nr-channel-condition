from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import evaluation.benchmark_partitions as partitions
import jsonschema
import numpy as np
import pytest
from datasets.channel import ChannelDataset
from datasets.split_policy import (
    LeakageNamespace,
    LeakageRelation,
    SplitParentType,
    SplitPolicy,
    SplitSubject,
    make_split_policy_registry,
)
from evaluation.benchmark_partitions import (
    BENCHMARK_COHORTS,
    materialize_benchmark_manifest,
    verify_benchmark_manifest_replay,
)
from scripts.materialize_benchmark_partitions import build_benchmark_partition_manifest

_SHA = "a" * 64


def _registry_fixture():
    subjects: list[SplitSubject] = []
    parent_to_index: dict[str, int] = {}
    for component_index in range(60):
        scene_id = f"scene-{component_index:02d}"
        for member in range(2):
            parent = f"parent-{component_index:02d}-{member}"
            parent_to_index[parent] = component_index * 2 + member
            subjects.append(
                SplitSubject(
                    subject_id=parent,
                    parent_type=SplitParentType.NETWORK_EPISODE,
                    source_family="fixture.family",
                    relations=(LeakageRelation(LeakageNamespace.SCENE_REALIZATION, scene_id),),
                )
            )
    registry = make_split_policy_registry(
        subjects,
        policy=SplitPolicy.release_v2(seed="benchmark-partition-tests"),
    )
    return registry, parent_to_index


def _component_members(registry, split: str) -> list[list[str]]:
    grouped: dict[str, list[str]] = {}
    for parent, component in registry.subject_to_component.items():
        if registry.split_for_parent(parent) == split:
            grouped.setdefault(component, []).append(parent)
    return [sorted(grouped[key]) for key in sorted(grouped)]


def _declaration(registry) -> dict:
    test_components = _component_members(registry, "test")
    assert len(test_components) >= 3
    site_held = sorted(test_components[0])
    site_array = sorted(test_components[1])
    external = sorted(parent for group in test_components[2:] for parent in group)
    parents = sorted(registry.subjects)
    return {
        "schema": "umcd.benchmark_cohort_declaration.v1",
        "schema_version": 1,
        "test_cohorts": {
            "site_held_out": site_held,
            "site_array_held_out": site_array,
            "real_external": external,
        },
        "site_ids": {parent: f"site-{parent}" for parent in parents},
        "array_ids": {parent: f"array-{parent}" for parent in parents},
        "external_evidence": {
            parent: {
                "measurement_inventory_sha256": "1" * 64,
                "measurement_protocol_sha256": "2" * 64,
                "rights_review_sha256": "3" * 64,
            }
            for parent in external
        },
    }


def _dataset(registry, parents: list[str] | None = None, *, bad_family_parent: str | None = None):
    selected = sorted(registry.subjects) if parents is None else list(parents)
    n = len(selected)
    metadata = []
    for parent in selected:
        subject = registry.subjects[parent]
        metadata.append(
            {
                "split_subject_id": parent,
                "split_source_family": (
                    "wrong.family" if parent == bad_family_parent else subject.source_family
                ),
                "split_parent_type": subject.parent_type.value,
            }
        )
    data = ChannelDataset(
        h=np.ones((n, 1, 8), dtype=np.complex64),
        y=np.zeros(n, dtype=np.uint8),
        group=np.asarray([f"group-{index}" for index in range(n)]),
        domain=np.asarray(["fixture"] * n),
        source=np.asarray(["synthetic"] * n),
        metadata=tuple(metadata),
        label_semantics="los_binary",
    )
    return [("fixture", _SHA, data)]


def _attestation(namespace: str) -> dict[str, str]:
    return {
        "artifact_sha256": "4" * 64,
        "signature_sha256": "5" * 64,
        "allowed_signers_sha256": "6" * 64,
        "signer_id": "fixture@example.com",
        "namespace": namespace,
    }


def _materialize(registry, declaration, datasets=None):
    return materialize_benchmark_manifest(
        registry,
        declaration,
        _dataset(registry) if datasets is None else datasets,
        registry_attestation=_attestation("umcd-split-policy-v2@github.com"),
        declaration_attestation=_attestation("umcd-benchmark-cohort-v1@github.com"),
    )


def test_valid_six_cohort_manifest_and_schema() -> None:
    registry, _ = _registry_fixture()
    declaration = _declaration(registry)

    manifest = _materialize(registry, declaration)

    assert manifest["status"] == "CANDIDATE_NOT_SCIENTIFICALLY_QUALIFIED"
    assert manifest["complete_six_cohort_coverage"] is True
    assert set(manifest["cohorts"]) == set(BENCHMARK_COHORTS)
    assert all(manifest["cohorts"][name]["rows"] > 0 for name in BENCHMARK_COHORTS)
    schema = json.loads(
        Path("datasets/schema/benchmark-partition-manifest.v1.schema.json").read_text()
    )
    jsonschema.validate(manifest, schema)


def test_manifest_replay_returns_exact_dataset_row_indices() -> None:
    registry, _ = _registry_fixture()
    declaration = _declaration(registry)
    datasets = _dataset(registry)
    registry_attestation = _attestation("umcd-split-policy-v2@github.com")
    declaration_attestation = _attestation("umcd-benchmark-cohort-v1@github.com")
    manifest = materialize_benchmark_manifest(
        registry,
        declaration,
        datasets,
        registry_attestation=registry_attestation,
        declaration_attestation=declaration_attestation,
    )

    replay = verify_benchmark_manifest_replay(
        manifest,
        registry,
        declaration,
        datasets,
        registry_attestation=registry_attestation,
        declaration_attestation=declaration_attestation,
    )

    expected = {
        cohort: [
            index
            for index, parent in enumerate(sorted(registry.subjects))
            if (
                registry.split_for_parent(parent) == cohort
                if cohort in {"train", "validation", "calibration"}
                else parent in declaration["test_cohorts"][cohort]
            )
        ]
        for cohort in BENCHMARK_COHORTS
    }
    assert replay["status"] == "PASS_INPUT_SCOPED_REPLAY"
    assert replay["qualification_status"] == "NOT_QUALIFIED"
    assert replay["rows_by_dataset"]["fixture"] == expected


def test_manifest_replay_rejects_rehashed_row_assignment_tampering() -> None:
    registry, _ = _registry_fixture()
    declaration = _declaration(registry)
    datasets = _dataset(registry)
    registry_attestation = _attestation("umcd-split-policy-v2@github.com")
    declaration_attestation = _attestation("umcd-benchmark-cohort-v1@github.com")
    manifest = materialize_benchmark_manifest(
        registry,
        declaration,
        datasets,
        registry_attestation=registry_attestation,
        declaration_attestation=declaration_attestation,
    )
    manifest["datasets"][0]["rows"][0]["cohort"] = "real_external"
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()

    with pytest.raises(ValueError, match="does not replay"):
        verify_benchmark_manifest_replay(
            manifest,
            registry,
            declaration,
            datasets,
            registry_attestation=registry_attestation,
            declaration_attestation=declaration_attestation,
        )


def test_component_cannot_cross_benchmark_cohorts() -> None:
    registry, _ = _registry_fixture()
    declaration = _declaration(registry)
    held = declaration["test_cohorts"]["site_held_out"]
    moved = held.pop()
    declaration["test_cohorts"]["site_array_held_out"].append(moved)
    declaration["test_cohorts"]["site_array_held_out"].sort()

    with pytest.raises(ValueError, match="independent component crosses"):
        _materialize(registry, declaration)


def test_site_held_out_identity_cannot_repeat_elsewhere() -> None:
    registry, _ = _registry_fixture()
    declaration = _declaration(registry)
    held_parent = declaration["test_cohorts"]["site_held_out"][0]
    train_parent = next(
        parent
        for parent in sorted(registry.subjects)
        if registry.split_for_parent(parent) == "train"
    )
    declaration["site_ids"][held_parent] = declaration["site_ids"][train_parent]

    with pytest.raises(ValueError, match="site_held_out sites are present"):
        _materialize(registry, declaration)


def test_site_array_held_out_array_cannot_repeat_elsewhere() -> None:
    registry, _ = _registry_fixture()
    declaration = _declaration(registry)
    held_parent = declaration["test_cohorts"]["site_array_held_out"][0]
    train_parent = next(
        parent
        for parent in sorted(registry.subjects)
        if registry.split_for_parent(parent) == "train"
    )
    declaration["array_ids"][held_parent] = declaration["array_ids"][train_parent]

    with pytest.raises(ValueError, match="arrays are present in another cohort"):
        _materialize(registry, declaration)


def test_test_parent_coverage_is_fail_closed() -> None:
    registry, _ = _registry_fixture()
    declaration = _declaration(registry)
    omitted = declaration["test_cohorts"]["real_external"].pop()
    declaration["external_evidence"].pop(omitted)

    with pytest.raises(ValueError, match="cover every frozen test parent exactly"):
        _materialize(registry, declaration)


def test_external_evidence_roster_and_hashes_are_exact() -> None:
    registry, _ = _registry_fixture()
    declaration = _declaration(registry)
    external = declaration["test_cohorts"]["real_external"][0]
    declaration["external_evidence"].pop(external)
    with pytest.raises(ValueError, match="cover exactly the real_external"):
        _materialize(registry, declaration)

    declaration = _declaration(registry)
    external = declaration["test_cohorts"]["real_external"][0]
    declaration["external_evidence"][external]["rights_review_sha256"] = "A" * 64
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _materialize(registry, declaration)


def test_dataset_metadata_must_match_frozen_parent() -> None:
    registry, _ = _registry_fixture()
    declaration = _declaration(registry)
    parent = sorted(registry.subjects)[0]

    with pytest.raises(ValueError, match="source family contradicts frozen parent"):
        _materialize(registry, declaration, _dataset(registry, bad_family_parent=parent))


def test_dataset_roster_must_cover_every_frozen_parent() -> None:
    registry, _ = _registry_fixture()
    declaration = _declaration(registry)
    parents = sorted(registry.subjects)[:-1]

    with pytest.raises(ValueError, match="cover every frozen parent exactly"):
        _materialize(registry, declaration, _dataset(registry, parents))


def test_declaration_schema_rejects_extra_fields() -> None:
    registry, _ = _registry_fixture()
    declaration = _declaration(registry)
    declaration["unexpected"] = True
    schema = json.loads(
        Path("datasets/schema/benchmark-cohort-declaration.v1.schema.json").read_text()
    )

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(declaration, schema)
    with pytest.raises(ValueError, match="invalid field roster"):
        _materialize(registry, declaration)


def test_materializer_refuses_existing_output_before_reading_inputs(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    output.write_text("existing", encoding="utf-8")
    paths = [tmp_path / f"input-{index}" for index in range(6)]

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_benchmark_partition_manifest(
            registry_path=paths[0],
            registry_signature=paths[1],
            registry_allowed_signers=paths[2],
            registry_signer="fixture@example.com",
            declaration_path=paths[3],
            declaration_signature=paths[4],
            declaration_allowed_signers=paths[5],
            declaration_signer="fixture@example.com",
            datasets=[("fixture", tmp_path / "dataset.npz")],
            output=output,
        )


def test_declaration_validator_requires_canonical_four_way_policy():
    registry = type(
        "RegistryStub",
        (),
        {"policy": type("PolicyStub", (), {"split_fractions": {"train": 1.0}})(), "subjects": {}},
    )()
    declaration = {
        "schema": "umcd.benchmark_cohort_declaration.v1",
        "schema_version": 1,
        "test_cohorts": {
            cohort: [] for cohort in ("site_held_out", "site_array_held_out", "real_external")
        },
        "site_ids": {},
        "array_ids": {},
        "external_evidence": {},
    }

    with pytest.raises(ValueError, match="canonical four-way split policy"):
        partitions._validate_declarations(registry, declaration)


def test_declaration_validator_rejects_parent_assigned_to_two_cohorts():
    registry, _ = _registry_fixture()
    declaration = _declaration(registry)
    duplicated = declaration["test_cohorts"]["site_held_out"][0]
    declaration["test_cohorts"]["site_array_held_out"] = sorted(
        [*declaration["test_cohorts"]["site_array_held_out"], duplicated]
    )

    with pytest.raises(ValueError, match="more than one cohort"):
        partitions._validate_declarations(registry, declaration)


def test_declaration_validator_enforces_frozen_site_relations():
    registry, _ = _registry_fixture()
    subjects = [
        replace(
            subject,
            relations=(
                *subject.relations,
                LeakageRelation(LeakageNamespace.SITE, f"locked-{subject.subject_id}"),
            ),
        )
        for subject in registry.subjects.values()
    ]
    registry = make_split_policy_registry(subjects, policy=registry.policy)
    declaration = _declaration(registry)

    with pytest.raises(ValueError, match="contradicts frozen site relation"):
        partitions._validate_declarations(registry, declaration)


def test_materializer_requires_at_least_one_dataset():
    registry, _ = _registry_fixture()

    with pytest.raises(ValueError, match="at least one ChannelDataset"):
        _materialize(registry, _declaration(registry), [])


def test_materializer_accepts_exact_frozen_leakage_relations():
    registry, _ = _registry_fixture()
    declaration = _declaration(registry)
    datasets = _dataset(registry)
    name, digest, data = datasets[0]
    metadata = list(data.metadata)
    first_parent = metadata[0]["split_subject_id"]
    metadata[0]["split_leakage_relations"] = [
        {"namespace": item.namespace.value, "identity": item.identity}
        for item in registry.subjects[first_parent].relations
    ]
    datasets[0] = (name, digest, replace(data, metadata=tuple(metadata)))

    manifest = _materialize(registry, declaration, datasets)

    assert manifest["datasets"][0]["rows"][0]["parent_id"] == first_parent


def test_benchmark_materializer_rejects_registry_attestation_namespace():
    registry, _ = _registry_fixture()

    with pytest.raises(ValueError, match="registry attestation namespace is invalid"):
        materialize_benchmark_manifest(
            registry,
            _declaration(registry),
            _dataset(registry),
            registry_attestation=_attestation("wrong"),
            declaration_attestation=_attestation("umcd-benchmark-cohort-v1@github.com"),
        )


def test_benchmark_manifest_replay_rejects_content_without_matching_digest():
    registry, _ = _registry_fixture()
    declaration = _declaration(registry)
    datasets = _dataset(registry)
    registry_attestation = _attestation("umcd-split-policy-v2@github.com")
    declaration_attestation = _attestation("umcd-benchmark-cohort-v1@github.com")
    manifest = _materialize(registry, declaration, datasets)
    manifest["status"] = "PASS"

    with pytest.raises(ValueError, match="digest does not match its contents"):
        verify_benchmark_manifest_replay(
            manifest,
            registry,
            declaration,
            datasets,
            registry_attestation=registry_attestation,
            declaration_attestation=declaration_attestation,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value.update(schema_version=True),
            "unsupported benchmark cohort declaration",
        ),
        (lambda value: value.update(site_ids={}), "site_ids must cover exactly"),
        (lambda value: value.update(array_ids={}), "array_ids must cover exactly"),
        (lambda value: value.update(test_cohorts=[]), "declare all three benchmark test cohorts"),
        (lambda value: value["test_cohorts"]["site_held_out"].append(1), "array of parent IDs"),
        (lambda value: value["test_cohorts"]["site_held_out"].reverse(), "sorted and unique"),
        (
            lambda value: value["test_cohorts"]["site_held_out"].append(
                value["test_cohorts"]["site_held_out"][0]
            ),
            "sorted and unique",
        ),
        (
            lambda value: value["test_cohorts"]["site_held_out"].append("unknown"),
            "unknown or non-test parent",
        ),
        (
            lambda value: value["test_cohorts"]["site_held_out"].append("unknown-z"),
            "unknown or non-test parent",
        ),
        (
            lambda value: value["external_evidence"].update({"unknown": {}}),
            "cover exactly the real_external",
        ),
        (
            lambda value: value["external_evidence"][next(iter(value["external_evidence"]))].update(
                extra=True
            ),
            "external evidence is incomplete",
        ),
        (
            lambda value: value["site_ids"].update({next(iter(value["site_ids"])): " "}),
            "canonical nonempty string",
        ),
    ],
)
def test_benchmark_declaration_validator_rejects_invalid_rosters(mutate, message):
    registry, _ = _registry_fixture()
    declaration = _declaration(registry)
    mutate(declaration)

    with pytest.raises(ValueError, match=message):
        partitions._validate_declarations(registry, declaration)


@pytest.mark.parametrize(
    ("name", "digest", "message"),
    [
        (" ", _SHA, "dataset name must be a canonical nonempty string"),
        ("fixture", "invalid", "dataset digest must be a lowercase SHA-256"),
    ],
)
def test_benchmark_materializer_rejects_invalid_dataset_identity(name, digest, message):
    registry, _ = _registry_fixture()
    declaration = _declaration(registry)
    datasets = _dataset(registry)
    datasets[0] = (name, digest, datasets[0][2])

    with pytest.raises(ValueError, match=message):
        _materialize(registry, declaration, datasets)


def test_benchmark_materializer_rejects_duplicate_dataset_names():
    registry, _ = _registry_fixture()
    declaration = _declaration(registry)
    item = _dataset(registry)[0]

    with pytest.raises(ValueError, match="duplicate dataset name"):
        _materialize(registry, declaration, [item, item])


@pytest.mark.parametrize(
    ("metadata_change", "message"),
    [
        (lambda rows: rows.__setitem__(0, None), "metadata must be an object"),
        (lambda rows: rows[0].update(split_subject_id="unknown"), "absent from the lock"),
        (lambda rows: rows[0].update(split_parent_type="other"), "parent type contradicts"),
        (lambda rows: rows[0].update(split_partition_id="other"), "partition identity contradicts"),
        (
            lambda rows: rows[0].update(
                split_leakage_relations=[{"namespace": "site", "identity": "fake"}]
            ),
            "leakage relation contradicts",
        ),
    ],
)
def test_benchmark_materializer_rejects_dataset_row_identity_drift(metadata_change, message):
    registry, _ = _registry_fixture()
    declaration = _declaration(registry)
    datasets = _dataset(registry)
    data = datasets[0][2]
    metadata = list(data.metadata)
    metadata_change(metadata)
    datasets[0] = (datasets[0][0], datasets[0][1], replace(data, metadata=tuple(metadata)))

    with pytest.raises(ValueError, match=message):
        _materialize(registry, declaration, datasets)


@pytest.mark.parametrize(
    ("attestation_key", "field", "value", "message"),
    [
        ("registry", "extra", "x", "registry attestation field roster"),
        ("declaration", "artifact_sha256", "bad", "lowercase SHA-256"),
        ("registry", "signer_id", " ", "canonical nonempty string"),
        ("declaration", "namespace", "wrong", "cohort attestation namespace"),
    ],
)
def test_benchmark_materializer_rejects_invalid_attestations(
    attestation_key, field, value, message
):
    registry, _ = _registry_fixture()
    declaration = _declaration(registry)
    attestations = {
        "registry": _attestation("umcd-split-policy-v2@github.com"),
        "declaration": _attestation("umcd-benchmark-cohort-v1@github.com"),
    }
    attestations[attestation_key][field] = value

    with pytest.raises(ValueError, match=message):
        materialize_benchmark_manifest(
            registry,
            declaration,
            _dataset(registry),
            registry_attestation=attestations["registry"],
            declaration_attestation=attestations["declaration"],
        )


@pytest.mark.parametrize("manifest", ([], {"manifest_sha256": "bad"}))
def test_benchmark_manifest_replay_rejects_invalid_digest_binding(manifest):
    registry, _ = _registry_fixture()
    declaration = _declaration(registry)

    with pytest.raises(ValueError, match="benchmark manifest"):
        verify_benchmark_manifest_replay(
            manifest,
            registry,
            declaration,
            _dataset(registry),
            registry_attestation=_attestation("umcd-split-policy-v2@github.com"),
            declaration_attestation=_attestation("umcd-benchmark-cohort-v1@github.com"),
        )
