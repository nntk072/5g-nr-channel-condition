"""UMCD provenance records, lineage claims and bundle integrity."""

from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import datasets.targets as target_contracts
import numpy as np
import pytest
from datasets.channel_view import (
    PhysicalChannelView,
)
from datasets.counterfactual import remove_direct_path_component
from datasets.episode_store import (
    _READ_ONLY_UMCD_TARGET_VERSIONS,
    EpisodeBundle,
    _direct_path_ablation_payload,
    _episode_payload,
    _json_bytes,
    _object_ref,
    _pack_npz,
    _path_payload,
    _publish_immutable,
    _read_object,
    _validate_path_oracle_layer,
    read_episode_bundle,
    write_episode_bundle,
)
from datasets.lineage import (
    LineageStatusClaim,
    audit_lineage_evidence_files,
    audit_lineage_evidence_payloads,
)
from datasets.path_oracle import PathOracleTable, PathSolveStatus
from datasets.provenance import UMCDRunProvenance
from datasets.split_policy import (
    SplitChildBinding,
    SplitParentType,
    SplitPolicy,
    SplitSubject,
    bind_split_children,
    make_split_policy_registry,
)
from datasets.targets import (
    TargetDefinition,
    TargetValidity,
    TargetValue,
)
from tests.test_datasets.umcd_schema_support import (
    _bundle,
    _dynamic_episode,
    _provenance_record,
)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"link_ids": (" link-1",)}, "canonical link IDs"),
        ({"path_kind": (" specular ",)}, "canonical value per path"),
        ({"coefficient_convention": " sionna_rt_path_coefficient_v1"}, "canonical string"),
    ],
)
def test_path_oracle_rejects_padded_join_and_provenance_identities(change, message):
    arguments = {
        "link_ids": ("link-1",),
        "frame_indices": np.asarray([0]),
        "rx_port_indices": np.asarray([0]),
        "tx_port_indices": np.asarray([0]),
        "path_offsets": np.asarray([0, 1]),
        "solve_status": (PathSolveStatus.SOLVED,),
        "delay_s": np.asarray([1e-9]),
        "complex_gain": np.asarray([1.0 + 0j]),
        "interaction_order": np.asarray([0]),
    }
    with pytest.raises(ValueError, match=message):
        PathOracleTable(**{**arguments, **change})


@pytest.mark.parametrize(
    "version",
    sorted(_READ_ONLY_UMCD_TARGET_VERSIONS | {"umcd.rms_delay_spread.v1"}),
)
def test_all_read_only_canonical_targets_keep_legacy_rank_on_bundle_read(tmp_path, version):
    name, semantic, units, domain = target_contracts._CANONICAL_TARGET_CONTRACTS[version]
    source, references, aggregations = target_contracts._CANONICAL_TARGET_PROVENANCE[version]
    reference = sorted(references)[0]
    aggregation = sorted(aggregations)[0] if isinstance(aggregations, frozenset) else aggregations
    if domain == "complex":
        values = np.asarray([[1.0 + 2.0j]], dtype=np.complex64)
    elif domain == "nonnegative_real":
        values = np.asarray([[2.5e-8]], dtype=np.float64)
    elif domain == "binary_scalar" and version == "umcd.f03_rt_static.direct_visibility.v1":
        values = np.asarray([[1]], dtype=np.uint8)
    elif domain == "binary_scalar":
        values = np.asarray([1], dtype=np.uint8)
    else:
        values = np.asarray([1], dtype=np.uint8)
    definition = TargetDefinition.versioned(
        semantic=semantic,
        units=units,
        source=source,
        reference=reference,
        aggregation_rule=aggregation,
        definition_version=version,
    )
    target = TargetValue(
        entity_id="gNB-1-to-UE-1",
        definition=definition,
        values=values,
        valid_mask=np.ones(values.shape, dtype=np.bool_),
        reason_codes=(TargetValidity.VALID,) * values.size,
        confidence="historical_rank_fixture",
    )
    base = replace(_bundle(), targets=())
    stored = write_episode_bundle(tmp_path, base)
    manifest_path = tmp_path / stored.relative_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = _pack_npz(
        {
            "schema": "umcd.target.v2",
            "entity_id": target.entity_id,
            "definition": asdict(definition),
            "reason_codes": [reason.value for reason in target.reason_codes],
            "confidence": target.confidence,
            "censoring_intervals": [None] * values.size,
        },
        values=values,
        valid_mask=np.ones(values.shape, dtype=np.bool_),
    )
    reference_object = _object_ref("targets", payload)
    _publish_immutable(tmp_path / reference_object["path"], payload)
    manifest["layers"].append({"layer": "targets", **reference_object})
    core = {key: manifest[key] for key in ("schema", "schema_version", "episode_id", "layers")}
    manifest["bundle_sha256"] = hashlib.sha256(_json_bytes(core)).hexdigest()
    manifest_path.write_bytes(_json_bytes(manifest) + b"\n")

    restored = read_episode_bundle(tmp_path, manifest_path)
    historical = next(
        item for item in restored.targets if item.definition.definition_version == version
    )
    assert historical.values.shape == values.shape
    assert historical.definition.value_axes == ()
    np.testing.assert_array_equal(historical.values, values)


def test_episode_bundle_roundtrip_keeps_layers_content_addressed_and_pickle_free(tmp_path):
    bundle = _bundle()
    stored = write_episode_bundle(tmp_path, bundle)
    loaded = read_episode_bundle(tmp_path, stored)

    assert loaded.episode.episode_id == bundle.episode.episode_id
    np.testing.assert_array_equal(loaded.path_oracle.complex_gain, bundle.path_oracle.complex_gain)
    np.testing.assert_array_equal(
        loaded.path_oracle.interaction_offsets, bundle.path_oracle.interaction_offsets
    )
    np.testing.assert_array_equal(loaded.path_oracle.object_ids, bundle.path_oracle.object_ids)
    np.testing.assert_array_equal(
        loaded.path_oracle.interaction_vertices_m, bundle.path_oracle.interaction_vertices_m
    )
    assert loaded.path_oracle.interaction_vertex_frame == "test_scene_xyz_m"
    assert loaded.targets[0].definition.aggregation_rule == (
        "one visibility value per link and frame; no pooling"
    )
    assert loaded.targets[0].definition.definition_version == "umcd.geometric_direct_path.v1"
    assert loaded.targets[0].definition.definition_sha256 == (
        bundle.targets[0].definition.definition_sha256
    )
    np.testing.assert_array_equal(loaded.channel_views[0].channel, bundle.channel_views[0].channel)
    np.testing.assert_array_equal(loaded.targets[0].valid_mask, bundle.targets[0].valid_mask)
    assert (tmp_path / "objects/oracle_paths").is_dir()
    assert (tmp_path / "objects/oracle_channels").is_dir()
    assert (tmp_path / "objects/targets").is_dir()
    manifest = json.loads((tmp_path / stored.relative_path).read_text())
    assert {item["layer"] for item in manifest["layers"]} == {
        "episode",
        "oracle_paths",
        "oracle_channels",
        "targets",
    }


def test_path_oracle_storage_metadata_schema_and_exact_array_roster():
    payload = _path_payload(_bundle().path_oracle)
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata_json"].tobytes())
        array_names = set(archive.files) - {"metadata_json"}

    schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets/schema/umcd-path-oracle-layer.v1.schema.json"
    )
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(metadata)
    _validate_path_oracle_layer(metadata, array_names)

    unsupported_convention = {
        **metadata,
        "coefficient_convention": "arbitrary_complex_amplitude_v1",
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(unsupported_convention)
    with pytest.raises(ValueError, match="coefficient_convention is unsupported"):
        _validate_path_oracle_layer(unsupported_convention, array_names)

    repeated_link_groups = {
        **metadata,
        "link_ids": [metadata["link_ids"][0], metadata["link_ids"][0]],
        "solve_status": [metadata["solve_status"][0], metadata["solve_status"][0]],
    }
    jsonschema.Draft202012Validator(schema).validate(repeated_link_groups)
    _validate_path_oracle_layer(repeated_link_groups, array_names)

    extra_metadata = {**metadata, "unrecognized": True}
    with pytest.raises(ValueError, match="metadata fields"):
        _validate_path_oracle_layer(extra_metadata, array_names)
    with pytest.raises(ValueError, match="arrays do not match"):
        _validate_path_oracle_layer(metadata, array_names - {"delay_s"})
    with pytest.raises(ValueError, match="arrays do not match"):
        _validate_path_oracle_layer(metadata, array_names | {"unexpected"})


def test_episode_bundle_roundtrip_accepts_repeated_path_link_ids_by_port(tmp_path):
    bundle = _bundle()
    paths = PathOracleTable(
        link_ids=("gNB-1-to-UE-1", "gNB-1-to-UE-1"),
        frame_indices=np.asarray([0, 0]),
        rx_port_indices=np.asarray([0, 0]),
        tx_port_indices=np.asarray([0, 1]),
        path_offsets=np.asarray([0, 1, 2]),
        solve_status=(PathSolveStatus.SOLVED, PathSolveStatus.SOLVED),
        delay_s=np.asarray([10e-9, 11e-9]),
        complex_gain=np.asarray([0.01 + 0.02j, 0.02 + 0.01j]),
        interaction_order=np.asarray([1, 1]),
        interaction_offsets=np.asarray([0, 1, 2]),
        interaction_types=np.asarray([1, 1]),
        object_ids=np.asarray([7, 7]),
        primitive_ids=np.asarray([3, 4]),
        interaction_vertices_m=np.asarray([[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]]),
        interaction_vertex_frame="test_scene_xyz_m",
    )
    stored = write_episode_bundle(tmp_path, replace(bundle, path_oracle=paths))
    loaded = read_episode_bundle(tmp_path, stored)

    assert loaded.path_oracle.link_ids == paths.link_ids
    np.testing.assert_array_equal(loaded.path_oracle.tx_port_indices, [0, 1])


def test_episode_bundle_roundtrips_split_bound_direct_path_ablation(tmp_path):
    bundle = _bundle()
    paths = PathOracleTable(
        link_ids=("gNB-1-to-UE-1", "gNB-1-to-UE-1"),
        frame_indices=np.asarray([0, 0]),
        rx_port_indices=np.asarray([0, 0]),
        tx_port_indices=np.asarray([0, 1]),
        path_offsets=np.asarray([0, 2, 2]),
        solve_status=(PathSolveStatus.SOLVED, PathSolveStatus.NO_PATH),
        delay_s=np.asarray([10e-9, 20e-9]),
        complex_gain=np.asarray([0.7 + 0.2j, 0.1 - 0.3j]),
        interaction_order=np.asarray([0, 1]),
    )
    policy = SplitPolicy(
        seed="episode-ablation-store",
        split_fractions={"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3},
    )
    subjects = [
        SplitSubject(
            subject_id="episode-1",
            parent_type=SplitParentType.NETWORK_EPISODE,
            source_family="rt.city",
        ),
        *[
            SplitSubject(
                subject_id=f"independent-{index}",
                parent_type=SplitParentType.NETWORK_EPISODE,
                source_family="rt.city",
            )
            for index in range(2)
        ],
    ]
    registry = bind_split_children(
        make_split_policy_registry(subjects, policy=policy),
        [
            SplitChildBinding("base-view", "episode-1", "counterfactual_base"),
            SplitChildBinding("minus-direct-view", "episode-1", "counterfactual_math_ablation"),
        ],
    )
    ablation = remove_direct_path_component(
        paths,
        frequency_offset_hz=bundle.episode.subcarrier_offsets_hz,
        carrier_hz=bundle.episode.carrier_hz,
        registry=registry,
        parent_subject_id="episode-1",
        base_child_id="base-view",
        ablated_child_id="minus-direct-view",
    )
    tampered_ablation = replace(ablation, full_cfr=ablation.full_cfr + 1.0)
    with pytest.raises(ValueError, match="do not replay"):
        replace(bundle, path_oracle=paths, direct_path_ablations=(tampered_ablation,))
    wrong_carrier_ablation = replace(ablation, carrier_hz=bundle.episode.carrier_hz + 1.0)
    with pytest.raises(ValueError, match="carrier differs from its parent episode"):
        replace(bundle, path_oracle=paths, direct_path_ablations=(wrong_carrier_ablation,))
    with pytest.raises(ValueError, match="child IDs must be unique within the bundle"):
        replace(bundle, path_oracle=paths, direct_path_ablations=(ablation, ablation))
    bundle = replace(bundle, path_oracle=paths, direct_path_ablations=(ablation,))

    stored = write_episode_bundle(tmp_path, bundle)
    loaded = read_episode_bundle(tmp_path, stored)

    assert len(loaded.direct_path_ablations) == 1
    actual = loaded.direct_path_ablations[0]
    assert actual.path_table_sha256 == ablation.path_table_sha256
    assert actual.split == ablation.split
    assert actual.carrier_hz == bundle.episode.carrier_hz
    assert actual.link_ids == ("gNB-1-to-UE-1", "gNB-1-to-UE-1")
    assert actual.removed_direct_path_count.tolist() == [1, 0]
    np.testing.assert_array_equal(actual.frame_indices, [0, 0])
    np.testing.assert_array_equal(actual.frequency_offset_hz, bundle.episode.subcarrier_offsets_hz)
    np.testing.assert_array_equal(actual.full_cfr, ablation.full_cfr)
    np.testing.assert_array_equal(actual.without_direct_cfr, ablation.without_direct_cfr)
    manifest = json.loads((tmp_path / stored.relative_path).read_text())
    assert "counterfactual_ablations" in {item["layer"] for item in manifest["layers"]}

    legacy_manifest = json.loads(json.dumps(manifest))
    reference = next(
        item for item in legacy_manifest["layers"] if item["layer"] == "counterfactual_ablations"
    )
    legacy_metadata, legacy_arrays = _read_object(tmp_path, reference)
    legacy_metadata["schema"] = "umcd.direct_path_ablation.v1"
    legacy_metadata.pop("carrier_hz")
    legacy_payload = _pack_npz(legacy_metadata, **legacy_arrays)
    legacy_digest = hashlib.sha256(legacy_payload).hexdigest()
    legacy_object_path = (
        tmp_path
        / "objects"
        / "counterfactual_ablations"
        / legacy_digest[:2]
        / f"{legacy_digest}.npz"
    )
    legacy_object_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_object_path.write_bytes(legacy_payload)
    reference["path"] = str(legacy_object_path.relative_to(tmp_path))
    reference["sha256"] = legacy_digest
    manifest_core = {
        key: legacy_manifest[key] for key in ("schema", "schema_version", "episode_id", "layers")
    }
    legacy_manifest["bundle_sha256"] = hashlib.sha256(_json_bytes(manifest_core)).hexdigest()
    legacy_manifest_path = tmp_path / "bundles" / "legacy-ablation-v1.json"
    legacy_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_manifest_path.write_bytes(_json_bytes(legacy_manifest))
    legacy_loaded = read_episode_bundle(tmp_path, str(legacy_manifest_path.relative_to(tmp_path)))
    legacy_ablation = legacy_loaded.direct_path_ablations[0]
    assert legacy_ablation.carrier_hz is None
    np.testing.assert_array_equal(legacy_ablation.full_cfr, ablation.full_cfr)
    with pytest.raises(ValueError, match="read-only"):
        _direct_path_ablation_payload(legacy_ablation)

    jsonschema = pytest.importorskip("jsonschema")
    schema_path = Path(__file__).resolve().parents[2] / "datasets/schema/umcd-bundle-v1.schema.json"
    schema = json.loads(schema_path.read_text())
    jsonschema.Draft202012Validator(schema).validate(manifest)


def test_episode_metadata_schema_covers_lineage_and_legacy_payloads():
    jsonschema = pytest.importorskip("jsonschema")
    episode = replace(
        _dynamic_episode(),
        split_leakage_relations=(
            ("asset_family", "asset:mesh-sha256-123"),
            ("site", "site:city-1"),
            ("scene_realization", "scene:odbl-etoile-v1"),
        ),
    )
    payload = _episode_payload(episode)
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata_json"].tobytes())

    schema_path = (
        Path(__file__).resolve().parents[2] / "datasets/schema/umcd-episode-metadata.v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(metadata)

    legacy_metadata = dict(metadata)
    legacy_metadata.pop("split_leakage_relations")
    validator.validate(legacy_metadata)

    malformed = dict(metadata)
    malformed["split_leakage_relations"] = [{"namespace": "unrecognized", "identity": "unsafe"}]
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(malformed)


def test_episode_bundle_rejects_tampered_content_and_immutable_id_reuse(tmp_path):
    bundle = _bundle()
    stored = write_episode_bundle(tmp_path, bundle)
    with pytest.raises(FileExistsError):
        write_episode_bundle(
            tmp_path,
            EpisodeBundle(
                bundle.episode,
                bundle.path_oracle,
                (
                    PhysicalChannelView(
                        **{
                            **{
                                name: getattr(bundle.channel_views[0], name)
                                for name in (
                                    "view_id",
                                    "link_id",
                                    "channel",
                                    "frame_times_s",
                                    "rx_port_ids",
                                    "tx_port_ids",
                                    "carrier_hz",
                                    "subcarrier_offsets_hz",
                                    "delay_reference",
                                    "power_scale",
                                )
                            },
                            "channel": np.zeros_like(bundle.channel_views[0].channel),
                        }
                    ),
                ),
                bundle.targets,
            ),
        )
    manifest = json.loads((tmp_path / stored.relative_path).read_text())
    path_ref = next(item for item in manifest["layers"] if item["layer"] == "oracle_paths")
    path_object = tmp_path / path_ref["path"]
    path_object.write_bytes(path_object.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="content hash"):
        read_episode_bundle(tmp_path, stored)


def test_provenance_record_snapshots_nested_configuration_and_checks_its_hash():
    material_config = {"wall": {"permittivity": 4.2, "roughness": [0.1, 0.2]}}
    record = _provenance_record()
    original = UMCDRunProvenance(
        record_id="material-snapshot",
        subject_layer="oracle_paths",
        source_family="F03-RT-STATIC",
        producer="test",
        package_source_sha256="c" * 64,
        software_versions={"sionna-rt": "2.2.0"},
        backend={"backend": "llvm"},
        scene_asset_sha256={},
        seed=None,
        physical_config={},
        solver_config={},
        material_config=material_config,
    )
    material_config["wall"]["permittivity"] = 99.0
    assert original.material_config["wall"]["permittivity"] == 4.2
    assert UMCDRunProvenance.from_dict(record.to_dict()).to_dict() == record.to_dict()
    tampered = record.to_dict()
    tampered["solver_config"]["max_depth"] = 99
    with pytest.raises(ValueError, match="hash is invalid"):
        UMCDRunProvenance.from_dict(tampered)


def test_provenance_lineage_claims_are_tiered_hash_bound_and_schema_valid(tmp_path):
    claims = (
        LineageStatusClaim(status="proposed", scope="city43-phase0", evidence_sha256={}),
        LineageStatusClaim(
            status="synthetic_debug",
            scope="network-phy-composer",
            evidence_sha256={"regression_report_sha256": "d" * 64},
        ),
        LineageStatusClaim(
            status="native_verified",
            scope="city43-phase0",
            evidence_sha256={
                "native_run_manifest_sha256": "e" * 64,
                "numeric_shape_gauge_report_sha256": "f" * 64,
            },
        ),
    )
    record = replace(_provenance_record(), lineage_claims=claims)
    payload = record.to_dict()
    assert payload["schema"] == "umcd.run_provenance.v2"
    assert UMCDRunProvenance.from_dict(payload).to_dict() == payload

    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[2] / "datasets/schema/umcd-run-provenance.v2.schema.json"
    )
    schema = json.loads(schema_path.read_text())
    jsonschema.Draft202012Validator(schema).validate(payload)
    missing_evidence = json.loads(json.dumps(payload))
    missing_evidence["lineage_claims"][1]["evidence_sha256"] = {}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(missing_evidence)

    bundle = replace(_bundle(), provenance_records=(record,))
    stored = write_episode_bundle(tmp_path, bundle)
    loaded = read_episode_bundle(tmp_path, stored)
    assert loaded.provenance_records[0].to_dict() == payload


@pytest.mark.parametrize("subject_layer", ("scenario_gbsm", "counterfactual_ablations"))
def test_provenance_schema_admits_provenance_for_all_episode_bundle_layers(subject_layer):
    record = replace(
        _provenance_record(),
        subject_layer=subject_layer,
        lineage_claims=(),
    )
    payload = record.to_dict()
    assert UMCDRunProvenance.from_dict(payload).to_dict() == payload

    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[2] / "datasets/schema/umcd-run-provenance.v2.schema.json"
    )
    schema = json.loads(schema_path.read_text())
    jsonschema.Draft202012Validator(schema).validate(payload)


@pytest.mark.parametrize(
    "status,evidence",
    [
        ("scenario_qualified", {"native_run_manifest_sha256": "a" * 64}),
        ("geometry_qualified", {"convergence_audit_sha256": "a" * 64}),
        ("receiver_qualified", {"receiver_contract_sha256": "a" * 64}),
        ("external_tested", {"external_test_report_sha256": "not-a-digest"}),
    ],
)
def test_lineage_claim_rejects_incomplete_or_invalid_evidence(status, evidence):
    with pytest.raises(ValueError):
        LineageStatusClaim(status=status, scope="task", evidence_sha256=evidence)


def test_lineage_evidence_audit_verifies_exact_payload_roster_and_schema():
    evidence = b"independent numeric-shape-gauge report\n"
    digest = hashlib.sha256(evidence).hexdigest()
    claims = (
        LineageStatusClaim(
            status="synthetic_debug",
            scope="network-phy",
            evidence_sha256={"regression_report_sha256": digest},
        ),
    )
    report = audit_lineage_evidence_payloads(claims, {digest: evidence})
    assert report["status"] == "EVIDENCE_BYTES_VERIFIED"
    assert report["scientific_status"] == "NOT_QUALIFIED"
    assert report["verified_evidence_bytes"] == len(evidence)

    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets/schema/umcd-lineage-evidence-audit.v2.schema.json"
    )
    jsonschema.Draft202012Validator(json.loads(schema_path.read_text())).validate(report)


def test_lineage_evidence_audit_rejects_missing_extra_tampered_and_oversized_payloads():
    evidence = b"qualification report"
    digest = hashlib.sha256(evidence).hexdigest()
    claim = LineageStatusClaim(
        status="synthetic_debug",
        scope="task",
        evidence_sha256={"regression_report_sha256": digest},
    )
    claims = (claim,)
    with pytest.raises(ValueError, match="roster"):
        audit_lineage_evidence_payloads(claims, {})
    with pytest.raises(ValueError, match="roster"):
        audit_lineage_evidence_payloads(claims, {digest: evidence, "0" * 64: b"extra"})
    with pytest.raises(ValueError, match="does not match digest"):
        audit_lineage_evidence_payloads(claims, {digest: b"tampered"})
    with pytest.raises(ValueError, match="exceed max_total_bytes"):
        audit_lineage_evidence_payloads(claims, {digest: evidence}, max_total_bytes=4)


def test_lineage_evidence_audit_reports_proposed_claim_without_evidence():
    claim = LineageStatusClaim(status="proposed", scope="planned-task", evidence_sha256={})
    report = audit_lineage_evidence_payloads((claim,), {})
    assert report["status"] == "NO_EVIDENCE_REQUIRED"
    assert report["scientific_status"] == "NOT_QUALIFIED"
    assert report["verified_evidence_sha256"] == []


def test_lineage_evidence_file_audit_streams_stable_files_under_root(tmp_path):
    evidence = b"large-enough-evidence\n" * 8192
    digest = hashlib.sha256(evidence).hexdigest()
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    (evidence_root / "audit.bin").write_bytes(evidence)
    claim = LineageStatusClaim(
        status="synthetic_debug",
        scope="task",
        evidence_sha256={"regression_report_sha256": digest},
    )

    report = audit_lineage_evidence_files(
        (claim,), {digest: "audit.bin"}, evidence_root=evidence_root
    )
    assert report["status"] == "EVIDENCE_BYTES_VERIFIED"
    assert report["verified_evidence_bytes"] == len(evidence)

    with pytest.raises(ValueError, match="max_total_bytes"):
        audit_lineage_evidence_files(
            (claim,),
            {digest: "audit.bin"},
            evidence_root=evidence_root,
            max_total_bytes=len(evidence) - 1,
        )
    with pytest.raises(ValueError, match="beneath evidence_root"):
        audit_lineage_evidence_files(
            (claim,), {digest: "../audit.bin"}, evidence_root=evidence_root
        )

    (evidence_root / "alias.bin").symlink_to(evidence_root / "audit.bin")
    with pytest.raises(OSError):
        audit_lineage_evidence_files((claim,), {digest: "alias.bin"}, evidence_root=evidence_root)

    os.link(evidence_root / "audit.bin", evidence_root / "hardlink.bin")
    with pytest.raises(ValueError, match="single-link"):
        audit_lineage_evidence_files(
            (claim,), {digest: "hardlink.bin"}, evidence_root=evidence_root
        )


def test_provenance_reader_rejects_boolean_schema_version_even_when_one():
    payload = _provenance_record().to_dict()
    payload["schema_version"] = True
    with pytest.raises(ValueError, match="schema_version must be an integer"):
        UMCDRunProvenance.from_dict(payload)
