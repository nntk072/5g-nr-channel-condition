from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from datasets.channel_view import (
    ChannelDelayReference,
    ChannelPowerScale,
    PhysicalChannelView,
)
from datasets.episode_store import EpisodeBundle, write_episode_bundle
from datasets.network import (
    AntennaArraySpec,
    DirectedLink,
    LinkDirection,
    LinkRole,
    NetworkEpisode,
    NetworkNode,
    NodeRole,
)
from datasets.nr import NRChannelRecord, NRObservation, NRTrajectory
from datasets.nr_store import write_nr_trajectory
from datasets.path_oracle import PathOracleTable, PathSolveStatus
from datasets.split_policy import (
    SPLIT_POLICY_SIGNATURE_NAMESPACE,
    LeakageNamespace,
    LeakageRelation,
    SplitPolicy,
    freeze_split_policy_registry,
    make_split_policy_registry,
    network_episode_subject,
)
from datasets.targets import TargetDefinition, TargetValidity, TargetValue, target_definition_digest
from receivers.nr import NRChannelEstimate, NRPUSCHConfig
from scripts.audit_umcd_corpus import (
    CORPUS_INVENTORY_SIGNATURE_NAMESPACE,
    NEAR_DUPLICATE_DISPOSITION_SIGNATURE_NAMESPACE,
    SCENE_ROSTER_INVENTORY_SIGNATURE_NAMESPACE,
    _episode_antenna_config_id,
    _episode_topology_id,
    _validate_report,
    audit_inventory,
    main,
)

_SIGNER = "umcd-corpus-audit-test"


def _sign_inventory_split_registry(
    inventory_path: Path, tmp_path: Path, *, namespace: str = SPLIT_POLICY_SIGNATURE_NAMESPACE
) -> Path:
    ssh_keygen = shutil.which("ssh-keygen")
    if ssh_keygen is None:
        pytest.skip("OpenSSH is unavailable")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    registry_path = inventory_path.parent / inventory["split_registry"]["path"]
    private_key = tmp_path / "corpus_audit_signer"
    subprocess.run(
        [ssh_keygen, "-q", "-t", "ed25519", "-N", "", "-f", str(private_key)],
        check=True,
        capture_output=True,
        text=True,
    )
    public_key = private_key.with_suffix(".pub").read_text(encoding="utf-8").strip()
    allowed_signers = tmp_path.parent / f"{tmp_path.name}-allowed_signers"
    allowed_signers.write_text(
        f'{_SIGNER} namespaces="{SPLIT_POLICY_SIGNATURE_NAMESPACE}" {public_key}\n',
        encoding="utf-8",
    )
    subprocess.run(
        [ssh_keygen, "-Y", "sign", "-n", namespace, "-f", str(private_key), str(registry_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    inventory["external_lock_attestation"] = {
        "status": "provided",
        "signature_path": f"{registry_path.name}.sig",
        "signer_id": _SIGNER,
    }
    inventory_path.write_text(json.dumps(inventory, sort_keys=True), encoding="utf-8")
    return allowed_signers


def _sign_corpus_inventory(inventory_path: Path, tmp_path: Path) -> Path:
    ssh_keygen = shutil.which("ssh-keygen")
    if ssh_keygen is None:
        pytest.skip("OpenSSH is unavailable")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    namespace = (
        SCENE_ROSTER_INVENTORY_SIGNATURE_NAMESPACE
        if inventory.get("schema") == "umcd.corpus_audit_inventory.v4"
        else CORPUS_INVENTORY_SIGNATURE_NAMESPACE
    )
    inventory["inventory_attestation"] = {
        "status": "provided",
        "signature_path": f"{inventory_path.name}.sig",
        "signer_id": _SIGNER,
    }
    inventory_path.write_text(json.dumps(inventory, sort_keys=True), encoding="utf-8")
    private_key = tmp_path / "corpus_inventory_signer"
    subprocess.run(
        [ssh_keygen, "-q", "-t", "ed25519", "-N", "", "-f", str(private_key)],
        check=True,
        capture_output=True,
        text=True,
    )
    public_key = private_key.with_suffix(".pub").read_text(encoding="utf-8").strip()
    allowed_signers = tmp_path.parent / f"{tmp_path.name}-inventory-allowed_signers"
    allowed_signers.write_text(
        f'{_SIGNER} namespaces="{namespace}" {public_key}\n',
        encoding="utf-8",
    )
    subprocess.run(
        [
            ssh_keygen,
            "-Y",
            "sign",
            "-n",
            namespace,
            "-f",
            str(private_key),
            str(inventory_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return allowed_signers


def _sign_near_duplicate_dispositions(
    inventory_path: Path,
    review: dict,
    inventory_sha256: str,
    tmp_path: Path,
    *,
    outcome: str = "legitimate_similarity",
    replace_candidate_ids: list[str] | None = None,
) -> tuple[str, str, str, Path]:
    ssh_keygen = shutil.which("ssh-keygen")
    if ssh_keygen is None:
        pytest.skip("OpenSSH is unavailable")
    reviewer_id = "independent-near-duplicate-reviewer"
    candidate_ids = (
        replace_candidate_ids
        if replace_candidate_ids is not None
        else [item["candidate_id"] for item in review["candidates"]]
    )
    evidence_by_candidate = {}
    for candidate_id in candidate_ids:
        relative = f"review-evidence/{candidate_id}.json"
        evidence_path = inventory_path.parent / relative
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"candidate_id": candidate_id, "review_note": "geometry and provenance reviewed"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        evidence_path.write_bytes(payload)
        evidence_by_candidate[candidate_id] = {
            "path": relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    manifest = {
        "schema": "umcd.near_duplicate_dispositions.v2",
        "inventory_sha256": inventory_sha256,
        "candidate_set_sha256": review["candidate_set_sha256"],
        "reviewer_id": reviewer_id,
        "dispositions": [
            {
                "candidate_id": candidate_id,
                "outcome": outcome,
                "rationale": "Reviewed geometry and acquisition provenance for this pair.",
                "evidence_files": [evidence_by_candidate[candidate_id]],
            }
            for candidate_id in candidate_ids
        ],
    }
    manifest_path = inventory_path.parent / "near-duplicate-dispositions.json"
    manifest_path.write_bytes(
        (
            json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()
    )
    private_key = tmp_path / "near_duplicate_reviewer_key"
    subprocess.run(
        [ssh_keygen, "-q", "-t", "ed25519", "-N", "", "-f", str(private_key)],
        check=True,
        capture_output=True,
        text=True,
    )
    public_key = private_key.with_suffix(".pub").read_text(encoding="utf-8").strip()
    allowed_signers = tmp_path.parent / f"{tmp_path.name}-reviewers-allowed_signers"
    allowed_signers.write_text(
        f'{reviewer_id} namespaces="{NEAR_DUPLICATE_DISPOSITION_SIGNATURE_NAMESPACE}" '
        f"{public_key}\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            ssh_keygen,
            "-Y",
            "sign",
            "-n",
            NEAR_DUPLICATE_DISPOSITION_SIGNATURE_NAMESPACE,
            "-f",
            str(private_key),
            str(manifest_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return (
        manifest_path.name,
        f"{manifest_path.name}.sig",
        reviewer_id,
        allowed_signers,
    )


def _bundle(episode_id: str, asset_family: str) -> EpisodeBundle:
    arrays = (
        AntennaArraySpec(
            "gNB-array",
            np.asarray([[0.0, 0.0, 0.0]]),
            ("V",),
            np.eye(3),
            ("gNB-V",),
            np.ones((1, 1), dtype=np.complex128),
        ),
        AntennaArraySpec(
            "UE-array",
            np.asarray([[0.0, 0.0, 0.0]]),
            ("V",),
            np.eye(3),
            ("UE-V",),
            np.ones((1, 1), dtype=np.complex128),
        ),
    )
    nodes = (
        NetworkNode("gNB-1", NodeRole.GNB, "gNB-array", (0.0, 0.0, 10.0), 1),
        NetworkNode("UE-1", NodeRole.UE, "UE-array", (20.0, 0.0, 1.5), 1),
    )
    episode = NetworkEpisode(
        episode_id,
        asset_family,
        "F03-RT-STATIC",
        3.5e9,
        arrays,
        nodes,
        (
            DirectedLink(
                "gNB-1-to-UE-1",
                "gNB-1",
                "UE-1",
                (0,),
                (0,),
                LinkDirection.DOWNLINK,
                LinkRole.SERVING,
            ),
        ),
        (0.0,),
        np.asarray([-15_000.0, 0.0, 15_000.0]),
    )
    path_oracle = PathOracleTable(
        link_ids=("gNB-1-to-UE-1",),
        frame_indices=np.asarray([0]),
        rx_port_indices=np.asarray([0]),
        tx_port_indices=np.asarray([0]),
        path_offsets=np.asarray([0, 1]),
        solve_status=(PathSolveStatus.SOLVED,),
        delay_s=np.asarray([10e-9]),
        complex_gain=np.asarray([0.01 + 0.02j]),
        interaction_order=np.asarray([1]),
    )
    channel = PhysicalChannelView(
        view_id="oracle-cfr-v1",
        link_id="gNB-1-to-UE-1",
        channel=np.ones((1, 1, 1, 3), dtype=np.complex64),
        frame_times_s=episode.frame_times_s,
        rx_port_ids=("UE-V",),
        tx_port_ids=("gNB-V",),
        carrier_hz=episode.carrier_hz,
        subcarrier_offsets_hz=episode.subcarrier_offsets_hz,
        delay_reference=ChannelDelayReference.ABSOLUTE_PROPAGATION,
        power_scale=ChannelPowerScale.ABSOLUTE,
    )
    fields = {
        "semantic": "geometric_direct_path",
        "units": "boolean",
        "source": "path_geometry_oracle",
        "reference": "3d_segment_visibility",
        "aggregation_rule": "one visibility value per link and frame",
        "definition_version": "test.v1",
    }
    definition = TargetDefinition(
        **fields,
        definition_sha256=target_definition_digest(**fields),
    )
    target = TargetValue(
        "gNB-1-to-UE-1",
        definition,
        np.asarray([1.0]),
        np.asarray([True]),
        (TargetValidity.VALID,),
        "geometry_verified",
    )
    return EpisodeBundle(episode, path_oracle, (channel,), (target,))


def _nr_trajectory(trajectory_id: str) -> NRTrajectory:
    records = []
    for index in range(2):
        estimate = NRChannelEstimate(
            h_hat=np.zeros((1, 1, 1, 1, 1, 14, 12), dtype=np.complex64),
            error_variance=np.zeros((1, 1, 1, 1, 1, 14, 12), dtype=np.float32),
            pilot_mask=np.zeros((1, 1, 14, 12), dtype=np.bool_),
            metadata={"receiver": "corpus-audit-test"},
        )
        observation = NRObservation(
            sample_id=f"{trajectory_id}-sample-{index}",
            received_grid=np.zeros((1, 1, 1, 14, 12), dtype=np.complex64),
            channel_estimate=estimate,
            fft_bin_indices=tuple(range(12)),
            receiver_config=NRPUSCHConfig(n_size_bwp=1, subcarrier_spacing_khz=30),
        )
        targets = {"los": np.asarray(True)} if index == 0 else {"quality": np.asarray(1.0)}
        records.append(
            NRChannelRecord(
                observation=observation,
                targets=targets,
                target_masks={},
                target_definitions={
                    name: TargetDefinition.versioned(
                        semantic=f"corpus audit test target {name}",
                        units="fixture value",
                        source="corpus audit test fixture",
                        reference="fixed synthetic test value",
                        aggregation_rule="one scalar per trajectory record",
                        definition_version="tests.corpus-audit.target.v1",
                    )
                    for name in targets
                },
                provenance={"source": "corpus-audit-fixture"},
            )
        )
    return NRTrajectory(trajectory_id, (0.0, 0.1), tuple(records))


def _fixture(
    tmp_path: Path,
    *,
    stratified: bool = False,
    scene_roster: bool = False,
    zero_channels: bool = False,
    incompatible_channel_context: bool = False,
) -> Path:
    store = tmp_path / "store"
    bundles = tuple(_bundle(f"episode-{index}", f"asset-family-{index}") for index in range(3))
    if incompatible_channel_context:
        changed = bundles[1]
        offsets = np.asarray([-30_000.0, 0.0, 30_000.0])
        changed_episode = replace(changed.episode, subcarrier_offsets_hz=offsets)
        changed_views = tuple(
            replace(view, subcarrier_offsets_hz=offsets)
            for view in changed.channel_views
        )
        bundles = (bundles[0], replace(changed, episode=changed_episode, channel_views=changed_views), bundles[2])
    if zero_channels:
        bundles = tuple(
            replace(
                bundle,
                channel_views=tuple(
                    replace(view, channel=np.zeros_like(view.channel))
                    for view in bundle.channel_views
                ),
            )
            for bundle in bundles
        )
    stored_bundles = tuple(write_episode_bundle(store, bundle) for bundle in bundles)
    scene_relations = (
        {
            bundle.episode.episode_id: (
                LeakageRelation(
                    LeakageNamespace.SCENE_REALIZATION,
                    f"scene-{bundle.episode.episode_id}",
                ),
            )
            for bundle in bundles
        }
        if stratified
        else {}
    )
    subjects = [
        network_episode_subject(
            bundle.episode,
            relations=scene_relations.get(bundle.episode.episode_id, ()),
        )
        for bundle in bundles
    ]
    registry = make_split_policy_registry(
        subjects,
        policy=SplitPolicy(
            seed="corpus-audit-test",
            split_fractions={"train": 0.7, "validation": 0.15, "test": 0.15},
        ),
    )
    registry_path = tmp_path / "split-registry.json"
    freeze_split_policy_registry(registry_path, registry)
    manifest_rows = []
    for bundle, stored in zip(bundles, stored_bundles, strict=True):
        manifest_path = store / stored.relative_path
        manifest_rows.append(
            {
                "episode_id": bundle.episode.episode_id,
                "manifest": stored.relative_path,
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                **({"band_id": "n78-fixture"} if stratified else {}),
                "relations": [
                    relation.to_dict()
                    for relation in scene_relations.get(bundle.episode.episode_id, ())
                ],
            }
        )
    target_id = "gNB-1-to-UE-1::geometric_direct_path"
    inventory = {
        "schema": (
            "umcd.corpus_audit_inventory.v4"
            if scene_roster
            else "umcd.corpus_audit_inventory.v3"
            if stratified
            else "umcd.corpus_audit_inventory.v2"
        ),
        "frozen": True,
        "inventory_id": "synthetic-fixture-only",
        "store": "store",
        "split_registry": {
            "path": registry_path.name,
            "sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        },
        "bundles": manifest_rows,
        "expected_coverage": [],
        "duplicate_threshold": 0.999,
        "legal_assessment": {"status": "pending"},
        "external_lock_attestation": {"status": "missing"},
    }
    if stratified:
        inventory["inventory_attestation"] = {"status": "missing"}
        inventory["band_registry"] = [
            {"band_id": "n78-fixture", "lower_hz": 3.3e9, "upper_hz": 3.8e9}
        ]
        inventory["target_task_bindings"] = [
            {"target_id": target_id, "task_id": "direct-path-detection"}
        ]
        inventory["expected_coverage"] = [
            {
                "split_id": registry.split_for_parent(bundle.episode.episode_id),
                "family_id": bundle.episode.family_id,
                "target_id": target_id,
                "minimum_independent_groups": 1,
                "scene_id": f"scene-{bundle.episode.episode_id}",
                "band_id": "n78-fixture",
                "topology_id": _episode_topology_id(bundle),
                "antenna_config_id": _episode_antenna_config_id(bundle),
                "task_id": "direct-path-detection",
            }
            for bundle in bundles
        ]
    if scene_roster:
        inventory["scene_inventory"] = [
            {
                "scene_id": f"scene-{bundle.episode.episode_id}",
                "parent_episode_ids": [bundle.episode.episode_id],
            }
            for bundle in bundles
        ]
    if not stratified:
        inventory["expected_coverage"] = [
            {
                "split_id": registry.split_for_parent(bundle.episode.episode_id),
                "family_id": bundle.episode.family_id,
                "target_id": target_id,
                "minimum_independent_groups": 1,
            }
            for bundle in bundles
        ]
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory, sort_keys=True), encoding="utf-8")
    return inventory_path


def test_audit_report_is_deterministic_hash_bound_and_nonqualifying(tmp_path):
    inventory_path = _fixture(tmp_path)

    first = audit_inventory(inventory_path)
    second = audit_inventory(inventory_path)

    assert first == second
    assert first["qualification"] == "not_claimed"
    assert first["schema"] == "umcd.corpus_audit_report.v9"
    assert first["external_gates"]["legal_assessment"]["status"] == "external_unverified"
    assert first["external_gates"]["external_lock_attestation"]["status"] == "external_unverified"
    assert first["checks"]["bundle_store_integrity"]["bundle_count"] == 3
    assert first["checks"]["split_lineage"]["child_rosters"] == {}
    assert all(
        cell["independent_group_count"] >= cell["minimum_independent_group_count"]
        for cell in first["checks"]["target_coverage"]["cells"]
    )
    assert first["checks"]["near_duplicate_review"]["status"] == "review_required"
    core = {key: value for key, value in first.items() if key != "report_sha256"}
    assert (
        first["report_sha256"]
        == hashlib.sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


def test_corpus_audit_report_v9_schema_rejects_unknown_near_duplicate_status(tmp_path):
    report = audit_inventory(_fixture(tmp_path))
    malformed = json.loads(json.dumps(report))
    malformed["checks"]["near_duplicate_review"]["status"] = "skipped_but_clean"

    with pytest.raises(ValueError, match="corpus audit report violates"):
        _validate_report(malformed)


def test_zero_norm_cross_split_cfrs_mark_near_duplicate_review_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.audit_umcd_corpus as corpus_audit

    monkeypatch.setattr(
        corpus_audit,
        "_bundle_geometry_hash",
        lambda bundle: bundle.episode.episode_id,
    )
    report = audit_inventory(_fixture(tmp_path, zero_channels=True))
    review = report["checks"]["near_duplicate_review"]

    assert report["schema"] == "umcd.corpus_audit_report.v9"
    assert review["status"] == "incomplete_comparisons"
    assert review["candidate_count"] == 0
    assert review["cross_split_pair_count"] > 0
    assert review["skipped_comparison_count"] > 0


def test_incompatible_cfr_frequency_grids_are_reported_as_incomplete(
    tmp_path: Path,
) -> None:
    report = audit_inventory(_fixture(tmp_path, incompatible_channel_context=True))
    review = report["checks"]["near_duplicate_review"]

    assert review["status"] == "incomplete_comparisons"
    assert review["skipped_comparison_reasons"]["incompatible_channel_context"] > 0
    assert any(
        item["comparison_context_id_a"] != item["comparison_context_id_b"]
        and item["normalized_complex_inner_product_magnitude"] is None
        for item in review["candidates"]
    )


@pytest.mark.parametrize(
    ("outcome", "expected_status"),
    (
        ("legitimate_similarity", "adjudicated_no_leakage"),
        ("confirmed_leakage", "confirmed_cross_split_leakage"),
    ),
)
def test_signed_near_duplicate_dispositions_bind_exact_inventory_and_candidate_set(
    tmp_path: Path, outcome: str, expected_status: str
) -> None:
    inventory_path = _fixture(tmp_path)
    initial = audit_inventory(inventory_path)
    review = initial["checks"]["near_duplicate_review"]
    assert initial["schema"] == "umcd.corpus_audit_report.v9"
    assert review["candidate_set_sha256"]
    assert all(len(item["candidate_id"]) == 64 for item in review["candidates"])
    assert review["disposition_status"] == "not_provided"

    disposition_path, signature_path, reviewer_id, allowed_signers = (
        _sign_near_duplicate_dispositions(
            inventory_path, review, initial["inventory_sha256"], tmp_path, outcome=outcome
        )
    )
    report = audit_inventory(
        inventory_path,
        near_duplicate_dispositions=disposition_path,
        near_duplicate_dispositions_signature=signature_path,
        near_duplicate_reviewer_id=reviewer_id,
        trusted_near_duplicate_reviewers_path=allowed_signers,
    )
    reviewed = report["checks"]["near_duplicate_review"]

    assert reviewed["status"] == expected_status
    assert reviewed["disposition_status"] == "verified"
    assert reviewed["disposition_signer_id"] == reviewer_id
    assert reviewed["disposition_signature_sha256"]
    assert reviewed["disposition_allowed_signers_sha256"]
    assert (
        reviewed["disposition_signature_namespace"]
        == NEAR_DUPLICATE_DISPOSITION_SIGNATURE_NAMESPACE
    )
    assert len(reviewed["dispositions"]) == reviewed["candidate_count"]
    assert all(
        (inventory_path.parent / evidence["path"]).is_file()
        and len(evidence["sha256"]) == 64
        for row in reviewed["dispositions"]
        for evidence in row["evidence_files"]
    )


@pytest.mark.parametrize("damage", ["missing", "digest_mismatch"])
def test_signed_near_duplicate_dispositions_require_matching_evidence_bytes(
    tmp_path: Path, damage: str
) -> None:
    inventory_path = _fixture(tmp_path)
    initial = audit_inventory(inventory_path)
    review = initial["checks"]["near_duplicate_review"]
    disposition_path, signature_path, reviewer_id, allowed_signers = (
        _sign_near_duplicate_dispositions(
            inventory_path, review, initial["inventory_sha256"], tmp_path
        )
    )
    evidence_path = next((tmp_path / "review-evidence").glob("*.json"))
    if damage == "missing":
        evidence_path.unlink()
    else:
        evidence_path.write_bytes(b"tampered after reviewer signature")

    with pytest.raises(ValueError, match="near-duplicate evidence"):
        audit_inventory(
            inventory_path,
            near_duplicate_dispositions=disposition_path,
            near_duplicate_dispositions_signature=signature_path,
            near_duplicate_reviewer_id=reviewer_id,
            trusted_near_duplicate_reviewers_path=allowed_signers,
        )


def test_signed_near_duplicate_dispositions_reject_candidate_roster_drift(tmp_path: Path) -> None:
    inventory_path = _fixture(tmp_path)
    report = audit_inventory(inventory_path)
    review = report["checks"]["near_duplicate_review"]
    disposition_path, signature_path, reviewer_id, allowed_signers = (
        _sign_near_duplicate_dispositions(
            inventory_path,
            review,
            report["inventory_sha256"],
            tmp_path,
            replace_candidate_ids=[
                "f" * 64,
                *[item["candidate_id"] for item in review["candidates"][1:]],
            ],
        )
    )

    with pytest.raises(ValueError, match="cover exactly the frozen candidate roster"):
        audit_inventory(
            inventory_path,
            near_duplicate_dispositions=disposition_path,
            near_duplicate_dispositions_signature=signature_path,
            near_duplicate_reviewer_id=reviewer_id,
            trusted_near_duplicate_reviewers_path=allowed_signers,
        )


def test_signed_near_duplicate_dispositions_reject_inventory_digest_drift(tmp_path: Path) -> None:
    inventory_path = _fixture(tmp_path)
    report = audit_inventory(inventory_path)
    review = report["checks"]["near_duplicate_review"]
    disposition_path, signature_path, reviewer_id, allowed_signers = (
        _sign_near_duplicate_dispositions(
            inventory_path,
            review,
            "0" * 64,
            tmp_path,
        )
    )

    with pytest.raises(ValueError, match="different inventory digest"):
        audit_inventory(
            inventory_path,
            near_duplicate_dispositions=disposition_path,
            near_duplicate_dispositions_signature=signature_path,
            near_duplicate_reviewer_id=reviewer_id,
            trusted_near_duplicate_reviewers_path=allowed_signers,
        )


def test_signed_near_duplicate_reviewer_requires_distinct_upstream_keyring(tmp_path: Path) -> None:
    inventory_path = _fixture(tmp_path, stratified=True)
    upstream_keyring = _sign_corpus_inventory(inventory_path, tmp_path)
    report = audit_inventory(
        inventory_path,
        trusted_allowed_signers_path=upstream_keyring,
    )
    review = report["checks"]["near_duplicate_review"]
    disposition_path, signature_path, reviewer_id, reviewer_keyring = (
        _sign_near_duplicate_dispositions(
            inventory_path,
            review,
            report["inventory_sha256"],
            tmp_path,
        )
    )
    upstream_keyring.write_text(
        upstream_keyring.read_text(encoding="utf-8") + reviewer_keyring.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="keyring must be distinct"):
        audit_inventory(
            inventory_path,
            trusted_allowed_signers_path=upstream_keyring,
            near_duplicate_dispositions=disposition_path,
            near_duplicate_dispositions_signature=signature_path,
            near_duplicate_reviewer_id=reviewer_id,
            trusted_near_duplicate_reviewers_path=upstream_keyring,
        )


def test_stratified_inventory_v3_binds_derived_physical_strata_and_task(tmp_path):
    inventory_path = _fixture(tmp_path, stratified=True)

    report = audit_inventory(inventory_path)

    cells = report["checks"]["target_coverage"]["cells"]
    assert len(cells) == 3
    assert {cell["scene_id"] for cell in cells} == {
        "scene-episode-0",
        "scene-episode-1",
        "scene-episode-2",
    }
    assert {cell["band_id"] for cell in cells} == {"n78-fixture"}
    assert {cell["task_id"] for cell in cells} == {"direct-path-detection"}
    assert all(cell["topology_id"].startswith("topology:sha256:") for cell in cells)
    assert all(cell["antenna_config_id"].startswith("antenna:sha256:") for cell in cells)
    assert report["qualification"] == "not_claimed"


def test_scene_inventory_v4_reports_exact_whole_scene_holdout(tmp_path):
    inventory_path = _fixture(tmp_path, stratified=True, scene_roster=True)

    report = audit_inventory(inventory_path)

    scene_report = report["checks"]["split_lineage"]["whole_scene_holdout"]
    assert scene_report["status"] == "PASS_INPUT_SCOPED"
    assert scene_report["scene_count"] == 3
    assert scene_report["parent_subject_count"] == 3
    assert set(scene_report["scene_splits"]) == {
        "scene-episode-0",
        "scene-episode-1",
        "scene-episode-2",
    }
    assert report["qualification"] == "not_claimed"


def test_scene_inventory_v4_rejects_omitted_scene_parent(tmp_path):
    inventory_path = _fixture(tmp_path, stratified=True, scene_roster=True)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["scene_inventory"].pop(0)
    inventory_path.write_text(json.dumps(inventory, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="scene memberships do not cover exactly"):
        audit_inventory(inventory_path)


def test_stratified_inventory_v3_rejects_carrier_outside_frozen_band(tmp_path):
    inventory_path = _fixture(tmp_path, stratified=True)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["band_registry"][0]["upper_hz"] = 3.4e9
    inventory_path.write_text(json.dumps(inventory, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="outside frozen band"):
        audit_inventory(inventory_path)


def test_stratified_inventory_signature_binds_exact_inventory_bytes(tmp_path):
    inventory_path = _fixture(tmp_path, stratified=True)
    allowed_signers = _sign_corpus_inventory(inventory_path, tmp_path)

    report = audit_inventory(
        inventory_path,
        trusted_allowed_signers_path=allowed_signers,
    )

    assert (
        report["external_gates"]["inventory_attestation"]["status"]
        == "signature_valid_against_supplied_trust_file"
    )
    assert report["external_gates"]["inventory_attestation"]["signer_id"] == _SIGNER


def test_scene_inventory_v4_signature_uses_versioned_purpose_namespace(tmp_path):
    inventory_path = _fixture(tmp_path, stratified=True, scene_roster=True)
    allowed_signers = _sign_corpus_inventory(inventory_path, tmp_path)

    report = audit_inventory(
        inventory_path,
        trusted_allowed_signers_path=allowed_signers,
    )

    attestation = report["external_gates"]["inventory_attestation"]
    assert attestation["status"] == "signature_valid_against_supplied_trust_file"
    assert attestation["signature_namespace"] == SCENE_ROSTER_INVENTORY_SIGNATURE_NAMESPACE
    assert report["checks"]["split_lineage"]["whole_scene_holdout"]["status"] == (
        "PASS_INPUT_SCOPED"
    )


def test_stratified_inventory_signature_rejects_post_signing_edits(tmp_path):
    inventory_path = _fixture(tmp_path, stratified=True)
    allowed_signers = _sign_corpus_inventory(inventory_path, tmp_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["inventory_id"] = "edited-after-signing"
    inventory_path.write_text(json.dumps(inventory, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="signature is invalid"):
        audit_inventory(
            inventory_path,
            trusted_allowed_signers_path=allowed_signers,
        )


def test_corpus_audit_rejects_target_cell_below_frozen_group_minimum(tmp_path):
    inventory_path = _fixture(tmp_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    for cell in inventory["expected_coverage"]:
        cell["minimum_independent_groups"] = 2
    inventory_path.write_text(json.dumps(inventory, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="minimum independent-group support"):
        audit_inventory(inventory_path)


def test_corpus_audit_v2_requires_frozen_group_minimum_in_every_target_cell(tmp_path):
    inventory_path = _fixture(tmp_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["expected_coverage"][0].pop("minimum_independent_groups")
    inventory_path.write_text(json.dumps(inventory, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="corpus inventory violates .*v2 schema"):
        audit_inventory(inventory_path)


def test_corpus_audit_reports_nr_target_support_bound_to_locked_parent(tmp_path):
    inventory_path = _fixture(tmp_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    store = tmp_path / inventory["store"]
    stored = write_nr_trajectory(store, _nr_trajectory("route-episode-0"))
    parent_episode_id = "episode-0"
    registry = json.loads((tmp_path / inventory["split_registry"]["path"]).read_text())
    parent_split = registry["subjects"][parent_episode_id]["split"]
    inventory["nr_trajectory_inventory"] = [
        {
            "trajectory_id": stored.trajectory_id,
            "parent_episode_id": parent_episode_id,
            "manifest": stored.manifest_relative_path,
            "manifest_sha256": hashlib.sha256(
                (store / stored.manifest_relative_path).read_bytes()
            ).hexdigest(),
        }
    ]
    inventory["nr_expected_coverage"] = [
        {"split_id": parent_split, "family_id": "F03-RT-STATIC", "target_id": "los"},
        {"split_id": parent_split, "family_id": "F03-RT-STATIC", "target_id": "quality"},
    ]
    inventory_path.write_text(json.dumps(inventory, sort_keys=True), encoding="utf-8")

    report = audit_inventory(inventory_path)

    support = report["checks"]["nr_trajectory_target_support"]
    assert support["status"] == "INPUT_SCOPED_NOT_RELEASE_QUALIFIED"
    assert support["trajectory_count"] == 1
    assert support["cells"] == [
        {
            "split_id": parent_split,
            "family_id": "F03-RT-STATIC",
            "target_id": "los",
            "trajectory_count": 1,
            "record_count": 2,
            "present_record_count": 1,
            "missing_record_count": 1,
            "element_count": 1,
            "selected_element_count": 1,
            "masked_element_count": 0,
            "implicit_all_valid_mask_record_count": 1,
        },
        {
            "split_id": parent_split,
            "family_id": "F03-RT-STATIC",
            "target_id": "quality",
            "trajectory_count": 1,
            "record_count": 2,
            "present_record_count": 1,
            "missing_record_count": 1,
            "element_count": 1,
            "selected_element_count": 1,
            "masked_element_count": 0,
            "implicit_all_valid_mask_record_count": 1,
        },
    ]


def test_corpus_audit_rejects_nr_trajectory_missing_from_frozen_inventory(tmp_path):
    inventory_path = _fixture(tmp_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    store = tmp_path / inventory["store"]
    write_nr_trajectory(store, _nr_trajectory("unlisted-route"))

    with pytest.raises(ValueError, match="absent from the frozen inventory"):
        audit_inventory(inventory_path)


def test_corpus_audit_keeps_explicit_zero_trajectory_coverage(tmp_path):
    inventory_path = _fixture(tmp_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    registry = json.loads((tmp_path / inventory["split_registry"]["path"]).read_text())
    split_id = registry["subjects"]["episode-0"]["split"]
    inventory["nr_trajectory_inventory"] = []
    inventory["nr_expected_coverage"] = [
        {"split_id": split_id, "family_id": "F03-RT-STATIC", "target_id": "los"}
    ]
    inventory_path.write_text(json.dumps(inventory, sort_keys=True), encoding="utf-8")

    support = audit_inventory(inventory_path)["checks"]["nr_trajectory_target_support"]

    assert support["trajectory_count"] == 0
    assert support["cells"][0]["missing_record_count"] == 0
    assert support["cells"][0]["element_count"] == 0


def test_cli_cryptographically_verifies_external_split_lock_attestation(tmp_path):
    inventory_path = _fixture(tmp_path)
    allowed_signers = _sign_inventory_split_registry(inventory_path, tmp_path)
    report_path = tmp_path / "reports" / "signed-audit.json"

    assert (
        main(
            [
                str(inventory_path),
                "--output",
                str(report_path),
                "--trusted-allowed-signers",
                str(allowed_signers),
            ]
        )
        == 1
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    attestation = report["external_gates"]["external_lock_attestation"]
    assert attestation["declared_status"] == "provided"
    assert attestation["status"] == "signature_valid_against_supplied_trust_file"
    assert report["checks"]["near_duplicate_review"]["status"] == "review_required"
    assert attestation["trust_file_custody"] == "not_established_by_this_audit"
    assert attestation["signer_id"] == _SIGNER
    assert attestation["signature_namespace"] == SPLIT_POLICY_SIGNATURE_NAMESPACE
    assert len(attestation["signature_sha256"]) == 64
    assert len(attestation["allowed_signers_sha256"]) == 64
    assert report["qualification"] == "not_claimed"


def test_audit_requires_external_trust_file_for_provided_split_lock_signature(tmp_path):
    inventory_path = _fixture(tmp_path)
    _sign_inventory_split_registry(inventory_path, tmp_path)

    with pytest.raises(ValueError, match="requires an independent allowed-signers file"):
        audit_inventory(inventory_path)


def test_audit_rejects_split_lock_signature_from_another_namespace(tmp_path):
    inventory_path = _fixture(tmp_path)
    allowed_signers = _sign_inventory_split_registry(
        inventory_path, tmp_path, namespace="unrelated-artifact-v1@github.com"
    )

    with pytest.raises(ValueError, match="signature is invalid or signer is not trusted"):
        audit_inventory(inventory_path, trusted_allowed_signers_path=allowed_signers)


def test_audit_rejects_symlinked_split_registry(tmp_path):
    inventory_path = _fixture(tmp_path)
    registry_path = (
        tmp_path / json.loads(inventory_path.read_text(encoding="utf-8"))["split_registry"]["path"]
    )
    target_path = tmp_path / "actual-split-registry.json"
    registry_path.rename(target_path)
    registry_path.symlink_to(target_path.name)

    with pytest.raises(ValueError, match="split registry path must not be a symlink"):
        audit_inventory(inventory_path)


def test_audit_rejects_symlinked_split_registry_signature(tmp_path):
    inventory_path = _fixture(tmp_path)
    allowed_signers = _sign_inventory_split_registry(inventory_path, tmp_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    signature_path = tmp_path / inventory["external_lock_attestation"]["signature_path"]
    target_path = tmp_path / "actual-split-registry.sig"
    signature_path.rename(target_path)
    signature_path.symlink_to(target_path.name)
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

    with pytest.raises(ValueError, match="split-lock signature path must not be a symlink"):
        audit_inventory(inventory_path, trusted_allowed_signers_path=allowed_signers)


def test_audit_rejects_manifest_hash_drift(tmp_path):
    inventory_path = _fixture(tmp_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["bundles"][0]["manifest_sha256"] = "0" * 64
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest hash mismatch"):
        audit_inventory(inventory_path)


def test_audit_rejects_same_basename_manifest_outside_canonical_directory(tmp_path):
    inventory_path = _fixture(tmp_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    store = tmp_path / "store"
    declared = inventory["bundles"][0]["manifest"]
    source = store / declared
    alias = store / "alternate" / source.name
    alias.parent.mkdir(parents=True)
    alias.write_bytes(source.read_bytes())
    inventory["bundles"][0]["manifest"] = alias.relative_to(store).as_posix()
    inventory_path.write_text(json.dumps(inventory, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="corpus inventory violates .*schema"):
        audit_inventory(inventory_path)


def test_audit_rejects_symlinked_episode_manifest(tmp_path):
    inventory_path = _fixture(tmp_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    store = tmp_path / "store"
    relative = inventory["bundles"][0]["manifest"]
    manifest_path = store / relative
    target = store / "alternate" / manifest_path.name
    target.parent.mkdir(parents=True)
    manifest_path.replace(target)
    manifest_path.symlink_to(target)

    with pytest.raises(ValueError, match="single-link regular non-symlink"):
        audit_inventory(inventory_path)


def test_audit_requires_explicit_legal_and_external_attestation_statuses(tmp_path):
    inventory_path = _fixture(tmp_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    del inventory["legal_assessment"]
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

    with pytest.raises(ValueError, match="legal_assessment"):
        audit_inventory(inventory_path)


def test_cli_writes_success_and_failure_reports(tmp_path):
    inventory_path = _fixture(tmp_path)
    report_path = tmp_path / "reports" / "audit.json"
    resolved_report_path = tmp_path / "reports" / "audit-resolved.json"
    failed_report_path = tmp_path / "reports" / "failed.json"

    assert main([str(inventory_path), "--output", str(report_path)]) == 1
    first_bytes = report_path.read_bytes()
    assert main([str(inventory_path), "--output", str(report_path)]) == 1
    assert report_path.read_bytes() == first_bytes

    initial_report = audit_inventory(inventory_path)
    near_duplicate_review = initial_report["checks"]["near_duplicate_review"]
    dispositions, signature, reviewer_id, reviewer_keyring = _sign_near_duplicate_dispositions(
        inventory_path,
        near_duplicate_review,
        initial_report["inventory_sha256"],
        tmp_path,
    )
    assert (
        main(
            [
                str(inventory_path),
                "--output",
                str(resolved_report_path),
                "--near-duplicate-dispositions",
                dispositions,
                "--near-duplicate-dispositions-signature",
                signature,
                "--near-duplicate-reviewer-id",
                reviewer_id,
                "--trusted-near-duplicate-reviewers",
                str(reviewer_keyring),
            ]
        )
        == 0
    )
    resolved = json.loads(resolved_report_path.read_text(encoding="utf-8"))
    assert resolved["checks"]["near_duplicate_review"]["status"] == "adjudicated_no_leakage"

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["legal_assessment"]["status"] = "unrecognized"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    assert main([str(inventory_path), "--output", str(failed_report_path)]) == 2
    failed = json.loads(failed_report_path.read_text(encoding="utf-8"))
    assert failed["execution"]["status"] == "failed"
    assert failed["qualification"] == "not_claimed"


def test_cli_returns_nonzero_for_signed_confirmed_cross_split_leakage(tmp_path: Path) -> None:
    inventory_path = _fixture(tmp_path)
    report = audit_inventory(inventory_path)
    review = report["checks"]["near_duplicate_review"]
    dispositions, signature, reviewer_id, reviewer_keyring = _sign_near_duplicate_dispositions(
        inventory_path,
        review,
        report["inventory_sha256"],
        tmp_path,
        outcome="confirmed_leakage",
    )
    output_path = tmp_path / "confirmed-leakage-report.json"

    exit_code = main(
        [
            str(inventory_path),
            "--output",
            str(output_path),
            "--near-duplicate-dispositions",
            dispositions,
            "--near-duplicate-dispositions-signature",
            signature,
            "--near-duplicate-reviewer-id",
            reviewer_id,
            "--trusted-near-duplicate-reviewers",
            str(reviewer_keyring),
        ]
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert saved["checks"]["near_duplicate_review"]["status"] == "confirmed_cross_split_leakage"


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_audit_rejects_invalid_pair_comparison_budgets(tmp_path, budget):
    inventory_path = _fixture(tmp_path)

    with pytest.raises((TypeError, ValueError), match="max_pair_comparisons"):
        audit_inventory(inventory_path, max_pair_comparisons=budget)


def test_audit_enforces_total_duplicate_comparison_budget(tmp_path):
    inventory_path = _fixture(tmp_path)

    with pytest.raises(ValueError, match="duplicate pair comparison budget"):
        audit_inventory(inventory_path, max_pair_comparisons=2)


def test_cli_refuses_to_replace_a_different_report(tmp_path):
    inventory_path = _fixture(tmp_path)
    report_path = tmp_path / "reports" / "audit.json"
    assert main([str(inventory_path), "--output", str(report_path)]) == 1
    report_path.write_text("user data\n", encoding="utf-8")

    assert main([str(inventory_path), "--output", str(report_path)]) == 2
    assert report_path.read_text(encoding="utf-8") == "user data\n"
