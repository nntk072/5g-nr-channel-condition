from __future__ import annotations

import hashlib
import json
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
from datasets.path_oracle import PathOracleTable, PathSolveStatus
from datasets.split_policy import (
    SplitPolicy,
    freeze_split_policy_registry,
    make_split_policy_registry,
    network_episode_subject,
)
from datasets.targets import TargetDefinition, TargetValidity, TargetValue, target_definition_digest
from scripts.audit_umcd_corpus import audit_inventory, main


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


def _fixture(tmp_path: Path) -> Path:
    store = tmp_path / "store"
    bundles = tuple(_bundle(f"episode-{index}", f"asset-family-{index}") for index in range(3))
    stored_bundles = tuple(write_episode_bundle(store, bundle) for bundle in bundles)
    subjects = [network_episode_subject(bundle.episode) for bundle in bundles]
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
                "relations": [],
            }
        )
    target_id = "gNB-1-to-UE-1::geometric_direct_path"
    inventory = {
        "schema": "umcd.corpus_audit_inventory.v1",
        "frozen": True,
        "inventory_id": "synthetic-fixture-only",
        "store": "store",
        "split_registry": {
            "path": registry_path.name,
            "sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        },
        "bundles": manifest_rows,
        "expected_coverage": [
            {
                "split_id": registry.split_for_parent(bundle.episode.episode_id),
                "family_id": bundle.episode.family_id,
                "target_id": target_id,
            }
            for bundle in bundles
        ],
        "duplicate_threshold": 0.999,
        "legal_assessment": {"status": "pending"},
        "external_lock_attestation": {"status": "missing"},
    }
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory, sort_keys=True), encoding="utf-8")
    return inventory_path


def test_audit_report_is_deterministic_hash_bound_and_nonqualifying(tmp_path):
    inventory_path = _fixture(tmp_path)

    first = audit_inventory(inventory_path)
    second = audit_inventory(inventory_path)

    assert first == second
    assert first["qualification"] == "not_claimed"
    assert first["external_gates"]["legal_assessment"]["status"] == "external_unverified"
    assert first["external_gates"]["external_lock_attestation"]["status"] == "external_unverified"
    assert first["checks"]["bundle_store_integrity"]["bundle_count"] == 3
    assert first["checks"]["near_duplicate_review"]["status"] == "review_required"
    core = {key: value for key, value in first.items() if key != "report_sha256"}
    assert (
        first["report_sha256"]
        == hashlib.sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


def test_audit_rejects_manifest_hash_drift(tmp_path):
    inventory_path = _fixture(tmp_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["bundles"][0]["manifest_sha256"] = "0" * 64
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest hash mismatch"):
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
    failed_report_path = tmp_path / "reports" / "failed.json"

    assert main([str(inventory_path), "--output", str(report_path)]) == 0
    first_bytes = report_path.read_bytes()
    assert main([str(inventory_path), "--output", str(report_path)]) == 0
    assert report_path.read_bytes() == first_bytes

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["legal_assessment"]["status"] = "unrecognized"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    assert main([str(inventory_path), "--output", str(failed_report_path)]) == 2
    failed = json.loads(failed_report_path.read_text(encoding="utf-8"))
    assert failed["execution"]["status"] == "failed"
    assert failed["qualification"] == "not_claimed"


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
    assert main([str(inventory_path), "--output", str(report_path)]) == 0
    report_path.write_text("user data\n", encoding="utf-8")

    assert main([str(inventory_path), "--output", str(report_path)]) == 2
    assert report_path.read_text(encoding="utf-8") == "user data\n"
