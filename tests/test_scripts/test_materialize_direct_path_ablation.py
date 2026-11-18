"""Canonical stored-bundle workflow for mathematical direct-path ablation."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from datasets.channel_view import ChannelDelayReference, ChannelPowerScale, PhysicalChannelView
from datasets.episode_store import EpisodeBundle, read_episode_bundle, write_episode_bundle
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
    SplitChildBinding,
    SplitParentType,
    SplitPolicy,
    SplitSubject,
    bind_split_children,
    freeze_split_policy_registry,
    make_split_policy_registry,
)
from scripts.materialize_direct_path_ablation import materialize_direct_path_ablation


def _bundle() -> EpisodeBundle:
    tx = AntennaArraySpec("tx", np.zeros((1, 3)), ("V",), np.eye(3), ("tx-port",), np.ones((1, 1)))
    rx = AntennaArraySpec("rx", np.zeros((1, 3)), ("V",), np.eye(3), ("rx-port",), np.ones((1, 1)))
    episode = NetworkEpisode(
        "episode-1",
        "split-group-1",
        "synthetic-fixture",
        3.5e9,
        (tx, rx),
        (
            NetworkNode("gNB", NodeRole.GNB, "tx", (0.0, 0.0, 2.0), 1),
            NetworkNode("UE", NodeRole.UE, "rx", (10.0, 0.0, 1.0), 1),
        ),
        (
            DirectedLink(
                "link-1", "gNB", "UE", (0,), (0,), LinkDirection.DOWNLINK, LinkRole.SERVING
            ),
        ),
        (0.0,),
        np.asarray([-15_000.0, 0.0, 15_000.0]),
    )
    paths = PathOracleTable(
        link_ids=("link-1",),
        frame_indices=np.asarray([0]),
        rx_port_indices=np.asarray([0]),
        tx_port_indices=np.asarray([0]),
        path_offsets=np.asarray([0, 2]),
        solve_status=(PathSolveStatus.SOLVED,),
        delay_s=np.asarray([30e-9, 45e-9]),
        complex_gain=np.asarray([1.0 + 0.0j, 0.2 - 0.1j]),
        interaction_order=np.asarray([0, 1]),
    )
    channel = PhysicalChannelView(
        view_id="view-1",
        link_id="link-1",
        channel=np.ones((1, 1, 1, 3), dtype=np.complex64),
        frame_times_s=(0.0,),
        rx_port_ids=("rx-port",),
        tx_port_ids=("tx-port",),
        carrier_hz=episode.carrier_hz,
        subcarrier_offsets_hz=episode.subcarrier_offsets_hz,
        delay_reference=ChannelDelayReference.ABSOLUTE_PROPAGATION,
        power_scale=ChannelPowerScale.ABSOLUTE,
    )
    return EpisodeBundle(episode, paths, (channel,), ())


def _registry(path: Path):
    subjects = (
        SplitSubject("episode-1", SplitParentType.NETWORK_EPISODE, "synthetic-fixture"),
        SplitSubject("independent-1", SplitParentType.NETWORK_EPISODE, "synthetic-fixture"),
        SplitSubject("independent-2", SplitParentType.NETWORK_EPISODE, "synthetic-fixture"),
    )
    registry = bind_split_children(
        make_split_policy_registry(
            subjects,
            policy=SplitPolicy(
                seed="ablation-cli-seed",
                split_fractions={"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3},
            ),
        ),
        (
            SplitChildBinding("base-1", "episode-1", "counterfactual_base"),
            SplitChildBinding("ablated-1", "episode-1", "counterfactual_math_ablation"),
        ),
    )
    freeze_split_policy_registry(path, registry)


def test_workflow_replays_and_publishes_derived_bundle_without_mutating_source(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "derived"
    stored = write_episode_bundle(source, _bundle())
    source_manifest = source / stored.relative_path
    original_manifest = source_manifest.read_bytes()
    registry_path = tmp_path / "split-lock.json"
    _registry(registry_path)

    report = materialize_direct_path_ablation(
        source_store=source,
        output_store=output,
        split_registry_path=registry_path,
        episode_id=stored.episode_id,
        bundle_sha256=stored.bundle_sha256,
        parent_subject_id="episode-1",
        base_child_id="base-1",
        ablated_child_id="ablated-1",
    )

    assert report["status"] == "PASS_SOFTWARE_REPLAY"
    assert report["qualification_status"] == "NOT_QUALIFIED"
    assert report["physical_blocker_modeled"] is False
    assert report["rt_generation_executed"] is False
    report_path = output / str(report["report_path"])
    receipt = json.loads(report_path.read_text())
    schema = json.loads(
        (
            Path(__file__).parents[2]
            / "datasets/schema/direct-path-ablation-materialization-report.v1.schema.json"
        ).read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(receipt)
    assert receipt["split_registry_sha256"] == report["split_registry_sha256"]
    assert source_manifest.read_bytes() == original_manifest
    derived_manifest = output / str(report["derived_manifest_path"])
    manifest = json.loads(derived_manifest.read_text())
    derived_ref = type(stored)(stored.episode_id, manifest["bundle_sha256"], stored.relative_path)
    replayed = read_episode_bundle(output, derived_ref)
    assert len(replayed.direct_path_ablations) == 1
    assert replayed.direct_path_ablations[0].removed_direct_path_count.tolist() == [1]


def test_workflow_rejects_overlapping_stores_before_writing(tmp_path):
    source = tmp_path / "source"
    stored = write_episode_bundle(source, _bundle())
    registry_path = tmp_path / "split-lock.json"
    _registry(registry_path)

    with pytest.raises(ValueError, match="separate, non-overlapping"):
        materialize_direct_path_ablation(
            source_store=source,
            output_store=source / "derived",
            split_registry_path=registry_path,
            episode_id=stored.episode_id,
            bundle_sha256=stored.bundle_sha256,
            parent_subject_id="episode-1",
            base_child_id="base-1",
            ablated_child_id="ablated-1",
        )
    assert not (source / "derived").exists()
