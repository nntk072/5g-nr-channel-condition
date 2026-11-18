"""Runnable workflow for pairing existing frequency bundles under split policy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from datasets.channel_view import ChannelDelayReference, ChannelPowerScale, PhysicalChannelView
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
from datasets.provenance import UMCDRunProvenance
from datasets.split_policy import (
    LeakageNamespace,
    LeakageRelation,
    SplitChildBinding,
    SplitParentType,
    SplitPolicy,
    SplitSubject,
    bind_split_children,
    freeze_split_policy_registry,
    make_split_policy_registry,
)
from scripts.pair_stored_frequency_views import pair_stored_frequency_views

_GEOMETRY_SHA256 = "a" * 64


def _stored_bundle(view_id: str, carrier_hz: float, episode_id: str) -> EpisodeBundle:
    tx_array = AntennaArraySpec(
        "tx-array",
        np.asarray([[0.0, 0.0, 0.0]]),
        ("V",),
        np.eye(3),
        ("tx-port",),
        np.ones((1, 1), dtype=np.complex64),
    )
    rx_array = AntennaArraySpec(
        "rx-array",
        np.asarray([[0.0, 0.0, 0.0]]),
        ("V",),
        np.eye(3),
        ("rx-port",),
        np.ones((1, 1), dtype=np.complex64),
    )
    nodes = (
        NetworkNode("tx", NodeRole.GNB, "tx-array", (0.0, 0.0, 1.0), 1),
        NetworkNode("rx", NodeRole.UE, "rx-array", (1.0, 0.0, 1.0), 1),
    )
    link = DirectedLink(
        "tx-to-rx", "tx", "rx", (0,), (0,), LinkDirection.DOWNLINK, LinkRole.SERVING
    )
    episode = NetworkEpisode(
        episode_id,
        "shared-site-session",
        "F04-RT-DYNAMIC",
        carrier_hz,
        (tx_array, rx_array),
        nodes,
        (link,),
        (0.0,),
        np.asarray([-15_000.0, 0.0, 15_000.0]),
    )
    view = PhysicalChannelView(
        view_id=view_id,
        link_id="tx-to-rx",
        channel=np.ones((1, 1, 1, 3), dtype=np.complex64),
        frame_times_s=(0.0,),
        rx_port_ids=("rx-port",),
        tx_port_ids=("tx-port",),
        carrier_hz=carrier_hz,
        subcarrier_offsets_hz=episode.subcarrier_offsets_hz,
        delay_reference=ChannelDelayReference.ABSOLUTE_PROPAGATION,
        power_scale=ChannelPowerScale.ABSOLUTE,
    )
    provenance = UMCDRunProvenance(
        record_id=f"{view_id}-run",
        subject_layer="oracle_channels",
        source_family="F04-RT-DYNAMIC",
        producer="Sionna RT PathSolver",
        package_source_sha256="b" * 64,
        software_versions={"sionna-rt": "2.2.0"},
        backend={"backend": "llvm"},
        scene_asset_sha256={"scene.xml": "c" * 64},
        seed=12,
        physical_config={
            "carrier_hz": carrier_hz,
            "scene_geometry": {"scene_geometry_sha256": _GEOMETRY_SHA256},
        },
        solver_config={"max_depth": 4},
        material_config={"concrete": {"frequency_hz": carrier_hz}},
    )
    return EpisodeBundle(episode, None, (view,), (), provenance_records=(provenance,))


def _registry(path):
    parent = SplitSubject(
        "drop-1",
        SplitParentType.NETWORK_EPISODE,
        "F04-RT-DYNAMIC",
        (LeakageRelation(LeakageNamespace.SCENE_REALIZATION, "munich-block-3"),),
    )
    others = (
        SplitSubject("independent-1", SplitParentType.NETWORK_EPISODE, "F04-RT-DYNAMIC"),
        SplitSubject("independent-2", SplitParentType.NETWORK_EPISODE, "F04-RT-DYNAMIC"),
    )
    registry = bind_split_children(
        make_split_policy_registry(
            (parent, *others),
            policy=SplitPolicy(
                seed="pair-cli-seed",
                split_fractions={"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3},
            ),
        ),
        (
            SplitChildBinding("drop-low", "drop-1", "episode_bundle"),
            SplitChildBinding("drop-high", "drop-1", "episode_bundle"),
        ),
    )
    freeze_split_policy_registry(path, registry)


def test_cli_workflow_publishes_pair_and_input_scoped_audit(tmp_path):
    store = tmp_path / "store"
    low = write_episode_bundle(store, _stored_bundle("view-low", 3.5e9, "drop-low"))
    high = write_episode_bundle(store, _stored_bundle("view-high", 28e9, "drop-high"))
    lock = tmp_path / "split-lock.json"
    _registry(lock)

    result = pair_stored_frequency_views(
        store_root=store,
        split_registry_path=lock,
        pair_id="pair-1",
        source_scene_id="munich-block-3",
        parent_episode_id="drop-1",
        lower_episode_id=low.episode_id,
        lower_bundle_sha256=low.bundle_sha256,
        lower_view_id="view-low",
        upper_episode_id=high.episode_id,
        upper_bundle_sha256=high.bundle_sha256,
        upper_view_id="view-high",
    )

    assert result["status"] == "PASS_INPUT_SCOPED"
    assert result["qualification_status"] == "NOT_QUALIFIED"
    assert result["rt_generation_executed"] is False
    assert (store / result["pair_manifest_path"]).is_file()
    audit_path = store / str(result["audit_path"])
    audit = json.loads(audit_path.read_text())
    schema = json.loads(
        (
            Path(__file__).parents[2]
            / "datasets/schema/paired-frequency-split-binding-audit.v1.schema.json"
        ).read_text()
    )
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator(schema).validate(audit)
    assert audit["child_bindings"][0]["split"] == audit["child_bindings"][1]["split"]


def test_cli_workflow_rejects_split_lineage_before_publication(tmp_path):
    store = tmp_path / "store"
    low = write_episode_bundle(store, _stored_bundle("view-low", 3.5e9, "drop-low"))
    high = write_episode_bundle(store, _stored_bundle("view-high", 28e9, "drop-high"))
    lock = tmp_path / "split-lock.json"
    _registry(lock)

    with pytest.raises(ValueError, match="does not bind the paired source scene"):
        pair_stored_frequency_views(
            store_root=store,
            split_registry_path=lock,
            pair_id="pair-1",
            source_scene_id="other-scene",
            parent_episode_id="drop-1",
            lower_episode_id=low.episode_id,
            lower_bundle_sha256=low.bundle_sha256,
            lower_view_id="view-low",
            upper_episode_id=high.episode_id,
            upper_bundle_sha256=high.bundle_sha256,
            upper_view_id="view-high",
        )
    assert not list((store / "manifests").glob("paired_frequency_views/**/*.json"))
