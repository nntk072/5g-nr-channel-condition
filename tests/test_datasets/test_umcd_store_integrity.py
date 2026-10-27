"""Integrity regression tests for the public UMCD episode store API."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from datasets.channel_view import (
    ChannelDelayReference,
    ChannelPowerScale,
    PhysicalChannelView,
)
from datasets.counterfactual import remove_direct_path_component
from datasets.episode_store import (
    EpisodeBundle,
    _episode_object_size_estimate,
    iter_episode_bundle_shards,
    read_episode_bundle,
    write_episode_bundle,
)
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
    make_split_policy_registry,
)
from datasets.targets import (
    TargetDefinition,
    TargetValidity,
    TargetValue,
    target_definition_digest,
)


def _versioned_delay_target() -> TargetValue:
    fields = {
        "semantic": "rms_delay_spread",
        "units": "s",
        "source": "path_oracle",
        "reference": "absolute_propagation_delay_v1",
        "aggregation_rule": "power_weighted_rms_over_valid_paths",
        "definition_version": "umcd.rms_delay_spread.v1",
    }
    definition = TargetDefinition(
        **fields,
        definition_sha256=target_definition_digest(**fields),
    )
    return TargetValue(
        entity_id="gNB-0-to-UE-0",
        definition=definition,
        values=np.asarray([1.234567890123456e-9, 0.0, 9.876543210987654e-9], dtype=np.float64),
        valid_mask=np.asarray([True, False, True]),
        reason_codes=(TargetValidity.VALID, TargetValidity.CENSORED, TargetValidity.VALID),
        confidence="synthetic_fixture",
    )


def _versioned_path_loss_target() -> TargetValue:
    fields = {
        "semantic": "path_loss",
        "units": "dB",
        "source": "measured_link_budget",
        "reference": "free_space_reference_v1",
        "aggregation_rule": "wideband_received_power_ratio",
        "definition_version": "umcd.path_loss.v1",
    }
    definition = TargetDefinition(
        **fields,
        definition_sha256=target_definition_digest(**fields),
    )
    return TargetValue(
        entity_id="gNB-0-to-UE-0",
        definition=definition,
        values=np.asarray([91.23456, 0.0], dtype=np.float32),
        valid_mask=np.asarray([True, False]),
        reason_codes=(TargetValidity.VALID, TargetValidity.NOT_OBSERVABLE),
        confidence="synthetic_fixture",
    )


def _versioned_complex_cfr_target() -> TargetValue:
    fields = {
        "semantic": "perfect synchronized effective-subcarrier CFR",
        "units": "complex channel gain (dimensionless)",
        "source": "sionna_phy_ofdmchannel_return_channel",
        "reference": (
            "OFDMChannel return_channel with the configured TDL profile, carrier, "
            "sample rate, and effective FFT-bin mapping"
        ),
        "aggregation_rule": "retain each effective subcarrier coefficient; "
        "no antenna, symbol, or power averaging",
        "definition_version": "umcd.tdl-pusch.effective-cfr.v1",
    }
    definition = TargetDefinition(
        **fields,
        definition_sha256=target_definition_digest(**fields),
    )
    return TargetValue(
        entity_id="gNB-0-to-UE-0",
        definition=definition,
        values=np.asarray([1.25 - 0.5j, 0.0 + 0.0j], dtype=np.complex64),
        valid_mask=np.asarray([True, False]),
        reason_codes=(TargetValidity.VALID, TargetValidity.NOT_OBSERVABLE),
        confidence="synthetic_fixture",
    )


def _episode_bundle() -> EpisodeBundle:
    gnb_array = AntennaArraySpec(
        array_id="gNB-array",
        element_positions_local_m=np.zeros((1, 3)),
        polarization_per_element=("V",),
        orientation_local_to_world=np.eye(3),
        port_ids=("gNB-port-0",),
        port_element_weights=np.ones((1, 1), dtype=np.complex128),
    )
    ue_array = AntennaArraySpec(
        array_id="UE-array",
        element_positions_local_m=np.zeros((1, 3)),
        polarization_per_element=("V",),
        orientation_local_to_world=np.eye(3),
        port_ids=("UE-port-0",),
        port_element_weights=np.ones((1, 1), dtype=np.complex128),
    )
    episode = NetworkEpisode(
        episode_id="g7-06-integrity-episode",
        split_group_id="integrity-group-0",
        family_id="integrity-test-scene",
        carrier_hz=3.5e9,
        arrays=(gnb_array, ue_array),
        nodes=(
            NetworkNode("gNB-0", NodeRole.GNB, "gNB-array", (0.0, 0.0, 5.0), 1),
            NetworkNode("UE-0", NodeRole.UE, "UE-array", (10.0, 0.0, 1.5), 1),
        ),
        links=(
            DirectedLink(
                "gNB-0-to-UE-0",
                "gNB-0",
                "UE-0",
                (0,),
                (0,),
                LinkDirection.DOWNLINK,
                LinkRole.SERVING,
            ),
        ),
        frame_times_s=(0.0,),
        subcarrier_offsets_hz=np.asarray([0.0]),
    )
    paths = PathOracleTable(
        link_ids=("gNB-0-to-UE-0",),
        frame_indices=np.asarray([0]),
        rx_port_indices=np.asarray([0]),
        tx_port_indices=np.asarray([0]),
        path_offsets=np.asarray([0, 1]),
        solve_status=(PathSolveStatus.SOLVED,),
        delay_s=np.asarray([35e-9]),
        complex_gain=np.asarray([0.25 + 0.5j]),
        interaction_order=np.asarray([0]),
    )
    channel = PhysicalChannelView(
        view_id="g7-12-precision-channel",
        link_id="gNB-0-to-UE-0",
        channel=np.asarray([[[[0.123456789012345 + 0.987654321098765j]]]], dtype=np.complex128),
        frame_times_s=(0.0,),
        rx_port_ids=("UE-port-0",),
        tx_port_ids=("gNB-port-0",),
        carrier_hz=3.5e9,
        subcarrier_offsets_hz=np.asarray([0.0], dtype=np.float64),
        delay_reference=ChannelDelayReference.ABSOLUTE_PROPAGATION,
        power_scale=ChannelPowerScale.ABSOLUTE,
    )
    channel_complex64 = PhysicalChannelView(
        view_id="g7-12-precision-channel-complex64",
        link_id="gNB-0-to-UE-0",
        channel=np.asarray([[[[0.1234567 + 0.9876543j]]]], dtype=np.complex64),
        frame_times_s=(0.0,),
        rx_port_ids=("UE-port-0",),
        tx_port_ids=("gNB-port-0",),
        carrier_hz=3.5e9,
        subcarrier_offsets_hz=np.asarray([0.0], dtype=np.float64),
        delay_reference=ChannelDelayReference.ABSOLUTE_PROPAGATION,
        power_scale=ChannelPowerScale.ABSOLUTE,
    )
    return EpisodeBundle(
        episode,
        paths,
        (channel, channel_complex64),
        (_versioned_delay_target(), _versioned_path_loss_target()),
    )


def _episode_bundle_with_counterfactual() -> EpisodeBundle:
    bundle = _episode_bundle()
    parent_id = bundle.episode.episode_id
    policy = SplitPolicy(
        seed="umcd-store-counterfactual-retry",
        split_fractions={"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3},
    )
    subjects = [
        SplitSubject(
            subject_id=parent_id,
            parent_type=SplitParentType.NETWORK_EPISODE,
            source_family=bundle.episode.family_id,
        ),
        *[
            SplitSubject(
                subject_id=f"independent-{index}",
                parent_type=SplitParentType.NETWORK_EPISODE,
                source_family=bundle.episode.family_id,
            )
            for index in range(2)
        ],
    ]
    registry = bind_split_children(
        make_split_policy_registry(subjects, policy=policy),
        [
            SplitChildBinding("counterfactual-base", parent_id, "counterfactual_base"),
            SplitChildBinding("counterfactual-ablated", parent_id, "counterfactual_math_ablation"),
        ],
    )
    ablation = remove_direct_path_component(
        bundle.path_oracle,
        frequency_offset_hz=bundle.episode.subcarrier_offsets_hz,
        registry=registry,
        parent_subject_id=parent_id,
        base_child_id="counterfactual-base",
        ablated_child_id="counterfactual-ablated",
    )
    return replace(bundle, direct_path_ablations=(ablation,))


def test_target_value_rejects_empty_truth_instead_of_losing_unavailability_reason():
    definition = _versioned_delay_target().definition

    with pytest.raises(ValueError, match="at least one element.*explicit reason code"):
        TargetValue(
            entity_id="gNB-0-to-UE-0",
            definition=definition,
            values=np.asarray([], dtype=np.float64),
            valid_mask=np.asarray([], dtype=np.bool_),
            reason_codes=(),
            confidence="synthetic_fixture",
        )


@pytest.mark.parametrize(
    "layer_name",
    (
        "episode",
        "oracle_paths",
        "counterfactual_ablations",
        "oracle_channels",
        "targets",
    ),
)
def test_episode_bundle_reader_rejects_changed_compressed_object_bytes(tmp_path, layer_name):
    store = tmp_path / "umcd-store"
    bundle = (
        _episode_bundle_with_counterfactual()
        if layer_name == "counterfactual_ablations"
        else _episode_bundle()
    )

    stored = write_episode_bundle(store, bundle)
    loaded = read_episode_bundle(store, stored)

    assert loaded.episode.episode_id == bundle.episode.episode_id
    np.testing.assert_array_equal(loaded.path_oracle.complex_gain, bundle.path_oracle.complex_gain)

    manifest_path = store / stored.relative_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    layer_reference = next(item for item in manifest["layers"] if item["layer"] == layer_name)
    object_path = store / layer_reference["path"]
    assert object_path.suffix == ".npz"

    tampered = bytearray(object_path.read_bytes())
    byte_index = len(tampered) // 2
    tampered[byte_index] ^= 0x01
    object_path.write_bytes(tampered)

    with pytest.raises(ValueError, match="content hash"):
        read_episode_bundle(store, stored)


def test_episode_bundle_reader_applies_optional_shard_byte_ceiling(tmp_path):
    store = tmp_path / "umcd-bounded-store"
    stored = write_episode_bundle(store, _episode_bundle())

    with pytest.raises(ValueError, match="configured byte limit"):
        read_episode_bundle(store, stored, max_shard_bytes=1)


def test_episode_bundle_manifest_publication_failure_is_retryable(tmp_path, monkeypatch):
    import datasets.episode_store as episode_store

    store = tmp_path / "umcd-retry-store"
    bundle = _episode_bundle()
    publish = episode_store._publish_immutable
    failed = False

    def fail_before_manifest(path, payload):
        nonlocal failed
        if not failed and "manifests" in path.parts:
            failed = True
            raise OSError("injected manifest publication failure")
        publish(path, payload)

    monkeypatch.setattr(episode_store, "_publish_immutable", fail_before_manifest)
    with pytest.raises(OSError, match="injected manifest publication failure"):
        write_episode_bundle(store, bundle)

    manifest_dir = store / "manifests" / "episodes"
    assert not manifest_dir.exists()
    assert list((store / "objects").rglob("*.npz"))

    stored = write_episode_bundle(store, bundle)
    assert read_episode_bundle(store, stored).episode.episode_id == bundle.episode.episode_id
    assert not [path for path in store.rglob(".*") if path.is_file()]


def test_episode_bundle_mid_object_publication_failure_is_retryable(tmp_path, monkeypatch):
    import datasets.episode_store as episode_store

    store = tmp_path / "umcd-mid-object-retry-store"
    bundle = _episode_bundle()
    publish = episode_store._publish_immutable
    object_publications = 0
    injected = False

    def fail_during_object_publication(path, payload):
        nonlocal object_publications, injected
        if "objects" in path.parts:
            object_publications += 1
            if object_publications == 2 and not injected:
                injected = True
                raise OSError("injected mid-object publication failure")
        publish(path, payload)

    monkeypatch.setattr(episode_store, "_publish_immutable", fail_during_object_publication)
    with pytest.raises(OSError, match="injected mid-object publication failure"):
        write_episode_bundle(store, bundle)

    assert list((store / "objects").rglob("*.npz"))
    assert not (store / "manifests" / "episodes").exists()

    stored = write_episode_bundle(store, bundle)
    loaded = read_episode_bundle(store, stored)
    assert loaded.episode.episode_id == bundle.episode.episode_id
    assert stored.episode_id == bundle.episode.episode_id
    assert not [path for path in store.rglob(".*") if path.is_file()]


def test_episode_bundle_counterfactual_object_publication_failure_is_retryable(
    tmp_path, monkeypatch
):
    import datasets.episode_store as episode_store

    store = tmp_path / "umcd-counterfactual-retry-store"
    bundle = _episode_bundle_with_counterfactual()
    publish = episode_store._publish_immutable
    injected = False

    def fail_counterfactual_object(path, payload):
        nonlocal injected
        if "counterfactual_ablations" in path.parts and not injected:
            injected = True
            raise OSError("injected counterfactual object publication failure")
        publish(path, payload)

    monkeypatch.setattr(episode_store, "_publish_immutable", fail_counterfactual_object)
    with pytest.raises(OSError, match="counterfactual object publication failure"):
        write_episode_bundle(store, bundle)

    assert list((store / "objects" / "episode").rglob("*.npz"))
    assert list((store / "objects" / "oracle_paths").rglob("*.npz"))
    assert not (store / "objects" / "counterfactual_ablations").exists()
    assert not (store / "manifests" / "episodes").exists()

    monkeypatch.setattr(episode_store, "_publish_immutable", publish)
    stored = write_episode_bundle(store, bundle)
    loaded = read_episode_bundle(store, stored)
    assert len(loaded.direct_path_ablations) == 1
    assert loaded.direct_path_ablations[0].removed_direct_path_count.tolist() == [1]
    np.testing.assert_array_equal(
        loaded.direct_path_ablations[0].without_direct_cfr,
        bundle.direct_path_ablations[0].without_direct_cfr,
    )
    assert not [path for path in store.rglob(".*") if path.is_file()]


def test_episode_bundle_shards_are_deterministic_and_budgeted(tmp_path):
    store = tmp_path / "umcd-store"
    template = _episode_bundle()
    second_episode = replace(template.episode, episode_id="g7-06-integrity-episode-b")
    write_episode_bundle(store, template)
    write_episode_bundle(
        store,
        EpisodeBundle(
            second_episode,
            template.path_oracle,
            template.channel_views,
            template.targets,
        ),
    )

    shards = list(iter_episode_bundle_shards(store, max_shard_bytes=10_000_000))

    assert len(shards) == 1
    assert [bundle.episode.episode_id for bundle in shards[0]] == [
        "g7-06-integrity-episode",
        "g7-06-integrity-episode-b",
    ]


def test_episode_bundle_shards_split_before_exceeding_accounting_budget(tmp_path):
    store = tmp_path / "umcd-store"
    template = _episode_bundle()
    second_episode = replace(template.episode, episode_id="g7-06-integrity-episode-b")
    write_episode_bundle(store, template)
    write_episode_bundle(
        store,
        EpisodeBundle(
            second_episode,
            template.path_oracle,
            template.channel_views,
            template.targets,
        ),
    )

    per_bundle_estimates = []
    for manifest_path in sorted((store / "manifests/episodes").glob("*.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        per_bundle_estimates.append(
            sum(
                _episode_object_size_estimate(store, reference, max_shard_bytes=10_000_000)
                for reference in manifest["layers"]
            )
        )
    budget = max(per_bundle_estimates)

    shards = list(iter_episode_bundle_shards(store, max_shard_bytes=budget))

    assert [[bundle.episode.episode_id for bundle in shard] for shard in shards] == [
        ["g7-06-integrity-episode"],
        ["g7-06-integrity-episode-b"],
    ]


def test_episode_bundle_shards_reject_a_budget_smaller_than_one_bundle(tmp_path):
    store = tmp_path / "umcd-store"
    write_episode_bundle(store, _episode_bundle())

    with pytest.raises(ValueError, match="max_shard_bytes"):
        list(iter_episode_bundle_shards(store, max_shard_bytes=1))


def test_episode_bundle_reader_rejects_duplicate_manifest_keys(tmp_path):
    store = tmp_path / "umcd-store"
    stored = write_episode_bundle(store, _episode_bundle())
    manifest_path = store / stored.relative_path
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        manifest_text.replace(
            '"schema":"umcd.bundle.v1"',
            '"schema":"umcd.bundle.v1","schema":"umcd.bundle.v1"',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        read_episode_bundle(store, stored)


def test_g7_12_compressed_round_trip_preserves_physical_values_and_dtypes(tmp_path):
    store = tmp_path / "precision-store"
    bundle = _episode_bundle()
    stored = write_episode_bundle(store, bundle)
    loaded = read_episode_bundle(store, stored)

    source_paths = bundle.path_oracle.complex_gain
    stored_paths = loaded.path_oracle.complex_gain
    assert stored_paths.dtype == source_paths.dtype == np.dtype(np.complex128)
    assert stored_paths.tobytes() == source_paths.tobytes()

    stored_channels = {item.view_id: item for item in loaded.channel_views}
    for source_channel in bundle.channel_views:
        stored_channel = stored_channels[source_channel.view_id].channel
        assert stored_channel.dtype == source_channel.channel.dtype
        assert stored_channel.tobytes() == source_channel.channel.tobytes()

    stored_targets = {item.definition.semantic: item for item in loaded.targets}
    for source_target in bundle.targets:
        stored_target = stored_targets[source_target.definition.semantic]
        assert stored_target.values.dtype == source_target.values.dtype
        assert stored_target.values.tobytes() == source_target.values.tobytes()
        assert stored_target.valid_mask.tobytes() == source_target.valid_mask.tobytes()
        assert stored_target.reason_codes == source_target.reason_codes
        assert stored_target.definition == source_target.definition


def test_g7_complex_cfr_target_survives_episode_store_round_trip(tmp_path):
    store = tmp_path / "complex-cfr-store"
    source_bundle = _episode_bundle()
    target = _versioned_complex_cfr_target()
    bundle = replace(source_bundle, targets=source_bundle.targets + (target,))

    stored = write_episode_bundle(store, bundle)
    loaded = read_episode_bundle(store, stored)

    loaded_target = next(
        item
        for item in loaded.targets
        if item.definition.definition_version == target.definition.definition_version
    )
    assert loaded_target.values.dtype == target.values.dtype == np.dtype(np.complex64)
    assert loaded_target.values.tobytes() == target.values.tobytes()
    assert loaded_target.valid_mask.tobytes() == target.valid_mask.tobytes()
    assert loaded_target.reason_codes == target.reason_codes
