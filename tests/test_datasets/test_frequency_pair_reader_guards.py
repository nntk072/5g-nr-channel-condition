"""Failure paths for reading provenance-bound frequency-pair artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest
from datasets import frequency_pair
from datasets.episode_store import EpisodeBundle, write_episode_bundle
from datasets.frequency_pair import (
    StoredPairedFrequencyManifest,
    pair_stored_frequency_bundles,
    read_paired_frequency_manifest,
    write_paired_frequency_manifest,
)
from datasets.split_policy import (
    LeakageNamespace,
    LeakageRelation,
    SplitParentType,
    SplitSubject,
    make_split_policy_registry,
)
from tests.test_datasets.test_frequency_pair import (
    _pair,
    _paired_split_registry,
    _stored_bundle,
)


def _stored_pair(tmp_path):
    low = write_episode_bundle(tmp_path, _stored_bundle("view-low", 3.5e9, "drop-low"))
    high = write_episode_bundle(tmp_path, _stored_bundle("view-high", 28e9, "drop-high"))
    pair = pair_stored_frequency_bundles(
        tmp_path,
        pair_id="pair-1",
        source_scene_id="munich-block-3",
        parent_episode_id="drop-1",
        lower_bundle=low,
        upper_bundle=high,
        lower_view_id="view-low",
        upper_view_id="view-high",
    )
    return low, high, pair, write_paired_frequency_manifest(tmp_path, pair)


def test_pair_reader_requires_stored_bundle_references(tmp_path):
    low, high, _, _ = _stored_pair(tmp_path)
    with pytest.raises(TypeError, match="StoredEpisodeBundle references"):
        pair_stored_frequency_bundles(
            tmp_path,
            pair_id="pair-1",
            source_scene_id="scene",
            parent_episode_id="drop-1",
            lower_bundle=object(),
            upper_bundle=high,
            lower_view_id="view-low",
            upper_view_id="view-high",
        )
    with pytest.raises(ValueError, match="lower_view_id"):
        pair_stored_frequency_bundles(
            tmp_path,
            pair_id="pair-1",
            source_scene_id="scene",
            parent_episode_id="drop-1",
            lower_bundle=low,
            upper_bundle=high,
            lower_view_id=" ",
            upper_view_id="view-high",
        )
    with pytest.raises(ValueError, match="upper_view_id"):
        pair_stored_frequency_bundles(
            tmp_path,
            pair_id="pair-1",
            source_scene_id="scene",
            parent_episode_id="drop-1",
            lower_bundle=low,
            upper_bundle=high,
            lower_view_id="view-low",
            upper_view_id=" ",
        )
    with pytest.raises(ValueError, match="distinct content hashes"):
        pair_stored_frequency_bundles(
            tmp_path,
            pair_id="pair-1",
            source_scene_id="scene",
            parent_episode_id="drop-1",
            lower_bundle=low,
            upper_bundle=low,
            lower_view_id="view-low",
            upper_view_id="view-low",
        )


def test_pair_reader_requires_exactly_one_selected_view_and_provenance(tmp_path):
    low, high, _, _ = _stored_pair(tmp_path)
    with pytest.raises(ValueError, match="exactly one channel view"):
        pair_stored_frequency_bundles(
            tmp_path,
            pair_id="pair-1",
            source_scene_id="scene",
            parent_episode_id="drop-1",
            lower_bundle=low,
            upper_bundle=high,
            lower_view_id="missing-view",
            upper_view_id="view-high",
        )

    source_bundle = _stored_bundle("view-low", 3.5e9, "drop-low")
    low_bundle = EpisodeBundle(
        replace(source_bundle.episode, episode_id="drop-low-without-provenance"),
        None,
        source_bundle.channel_views,
        (),
    )
    low_without_provenance = write_episode_bundle(tmp_path, low_bundle)
    with pytest.raises(ValueError, match="carrier-matched oracle-channel provenance"):
        pair_stored_frequency_bundles(
            tmp_path,
            pair_id="pair-1",
            source_scene_id="scene",
            parent_episode_id="drop-1",
            lower_bundle=low_without_provenance,
            upper_bundle=high,
            lower_view_id="view-low",
            upper_view_id="view-high",
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"not-json", "not valid UTF-8 JSON"),
        (b"\xff", "not valid UTF-8 JSON"),
        (b"[]", "unsupported paired frequency manifest schema"),
    ],
)
def test_pair_manifest_reader_rejects_invalid_documents(tmp_path, payload, message):
    _, _, _, stored = _stored_pair(tmp_path)
    (tmp_path / stored.relative_path).write_bytes(payload)
    with pytest.raises(ValueError, match=message):
        read_paired_frequency_manifest(
            tmp_path, stored, lower_bundle=object(), upper_bundle=object()
        )


def test_pair_manifest_reader_checks_canonical_encoding_and_view_roster(tmp_path):
    low, high, pair, stored = _stored_pair(tmp_path)
    path = tmp_path / stored.relative_path
    document = pair.to_dict()
    path.write_text(json.dumps(document, indent=2))
    with pytest.raises(ValueError, match="not canonically encoded"):
        read_paired_frequency_manifest(tmp_path, stored, lower_bundle=low, upper_bundle=high)

    document["views"] = document["views"][:1]
    core = {key: value for key, value in document.items() if key != "pair_sha256"}
    document["pair_sha256"] = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    new_stored = StoredPairedFrequencyManifest(
        document["pair_sha256"],
        f"manifests/paired_frequency_views/{document['pair_sha256'][:2]}/{document['pair_sha256']}.json",
    )
    path = tmp_path / new_stored.relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")))
    with pytest.raises(ValueError, match="exactly two channel views"):
        read_paired_frequency_manifest(
            tmp_path, new_stored, lower_bundle=low, upper_bundle=high
        )


def test_pair_manifest_reader_rejects_path_escape_and_wrong_stored_type(tmp_path):
    _, _, _, stored = _stored_pair(tmp_path)
    with pytest.raises(TypeError, match="StoredPairedFrequencyManifest"):
        read_paired_frequency_manifest(tmp_path, object(), lower_bundle=object(), upper_bundle=object())
    path = tmp_path / stored.relative_path
    path.unlink()
    outside = tmp_path.parent / "outside-pair-manifest.json"
    outside.write_text("{}")
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="resolves outside its store"):
        read_paired_frequency_manifest(
            tmp_path, stored, lower_bundle=object(), upper_bundle=object()
        )


def test_pair_writer_rejects_wrong_pair_type():
    from datasets.frequency_pair import write_paired_frequency_manifest

    with pytest.raises(TypeError, match="PairedFrequencyViews"):
        write_paired_frequency_manifest("unused", object())


def test_pair_writer_enforces_configured_manifest_size_limit(tmp_path, monkeypatch):
    import datasets.frequency_pair as frequency_pair

    monkeypatch.setattr(frequency_pair, "_MAX_PAIR_MANIFEST_BYTES", 1)
    with pytest.raises(ValueError, match="byte ceiling"):
        frequency_pair.write_paired_frequency_manifest(tmp_path, _pair())


def test_frequency_pair_private_normalizers_reject_or_preserve_supported_shapes():
    base = _pair().lower_provenance
    assert frequency_pair._plain((1, (2,))) == [1, [2]]
    with_trajectory = replace(
        base,
        physical_config={"trajectory": {"carrier_hz": 3.5e9}, "carrier_hz": 99.0},
    )
    assert frequency_pair._bound_carrier_hz(with_trajectory) == 3.5e9
    without_geometry = replace(base, physical_config={"carrier_hz": base.physical_config["carrier_hz"]})
    assert frequency_pair._scene_geometry_digest(without_geometry) is None
    for carrier in (True, "3.5 GHz"):
        invalid = replace(base, physical_config={"carrier_hz": carrier})
        assert frequency_pair._bound_carrier_hz(invalid) is None


def test_pair_split_audit_rejects_invalid_registry_relationships():
    from datasets.frequency_pair import audit_paired_frequency_split_binding

    pair = _pair()
    registry = _paired_split_registry()
    with pytest.raises(TypeError, match="PairedFrequencyViews"):
        audit_paired_frequency_split_binding(object(), registry)
    with pytest.raises(TypeError, match="SplitPolicyRegistry"):
        audit_paired_frequency_split_binding(pair, object())

    extras = (
        SplitSubject("audit-extra-a", SplitParentType.NETWORK_EPISODE, "F04-RT-DYNAMIC"),
        SplitSubject("audit-extra-b", SplitParentType.NETWORK_EPISODE, "F04-RT-DYNAMIC"),
    )
    missing_parent = make_split_policy_registry(
        tuple(subject for key, subject in registry.subjects.items() if key != pair.parent_episode_id)
        + (SplitSubject("unrelated", SplitParentType.NETWORK_EPISODE, "F04-RT-DYNAMIC"),),
        policy=registry.policy,
    )
    with pytest.raises(ValueError, match="parent is absent"):
        audit_paired_frequency_split_binding(pair, missing_parent)

    wrong_type = SplitSubject(
        pair.parent_episode_id,
        SplitParentType.STANDARDIZED_CHANNEL_DRAW,
        "F04-RT-DYNAMIC",
        (LeakageRelation(LeakageNamespace.SCENE_REALIZATION, pair.source_scene_id),),
    )
    wrong_type_registry = make_split_policy_registry(
        (wrong_type, *extras), policy=registry.policy
    )
    with pytest.raises(ValueError, match="must be a network episode"):
        audit_paired_frequency_split_binding(pair, wrong_type_registry)

    no_scene = SplitSubject(
        pair.parent_episode_id,
        SplitParentType.NETWORK_EPISODE,
        "F04-RT-DYNAMIC",
        (LeakageRelation(LeakageNamespace.SCENE_REALIZATION, "another-scene"),),
    )
    no_scene_registry = make_split_policy_registry((no_scene, *extras), policy=registry.policy)
    with pytest.raises(ValueError, match="does not bind"):
        audit_paired_frequency_split_binding(pair, no_scene_registry)

    missing_children = make_split_policy_registry(
        (
            SplitSubject(
                pair.parent_episode_id,
                SplitParentType.NETWORK_EPISODE,
                "F04-RT-DYNAMIC",
                (LeakageRelation(LeakageNamespace.SCENE_REALIZATION, pair.source_scene_id),),
            ),
            *extras,
        ),
        policy=registry.policy,
    )
    with pytest.raises(ValueError, match="absent from the split registry"):
        audit_paired_frequency_split_binding(pair, missing_children)


def test_pair_manifest_reader_rejects_rehashed_unknown_fields(tmp_path):
    low, high, pair, _ = _stored_pair(tmp_path)
    document = pair.to_dict()
    document["unexpected"] = "not part of the canonical pair representation"
    core = {key: value for key, value in document.items() if key != "pair_sha256"}
    document["pair_sha256"] = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    stored = StoredPairedFrequencyManifest(
        document["pair_sha256"],
        f"manifests/paired_frequency_views/{document['pair_sha256'][:2]}/{document['pair_sha256']}.json",
    )
    path = tmp_path / stored.relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")))
    with pytest.raises(ValueError, match="differs from replayed source bundles"):
        read_paired_frequency_manifest(tmp_path, stored, lower_bundle=low, upper_bundle=high)
