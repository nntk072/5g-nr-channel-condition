"""UMCD target definitions, visibility and censoring targets, and legacy target migration."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict, replace
from pathlib import Path

import datasets.targets as target_contracts
import numpy as np
import pytest
from datasets.episode_store import (
    EpisodeBundle,
    _json_bytes,
    _object_ref,
    _pack_npz,
    _publish_immutable,
    _target_payload,
    read_episode_bundle,
    write_episode_bundle,
)
from datasets.targets import (
    CensoringInterval,
    TargetDefinition,
    TargetValidity,
    TargetValue,
    definition_digest,
    migrate_legacy_target_definition,
    target_definition_digest,
)
from tests.test_datasets.umcd_schema_support import (
    _bundle,
    _episode,
)


def test_censoring_interval_is_preserved_in_content_addressed_target_layer(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    bundle = _bundle()
    target = replace(
        bundle.targets[0],
        values=np.asarray([0.0]),
        valid_mask=np.asarray([False]),
        reason_codes=(TargetValidity.CENSORED,),
        censoring_intervals=(CensoringInterval(lower=10e-9),),
    )
    with np.load(io.BytesIO(_target_payload(target)), allow_pickle=False) as archive:
        target_metadata = json.loads(archive["metadata_json"].tobytes())
    target_schema = json.loads(
        (Path(__file__).parents[2] / "datasets/schema/umcd-target-layer.v1.schema.json").read_text()
    )
    jsonschema.Draft202012Validator(target_schema).validate(target_metadata)
    stored = write_episode_bundle(tmp_path, replace(bundle, targets=(target,)))
    loaded = read_episode_bundle(tmp_path, stored)

    assert loaded.targets[0].reason_codes == (TargetValidity.CENSORED,)
    assert loaded.targets[0].censoring_intervals == (CensoringInterval(lower=10e-9),)


def test_target_values_require_definition_hash_and_reasoned_element_masks():
    fields = {
        "semantic": "rms_delay_spread",
        "units": "s",
        "source": "path_oracle",
        "reference": "absolute_physical_channel_path_oracle",
        "aggregation_rule": "RMS over valid paths after power normalization",
        "definition_version": "umcd.rms_delay_spread.v1",
    }
    definition = TargetDefinition(
        **fields,
        definition_sha256=target_definition_digest(**fields),
    )
    target = TargetValue(
        entity_id="link-1",
        definition=definition,
        values=np.asarray([12e-9, 0.0]),
        valid_mask=np.asarray([True, False]),
        reason_codes=(TargetValidity.VALID, TargetValidity.NOT_OBSERVABLE),
        confidence="rt_solver_truth",
    )
    assert target.valid_count == 1
    assert target.unavailable_counts == {TargetValidity.NOT_OBSERVABLE: 1}
    with pytest.raises(ValueError, match="masked target element"):
        TargetValue(
            entity_id="link-1",
            definition=definition,
            values=np.asarray([12e-9, 3e-9]),
            valid_mask=np.asarray([True, False]),
            reason_codes=(TargetValidity.VALID, TargetValidity.NOT_OBSERVABLE),
            confidence="rt_solver_truth",
        )


@pytest.mark.parametrize(
    "changed_field,changed_value",
    [
        ("semantic", "rms_delay_spread_changed"),
        ("units", "ns"),
        ("source", "different_oracle"),
        ("reference", "different_gauge"),
        ("aggregation_rule", "maximum over paths"),
        ("definition_version", "umcd.rms_delay_spread.v2"),
    ],
)
def test_target_definition_digest_binds_every_canonical_field(changed_field, changed_value):
    fields = {
        "semantic": "rms_delay_spread",
        "units": "s",
        "source": "path_oracle",
        "reference": "absolute_physical_channel_path_oracle",
        "aggregation_rule": "RMS over valid paths after power normalization",
        "definition_version": "umcd.rms_delay_spread.v1",
    }
    definition = TargetDefinition(
        **fields,
        definition_sha256=target_definition_digest(**fields),
    )
    changed = {**fields, changed_field: changed_value}
    tampered = TargetDefinition(
        **changed,
        definition_sha256=definition.definition_sha256,
    )
    with pytest.raises(ValueError, match="does not match its canonical fields"):
        TargetValue(
            entity_id="link-1",
            definition=tampered,
            values=np.asarray([1.0]),
            valid_mask=np.asarray([True]),
            reason_codes=(TargetValidity.VALID,),
            confidence="verified",
        )


def test_definition_digest_hashes_exact_utf8_bytes_as_public_utility():
    canonical_definition = '{"semantic":"LOS"}'
    assert (
        definition_digest(canonical_definition)
        == hashlib.sha256(canonical_definition.encode("utf-8")).hexdigest()
    )
    with pytest.raises(ValueError, match="non-empty string"):
        definition_digest("  ")


def test_legacy_definition_defaults_remain_constructible_but_not_canonical_targets():
    legacy = TargetDefinition(semantic="legacy_label", units="class", source="legacy_source")
    assert legacy.reference == "unspecified_legacy"
    assert legacy.aggregation_rule == "unspecified_legacy"
    assert legacy.definition_version == "unversioned_legacy"
    with pytest.raises(ValueError, match="SHA-256 digest"):
        TargetValue(
            entity_id="link-1",
            definition=legacy,
            values=np.asarray([1.0]),
            valid_mask=np.asarray([True]),
            reason_codes=(TargetValidity.VALID,),
            confidence="legacy",
        )


def test_v1_target_definition_migrates_with_explicitly_unknown_contract_fields():
    migrated = migrate_legacy_target_definition(
        {
            "semantic": "legacy_delay",
            "units": "s",
            "source": "old_path_solver",
            "reference": "unspecified_legacy",
            "definition_sha256": "old-format-digest",
        }
    )

    assert migrated.reference == "not recorded by legacy target schema v1"
    assert migrated.aggregation_rule == "not recorded by legacy target schema v1"
    assert migrated.definition_version == "migrated_from_legacy_target_v1"
    migrated.require_versioned()


@pytest.mark.parametrize(
    ("version", "shape", "axes"),
    [
        (
            "umcd.standardized.effective-cfr.v2",
            (1, 1, 1, 14, 1, 1, 2),
            (
                "batch",
                "rx_antenna",
                "rx_stream",
                "ofdm_symbol",
                "tx_stream",
                "tx_antenna",
                "effective_subcarrier",
            ),
        ),
        (
            "umcd.tdl-pusch.effective-cfr.v2",
            (1, 1, 1, 14, 1, 1, 2),
            (
                "batch",
                "rx_antenna",
                "rx_stream",
                "ofdm_symbol",
                "tx_stream",
                "tx_antenna",
                "effective_subcarrier",
            ),
        ),
        (
            "umcd.rt-trajectory.absolute-cfr.v2",
            (2,),
            ("effective_subcarrier",),
        ),
        (
            "umcd.rt-trajectory.symbol-cfr.v2",
            (14, 2),
            ("ofdm_symbol", "effective_subcarrier"),
        ),
    ],
)
def test_axis_bound_cfr_target_must_match_episode_subcarrier_grid(tmp_path, version, shape, axes):
    _name, semantic, units, _domain = target_contracts._CANONICAL_TARGET_CONTRACTS[version]
    source, references, aggregations = target_contracts._CANONICAL_TARGET_PROVENANCE[version]
    definition = TargetDefinition.versioned(
        semantic=semantic,
        units=units,
        source=source,
        reference=next(iter(references)),
        aggregation_rule=(
            next(iter(aggregations)) if isinstance(aggregations, frozenset) else aggregations
        ),
        definition_version=version,
        value_axes=axes,
    )
    values = np.zeros(shape, dtype=np.complex128)
    target = TargetValue(
        entity_id="gNB-1-to-UE-1",
        definition=definition,
        values=values,
        valid_mask=np.ones(shape, dtype=np.bool_),
        reason_codes=(TargetValidity.VALID,) * values.size,
        confidence="synthetic_contract_check",
    )
    with pytest.raises(ValueError, match="effective-subcarrier axis must match"):
        EpisodeBundle(_episode(), None, (), (target,))

    matching_values = np.zeros((*shape[:-1], 3), dtype=np.complex128)
    matching_target = replace(
        target,
        values=matching_values,
        valid_mask=np.ones(matching_values.shape, dtype=np.bool_),
        reason_codes=(TargetValidity.VALID,) * matching_values.size,
        censoring_intervals=(None,) * matching_values.size,
    )
    bundle = EpisodeBundle(_episode(), None, (), (matching_target,))
    assert bundle.targets[0].values.shape[-1] == len(bundle.episode.subcarrier_offsets_hz)
    stored = write_episode_bundle(tmp_path, bundle)
    restored = read_episode_bundle(tmp_path, stored)
    assert restored.targets[0].definition.value_axes == axes
    np.testing.assert_array_equal(restored.targets[0].values, matching_values)


def test_legacy_v1_target_definition_cannot_be_republished_as_target_v2(tmp_path):
    bundle = _bundle()
    legacy_definition = migrate_legacy_target_definition(
        {
            "semantic": "geometric_direct_path",
            "units": "boolean",
            "source": "path_geometry_oracle",
        }
    )
    legacy_target = TargetValue(
        entity_id=bundle.targets[0].entity_id,
        definition=legacy_definition,
        values=np.asarray([1.0]),
        valid_mask=np.asarray([True]),
        reason_codes=(TargetValidity.VALID,),
        confidence="legacy_geometry_record",
    )
    legacy_bundle = replace(bundle, targets=(legacy_target,))

    with pytest.raises(ValueError, match="read-only for canonical UMCD publication"):
        write_episode_bundle(tmp_path, legacy_bundle)
    assert list(tmp_path.iterdir()) == []


def test_historical_frame_visibility_v1_layer_reads_without_axis_upgrade(tmp_path):
    bundle = _bundle()
    episode = bundle.episode
    stored = write_episode_bundle(tmp_path, bundle)
    manifest_path = tmp_path / stored.relative_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    definition = TargetDefinition.versioned(
        semantic="independent Mitsuba segment-intersection geometry oracle",
        units="boolean",
        source="mitsuba_segment_intersection_oracle",
        reference=(
            "independent finite-segment visibility query for the recorded TX/RX poses "
            "and scene geometry"
        ),
        aggregation_rule="one independent direct-path visibility value per frame",
        definition_version="umcd.rt-trajectory.geometric-los.v1",
    )
    values = np.asarray([1], dtype=np.uint8)
    payload = _pack_npz(
        {
            "schema": "umcd.target.v2",
            "entity_id": episode.links[0].link_id,
            "definition": asdict(definition),
            "reason_codes": [TargetValidity.VALID.value] * values.size,
            "confidence": "historical_geometry_record",
            "censoring_intervals": [None] * values.size,
        },
        values=values,
        valid_mask=np.ones(values.shape, dtype=np.bool_),
    )
    reference = _object_ref("targets", payload)
    _publish_immutable(tmp_path / reference["path"], payload)
    manifest["layers"].append({"layer": "targets", **reference})
    core = {key: manifest[key] for key in ("schema", "schema_version", "episode_id", "layers")}
    manifest["bundle_sha256"] = hashlib.sha256(_json_bytes(core)).hexdigest()
    manifest_path.write_bytes(_json_bytes(manifest) + b"\n")

    loaded = read_episode_bundle(tmp_path, manifest_path)
    historical = next(
        target
        for target in loaded.targets
        if target.definition.definition_version == "umcd.rt-trajectory.geometric-los.v1"
    )
    assert historical.definition.value_axes == ()
    np.testing.assert_array_equal(historical.values, values)


@pytest.mark.parametrize(
    ("version", "source", "reference"),
    [
        (
            "umcd.f04_rt_dynamic.direct_visibility.v2",
            "mitsuba_segment_intersection_oracle",
            "independent finite-segment Mitsuba geometry query for the RT scene and poses",
        ),
        (
            "umcd.f04_rt_dynamic.direct_visibility.v3",
            "numpy_loaded_scene_triangle_inventory",
            "digest-bound loaded-scene triangle query cross-checked with Mitsuba segment intersection and depth-zero Sionna PathSolver at each recorded pose",
        ),
    ],
)
def test_historical_dynamic_visibility_v2_v3_reads_without_axis_upgrade(
    tmp_path, version, source, reference
):
    bundle = _bundle()
    episode = bundle.episode
    stored = write_episode_bundle(tmp_path, bundle)
    manifest_path = tmp_path / stored.relative_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    definition = TargetDefinition.versioned(
        semantic="geometric direct-path visibility",
        units="boolean",
        source=source,
        reference=reference,
        aggregation_rule="one visibility value per trajectory frame; no spatial or temporal pooling",
        definition_version=version,
    )
    values = np.asarray([1], dtype=np.uint8)
    payload = _pack_npz(
        {
            "schema": "umcd.target.v2",
            "entity_id": episode.links[0].link_id,
            "definition": asdict(definition),
            "reason_codes": [TargetValidity.VALID.value],
            "confidence": "historical_geometry_record",
            "censoring_intervals": [None],
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

    loaded = read_episode_bundle(tmp_path, manifest_path)
    historical = next(
        target for target in loaded.targets if target.definition.definition_version == version
    )
    assert historical.definition.value_axes == ()
    assert historical.values.shape == (1,)
    np.testing.assert_array_equal(historical.values, values)


@pytest.mark.parametrize(
    ("version", "semantic", "units", "source", "reference", "aggregation", "value"),
    [
        (
            "umcd.standardized.effective-cfr.v1",
            "perfect_synchronized_effective_subcarrier_cfr",
            "complex channel gain (dimensionless)",
            "standardized Sionna TDL/CDL CFR adapter",
            "profile-relative excess-delay path truth sinc-discretized for this NR sample rate; "
            "the static per-subcarrier target does not encode time-domain ISI/ICI",
            "retain each effective subcarrier coefficient in receiver FFT-bin order; "
            "no antenna, symbol, or power averaging",
            np.asarray([1.0 + 0j], dtype=np.complex64),
        ),
        (
            "umcd.tdl-pusch.effective-cfr.v1",
            "perfect synchronized effective-subcarrier CFR",
            "complex channel gain (dimensionless)",
            "sionna_phy_ofdmchannel_return_channel",
            "OFDMChannel return_channel with the configured TDL profile, carrier, "
            "sample rate, and effective FFT-bin mapping",
            "retain each effective subcarrier coefficient; no antenna, symbol, or power averaging",
            np.asarray([1.0 + 0j], dtype=np.complex64),
        ),
        (
            "umcd.rt-trajectory.absolute-cfr.v1",
            "absolute-delay full-path complex CFR on effective PUSCH subcarriers",
            "complex channel gain (dimensionless)",
            "sionna_rt_pathsolver",
            "Sionna RT full solver-valid path set evaluated with absolute propagation "
            "delays and configured carrier/subcarrier offsets",
            "coherent sum of all retained solver-valid paths per effective subcarrier and frame; "
            "no path-count or power normalization",
            np.asarray([1.0 + 0j], dtype=np.complex64),
        ),
        (
            "umcd.rt-trajectory.symbol-cfr.v1",
            "absolute-delay full-path CFR with path-Doppler phase evolution sampled at PUSCH OFDM symbols",
            "complex channel gain (dimensionless)",
            "sionna_rt_paths_cfr_doppler",
            "absolute-delay RT path channel evaluated at the declared NR OFDM symbol "
            "start timestamps",
            "coherent sum of retained solver-valid paths per effective subcarrier and OFDM symbol; "
            "no path or symbol averaging",
            np.asarray([1.0 + 0j], dtype=np.complex64),
        ),
        (
            "umcd.rt-trajectory.geometric-los.v1",
            "independent Mitsuba segment-intersection geometry oracle",
            "boolean",
            "mitsuba_segment_intersection_oracle",
            "independent finite-segment visibility query for the recorded TX/RX poses "
            "and scene geometry",
            "one independent direct-path visibility value per frame",
            np.asarray([1], dtype=np.uint8),
        ),
        (
            "umcd.rt-trajectory.geometric-los.v2",
            "digest-bound NumPy loaded-scene direct-visibility oracle",
            "boolean",
            "numpy_loaded_scene_triangle_inventory",
            "loaded-scene triangle query cross-checked with Mitsuba segment intersection "
            "and depth-zero Sionna PathSolver for each recorded TX/RX pose and geometry digest",
            "one independent direct-path visibility value per frame",
            np.asarray([1], dtype=np.uint8),
        ),
        (
            "umcd.f03_rt_static.direct_visibility.v1",
            "geometric direct-path visibility",
            "boolean",
            "independent_segment_visibility_oracle",
            "finite scene-segment visibility cross-checked by depth-zero PathSolver",
            "one direct-path visibility label per link; no temporal or port pooling",
            np.asarray([1], dtype=np.uint8),
        ),
        (
            "umcd.f04_rt_dynamic.direct_visibility.v2",
            "geometric direct-path visibility",
            "boolean",
            "mitsuba_segment_intersection_oracle",
            "independent finite-segment Mitsuba geometry query for the RT scene and poses",
            "one visibility value per trajectory frame; no spatial or temporal pooling",
            np.asarray([1], dtype=np.uint8),
        ),
        (
            "umcd.f04_rt_dynamic.direct_visibility.v3",
            "geometric direct-path visibility",
            "boolean",
            "numpy_loaded_scene_triangle_inventory",
            "digest-bound loaded-scene triangle query cross-checked with Mitsuba segment "
            "intersection and depth-zero Sionna PathSolver at each recorded pose",
            "one visibility value per trajectory frame; no spatial or temporal pooling",
            np.asarray([1], dtype=np.uint8),
        ),
    ],
)
def test_superseded_target_versions_are_read_only_for_umcd_publication(
    tmp_path, version, semantic, units, source, reference, aggregation, value
):
    target = TargetValue(
        entity_id="gNB-1-to-UE-1",
        definition=TargetDefinition.versioned(
            semantic=semantic,
            units=units,
            source=source,
            reference=reference,
            aggregation_rule=aggregation,
            definition_version=version,
        ),
        values=value,
        valid_mask=np.ones(value.shape, dtype=bool),
        reason_codes=(TargetValidity.VALID,) * value.size,
        confidence="historical_read_only_fixture",
    )

    with pytest.raises(ValueError, match="read-only for canonical UMCD publication"):
        _target_payload(target)


@pytest.mark.parametrize(
    ("version", "value_axes"),
    [
        ("umcd.f04_rt_dynamic.direct_visibility.v3", None),
        ("umcd.f04_rt_dynamic.direct_visibility.v4", ("frame",)),
    ],
)
def test_frame_visibility_target_matches_episode_timestamp_roster(version, value_axes):
    episode = replace(_episode(), frame_times_s=(0.0, 0.01, 0.02))
    definition = TargetDefinition.versioned(
        semantic="geometric direct-path visibility",
        units="boolean",
        source="numpy_loaded_scene_triangle_inventory",
        reference=(
            "digest-bound loaded-scene triangle query cross-checked with Mitsuba "
            "segment intersection and depth-zero Sionna PathSolver at each recorded pose"
        ),
        aggregation_rule="one visibility value per trajectory frame; no spatial or temporal pooling",
        definition_version=version,
        value_axes=value_axes,
    )
    target = TargetValue(
        entity_id="gNB-1-to-UE-1",
        definition=definition,
        values=np.asarray([1, 0, 1], dtype=np.uint8),
        valid_mask=np.ones(3, dtype=np.bool_),
        reason_codes=(TargetValidity.VALID,) * 3,
        confidence="geometry_verified",
    )
    bundle = EpisodeBundle(episode, None, (), (target,))
    assert len(bundle.targets[0].values) == len(episode.frame_times_s)

    truncated = TargetValue(
        entity_id=target.entity_id,
        definition=definition,
        values=np.asarray([1, 0], dtype=np.uint8),
        valid_mask=np.ones(2, dtype=np.bool_),
        reason_codes=(TargetValidity.VALID,) * 2,
        confidence="geometry_verified",
    )
    with pytest.raises(ValueError, match="frame-time roster"):
        EpisodeBundle(episode, None, (), (truncated,))


def test_static_visibility_target_requires_one_value_per_link():
    definition = TargetDefinition.versioned(
        semantic="geometric direct-path visibility",
        units="boolean",
        source="numpy_loaded_scene_triangle_inventory",
        reference=(
            "digest-bound loaded-scene triangle visibility cross-checked by Mitsuba "
            "segment intersection and depth-zero PathSolver"
        ),
        aggregation_rule="one direct-path visibility label per link; no temporal or port pooling",
        definition_version="umcd.f03_rt_static.direct_visibility.v2",
    )

    with pytest.raises(ValueError, match="exactly one value"):
        TargetValue(
            entity_id="gNB-1-to-UE-1",
            definition=definition,
            values=np.asarray([1, 0], dtype=np.uint8),
            valid_mask=np.ones(2, dtype=np.bool_),
            reason_codes=(TargetValidity.VALID,) * 2,
            confidence="geometry_verified",
        )

    with pytest.raises(ValueError, match=r"fixed \(1,\) storage shape"):
        TargetValue(
            entity_id="gNB-1-to-UE-1",
            definition=definition,
            values=np.asarray([[1]], dtype=np.uint8),
            valid_mask=np.ones((1, 1), dtype=np.bool_),
            reason_codes=(TargetValidity.VALID,),
            confidence="geometry_verified",
        )


def test_static_visibility_v2_roundtrips_through_canonical_umcd_store(tmp_path):
    definition = TargetDefinition.versioned(
        semantic="geometric direct-path visibility",
        units="boolean",
        source="numpy_loaded_scene_triangle_inventory",
        reference=(
            "digest-bound loaded-scene triangle visibility cross-checked by Mitsuba "
            "segment intersection and depth-zero PathSolver"
        ),
        aggregation_rule="one direct-path visibility label per link; no temporal or port pooling",
        definition_version="umcd.f03_rt_static.direct_visibility.v2",
    )
    target = TargetValue(
        entity_id="gNB-1-to-UE-1",
        definition=definition,
        values=np.asarray([1], dtype=np.uint8),
        valid_mask=np.asarray([True], dtype=np.bool_),
        reason_codes=(TargetValidity.VALID,),
        confidence="synthetic_geometry_contract_fixture",
    )
    bundle = EpisodeBundle(_episode(), None, (), (target,))

    stored = write_episode_bundle(tmp_path, bundle)
    restored = read_episode_bundle(tmp_path, stored)

    assert restored.targets[0].definition.definition_version == definition.definition_version
    assert restored.targets[0].definition.definition_sha256 == definition.definition_sha256
    assert restored.targets[0].values.shape == (1,)
    np.testing.assert_array_equal(restored.targets[0].values, [1])


def test_complex_target_values_preserve_dtype_through_episode_store_roundtrip(tmp_path):
    bundle = _bundle()
    definition = TargetDefinition.versioned(
        semantic="complex CFR coefficients",
        units="complex channel gain (dimensionless)",
        source="analytic test fixture",
        reference="declared fixture gauge",
        aggregation_rule="preserve each declared complex coefficient",
        definition_version="umcd.test.complex-cfr.v1",
    )
    values = np.asarray([1.25 - 0.5j, 0.0 + 0.0j], dtype=np.complex64)
    complex_target = TargetValue(
        entity_id="gNB-1-to-UE-1",
        definition=definition,
        values=values,
        valid_mask=np.asarray([True, False]),
        reason_codes=(TargetValidity.VALID, TargetValidity.NOT_OBSERVABLE),
        confidence="synthetic_fixture",
    )
    bundle = replace(bundle, targets=(complex_target,))

    stored = write_episode_bundle(tmp_path, bundle)
    restored = read_episode_bundle(tmp_path, stored)

    assert restored.targets[0].values.dtype == np.dtype(np.complex64)
    np.testing.assert_array_equal(restored.targets[0].values, values)
    np.testing.assert_array_equal(restored.targets[0].valid_mask, [True, False])
    assert restored.targets[0].reason_codes == complex_target.reason_codes
