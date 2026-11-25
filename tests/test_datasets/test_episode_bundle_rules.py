"""Episode bundle: layer identities, entity references and data-layer requirements."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from datasets.episode_store import EpisodeBundle
from datasets.path_oracle import rms_delay_spread_target
from datasets.targets import (
    _CANONICAL_TARGET_CONTRACTS,
    _CANONICAL_TARGET_PROVENANCE,
    TargetDefinition,
    TargetValidity,
    TargetValue,
)
from tests.test_datasets.umcd_schema_support import _bundle, _provenance_record


def test_the_fixture_bundle_is_valid() -> None:
    bundle = _bundle()
    assert len(bundle.targets) == 1 and len(bundle.channel_views) == 1


def test_channel_view_identities_must_be_unique() -> None:
    bundle = _bundle()
    view = bundle.channel_views[0]
    with pytest.raises(ValueError, match="channel view IDs must be unique"):
        EpisodeBundle(bundle.episode, bundle.path_oracle, (view, view), bundle.targets)


def test_target_semantics_must_be_unique_per_entity() -> None:
    bundle = _bundle()
    target = bundle.targets[0]
    with pytest.raises(ValueError, match="target semantic must be unique"):
        EpisodeBundle(bundle.episode, bundle.path_oracle, bundle.channel_views, (target, target))


def test_target_entities_must_be_the_episode_or_one_of_its_links() -> None:
    bundle = _bundle()
    ghost = replace(bundle.targets[0], entity_id="not-in-this-episode")
    with pytest.raises(ValueError, match="entity_id must reference the episode"):
        EpisodeBundle(bundle.episode, bundle.path_oracle, bundle.channel_views, (ghost,))


def test_provenance_records_must_be_run_provenance_values() -> None:
    bundle = _bundle()
    with pytest.raises(TypeError, match="provenance_records must contain UMCDRunProvenance values"):
        EpisodeBundle(
            bundle.episode,
            bundle.path_oracle,
            bundle.channel_views,
            bundle.targets,
            provenance_records=("not-a-record",),
        )


def test_provenance_record_identities_must_be_unique() -> None:
    bundle = _bundle()
    record = _provenance_record()
    with pytest.raises(ValueError, match="provenance record IDs must be unique"):
        EpisodeBundle(
            bundle.episode,
            bundle.path_oracle,
            bundle.channel_views,
            bundle.targets,
            provenance_records=(record, record),
        )


def test_provenance_subject_layer_must_exist_in_bundle() -> None:
    bundle = _bundle()
    record = replace(_provenance_record(), subject_layer="scenario_gbsm")
    with pytest.raises(ValueError, match="subject_layer must identify a populated bundle layer"):
        EpisodeBundle(
            bundle.episode,
            bundle.path_oracle,
            bundle.channel_views,
            bundle.targets,
            provenance_records=(record,),
        )


def test_scenario_gbsm_records_must_be_typed_and_single() -> None:
    bundle = _bundle()
    with pytest.raises(TypeError, match="scenario_gbsm_records must contain scenario GBSM records"):
        EpisodeBundle(
            bundle.episode,
            bundle.path_oracle,
            bundle.channel_views,
            bundle.targets,
            scenario_gbsm_records=("not-a-drop",),
        )


def test_episode_bundle_allows_at_most_one_scenario_gbsm_drop() -> None:
    bundle = _bundle()
    with pytest.raises(ValueError, match="can contain one scenario GBSM drop"):
        EpisodeBundle(
            bundle.episode,
            bundle.path_oracle,
            bundle.channel_views,
            bundle.targets,
            scenario_gbsm_records=(object(), object()),
        )


def test_a_bundle_with_no_data_layer_is_refused() -> None:
    bundle = _bundle()
    with pytest.raises(ValueError, match="needs at least one non-episode data layer"):
        EpisodeBundle(bundle.episode, None, (), ())


def test_direct_path_ablations_must_be_typed() -> None:
    bundle = _bundle()
    with pytest.raises(
        TypeError, match="direct_path_ablations must contain DirectPathAblation values"
    ):
        EpisodeBundle(
            bundle.episode,
            bundle.path_oracle,
            bundle.channel_views,
            bundle.targets,
            direct_path_ablations=("not-an-ablation",),
        )


def test_receiver_bundle_rejects_unrecognized_linked_observation_runtime_type() -> None:
    bundle = _bundle()
    binding = SimpleNamespace(
        link_id=bundle.episode.links[0].link_id,
        frame_index=0,
        observation=SimpleNamespace(sample_id="synthetic-sample"),
    )
    with pytest.raises(TypeError, match="supported linked contract"):
        EpisodeBundle(
            bundle.episode,
            bundle.path_oracle,
            (),
            (),
            receiver_observations=(binding,),  # type: ignore[arg-type]
        )


def test_receiver_sample_ids_must_be_unique_across_episode_frames() -> None:
    from tests.test_datasets.umcd_schema_support import _dynamic_episode, _receiver_binding

    binding = _receiver_binding()
    second_frame = replace(binding, frame_index=1)
    with pytest.raises(ValueError, match="sample IDs must be unique"):
        EpisodeBundle(
            _dynamic_episode(),
            None,
            (),
            (),
            receiver_observations=(binding, second_frame),
        )


def test_target_frequency_and_rms_replay_contracts_match_episode() -> None:
    from tests.test_datasets.test_umcd_store_integrity import (
        _episode_bundle,
        _versioned_complex_cfr_target,
    )

    bundle = _episode_bundle()
    frequency_target = replace(
        _versioned_complex_cfr_target(),
        entity_id=bundle.episode.links[0].link_id,
        values=np.asarray([1.0 + 0.0j, 2.0 + 0.0j]),
        valid_mask=np.asarray([True, True]),
        reason_codes=(TargetValidity.VALID, TargetValidity.VALID),
        censoring_intervals=(None, None),
    )
    with pytest.raises(ValueError, match="effective-subcarrier axis must match"):
        EpisodeBundle(bundle.episode, bundle.path_oracle, (), (frequency_target,))

    assert bundle.path_oracle is not None
    valid_rms = rms_delay_spread_target(
        bundle.path_oracle,
        link_id=bundle.episode.links[0].link_id,
        frame_count=len(bundle.episode.frame_times_s),
    )
    assert (
        EpisodeBundle(bundle.episode, bundle.path_oracle, (), (valid_rms,)).targets[0] is valid_rms
    )

    episode_entity_rms = replace(valid_rms, entity_id=bundle.episode.episode_id)
    with pytest.raises(ValueError, match="per-frame numeric target must identify an episode link"):
        EpisodeBundle(bundle.episode, bundle.path_oracle, (), (episode_entity_rms,))

    wrong_frame_count = replace(
        valid_rms,
        values=np.asarray([0.0, 1.0]),
        valid_mask=np.asarray([True, True]),
        reason_codes=(TargetValidity.VALID, TargetValidity.VALID),
        censoring_intervals=(None, None),
    )
    with pytest.raises(ValueError, match="per-frame numeric target shape"):
        EpisodeBundle(bundle.episode, bundle.path_oracle, (), (wrong_frame_count,))

    mismatched_values = replace(valid_rms, values=valid_rms.values + 1.0)
    with pytest.raises(ValueError, match="must exactly replay"):
        EpisodeBundle(bundle.episode, bundle.path_oracle, (), (mismatched_values,))


def test_direct_path_ablation_bindings_preserve_parent_grid_and_unique_children() -> None:
    from tests.test_datasets.test_umcd_store_integrity import (
        _episode_bundle_with_counterfactual,
    )

    bundle = _episode_bundle_with_counterfactual()
    ablation = bundle.direct_path_ablations[0]

    with pytest.raises(ValueError, match="require their source path-oracle layer"):
        replace(bundle, path_oracle=None, targets=())

    with pytest.raises(ValueError, match="parent must match the episode ID"):
        replace(
            bundle,
            direct_path_ablations=(replace(ablation, parent_subject_id="other-episode"),),
        )

    legacy_grid_mismatch = replace(
        ablation,
        carrier_hz=None,
        frequency_offset_hz=ablation.frequency_offset_hz + 1.0,
    )
    with pytest.raises(ValueError, match="frequency grid differs from episode grid"):
        replace(bundle, direct_path_ablations=(legacy_grid_mismatch,))

    with pytest.raises(ValueError, match="child IDs must be unique"):
        replace(bundle, direct_path_ablations=(ablation, ablation))


def _visibility_target(version: str, *, entity_id: str = "gNB-1-to-UE-1", values=(1.0,)):
    _name, semantic, units, _domain = _CANONICAL_TARGET_CONTRACTS[version]
    source, references, aggregations = _CANONICAL_TARGET_PROVENANCE[version]
    reference = next(iter(references))
    aggregation = next(iter(aggregations)) if isinstance(aggregations, frozenset) else aggregations
    definition = TargetDefinition.versioned(
        semantic=semantic,
        units=units,
        source=source,
        reference=reference,
        aggregation_rule=aggregation,
        definition_version=version,
        value_axes=("frame",) if version == "umcd.rms_delay_spread.v2" else None,
    )
    values_array = np.asarray(values, dtype=np.float64)
    return TargetValue(
        entity_id,
        definition,
        values_array,
        np.ones(values_array.shape, dtype=np.bool_),
        (TargetValidity.VALID,) * values_array.size,
        "synthetic_contract_check",
    )


def test_frame_visibility_targets_bind_to_link_and_frame_roster() -> None:
    bundle = _bundle()
    valid = _visibility_target("umcd.rt-trajectory.geometric-los.v2")
    validated = EpisodeBundle(bundle.episode, bundle.path_oracle, (), (valid,))
    assert validated.targets[0].entity_id == "gNB-1-to-UE-1"

    with pytest.raises(ValueError, match="frame visibility target must identify an episode link"):
        EpisodeBundle(
            bundle.episode,
            bundle.path_oracle,
            (),
            (
                _visibility_target(
                    "umcd.rt-trajectory.geometric-los.v2",
                    entity_id=bundle.episode.episode_id,
                ),
            ),
        )
    with pytest.raises(ValueError, match="frame visibility target shape"):
        EpisodeBundle(
            bundle.episode,
            bundle.path_oracle,
            (),
            (_visibility_target("umcd.rt-trajectory.geometric-los.v2", values=(1.0, 0.0)),),
        )


def test_static_visibility_targets_bind_to_episode_links() -> None:
    bundle = _bundle()

    with pytest.raises(ValueError, match="static visibility target must identify an episode link"):
        EpisodeBundle(
            bundle.episode,
            bundle.path_oracle,
            (),
            (
                _visibility_target(
                    "umcd.f03_rt_static.direct_visibility.v2",
                    entity_id=bundle.episode.episode_id,
                ),
            ),
        )


def test_rms_delay_spread_v2_requires_path_oracle_layer() -> None:
    bundle = _bundle()

    with pytest.raises(ValueError, match="requires its declared path-oracle source layer"):
        EpisodeBundle(
            bundle.episode,
            None,
            (),
            (_visibility_target("umcd.rms_delay_spread.v2"),),
        )
