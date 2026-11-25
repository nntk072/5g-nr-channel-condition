"""Episode bundle: layer identities, entity references and data-layer requirements."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from datasets.episode_store import EpisodeBundle
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
