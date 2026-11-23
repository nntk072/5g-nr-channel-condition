"""UMCD path-oracle, ragged path-table and path-interaction schemas."""

from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from datasets.episode_store import (
    EpisodeBundle,
    _target_payload,
    read_episode_bundle,
    write_episode_bundle,
)
from datasets.path_oracle import PathOracleTable, PathSolveStatus, rms_delay_spread_target
from datasets.targets import (
    TargetDefinition,
    TargetValidity,
    TargetValue,
)
from tests.test_datasets.umcd_schema_support import (
    _episode,
)


def test_ragged_path_oracle_keeps_no_path_distinct_from_solver_failure():
    paths = PathOracleTable(
        link_ids=("gNB-1-to-UE-1", "gNB-1-to-UE-1"),
        frame_indices=np.asarray([0, 0]),
        rx_port_indices=np.asarray([0, 0]),
        tx_port_indices=np.asarray([0, 1]),
        path_offsets=np.asarray([0, 2, 2]),
        solve_status=(PathSolveStatus.SOLVED, PathSolveStatus.NO_PATH),
        delay_s=np.asarray([10e-9, 15e-9]),
        complex_gain=np.asarray([0.1 + 0.2j, 0.03 - 0.01j]),
        interaction_order=np.asarray([0, 1]),
        doppler_hz=np.asarray([0.0, 14.0]),
        aoa_azimuth_elevation_rad=np.asarray([[0.0, 0.0], [0.4, -0.1]]),
        aod_azimuth_elevation_rad=np.asarray([[np.pi, 0.0], [-0.4, 0.1]]),
        path_kind=("direct", "reflection"),
    )

    paths.validate_for(_episode())
    assert paths.group_count == 2
    assert paths.path_count == 2
    assert paths.group_slice(0) == slice(0, 2)
    assert paths.group_slice(1) == slice(2, 2)
    assert paths.solve_status[1] is PathSolveStatus.NO_PATH
    assert not paths.delay_s.flags.writeable


def test_rms_delay_spread_producer_emits_frame_values_and_preserves_unavailable_frames():
    paths = PathOracleTable(
        link_ids=("link", "link", "link"),
        frame_indices=np.asarray([0, 1, 2]),
        rx_port_indices=np.asarray([0, 0, 0]),
        tx_port_indices=np.asarray([0, 0, 0]),
        path_offsets=np.asarray([0, 2, 2, 3]),
        solve_status=(PathSolveStatus.SOLVED, PathSolveStatus.NO_PATH, PathSolveStatus.SOLVED),
        delay_s=np.asarray([0.0, 10e-9, 20e-9]),
        complex_gain=np.asarray([1e308 + 1e308j, 1e308 + 1e308j, 0.0 + 0j]),
        interaction_order=np.asarray([0, 1, 0]),
    )

    target = rms_delay_spread_target(paths, link_id="link", frame_count=3)

    assert target.definition.definition_version == "umcd.rms_delay_spread.v2"
    assert target.definition.value_axes == ("frame",)
    assert target.values[0] == pytest.approx(5e-9)
    assert target.valid_mask.tolist() == [True, False, False]
    assert target.reason_codes == (
        TargetValidity.VALID,
        TargetValidity.NOT_OBSERVABLE,
        TargetValidity.NOT_OBSERVABLE,
    )


def test_rms_delay_spread_producer_rejects_multiport_and_failed_solves():
    multiport = PathOracleTable(
        link_ids=("link", "link"),
        frame_indices=np.asarray([0, 0]),
        rx_port_indices=np.asarray([0, 1]),
        tx_port_indices=np.asarray([0, 0]),
        path_offsets=np.asarray([0, 1, 2]),
        solve_status=(PathSolveStatus.SOLVED, PathSolveStatus.SOLVED),
        delay_s=np.asarray([0.0, 1e-9]),
        complex_gain=np.asarray([1.0 + 0j, 1.0 + 0j]),
        interaction_order=np.asarray([0, 0]),
    )
    with pytest.raises(ValueError, match="multiport aggregation is not defined"):
        rms_delay_spread_target(multiport, link_id="link", frame_count=1)

    failed = PathOracleTable(
        link_ids=("link",),
        frame_indices=np.asarray([0]),
        rx_port_indices=np.asarray([0]),
        tx_port_indices=np.asarray([0]),
        path_offsets=np.asarray([0, 0]),
        solve_status=(PathSolveStatus.SOLVER_FAILURE,),
        delay_s=np.asarray([], dtype=np.float64),
        complex_gain=np.asarray([], dtype=np.complex128),
        interaction_order=np.asarray([], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="solver_failure frame 0"):
        rms_delay_spread_target(failed, link_id="link", frame_count=1)


@pytest.mark.parametrize(
    ("offsets", "statuses", "path_count", "message"),
    [
        ([0, 1, 2], [PathSolveStatus.SOLVED], 2, "path_offsets must have"),
        ([0, 0], [PathSolveStatus.SOLVED], 0, "must contain at least one path"),
        ([0, 1], [PathSolveStatus.NO_PATH], 1, "must not contain fabricated"),
        ([0, 1], [PathSolveStatus.SOLVER_FAILURE], 1, "must not contain fabricated"),
    ],
)
def test_ragged_path_table_rejects_inconsistent_outcome_and_offsets(
    offsets, statuses, path_count, message
):
    with pytest.raises(ValueError, match=message):
        PathOracleTable(
            link_ids=("gNB-1-to-UE-1",),
            frame_indices=np.asarray([0]),
            rx_port_indices=np.asarray([0]),
            tx_port_indices=np.asarray([0]),
            path_offsets=np.asarray(offsets),
            solve_status=tuple(statuses),
            delay_s=np.full(path_count, 1e-9),
            complex_gain=np.full(path_count, 1.0 + 0j),
            interaction_order=np.zeros(path_count, dtype=np.int64),
        )


def test_ragged_path_table_validates_link_frame_and_port_references():
    episode = _episode()
    paths = PathOracleTable(
        link_ids=("missing",),
        frame_indices=np.asarray([0]),
        rx_port_indices=np.asarray([0]),
        tx_port_indices=np.asarray([0]),
        path_offsets=np.asarray([0, 1]),
        solve_status=(PathSolveStatus.SOLVED,),
        delay_s=np.asarray([1e-9]),
        complex_gain=np.asarray([1.0 + 0j]),
        interaction_order=np.asarray([0]),
    )
    with pytest.raises(ValueError, match="missing episode link"):
        paths.validate_for(episode)


def test_path_oracle_rejects_omitted_port_solve_outcome():
    episode = _episode()
    paths = PathOracleTable(
        link_ids=("gNB-1-to-UE-1",),
        frame_indices=np.asarray([0]),
        rx_port_indices=np.asarray([0]),
        tx_port_indices=np.asarray([0]),
        path_offsets=np.asarray([0, 1]),
        solve_status=(PathSolveStatus.SOLVED,),
        delay_s=np.asarray([1e-9]),
        complex_gain=np.asarray([1.0 + 0j]),
        interaction_order=np.asarray([0]),
    )

    with pytest.raises(ValueError, match="every episode link/frame/port pair exactly"):
        paths.validate_for(episode)


def test_path_interaction_trace_rejects_partial_or_misaligned_geometry():
    common = {
        "link_ids": ("link-1",),
        "frame_indices": np.asarray([0]),
        "rx_port_indices": np.asarray([0]),
        "tx_port_indices": np.asarray([0]),
        "path_offsets": np.asarray([0, 1]),
        "solve_status": (PathSolveStatus.SOLVED,),
        "delay_s": np.asarray([2e-9]),
        "complex_gain": np.asarray([0.25 + 0.1j]),
        "interaction_order": np.asarray([1]),
    }
    interactions = {
        "interaction_offsets": np.asarray([0, 1]),
        "interaction_types": np.asarray([1]),
        "object_ids": np.asarray([7]),
        "primitive_ids": np.asarray([3]),
        "interaction_vertices_m": np.asarray([[1.0, 2.0, 3.0]]),
        "interaction_vertex_frame": "test_scene_xyz_m",
    }
    with pytest.raises(ValueError, match="must be supplied together"):
        PathOracleTable(**common, interaction_offsets=interactions["interaction_offsets"])
    with pytest.raises(ValueError, match="agree with per-path interaction_order"):
        PathOracleTable(
            **common,
            **{
                **interactions,
                "interaction_offsets": np.asarray([0, 0]),
            },
        )
    with pytest.raises(ValueError, match=r"finite shape \[interaction, xyz\]"):
        PathOracleTable(
            **common,
            **{
                **interactions,
                "interaction_vertices_m": np.asarray([[1.0, np.nan, 3.0]]),
            },
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("aoa_azimuth_elevation_rad", np.asarray([[True, False]])),
        ("aod_azimuth_elevation_rad", np.asarray([["0.0", "0.0"]])),
        ("aoa_azimuth_elevation_rad", np.asarray([[0.0, 0.0]], dtype=object)),
        ("aod_azimuth_elevation_rad", np.asarray([[0.0 + 1j, 0.0 + 0j]])),
        ("interaction_vertices_m", np.asarray([[True, False, True]])),
        ("interaction_vertices_m", np.asarray([["1", "2", "3"]])),
        ("interaction_vertices_m", np.asarray([[1.0, 2.0, 3.0]], dtype=object)),
        ("interaction_vertices_m", np.asarray([[1.0 + 1j, 2.0, 3.0]])),
    ],
)
def test_path_oracle_rejects_non_real_numeric_angle_and_vertex_dtypes(field, value):
    arguments = {
        "link_ids": ("link-1",),
        "frame_indices": np.asarray([0]),
        "rx_port_indices": np.asarray([0]),
        "tx_port_indices": np.asarray([0]),
        "path_offsets": np.asarray([0, 1]),
        "solve_status": (PathSolveStatus.SOLVED,),
        "delay_s": np.asarray([1e-9]),
        "complex_gain": np.asarray([1.0 + 0j]),
        "interaction_order": np.asarray([0 if field != "interaction_vertices_m" else 1]),
    }
    if field == "interaction_vertices_m":
        arguments.update(
            {
                "interaction_offsets": np.asarray([0, 1]),
                "interaction_types": np.asarray([1]),
                "object_ids": np.asarray([0]),
                "primitive_ids": np.asarray([0]),
                "interaction_vertex_frame": "scene_xyz_m",
            }
        )

    with pytest.raises(TypeError, match="real numeric dtype"):
        PathOracleTable(**arguments, **{field: value})


@pytest.mark.parametrize("field", ("frame_indices", "path_offsets", "interaction_types"))
def test_path_oracle_rejects_integer_values_that_overflow_int64(field):
    arguments = {
        "link_ids": ("link-1",),
        "frame_indices": np.asarray([0], dtype=np.uint64),
        "rx_port_indices": np.asarray([0], dtype=np.uint64),
        "tx_port_indices": np.asarray([0], dtype=np.uint64),
        "path_offsets": np.asarray([0, 1], dtype=np.uint64),
        "solve_status": (PathSolveStatus.SOLVED,),
        "delay_s": np.asarray([1e-9]),
        "complex_gain": np.asarray([1.0 + 0j]),
        "interaction_order": np.asarray([0], dtype=np.uint64),
    }
    if field == "interaction_types":
        arguments.update(
            {
                "interaction_order": np.asarray([1], dtype=np.uint64),
                "interaction_offsets": np.asarray([0, 1], dtype=np.uint64),
                "interaction_types": np.asarray([1], dtype=np.uint64),
                "object_ids": np.asarray([0], dtype=np.uint64),
                "primitive_ids": np.asarray([0], dtype=np.uint64),
                "interaction_vertices_m": np.asarray([[0.0, 0.0, 0.0]]),
                "interaction_vertex_frame": "scene_xyz_m",
            }
        )
    arguments[field] = (
        np.asarray([0, 2**63], dtype=np.uint64)
        if field == "path_offsets"
        else np.asarray([2**63], dtype=np.uint64)
    )

    with pytest.raises(ValueError, match="fit int64"):
        PathOracleTable(**arguments)


def test_path_oracle_rejects_unregistered_coefficient_convention():
    with pytest.raises(ValueError, match="unsupported path coefficient convention"):
        PathOracleTable(
            link_ids=("link-1",),
            frame_indices=np.asarray([0]),
            rx_port_indices=np.asarray([0]),
            tx_port_indices=np.asarray([0]),
            path_offsets=np.asarray([0, 1]),
            solve_status=(PathSolveStatus.SOLVED,),
            delay_s=np.asarray([1e-9]),
            complex_gain=np.asarray([1.0 + 0j]),
            interaction_order=np.asarray([0]),
            coefficient_convention="arbitrary_complex_amplitude_v1",
        )


def test_path_oracle_rejects_padded_interaction_coordinate_frame():
    with pytest.raises(ValueError, match="canonical named coordinate frame"):
        PathOracleTable(
            link_ids=("link-1",),
            frame_indices=np.asarray([0]),
            rx_port_indices=np.asarray([0]),
            tx_port_indices=np.asarray([0]),
            path_offsets=np.asarray([0, 1]),
            solve_status=(PathSolveStatus.SOLVED,),
            delay_s=np.asarray([1e-9]),
            complex_gain=np.asarray([1.0 + 0j]),
            interaction_order=np.asarray([1]),
            interaction_offsets=np.asarray([0, 1]),
            interaction_types=np.asarray([1]),
            object_ids=np.asarray([0]),
            primitive_ids=np.asarray([0]),
            interaction_vertices_m=np.asarray([[0.0, 0.0, 0.0]]),
            interaction_vertex_frame=" scene_xyz_m ",
        )


def test_rms_delay_spread_v2_persists_frame_axis_and_binds_roster(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    episode = _episode()
    episode = replace(episode, links=(replace(episode.links[0], tx_port_indices=(0,)),))
    paths = PathOracleTable(
        link_ids=("gNB-1-to-UE-1",),
        frame_indices=np.asarray([0]),
        rx_port_indices=np.asarray([0]),
        tx_port_indices=np.asarray([0]),
        path_offsets=np.asarray([0, 2]),
        solve_status=(PathSolveStatus.SOLVED,),
        delay_s=np.asarray([10e-9, 14e-9]),
        complex_gain=np.asarray([1.0 + 0.0j, 1.0 + 0.0j]),
        interaction_order=np.asarray([0, 0]),
    )
    target = rms_delay_spread_target(paths, link_id="gNB-1-to-UE-1", frame_count=1)
    bundle = EpisodeBundle(episode, paths, (), (target,))
    with np.load(io.BytesIO(_target_payload(target)), allow_pickle=False) as archive:
        target_metadata = json.loads(archive["metadata_json"].tobytes())
    target_schema = json.loads(
        (Path(__file__).parents[2] / "datasets/schema/umcd-target-layer.v1.schema.json").read_text()
    )
    jsonschema.Draft202012Validator(target_schema).validate(target_metadata)
    stored = write_episode_bundle(tmp_path, replace(bundle, targets=(target,)))
    loaded = read_episode_bundle(tmp_path, stored)
    assert loaded.targets[0].definition.value_axes == ("frame",)
    np.testing.assert_array_equal(loaded.targets[0].values, target.values)

    wrong_entity = replace(target, entity_id="episode-1")
    with pytest.raises(ValueError, match="identify an episode link"):
        EpisodeBundle(bundle.episode, paths, (), (wrong_entity,))
    wrong_shape = replace(
        target,
        values=np.asarray([2e-9, 2e-9]),
        valid_mask=np.ones(2, dtype=bool),
        reason_codes=(TargetValidity.VALID, TargetValidity.VALID),
        censoring_intervals=(None, None),
    )
    with pytest.raises(ValueError, match="frame-time roster"):
        EpisodeBundle(bundle.episode, paths, (), (wrong_shape,))
    wrong_value = replace(target, values=np.asarray([12e-9]))
    with pytest.raises(ValueError, match="exactly replay from its path-oracle source"):
        EpisodeBundle(bundle.episode, paths, (), (wrong_value,))
    wrong_missingness = replace(
        target,
        values=np.asarray([0.0]),
        valid_mask=np.asarray([False]),
        reason_codes=(TargetValidity.NOT_OBSERVABLE,),
    )
    with pytest.raises(ValueError, match="exactly replay from its path-oracle source"):
        EpisodeBundle(bundle.episode, paths, (), (wrong_missingness,))
    with pytest.raises(ValueError, match="requires its declared path-oracle source layer"):
        EpisodeBundle(bundle.episode, None, (), (target,))

    legacy_definition = TargetDefinition.versioned(
        semantic="rms_delay_spread",
        units="s",
        source="path_oracle",
        reference="absolute_physical_channel_path_oracle",
        aggregation_rule="power_weighted_rms_over_valid_paths",
        definition_version="umcd.rms_delay_spread.v1",
    )
    legacy_target = TargetValue(
        entity_id="gNB-1-to-UE-1",
        definition=legacy_definition,
        values=np.asarray([12e-9]),
        valid_mask=np.asarray([True]),
        reason_codes=(TargetValidity.VALID,),
        confidence="legacy_contract_check",
    )
    with pytest.raises(ValueError, match="read-only for canonical UMCD publication"):
        write_episode_bundle(tmp_path / "legacy", replace(bundle, targets=(legacy_target,)))
