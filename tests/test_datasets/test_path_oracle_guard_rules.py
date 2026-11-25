"""Path-oracle tables: target reduction, frame rosters, ragged-field shapes and episode binding."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from datasets.path_oracle import PathOracleTable, PathSolveStatus, rms_delay_spread_target
from datasets.path_oracle import reconstruct_path_oracle_cfr as reconstruct
from tests.test_datasets.umcd_schema_support import _episode

LINK = "gNB-1-to-UE-1"


def _table(**overrides: Any) -> PathOracleTable:
    kwargs: dict[str, Any] = {
        "link_ids": (LINK,),
        "frame_indices": np.asarray([0]),
        "rx_port_indices": np.asarray([0]),
        "tx_port_indices": np.asarray([0]),
        "path_offsets": np.asarray([0, 2]),
        "solve_status": (PathSolveStatus.SOLVED,),
        "delay_s": np.asarray([1e-9, 3e-9]),
        "complex_gain": np.asarray([1.0 + 0j, 0.5 + 0j]),
        "interaction_order": np.zeros(2, dtype=np.int64),
    }
    kwargs.update(overrides)
    return PathOracleTable(**kwargs)


def test_rms_delay_spread_is_power_weighted_over_valid_frames() -> None:
    # Path powers 1 and 0.25 at delays 1 ns and 3 ns, weighted by normalized power.
    powers = np.asarray([1.0, 0.25])
    delays = np.asarray([1e-9, 3e-9])
    weights = powers / powers.sum()
    mean = float(np.sum(weights * delays))
    expected = float(np.sqrt(np.sum(weights * (delays - mean) ** 2)))
    target = rms_delay_spread_target(_table(), link_id=LINK, frame_count=1)
    assert target.valid_mask.tolist() == [True]
    assert float(target.values[0]) == pytest.approx(expected)


def test_frames_without_power_are_unavailable_not_zero() -> None:
    target = rms_delay_spread_target(
        _table(complex_gain=np.zeros(2, dtype=complex)), link_id=LINK, frame_count=1
    )
    assert target.valid_mask.tolist() == [False]


@pytest.mark.parametrize(
    ("link_id", "frame_count", "message"),
    [
        pytest.param("absent-link", 1, "no groups for link", id="unknown-link"),
        pytest.param(LINK, 0, "frame_count must be a positive integer", id="zero-frames"),
        pytest.param(LINK, True, "frame_count must be a positive integer", id="bool-frames"),
        pytest.param(LINK, 2, "exact frame roster", id="missing-frame"),
    ],
)
def test_target_reduction_checks_link_and_frame_roster(link_id, frame_count, message) -> None:
    with pytest.raises(ValueError, match=message):
        rms_delay_spread_target(_table(), link_id=link_id, frame_count=frame_count)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({"frame_indices": np.asarray(["0"])}, "must use a real or complex", id="str"),
        pytest.param(
            {"frame_indices": np.asarray([0.0 + 1j])},
            "must use an integer dtype",
            id="complex-to-int",
        ),
        pytest.param({"frame_indices": np.asarray([[0]])}, "one-dimensional array", id="2d"),
        pytest.param(
            {"frame_indices": np.asarray([2**63], dtype=np.uint64)}, "must fit int64", id="overflow"
        ),
        pytest.param(
            {"delay_s": np.asarray([1e-9 + 0j, 3e-9 + 0j])},
            "must use a real numeric dtype",
            id="complex-delay",
        ),
        pytest.param(
            {"path_offsets": np.asarray([0, 2, 1])},
            "one leading zero and one end per group",
            id="offset-count",
        ),
        pytest.param(
            {"frame_indices": np.asarray([0, 1])},
            "one frame, RX/TX port, and solve status",
            id="group-length",
        ),
        pytest.param(
            {"doppler_hz": np.asarray([1.0])},
            "must match the ragged path count",
            id="doppler-length",
        ),
        pytest.param(
            {"aoa_azimuth_elevation_rad": np.zeros((2, 3))},
            "must have finite shape \\[path, azimuth/elevation\\]",
            id="angle-shape",
        ),
        pytest.param(
            {"aoa_azimuth_elevation_rad": np.asarray([[4.0, 0.0], [0.0, 0.0]])},
            "azimuth must be in",
            id="azimuth-range",
        ),
        pytest.param(
            {"aod_azimuth_elevation_rad": np.asarray([[0.0, 2.0], [0.0, 0.0]])},
            "elevation must be in",
            id="elevation-range",
        ),
        pytest.param(
            {
                "interaction_offsets": np.asarray([0.0, 1.0, 2.0]),
                "interaction_types": np.asarray([1, 1]),
                "object_ids": np.asarray([0, 0]),
                "primitive_ids": np.asarray([0, 0]),
                "interaction_vertices_m": np.zeros((2, 3)),
                "interaction_vertex_frame": "scene",
                "interaction_order": np.asarray([1, 1]),
            },
            "interaction_offsets must use an integer dtype",
            id="interaction-dtype",
        ),
        pytest.param(
            {
                "interaction_offsets": np.asarray([0, 1]),
                "interaction_types": np.asarray([1, 1]),
                "object_ids": np.asarray([0, 0]),
                "primitive_ids": np.asarray([0, 0]),
                "interaction_vertices_m": np.zeros((2, 3)),
                "interaction_vertex_frame": "scene",
                "interaction_order": np.asarray([1, 0]),
            },
            "one leading zero per path table",
            id="interaction-offset-length",
        ),
        pytest.param(
            {
                "interaction_offsets": np.asarray([0, 2, 1]),
                "interaction_types": np.asarray([1, 1]),
                "object_ids": np.asarray([0, 0]),
                "primitive_ids": np.asarray([0, 0]),
                "interaction_vertices_m": np.zeros((2, 3)),
                "interaction_vertex_frame": "scene",
                "interaction_order": np.asarray([2, -1]),
            },
            "nonnegative",
            id="interaction-order-negative",
        ),
        pytest.param(
            {
                "interaction_offsets": np.asarray([0, 1, 2]),
                "interaction_types": np.asarray([1]),
                "object_ids": np.asarray([0, 0]),
                "primitive_ids": np.asarray([0, 0]),
                "interaction_vertices_m": np.zeros((2, 3)),
                "interaction_vertex_frame": "scene",
                "interaction_order": np.asarray([1, 1]),
            },
            "match interaction_offsets total",
            id="interaction-total",
        ),
        pytest.param(
            {
                "interaction_offsets": np.asarray([0, 1, 2]),
                "interaction_types": np.asarray([0, 1]),
                "object_ids": np.asarray([0, 0]),
                "primitive_ids": np.asarray([0, 0]),
                "interaction_vertices_m": np.zeros((2, 3)),
                "interaction_vertex_frame": "scene",
                "interaction_order": np.asarray([1, 1]),
            },
            "types and IDs must be nonnegative",
            id="interaction-type-zero",
        ),
        pytest.param(
            {
                "interaction_offsets": np.asarray([0, 1, 2]),
                "interaction_types": np.asarray([1, 1]),
                "object_ids": np.asarray([0, 0]),
                "primitive_ids": np.asarray([0, 0]),
                "interaction_vertices_m": np.zeros((1, 3)),
                "interaction_vertex_frame": "scene",
                "interaction_order": np.asarray([1, 1]),
            },
            "finite shape",
            id="vertex-shape",
        ),
        pytest.param(
            {
                "interaction_offsets": np.asarray([0, 1, 2]),
                "interaction_types": np.asarray([1, 1]),
                "object_ids": np.asarray([0, 0]),
                "primitive_ids": np.asarray([0, 0]),
                "interaction_vertices_m": np.zeros((2, 3)),
                "interaction_vertex_frame": " scene",
                "interaction_order": np.asarray([1, 1]),
            },
            "canonical named coordinate frame",
            id="vertex-frame-name",
        ),
        pytest.param(
            {"interaction_vertex_frame": "scene"},
            "requires interaction geometry arrays",
            id="frame-without-geometry",
        ),
        pytest.param(
            {"coefficient_convention": " spaced "},
            "non-empty canonical string",
            id="convention-blank",
        ),
        pytest.param(
            {"coefficient_convention": "other-convention"},
            "unsupported path coefficient convention",
            id="convention-unknown",
        ),
    ],
)
def test_path_table_rejects_ragged_and_optional_field_violations(overrides, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _table(**overrides)


def test_group_keys_must_be_unique_and_status_consistent() -> None:
    with pytest.raises(ValueError, match="path group keys must be unique"):
        _table(
            link_ids=(LINK, LINK),
            frame_indices=np.asarray([0, 0]),
            rx_port_indices=np.asarray([0, 0]),
            tx_port_indices=np.asarray([0, 0]),
            path_offsets=np.asarray([0, 1, 2]),
            solve_status=(PathSolveStatus.SOLVED, PathSolveStatus.SOLVED),
            delay_s=np.asarray([1e-9, 2e-9]),
            complex_gain=np.asarray([1.0, 1.0], dtype=complex),
            interaction_order=np.zeros(2, dtype=np.int64),
        )


def test_group_slices_are_bounded_and_typed() -> None:
    table = _table()
    assert table.group_slice(0) == slice(0, 2)
    with pytest.raises(TypeError, match="group_index must be an integer"):
        table.group_slice(True)
    with pytest.raises(IndexError):
        table.group_slice(1)


def test_reconstruction_refuses_overflowing_sums() -> None:
    table = _table(complex_gain=np.asarray([1e308 + 0j, 1e308 + 0j]))
    with pytest.raises(ValueError, match="reconstructed path-oracle CFR must be finite"):
        reconstruct(table, carrier_hz=3.5e9, frequency_offset_hz=np.asarray([0.0, 1.0]))


def test_episode_binding_checks_frame_and_port_indices_against_the_episode() -> None:
    episode = _episode()
    with pytest.raises(ValueError, match="outside the episode time axis"):
        _table(frame_indices=np.asarray([99])).validate_for(episode)
    with pytest.raises(ValueError, match="outside the link port map"):
        _table(rx_port_indices=np.asarray([99])).validate_for(episode)
    with pytest.raises(ValueError, match="outside the link port map"):
        _table(tx_port_indices=np.asarray([99])).validate_for(episode)
