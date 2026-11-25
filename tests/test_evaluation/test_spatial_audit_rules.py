"""Spatial buffer audit: buffer, field, frame, coordinate and roster rules."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from evaluation.audits.spatial_audit import audit_spatial_buffer
from tests.test_evaluation.test_spatial_audit import make_records


def _records() -> tuple[list[dict[str, Any]], Any]:
    rows, split = make_records()
    records = [
        {**row, "coordinates_m": [float(index), 0.0], "coordinate_frame_id": "EPSG:32632"}
        for index, row in enumerate(rows)
    ]
    return records, split


@pytest.mark.parametrize("buffer", [0.0, -1.0, float("nan"), True])
def test_buffer_must_be_finite_and_positive(buffer) -> None:
    records, split = _records()
    with pytest.raises(ValueError, match="buffer_m must be finite and positive"):
        audit_spatial_buffer(records, split, buffer_m=buffer)


def test_coordinate_field_names_must_be_distinct_and_canonical() -> None:
    records, split = _records()
    with pytest.raises(ValueError, match="distinct nonempty canonical strings"):
        audit_spatial_buffer(
            records, split, buffer_m=1.0, coordinate_field="x", coordinate_frame_field="x"
        )
    with pytest.raises(ValueError, match="distinct nonempty canonical strings"):
        audit_spatial_buffer(records, split, buffer_m=1.0, coordinate_field=" x")


def test_records_must_be_nonempty() -> None:
    _, split = _records()
    with pytest.raises(ValueError, match="must not be empty"):
        audit_spatial_buffer([], split, buffer_m=1.0)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param(
            lambda r: r[0].update(sample_id=" padded"),
            "sample IDs must be non-empty canonical",
            id="sample-canonical",
        ),
        pytest.param(
            lambda r: r[1].update(sample_id=r[0]["sample_id"]),
            "non-empty and unique",
            id="sample-duplicate",
        ),
        pytest.param(
            lambda r: r[0].update(sample_id="absent-sample"),
            "absent from frozen split",
            id="sample-absent",
        ),
        pytest.param(
            lambda r: r[0].update(coordinate_frame_id=""),
            "must be a non-empty canonical string",
            id="frame-empty",
        ),
        pytest.param(
            lambda r: r[1].update(coordinate_frame_id="LOCAL"),
            "same coordinate frame",
            id="frame-mixed",
        ),
        pytest.param(
            lambda r: r[0].update(coordinates_m=[1.0, 2.0, 3.0, 4.0]),
            "finite 2D/3D point",
            id="point-size",
        ),
        pytest.param(
            lambda r: r[0].update(coordinates_m=[float("nan"), 0.0]),
            "finite 2D/3D point",
            id="point-nan",
        ),
        pytest.param(
            lambda r: r[0].update(coordinates_m=[1.0, 2.0, 3.0]), "same dimension", id="point-dims"
        ),
    ],
)
def test_records_must_name_their_samples_frames_and_points(change, message) -> None:
    records, split = _records()
    records = copy.deepcopy(records)
    change(records)
    with pytest.raises(ValueError, match=message):
        audit_spatial_buffer(records, split, buffer_m=1.0)


def test_every_frozen_sample_must_be_provided() -> None:
    records, split = _records()
    with pytest.raises(ValueError, match="do not cover exactly"):
        audit_spatial_buffer(records[:-1], split, buffer_m=1.0)
