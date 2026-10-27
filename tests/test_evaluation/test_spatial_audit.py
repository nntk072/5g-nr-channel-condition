import pytest
from evaluation.independent_splits import make_independent_split
from evaluation.spatial_audit import audit_spatial_buffer


def make_records():
    rows = [
        {"sample_id": f"sample-{index}", "scene_group_id": f"scene-{index}"} for index in range(6)
    ]
    return rows, make_independent_split(rows, seed="spatial-audit")


def test_spatial_audit_reports_cross_split_neighbors_within_meter_buffer():
    rows, split = make_records()
    train_id = next(sample for sample, name in split.sample_to_split.items() if name == "train")
    validation_id = next(
        sample for sample, name in split.sample_to_split.items() if name == "validation"
    )
    test_id = next(sample for sample, name in split.sample_to_split.items() if name == "test")
    points = {row["sample_id"]: [1000.0 + index * 10, 0.0] for index, row in enumerate(rows)}
    points[train_id] = [0.0, 0.0]
    points[test_id] = [0.25, 0.0]
    points[validation_id] = [100.0, 0.0]
    spatial_records = [
        {
            **row,
            "coordinates_m": points[row["sample_id"]],
            "coordinate_frame_id": "EPSG:32632",
        }
        for row in rows
    ]

    report = audit_spatial_buffer(spatial_records, split, buffer_m=0.5)

    assert not report.is_clean
    assert len(report.pairs) == 1
    assert {report.pairs[0].sample_id_a, report.pairs[0].sample_id_b} == {
        train_id,
        test_id,
    }
    assert report.pairs[0].distance_m == pytest.approx(0.25)


def test_spatial_audit_is_clean_for_well_separated_projected_samples():
    rows, split = make_records()
    spatial_records = [
        {
            **row,
            "coordinates_m": [index * 100.0, index * 25.0, 4.0],
            "coordinate_frame_id": "EPSG:32632",
        }
        for index, row in enumerate(rows)
    ]

    report = audit_spatial_buffer(spatial_records, split, buffer_m=10.0)

    assert report.is_clean
    assert report.pairs == ()


@pytest.mark.parametrize(
    ("records", "message"),
    [
        (
            [
                {
                    "sample_id": "sample-0",
                    "coordinates_m": [0.0, 0.0],
                    "coordinate_frame_id": "EPSG:32632",
                }
            ],
            "exactly the samples",
        ),
        (
            [
                {
                    "sample_id": "sample-0",
                    "coordinates_m": [0.0, 0.0],
                    "coordinate_frame_id": "EPSG:32632",
                },
                {
                    "sample_id": "sample-0",
                    "coordinates_m": [1.0, 1.0],
                    "coordinate_frame_id": "EPSG:32632",
                },
            ],
            "unique",
        ),
    ],
)
def test_spatial_audit_rejects_incomplete_or_duplicate_records(records, message):
    _, split = make_records()
    with pytest.raises(ValueError, match=message):
        audit_spatial_buffer(records, split, buffer_m=1.0)


def test_spatial_audit_rejects_nonmetric_or_mixed_coordinate_shapes():
    rows, split = make_records()
    records = [
        {
            **row,
            "coordinates_m": [float(index), 0.0],
            "coordinate_frame_id": "EPSG:32632",
        }
        for index, row in enumerate(rows)
    ]
    records[2]["coordinates_m"] = [2.0, 0.0, 1.0]

    with pytest.raises(ValueError, match="same dimension"):
        audit_spatial_buffer(records, split, buffer_m=1.0)


def test_spatial_audit_rejects_mixed_coordinate_frames_and_noncanonical_ids():
    rows, split = make_records()
    records = [
        {
            **row,
            "coordinates_m": [float(index), 0.0],
            "coordinate_frame_id": "EPSG:32632",
        }
        for index, row in enumerate(rows)
    ]
    records[1]["coordinate_frame_id"] = "EPSG:32633"
    with pytest.raises(ValueError, match="same coordinate frame"):
        audit_spatial_buffer(records, split, buffer_m=1.0)

    records[1]["coordinate_frame_id"] = "EPSG:32632"
    records[0]["sample_id"] = f" {records[0]['sample_id']}"
    with pytest.raises(ValueError, match="canonical strings"):
        audit_spatial_buffer(records, split, buffer_m=1.0)
