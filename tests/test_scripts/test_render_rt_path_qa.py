import numpy as np
import pytest
from scripts.render_rt_path_qa import extract_path_polylines, select_stratified_indices


def test_extract_path_polylines_preserves_order_and_skips_padding_vertices():
    sources = np.array([-1.0, 0.0, 2.0])
    targets = np.array([2.0, 0.0, 1.0])
    vertices = np.zeros((2, 3, 3))
    vertices[0, 1] = [0.0, 2.0, 2.0]
    vertices[0, 2] = [0.0, -2.0, 2.0]
    vertices[1, 2] = [1.0, -2.0, 2.0]
    interactions = np.array([[0, 1, 4], [0, 0, 2]])
    valid = np.array([True, True, False])

    paths = extract_path_polylines(sources, targets, vertices, interactions, valid)

    assert len(paths) == 2
    assert paths[0]["interaction_label"] == "LOS candidate"
    assert paths[0]["direct_candidate"] is True
    np.testing.assert_array_equal(paths[0]["coordinates_m"], [sources, targets])
    assert paths[1]["interaction_label"] == "specular"
    assert paths[1]["interactions"] == ["specular"]
    np.testing.assert_array_equal(paths[1]["coordinates_m"], [sources, vertices[0, 1], targets])


def test_extract_path_polylines_marks_multi_mechanism_paths():
    points = extract_path_polylines(
        sources=np.zeros(3),
        targets=np.ones(3),
        vertices=np.array([[[0.2, 0.3, 0.4]], [[0.5, 0.6, 0.7]]]),
        interactions=np.array([[4], [8]]),
        valid=np.array([True]),
    )

    assert points[0]["interaction_label"] == "mixed"
    assert points[0]["interactions"] == ["diffraction", "refraction"]
    assert not points[0]["direct_candidate"]


@pytest.mark.parametrize(
    ("vertices", "interactions", "valid", "message"),
    [
        (np.zeros((2, 1, 2)), np.zeros((2, 1)), [True], "shape "),
        (np.zeros((2, 2, 3)), np.zeros((1, 2)), [True, True], "dimensions differ"),
        (np.full((1, 1, 3), np.nan), np.ones((1, 1)), [True], "finite"),
    ],
)
def test_extract_path_polylines_rejects_inconsistent_or_nonfinite_arrays(
    vertices, interactions, valid, message
):
    with pytest.raises(ValueError, match=message):
        extract_path_polylines(np.zeros(3), np.ones(3), vertices, interactions, valid)


def test_stratified_selection_uses_balanced_path_count_quantiles():
    metadata = [{"valid_paths_full": count} for count in (9, 1, 3, 8, 2, 4, 6, 10)]
    labels = np.array([1, 0, 0, 1, 0, 0, 1, 1])

    indices = select_stratified_indices(metadata, labels, examples_per_class=3)

    assert indices == [1, 2, 5, 6, 0, 7]
    assert [int(labels[index]) for index in indices] == [0, 0, 0, 1, 1, 1]


@pytest.mark.parametrize(
    ("metadata", "labels", "count", "message"),
    [
        ([{"valid_paths_full": 1}], [1], 1, "both binary"),
        ([{"valid_paths_full": 1}, {"valid_paths_full": 2}], [0], 1, "aligned"),
        ([{}, {}], [0, 1], 1, "valid_paths_full"),
        ([{"valid_paths_full": 1}, {"valid_paths_full": 2}], [0, 1], 0, "positive integer"),
    ],
)
def test_stratified_selection_rejects_missing_or_invalid_inputs(metadata, labels, count, message):
    with pytest.raises(ValueError, match=message):
        select_stratified_indices(metadata, np.asarray(labels), examples_per_class=count)
