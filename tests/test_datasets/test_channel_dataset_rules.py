"""Legacy channel dataset: observation arrays, labels, split bindings and concatenation rules."""

from __future__ import annotations

import numpy as np
import pytest
from datasets.channel import ChannelDataset, ChannelModelInputs


def _dataset(**overrides):
    n = 4
    kwargs = {
        "h": (np.ones((n, 2, 8)) + 1j * np.ones((n, 2, 8))).astype(np.complex64),
        "y": np.array([0, 1, 0, 1]),
        "group": np.array(["group0", "group0", "group1", "group1"]),
        "domain": np.array(["tdl"] * n),
        "source": np.array(["synthetic"] * n),
        "metadata": tuple({"split_subject_id": f"s{i}"} for i in range(n)),
        "generation_report": {"label_semantics": "x"},
        "label_semantics": "x",
    }
    kwargs.update(overrides)
    return ChannelDataset(**kwargs)


def test_dataset_round_trips_and_feeds_only_cfr_inputs() -> None:
    dataset = _dataset()
    assert dataset.model_inputs().observed_cfr.shape == (4, 2, 8)
    selected = dataset.select(np.array([0, 2]))
    assert len(selected.y) == 2


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({"h": np.ones((4, 2, 8))}, "h must be complex", id="real-h"),
        pytest.param({"h": (np.ones((4, 2, 7)) * 1j)}, "h must be complex", id="few-subcarriers"),
        pytest.param(
            {"h": np.full((4, 2, 8), np.nan + 0j)}, "channel must be finite", id="nonfinite"
        ),
        pytest.param({"y": np.array([0, 1, 2, 1])}, "labels must be 0 or 1", id="label-value"),
        pytest.param(
            {"group": np.array(["group0", "group0"])}, "group must be length 4", id="group-length"
        ),
        pytest.param({"metadata": ({},)}, "metadata count mismatch", id="metadata-count"),
        pytest.param(
            {"group": np.array([" group0", "group0", "group1", "group1"])},
            "group IDs must be nonempty",
            id="padded-group",
        ),
        pytest.param(
            {"label_semantics": " "},
            "label_semantics must be a non-empty string",
            id="semantics-blank",
        ),
        pytest.param(
            {"generation_report": {"label_semantics": "other"}},
            "disagrees with generation_report",
            id="semantics-mismatch",
        ),
        pytest.param(
            {"generation_report": {"value": float("nan")}},
            "finite JSON-compatible",
            id="report-nan",
        ),
        pytest.param(
            {"generation_report": ["not", "object"]}, "must be a JSON object", id="report-list"
        ),
    ],
)
def test_dataset_rejects_inconsistent_arrays_and_provenance(overrides, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _dataset(**overrides)


def test_split_child_binding_checks_identity_and_metadata() -> None:
    dataset = _dataset()
    with pytest.raises(ValueError, match="child_id must be a nonempty canonical string"):
        dataset.with_split_child_binding("", "kind")
    with pytest.raises(ValueError, match="child_kind must be a nonempty canonical string"):
        dataset.with_split_child_binding("child", " ")
    single = _dataset(metadata=tuple({"split_subject_id": "s0"} for _ in range(4)))
    bound = single.with_split_child_binding("child-a", "kind-a")
    assert all(row["split_child_id"] == "child-a" for row in bound.metadata)
    bad_rows = _dataset(metadata=("not-a-dict",) * 4)
    with pytest.raises(ValueError, match="metadata must be an object"):
        bad_rows.with_split_child_binding("child-a", "kind-a")
    blank_parent = _dataset(metadata=tuple({"split_subject_id": " "} for _ in range(4)))
    with pytest.raises(ValueError, match="split_subject_id"):
        blank_parent.with_split_child_binding("child-a", "kind-a")
    half_bound = _dataset(
        metadata=tuple({"split_subject_id": f"s{i}", "split_child_id": "child-a"} for i in range(4))
    )
    with pytest.raises(ValueError, match="incomplete split child metadata"):
        half_bound.with_split_child_binding("child-a", "kind-a")
    other = _dataset(
        metadata=tuple(
            {"split_subject_id": f"s{i}", "split_child_id": "other", "split_child_kind": "kind-a"}
            for i in range(4)
        )
    )
    with pytest.raises(ValueError, match="one parent|already bound"):
        other.with_split_child_binding("child-a", "kind-a")


def test_concatenate_requires_datasets_and_one_shape() -> None:
    with pytest.raises(ValueError, match="No datasets"):
        ChannelDataset.concatenate([])
    wide = _dataset(h=(np.ones((4, 2, 10)) * 1j).astype(np.complex64), metadata=_dataset().metadata)
    with pytest.raises(ValueError):
        ChannelDataset.concatenate([_dataset(), wide])
    assert len(ChannelDataset.concatenate([_dataset(), _dataset()]).y) == 8


def test_model_inputs_require_a_complex_three_axis_cfr() -> None:
    with pytest.raises(ValueError, match="observed_cfr must be complex"):
        ChannelModelInputs(observed_cfr=np.ones((2, 2)))
    with pytest.raises(ValueError, match="observed_cfr must be nonempty and finite"):
        ChannelModelInputs(observed_cfr=np.full((1, 1, 8), np.inf + 0j))
