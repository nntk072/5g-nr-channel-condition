"""Split lineage metadata: parent type, leakage relations, RT block size and position fields."""

from __future__ import annotations

import pytest
from datasets.split_lineage import derive_channel_dataset_split_evidence
from tests.test_datasets.test_split_lineage_metadata_rules import _with_metadata


def _derive(changes: dict[str, object]):
    return derive_channel_dataset_split_evidence(
        (("run", _with_metadata(changes)),), spatial_buffer_m=1.0
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        pytest.param(
            {"split_parent_type": "not-a-parent"}, "invalid split_parent_type", id="parent-type"
        ),
        pytest.param(
            {"split_leakage_relations": "site"},
            "split_leakage_relations must be an array",
            id="relations-type",
        ),
        pytest.param(
            {"split_leakage_relations": ["x"]}, "malformed leakage relation", id="relation-shape"
        ),
        pytest.param(
            {"split_leakage_relations": [["unknown-namespace", "x"]]},
            "malformed leakage relation",
            id="relation-namespace",
        ),
        pytest.param({"split_block_size_m": 0.0}, "invalid RT split block size", id="block"),
        pytest.param(
            {"split_positions_m": [1.0, 2.0]},
            "no finite RT receiver position",
            id="positions-shape",
        ),
        pytest.param(
            {"split_positions_m": [1.0, float("nan"), 0.0]},
            "no finite RT receiver position",
            id="positions-nan",
        ),
    ],
)
def test_lineage_metadata_fields_are_validated(changes, message) -> None:
    with pytest.raises(ValueError, match=message):
        _derive(changes)
