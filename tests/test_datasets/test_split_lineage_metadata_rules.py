"""Split lineage: scene-asset manifests, identity metadata and dataset-roster rules."""

from __future__ import annotations

import copy

import pytest
from datasets.channel import ChannelDataset
from datasets.split_lineage import derive_channel_dataset_split_evidence
from tests.test_datasets.test_split_lineage_audit import _dataset


def _with_metadata(changes: dict[str, object]):
    _, base = _dataset("run", [("parent-a", 0.0, "a" * 64)])
    metadata = copy.deepcopy(dict(base.metadata[0]))
    metadata.update(changes)
    return ChannelDataset(
        h=base.h,
        y=base.y,
        group=base.group,
        domain=base.domain,
        source=base.source,
        metadata=(metadata,),
        generation_report=base.generation_report,
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        pytest.param(
            {"scene_asset_set_sha256": "G" * 64}, "invalid RT scene asset digest", id="digest"
        ),
        pytest.param(
            {"scene_asset_manifest_complete": "yes"}, "completeness flag", id="complete-type"
        ),
    ],
)
def test_scene_asset_metadata_is_validated(changes, message) -> None:
    malformed = _with_metadata(changes)
    with pytest.raises(ValueError, match=message):
        derive_channel_dataset_split_evidence((("run", malformed),), spatial_buffer_m=1.0)


def test_duplicate_run_names_are_refused() -> None:
    _, base = _dataset("run", [("parent-a", 0.0, "a" * 64)])
    with pytest.raises(ValueError):
        derive_channel_dataset_split_evidence((("run", base), ("run", base)), spatial_buffer_m=1.0)


def test_spatial_buffer_must_be_finite_and_nonnegative() -> None:
    _, base = _dataset("run", [("parent-a", 0.0, "a" * 64)])
    with pytest.raises(ValueError):
        derive_channel_dataset_split_evidence((("run", base),), spatial_buffer_m=float("nan"))
    with pytest.raises(ValueError):
        derive_channel_dataset_split_evidence((("run", base),), spatial_buffer_m=-1.0)


def test_dataset_rosters_must_be_nonempty() -> None:
    with pytest.raises(ValueError):
        derive_channel_dataset_split_evidence((), spatial_buffer_m=1.0)
