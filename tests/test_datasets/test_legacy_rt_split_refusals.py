"""Legacy RT split migration: refuses non-RT rows, partial split metadata and malformed saved scene identity."""

from __future__ import annotations

import numpy as np
import pytest
from datasets.channel import ChannelDataset
from datasets.legacy_rt_split import upgrade_legacy_rt_split_metadata


def _dataset(source="sionna_rt", metadata=None, generation_report=None) -> ChannelDataset:
    return ChannelDataset(
        h=np.ones((1, 1, 8), dtype=np.complex64),
        y=np.asarray([0], dtype=np.uint8),
        group=np.asarray(["g"]),
        domain=np.asarray(["scene"]),
        source=np.asarray([source]),
        metadata=(
            metadata if metadata is not None else {"scene": "scene-a", "rx": [0.0, 0.0, 1.5]},
        ),
        generation_report=generation_report or {},
    )


def test_only_sionna_rt_rows_can_be_migrated() -> None:
    with pytest.raises(ValueError, match="requires only sionna_rt rows"):
        upgrade_legacy_rt_split_metadata(_dataset(source="standardized"), dataset_name="legacy")


def test_saved_scene_configuration_is_required() -> None:
    with pytest.raises(ValueError, match="lacks saved scene configuration or asset closure"):
        upgrade_legacy_rt_split_metadata(_dataset(generation_report={}), dataset_name="legacy")


def test_partial_split_metadata_is_refused() -> None:
    metadata = {
        "scene": "scene-a",
        "rx": [0.0, 0.0, 1.5],
        "split_subject_id": "rt:partial",
    }
    with pytest.raises(ValueError, match="partial or mixed RT split metadata"):
        upgrade_legacy_rt_split_metadata(_dataset(metadata=metadata), dataset_name="legacy")
