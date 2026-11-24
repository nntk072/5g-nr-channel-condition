"""NR record store: pointer, manifest and record-archive integrity rules on disk."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from datasets.nr import NRChannelRecord, TargetDefinition
from datasets.nr_store import read_nr_record, write_nr_record
from tests.test_datasets.test_nr import make_observation


def _record() -> NRChannelRecord:
    return NRChannelRecord(
        observation=make_observation(),
        targets={"geometric_los": np.asarray(True)},
        target_masks={"geometric_los": np.asarray(True)},
        provenance={"scene_hash": "sha256:abc", "simulator": "Sionna 2.2"},
        target_definitions={
            "geometric_los": TargetDefinition.versioned(
                semantic="visibility",
                units="boolean",
                source="sionna_rt",
                reference="fixture",
                aggregation_rule="one value per sample",
                definition_version="tests.nr.store-pointer.v1",
            )
        },
    )


def _written(tmp_path: Path):
    stored = write_nr_record(tmp_path, _record())
    return stored, tmp_path / stored.relative_path


def test_record_round_trips_through_its_manifest(tmp_path: Path) -> None:
    stored, _ = _written(tmp_path)
    restored = read_nr_record(tmp_path, stored)
    assert restored.observation.sample_id == _record().observation.sample_id
    assert restored.targets["geometric_los"] == np.asarray(True)


def test_pointer_path_must_stay_inside_the_store(tmp_path: Path) -> None:
    stored, _ = _written(tmp_path)
    with pytest.raises(ValueError, match="must remain inside"):
        read_nr_record(tmp_path, dataclasses.replace(stored, relative_path="../escape.json"))
