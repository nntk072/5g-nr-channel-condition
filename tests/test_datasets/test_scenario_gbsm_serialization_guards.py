"""Serialization and path-boundary checks for scenario GBSM records."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import datasets.scenario_gbsm as scenario_gbsm
import numpy as np
import pytest


@pytest.mark.parametrize("value", [object(), float("nan")])
def test_canonical_json_rejects_non_json_and_nonfinite_values(value) -> None:
    with pytest.raises(TypeError, match="must be finite JSON-compatible data"):
        scenario_gbsm._canonical_json(value, label="fixture")


@pytest.mark.parametrize(
    "relative",
    [PurePosixPath("/absolute/file"), PurePosixPath("../escape"), PurePosixPath(r"foo\bar")],
)
def test_store_path_rejects_noncanonical_relative_members(tmp_path: Path, relative) -> None:
    with pytest.raises(ValueError, match="canonical relative path"):
        scenario_gbsm._store_path(tmp_path, relative, label="fixture")


def test_store_path_rejects_a_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "object.bin").write_bytes(b"data")
    (root / "objects").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes the store root"):
        scenario_gbsm._store_path(root, PurePosixPath("objects/object.bin"), label="fixture")


@pytest.mark.parametrize("angles", [(0.0, 1.0), (0.0, float("nan"), 1.0)])
def test_orientation_matrix_rejects_invalid_euler_angles(angles) -> None:
    with pytest.raises(ValueError, match="three finite Euler angles"):
        scenario_gbsm._sionna_orientation_matrix(angles)


def test_record_serializer_enforces_uncompressed_size_limit(monkeypatch) -> None:
    record = SimpleNamespace(
        sample_id="sample",
        run_id="run",
        sample_index=0,
        scenario="UMi",
        scenario_los=True,
        metadata={},
        h_path=np.zeros((1,), dtype=np.complex64),
        tau_s=np.zeros((1,), dtype=np.float64),
        cfr=np.zeros((1,), dtype=np.complex64),
        subcarrier_offsets_hz=np.zeros((1,), dtype=np.float64),
    )
    monkeypatch.setattr(scenario_gbsm, "MAX_SCENARIO_GBSM_OBJECT_BYTES", 1)
    with pytest.raises(ValueError, match="uncompressed byte ceiling"):
        scenario_gbsm._record_bytes(record)
