"""Prediction snapshots: bounded, stable, single-link regular files read without following symlinks."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from evaluation.scoring import uncertainty_io


def test_snapshot_returns_exact_bytes(tmp_path: Path) -> None:
    source = tmp_path / "predictions.csv"
    source.write_bytes(b"label,score\n1,0.5\n")
    assert uncertainty_io._read_prediction_snapshot(source) == b"label,score\n1,0.5\n"


def test_snapshot_refuses_symlinks_and_hard_links(tmp_path: Path) -> None:
    target = tmp_path / "real.csv"
    target.write_bytes(b"x")
    link = tmp_path / "link.csv"
    link.symlink_to(target)
    with pytest.raises(ValueError):
        uncertainty_io._read_prediction_snapshot(link)
    alias = tmp_path / "alias.csv"
    os.link(target, alias)
    with pytest.raises(ValueError, match="single-link|regular"):
        uncertainty_io._read_prediction_snapshot(target)


def test_snapshot_enforces_the_byte_limit(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "large.csv"
    source.write_bytes(b"0123456789")
    monkeypatch.setattr(uncertainty_io, "_MAX_PREDICTION_FILE_BYTES", 4)
    with pytest.raises(ValueError, match="exceeds 4 bytes"):
        uncertainty_io._read_prediction_snapshot(source)


def test_snapshot_detects_a_file_that_changes_while_it_is_read(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "moving.csv"
    source.write_bytes(b"before")
    real_identity = uncertainty_io._file_identity
    calls = {"count": 0}

    def drifting(metadata):
        calls["count"] += 1
        identity = real_identity(metadata)
        # Change the identity seen after the read so that the snapshot is refused.
        return (0, *identity[1:]) if calls["count"] == 3 else identity

    monkeypatch.setattr(uncertainty_io, "_file_identity", drifting)
    with pytest.raises(ValueError, match="changed while"):
        uncertainty_io._read_prediction_snapshot(source)


def test_snapshot_detects_a_file_that_changes_while_opening(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "opening.csv"
    source.write_bytes(b"stable")
    real_identity = uncertainty_io._file_identity
    calls = {"count": 0}

    def drift_on_open(metadata):
        calls["count"] += 1
        identity = real_identity(metadata)
        return (0, *identity[1:]) if calls["count"] == 2 else identity

    monkeypatch.setattr(uncertainty_io, "_file_identity", drift_on_open)
    with pytest.raises(ValueError, match="changed while opening"):
        uncertainty_io._read_prediction_snapshot(source)


def test_snapshot_enforces_byte_limit_if_file_grows_during_read(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "growing.csv"
    source.write_bytes(b"x")
    monkeypatch.setattr(uncertainty_io, "_MAX_PREDICTION_FILE_BYTES", 4)
    monkeypatch.setattr(uncertainty_io.os, "read", lambda *_args: b"12345")

    with pytest.raises(ValueError, match="exceeds 4 bytes"):
        uncertainty_io._read_prediction_snapshot(source)
