"""Strict JSON and file-boundary helpers for joint evidence artifacts."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from evaluation import joint_evidence_artifact as artifact


@pytest.mark.parametrize(
    "value",
    [
        {"$complex": "1,2"},
        {"$complex": [1]},
        {"$complex": [True, 2]},
        {"$complex": ["1", 2]},
    ],
)
def test_complex_json_decoder_rejects_invalid_component_shapes(value):
    with pytest.raises(ValueError, match="exactly two real numeric components"):
        artifact._decode_json_value(value)


def test_strict_json_parser_rejects_duplicate_keys_constants_and_bad_utf8():
    for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b"\xff"):
        with pytest.raises(ValueError, match="not strict UTF-8 JSON"):
            artifact._parse_json(raw, label="probe")


def test_json_tree_normalizes_numpy_complex_and_mapping_values():
    assert artifact._json_tree(
        {"z": np.complex64(1 + 2j), "a": np.asarray([1, 2]), "nested": (np.int64(3),)}
    ) == {"a": [1, 2], "nested": [3], "z": {"$complex": [1.0, 2.0]}}


@pytest.mark.parametrize(
    "value",
    [
        {1: "non-string key"},
        float("inf"),
        object(),
    ],
)
def test_json_tree_rejects_values_outside_artifact_domain(value):
    with pytest.raises((TypeError, ValueError)):
        artifact._json_tree(value)


def test_json_tree_rejects_nonfinite_complex_components():
    with pytest.raises(ValueError, match="not JSON compliant"):
        artifact._canonical_json_bytes(1 + float("inf") * 1j)


def test_regular_file_reader_requires_no_follow_support(monkeypatch, tmp_path: Path):
    path = tmp_path / "input.json"
    path.write_bytes(b"{}")
    monkeypatch.delattr(artifact.os, "O_NOFOLLOW")
    with pytest.raises(RuntimeError, match="O_NOFOLLOW"):
        artifact._read_regular_file(path, limit=10, label="probe")


def test_regular_file_reader_rejects_hard_links_and_size_growth(monkeypatch, tmp_path: Path):
    path = tmp_path / "input.json"
    path.write_bytes(b"{}")
    alias = tmp_path / "alias.json"
    os.link(path, alias)
    with pytest.raises(ValueError, match="single-link regular file"):
        artifact._read_regular_file(path, limit=10, label="probe")
    alias.unlink()

    monkeypatch.setattr(artifact.os, "read", lambda _descriptor, _limit: b"12345")
    with pytest.raises(ValueError, match="exceeds 4 bytes"):
        artifact._read_regular_file(path, limit=4, label="probe")


def test_regular_file_reader_detects_descriptor_change_while_opening(monkeypatch, tmp_path: Path):
    path = tmp_path / "input.json"
    path.write_bytes(b"{}")
    signature = artifact._stat_signature
    calls = 0

    def changing_signature(value):
        nonlocal calls
        calls += 1
        current = signature(value)
        return current if calls == 1 else (*current, 1)

    monkeypatch.setattr(artifact, "_stat_signature", changing_signature)
    with pytest.raises(ValueError, match="changed while opening"):
        artifact._read_regular_file(path, limit=10, label="probe")
