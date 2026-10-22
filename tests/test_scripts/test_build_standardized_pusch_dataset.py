"""Immutable and replay-safe PUSCH dataset build-report tests."""

from __future__ import annotations

import json

import pytest
from scripts.build_standardized_pusch_dataset import _publish_immutable_json


def test_build_report_publish_is_idempotent_and_rejects_drift(tmp_path) -> None:
    path = tmp_path / "build-report.json"
    report = {"schema": "standardized-pusch-build.v1", "samples": 4}

    _publish_immutable_json(path, report)
    original_bytes = path.read_bytes()
    _publish_immutable_json(path, report)

    assert path.read_bytes() == original_bytes
    assert json.loads(path.read_text(encoding="utf-8")) == report
    with pytest.raises(ValueError, match="immutable build report"):
        _publish_immutable_json(path, {**report, "samples": 5})
    assert path.read_bytes() == original_bytes
