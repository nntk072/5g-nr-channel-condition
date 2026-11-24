"""Configuration resolution records the declared source path as text."""

from __future__ import annotations

from pathlib import Path

from configuration.loader import resolve_config


def test_resolved_config_records_its_source_path(tmp_path: Path) -> None:
    resolved = resolve_config({"seed": 1}, source=tmp_path / "experiment.json")
    assert resolved["source"] == str(tmp_path / "experiment.json")
    assert "source" not in resolve_config({"seed": 1})
