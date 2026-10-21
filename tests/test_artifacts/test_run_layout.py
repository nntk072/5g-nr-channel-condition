from __future__ import annotations

import pytest
from artifacts.layout import RunLayout


def test_layout_creates_canonical_directories(tmp_path) -> None:
    layout = RunLayout.at(tmp_path / "run-001")
    layout.ensure()

    for directory in (
        layout.config,
        layout.models,
        layout.predictions,
        layout.evaluation,
        layout.diagnostics,
        layout.logs,
    ):
        assert directory.is_dir()


def test_completed_run_cannot_be_reused(tmp_path) -> None:
    layout = RunLayout.at(tmp_path / "run-001")
    layout.ensure()
    layout.complete.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="immutable"):
        layout.ensure()
