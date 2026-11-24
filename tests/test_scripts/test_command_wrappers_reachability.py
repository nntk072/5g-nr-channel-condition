"""Command wrappers: each entry module imports, delegates to its package function and reports bad arguments."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from integrations import runtime_inventory


def test_runtime_inventory_wrapper_delegates_to_the_package_entry_point() -> None:
    module = importlib.import_module("scripts.report_runtime_inventory")
    assert module.main is runtime_inventory.main


def test_scene_family_sampler_requires_its_seed(tmp_path: Path, capsys) -> None:
    from scripts.sample_scene_families import main

    with pytest.raises(SystemExit) as exit_info:
        main(["--registry", str(tmp_path / "missing.json")])
    assert exit_info.value.code == 2
    assert "--seed" in capsys.readouterr().err
