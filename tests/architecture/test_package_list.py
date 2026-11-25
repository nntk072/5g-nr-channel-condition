"""The package list is defined once, in config/packages.toml, and derived everywhere else."""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_sync_module():
    spec = importlib.util.spec_from_file_location("sync_packages", ROOT / "tools/sync_packages.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pyproject_package_lists_match_the_single_source() -> None:
    sync = _load_sync_module()
    current = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert sync.render(current, sync.load_packages()) == current


def test_tools_check_reads_the_same_package_names() -> None:
    names = tomllib.loads((ROOT / "config/packages.toml").read_text(encoding="utf-8"))["packages"][
        "names"
    ]
    source = (ROOT / "tools/check").read_text(encoding="utf-8")
    assert "config" in source and "packages.toml" in source
    assert "PACKAGE_NAMES" in source
    # The agent gate has its own scope line; every other line must derive from the file.
    derived = [line for line in source.splitlines() if not line.startswith("AGENT = [")]
    for name in names:
        assert all(f'"{name}"' not in line for line in derived), f"{name} is hard-coded"
