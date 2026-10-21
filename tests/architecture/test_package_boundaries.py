from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ACTIVE_PACKAGES = (
    "artifacts",
    "configuration",
    "datasets",
    "propagation",
    "receivers",
    "models",
    "learning",
    "evaluation",
    "experiments",
    "integrations",
    "compute",
)


def test_active_modules_do_not_import_legacy_or_script_code() -> None:
    forbidden = {"archive", "notebooks", "scripts"}
    for package in ACTIVE_PACKAGES:
        for source in (ROOT / package).rglob("*.py"):
            tree = ast.parse(source.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                else:
                    continue
                assert not any(name.split(".")[0] in forbidden for name in names), source


def test_sionna_is_only_imported_inside_its_adapter() -> None:
    for package in ACTIVE_PACKAGES:
        for source in (ROOT / package).rglob("*.py"):
            if source.is_relative_to(ROOT / "integrations" / "sionna"):
                continue
            tree = ast.parse(source.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                else:
                    continue
                assert not any(
                    name.split(".")[0] in {"sionna", "mitsuba", "drjit"} for name in names
                ), source


def test_standard_package_imports_do_not_load_native_sionna() -> None:
    for module in (
        "artifacts",
        "configuration",
        "datasets",
        "propagation",
        "receivers",
        "models",
        "evaluation",
        "experiments",
    ):
        __import__(module)
    assert not {"sionna", "mitsuba", "drjit"}.intersection(sys.modules)
