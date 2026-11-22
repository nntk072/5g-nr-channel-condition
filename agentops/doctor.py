"""Read-only engineering capability inventory for the agent control plane."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import load_config
from .contracts import ROOT


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _is_ignored(repo_root: Path, relative_path: str) -> bool:
    probe = f"{relative_path.rstrip('/')}/.nr-agent-doctor-probe"
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", probe],
        cwd=repo_root,
        shell=False,
        check=False,
        timeout=5,
    )
    return result.returncode == 0


def _package_imports() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("agentops", "artifacts", "jsonschema", "yaml"):
        try:
            importlib.import_module(name)
        except Exception as error:  # pragma: no cover - environment-specific detail
            result[name] = f"UNAVAILABLE:{type(error).__name__}"
        else:
            result[name] = "AVAILABLE"
    return result


def _installed_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "UNAVAILABLE"


def doctor_report(repo_root: Path = ROOT) -> dict[str, Any]:
    config = load_config(repo_root=repo_root)
    uv = shutil.which("uv")
    fallback_uv = Path.home() / ".local/bin/uv"
    if uv is None and fallback_uv.is_file() and os.access(fallback_uv, os.X_OK):
        uv = str(fallback_uv)

    provider = "UNAVAILABLE"
    if config.provider is not None:
        candidate = Path(config.provider.executable)
        provider = (
            str(candidate)
            if candidate.is_absolute() and candidate.is_file() and os.access(candidate, os.X_OK)
            else (shutil.which(config.provider.executable) or "UNAVAILABLE")
        )

    roots: dict[str, dict[str, Any]] = {}
    for label, relative in (("state", config.state_root), ("log", config.log_root)):
        target = repo_root / relative
        parent = _nearest_existing_parent(target)
        roots[label] = {
            "path": relative,
            "ignored": _is_ignored(repo_root, relative),
            "writable_parent": os.access(parent, os.W_OK),
        }

    imports = _package_imports()
    return {
        "python": sys.version.split()[0],
        "uv": uv or "UNAVAILABLE",
        "package_imports": imports,
        "provider": provider,
        "state_roots": roots,
        "native_extras": {
            "sionna-no-rt": _installed_version("sionna-no-rt"),
            "sionna-rt": _installed_version("sionna-rt"),
        },
        "native_import_performed": False,
        "scientific_qualification": "UNCHANGED",
        "ok": bool(uv)
        and all(value == "AVAILABLE" for value in imports.values())
        and all(item["ignored"] and item["writable_parent"] for item in roots.values()),
    }
