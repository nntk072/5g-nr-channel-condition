"""Run tracking adapted from the author's Aalto_MS_Thesis artifact and hashing patterns."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(data: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def atomic_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix="." + path.name + ".")
    try:
        with os.fdopen(fd, "w", encoding="utf8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True, default=str, allow_nan=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def package_source_sha256() -> str:
    """Hash active research package source even without Git history."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    packages = (
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
        "loslab",
    )
    for package in packages:
        folder = root / package
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            digest.update(path.relative_to(root).as_posix().encode("utf8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def versions() -> dict[str, str]:
    d = {"python": sys.version.split()[0]}
    for p in ("numpy", "scikit-learn", "torch", "sionna", "sionna-rt", "sionna-phy"):
        try:
            d[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            d[p] = "not installed"
    try:
        d["git_sha"] = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=2
            )
            .decode()
            .strip()
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        d["git_sha"] = "unavailable"
    return d
