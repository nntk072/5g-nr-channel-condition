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


def require_file_matches_git_head(path: str | Path) -> None:
    """Fail when a provenance-bound source file differs from committed HEAD."""
    source = Path(path).resolve()
    try:
        root = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=source.parent,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            .decode()
            .strip()
        ).resolve()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("a Git checkout is required to verify the experiment source") from error
    try:
        relative = source.relative_to(root).as_posix()
    except ValueError as error:
        raise RuntimeError("the experiment source must be inside the Git checkout") from error
    try:
        committed = subprocess.check_output(
            ["git", "show", f"HEAD:{relative}"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(
            "the experiment source must be tracked in the current Git commit"
        ) from error
    if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(committed).digest():
        raise RuntimeError(
            f"experiment source {relative} differs from Git HEAD; commit it before generating results"
        )


def canonical_hash(data: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def atomic_json(path: str | Path, data: Any, *, overwrite: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix="." + path.name + ".")
    try:
        with os.fdopen(fd, "w", encoding="utf8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True, default=str, allow_nan=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        if overwrite:
            os.replace(tmp, path)
        else:
            os.link(tmp, path)
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
            distribution = (
                "sionna-no-rt"
                if p == "sionna-phy"
                else p
            )
            try:
                d[p] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                if p != "sionna-rt":
                    raise
                d[p] = importlib.metadata.version("sionna-no-rt")
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
