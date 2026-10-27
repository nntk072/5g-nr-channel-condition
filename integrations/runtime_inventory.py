"""Exact, read-only inventory of the software and model tables used by runs.

The report is an integrity receipt for the local environment. It is not a
hardware attestation and does not establish scientific provenance or trust.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

RUNTIME_INVENTORY_SCHEMA = "umcd.runtime_inventory.v2"
RUNTIME_INVENTORY_SCHEMA_V3 = "umcd.runtime_inventory.v3"
_DISTRIBUTIONS = (
    "numpy",
    "scipy",
    "scikit-learn",
    "torch",
    "sionna-no-rt",
    "sionna",
    "sionna-rt",
    "mitsuba",
    "drjit",
)
_MAX_IDENTITY_FILE_BYTES = 32 * 1024 * 1024
_MAX_NATIVE_LIBRARY_BYTES = 1024 * 1024 * 1024
_MAX_NATIVE_LIBRARY_TOTAL_BYTES = 8 * 1024 * 1024 * 1024
_MAX_PROC_MAPS_BYTES = 16 * 1024 * 1024
_NATIVE_LIBRARY_NAME = re.compile(r"\.so(?:\.\d+)*$")


def _sha256(path: Path, *, max_bytes: int = _MAX_IDENTITY_FILE_BYTES) -> str:
    before_path = os.lstat(path)
    if not stat.S_ISREG(before_path.st_mode):
        raise ValueError(f"identity input must be a regular file: {path}")
    if before_path.st_size > max_bytes:
        raise ValueError(f"identity input exceeds the bounded size limit: {path}")
    if not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("platform lacks safe no-symlink file opening")
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink < 1
            or before.st_size > max_bytes
            or (before.st_dev, before.st_ino) != (before_path.st_dev, before_path.st_ino)
        ):
            raise ValueError(f"identity input changed or violates its file contract: {path}")
        total = 0
        while block := os.read(descriptor, 1024 * 1024):
            total += len(block)
            if total > max_bytes:
                raise ValueError(f"identity input exceeds the bounded size limit: {path}")
            digest.update(block)
        after = os.fstat(descriptor)
        path_after = os.lstat(path)
    finally:
        os.close(descriptor)

    def stable_fields(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if (
        total != before.st_size
        or stable_fields(before) != stable_fields(after)
        or stable_fields(before) != stable_fields(path_after)
    ):
        raise ValueError(f"identity input changed while hashing: {path}")
    return digest.hexdigest()


def _loaded_native_library_identity(
    maps_path: str | Path = "/proc/self/maps",
) -> dict[str, Any]:
    """Hash mapped Linux shared objects under explicit per-file and total limits."""
    path = Path(maps_path)
    try:
        with path.open("rb") as stream:
            raw_maps = stream.read(_MAX_PROC_MAPS_BYTES + 1)
    except OSError:
        return {
            "status": "UNAVAILABLE",
            "capture_method": "linux_proc_self_maps_sha256",
            "reason": "process_maps_unavailable",
            "limits": {
                "max_library_bytes": _MAX_NATIVE_LIBRARY_BYTES,
                "max_total_bytes": _MAX_NATIVE_LIBRARY_TOTAL_BYTES,
            },
            "libraries": [],
            "unhashed": [],
            "total_hashed_bytes": 0,
        }
    if len(raw_maps) > _MAX_PROC_MAPS_BYTES:
        return {
            "status": "INCOMPLETE",
            "capture_method": "linux_proc_self_maps_sha256",
            "reason": "process_maps_exceed_byte_limit",
            "limits": {
                "max_library_bytes": _MAX_NATIVE_LIBRARY_BYTES,
                "max_total_bytes": _MAX_NATIVE_LIBRARY_TOTAL_BYTES,
            },
            "libraries": [],
            "unhashed": [],
            "total_hashed_bytes": 0,
        }

    def unescape_path(value: str) -> str:
        return re.sub(
            r"\\([0-7]{3})",
            lambda match: chr(int(match.group(1), 8)),
            value,
        )

    mapped_paths: set[str] = set()
    for raw_line in raw_maps.splitlines():
        fields = raw_line.decode("utf-8", errors="surrogateescape").split(maxsplit=5)
        if len(fields) != 6:
            continue
        mapped = unescape_path(fields[5])
        if mapped.endswith(" (deleted)"):
            mapped = mapped.removesuffix(" (deleted)")
        if mapped.startswith("/") and _NATIVE_LIBRARY_NAME.search(mapped):
            mapped_paths.add(mapped)

    libraries: list[dict[str, Any]] = []
    unhashed: list[dict[str, Any]] = []
    total_hashed = 0
    for library_path in sorted(mapped_paths):
        try:
            info = os.lstat(library_path)
        except OSError:
            unhashed.append({"path": library_path, "reason": "mapped_file_unavailable"})
            continue
        if not stat.S_ISREG(info.st_mode):
            unhashed.append({"path": library_path, "reason": "mapped_file_not_regular"})
            continue
        if info.st_size > _MAX_NATIVE_LIBRARY_BYTES:
            unhashed.append(
                {
                    "path": library_path,
                    "reason": "per_library_byte_limit_exceeded",
                    "size_bytes": info.st_size,
                }
            )
            continue
        if total_hashed + info.st_size > _MAX_NATIVE_LIBRARY_TOTAL_BYTES:
            unhashed.append(
                {
                    "path": library_path,
                    "reason": "total_byte_limit_exceeded",
                    "size_bytes": info.st_size,
                }
            )
            continue
        try:
            digest = _sha256(Path(library_path), max_bytes=_MAX_NATIVE_LIBRARY_BYTES)
            after_hash = os.lstat(library_path)
        except (OSError, ValueError):
            unhashed.append(
                {
                    "path": library_path,
                    "reason": "mapped_file_changed_or_unhashable",
                    "size_bytes": info.st_size,
                }
            )
            continue
        if (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_nlink,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        ) != (
            after_hash.st_dev,
            after_hash.st_ino,
            after_hash.st_mode,
            after_hash.st_nlink,
            after_hash.st_size,
            after_hash.st_mtime_ns,
            after_hash.st_ctime_ns,
        ):
            unhashed.append(
                {
                    "path": library_path,
                    "reason": "mapped_file_changed_or_unhashable",
                    "size_bytes": info.st_size,
                }
            )
            continue
        libraries.append(
            {
                "path": library_path,
                "size_bytes": info.st_size,
                "sha256": digest,
            }
        )
        total_hashed += info.st_size
    return {
        "status": "INCOMPLETE" if unhashed else "COMPLETE",
        "capture_method": "linux_proc_self_maps_sha256",
        "reason": None if not unhashed else "one_or_more_mapped_libraries_unhashed",
        "limits": {
            "max_library_bytes": _MAX_NATIVE_LIBRARY_BYTES,
            "max_total_bytes": _MAX_NATIVE_LIBRARY_TOTAL_BYTES,
        },
        "libraries": libraries,
        "unhashed": unhashed,
        "total_hashed_bytes": total_hashed,
    }


def _distribution_identity(name: str) -> dict[str, Any]:
    try:
        dist = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {"name": name, "installed": False}

    files = dist.files or ()
    identity_files: dict[str, str] = {}
    table_files: list[dict[str, str]] = []
    for entry in files:
        relative = str(entry).replace("\\", "/")
        if relative.endswith(
            (".dist-info/RECORD", ".dist-info/WHEEL", ".dist-info/direct_url.json")
        ):
            path = Path(dist.locate_file(entry))
            if path.exists():
                identity_files[relative.rsplit("/", 1)[-1]] = _sha256(path)
        if (
            name == "sionna-no-rt"
            and relative.startswith("sionna/phy/channel/tr38901/models/")
            and relative.endswith(".json")
        ):
            path = Path(dist.locate_file(entry))
            table_files.append({"path": relative, "sha256": _sha256(path)})
    table_files.sort(key=lambda item: item["path"])
    table_payload = json.dumps(table_files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "name": dist.metadata.get("Name", name),
        "installed": True,
        "version": dist.version,
        "identity_completeness": "COMPLETE" if "RECORD" in identity_files else "INCOMPLETE",
        "distribution_metadata_sha256": hashlib.sha256(
            json.dumps(identity_files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "identity_files": identity_files,
        "model_table_count": len(table_files),
        "model_tables_sha256": hashlib.sha256(table_payload).hexdigest(),
        "model_tables": table_files,
    }


def _source_identity(repository_root: Path) -> dict[str, Any]:
    def hash_if_present(name: str) -> str | None:
        path = repository_root / name
        try:
            os.lstat(path)
        except FileNotFoundError:
            return None
        return _sha256(path)

    result: dict[str, Any] = {}
    for name in ("pyproject.toml", "uv.lock"):
        result[name] = hash_if_present(name)
    result["runtime_inventory_sources"] = {
        name: hash_if_present(name)
        for name in (
            "integrations/runtime_inventory.py",
            "integrations/runtime_compatibility.py",
            "scripts/report_runtime_inventory.py",
        )
    }
    try:
        import subprocess

        commit = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repository_root), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        result["git_commit"] = commit
        result["tracked_worktree_clean"] = not bool(dirty)
    except (OSError, subprocess.SubprocessError):
        result["git_commit"] = None
        result["tracked_worktree_clean"] = None
    return result


def _accelerator_identity(probe: bool) -> dict[str, Any]:
    if not probe:
        return {"probed": False, "status": "NOT_PROBED"}
    try:
        import torch
    except ImportError:
        return {"probed": True, "status": "TORCH_UNAVAILABLE"}
    result: dict[str, Any] = {
        "probed": True,
        "torch_version": str(torch.__version__),
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "devices": [],
    }
    if result["cuda_available"]:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            result["devices"].append(
                {
                    "index": index,
                    "name": str(props.name),
                    "compute_capability": [int(props.major), int(props.minor)],
                    "total_memory_bytes": int(props.total_memory),
                }
            )
    result["status"] = "CUDA_AVAILABLE" if result["cuda_available"] else "CUDA_UNAVAILABLE"
    return result


def build_runtime_inventory(
    repository_root: str | Path,
    *,
    probe_accelerator: bool = False,
    probe_apis: bool = False,
    probe_rt_apis: bool = False,
    probe_loaded_native_libraries: bool = False,
) -> dict[str, Any]:
    """Build a JSON-ready software/model-table identity report.

    CUDA discovery is opt-in and does not allocate a device tensor. The
    default path uses package metadata only and never imports Sionna or Torch.
    """
    if type(probe_accelerator) is not bool:
        raise TypeError("probe_accelerator must be a Boolean")
    if type(probe_apis) is not bool or type(probe_rt_apis) is not bool:
        raise TypeError("API probe options must be Booleans")
    if type(probe_loaded_native_libraries) is not bool:
        raise TypeError("probe_loaded_native_libraries must be a Boolean")
    if probe_rt_apis and not probe_apis:
        raise ValueError("probe_rt_apis requires probe_apis")
    root = Path(repository_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("repository_root must be a directory")
    if probe_apis:
        from integrations.runtime_compatibility import build_runtime_compatibility

        api_compatibility = build_runtime_compatibility(probe_rt=probe_rt_apis)
    else:
        api_compatibility = {
            "schema": "umcd.runtime_compatibility.v1",
            "status": "NOT_PROBED",
            "phy": {"status": "NOT_PROBED", "distributions": {}, "modules": []},
            "rt": {"status": "NOT_PROBED", "distributions": {}, "modules": []},
        }
    report = {
        "schema": (
            RUNTIME_INVENTORY_SCHEMA_V3
            if probe_loaded_native_libraries
            else RUNTIME_INVENTORY_SCHEMA
        ),
        "status": "SELF_REPORTED_INTEGRITY_ONLY",
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "source": _source_identity(root),
        "distributions": [_distribution_identity(name) for name in _DISTRIBUTIONS],
        "accelerator": _accelerator_identity(probe_accelerator),
        "api_compatibility": api_compatibility,
        "qualification": "NOT_QUALIFIED",
    }
    if probe_loaded_native_libraries:
        report["loaded_native_libraries"] = _loaded_native_library_identity()
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """CLI for creating an exact, local runtime inventory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository containing pyproject.toml and uv.lock",
    )
    parser.add_argument(
        "--probe-accelerator",
        action="store_true",
        help="import Torch and report CUDA build/device metadata without allocating tensors",
    )
    parser.add_argument(
        "--probe-apis",
        action="store_true",
        help="import and check the pinned Sionna PHY exports used by this repository",
    )
    parser.add_argument(
        "--probe-rt-apis",
        action="store_true",
        help="also import/check pinned Sionna RT, Mitsuba, and Dr.Jit APIs",
    )
    parser.add_argument(
        "--probe-loaded-native-libraries",
        action="store_true",
        help="hash currently mapped Linux shared libraries under bounded byte limits",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="new output file; creation is exclusive and refuses to replace an existing path",
    )
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"refusing to replace runtime inventory output: {args.output}")
    report = build_runtime_inventory(
        args.repository_root,
        probe_accelerator=args.probe_accelerator,
        probe_apis=args.probe_apis,
        probe_rt_apis=args.probe_rt_apis,
        probe_loaded_native_libraries=args.probe_loaded_native_libraries,
    )
    payload = (json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    if args.output is None:
        sys.stdout.buffer.write(payload)
    else:
        with args.output.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    return 0
