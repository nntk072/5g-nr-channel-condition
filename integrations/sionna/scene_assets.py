"""Content identities for Mitsuba XML scenes and their referenced assets."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import stat
from collections.abc import Mapping
from pathlib import Path
from types import CodeType, FunctionType, ModuleType
from typing import Any
from xml.etree import ElementTree

import numpy as np

DEFAULT_MAX_SCENE_ASSET_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_MAX_SCENE_ASSET_SET_BYTES = 8 * 1024 * 1024 * 1024
MAX_SCENE_ASSET_TEXT_LINE_BYTES = 1024 * 1024


class _BoundedAssetReader:
    def __init__(self, stream: Any, *, max_bytes: int) -> None:
        self._stream = stream
        self._max_bytes = max_bytes
        self._bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        remaining = self._max_bytes + 1 - self._bytes_read
        if remaining <= 0:
            raise ValueError("scene asset exceeds the configured byte limit")
        request_size = remaining if size < 0 else min(size, remaining)
        data = self._stream.read(request_size)
        self._bytes_read += len(data)
        if self._bytes_read > self._max_bytes:
            raise ValueError("scene asset exceeds the configured byte limit")
        return data


def _stat_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _asset_references_and_digest(
    path: Path, *, max_bytes: int, limit_error: str
) -> tuple[list[str], int, str]:
    """Inspect and hash one regular asset through a stable open descriptor."""
    before_path = os.lstat(path)
    if not stat.S_ISREG(before_path.st_mode):
        raise ValueError(f"scene asset must be a regular file: {path}")
    if before_path.st_size > max_bytes:
        raise ValueError(f"{limit_error}: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or _stat_identity(before) != _stat_identity(
            before_path
        ):
            raise ValueError(f"scene asset changed while opening: {path}")

        references: list[str] = []
        if path.suffix.lower() == ".xml":
            with os.fdopen(os.dup(descriptor), "rb") as stream:
                bounded_stream = _BoundedAssetReader(stream, max_bytes=max_bytes)
                for _, element in ElementTree.iterparse(bounded_stream, events=("start",)):
                    filename = element.attrib.get("filename")
                    if filename is not None:
                        references.append(filename)
                    if (
                        element.tag == "string"
                        and element.attrib.get("name") == "filename"
                        and "value" in element.attrib
                    ):
                        references.append(element.attrib["value"])
                    element.clear()
        elif path.suffix.lower() in {".obj", ".mtl"}:
            with os.fdopen(os.dup(descriptor), "rb") as stream:
                parsed_bytes = 0
                while line := stream.readline(
                    min(MAX_SCENE_ASSET_TEXT_LINE_BYTES, max_bytes + 1 - parsed_bytes)
                ):
                    parsed_bytes += len(line)
                    if parsed_bytes > max_bytes:
                        raise ValueError(f"{limit_error}: {path}")
                    if len(line) == MAX_SCENE_ASSET_TEXT_LINE_BYTES and not line.endswith(b"\n"):
                        raise ValueError("scene asset text line exceeds the parser limit")
                    tokens = shlex.split(line.decode("utf-8", errors="replace"), comments=True)
                    if not tokens:
                        continue
                    directive = tokens[0].lower()
                    if path.suffix.lower() == ".obj" and directive == "mtllib":
                        references.extend(tokens[1:])
                    elif path.suffix.lower() == ".mtl" and (
                        directive.startswith("map_")
                        or directive in {"bump", "disp", "decal", "norm", "refl"}
                    ):
                        if len(tokens) > 1:
                            references.append(tokens[-1])

        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        byte_count = 0
        while block := os.read(descriptor, min(1024 * 1024, max_bytes + 1 - byte_count)):
            byte_count += len(block)
            if byte_count > max_bytes:
                raise ValueError(f"{limit_error}: {path}")
            digest.update(block)
        after = os.fstat(descriptor)
        after_path = os.lstat(path)
        if (
            byte_count != before.st_size
            or _stat_identity(before) != _stat_identity(after)
            or _stat_identity(before) != _stat_identity(after_path)
        ):
            raise ValueError(f"scene asset changed while being inspected: {path}")
        return references, byte_count, digest.hexdigest()
    finally:
        os.close(descriptor)


def scene_asset_manifest(
    scene_xml: str | Path,
    *,
    max_asset_bytes: int = DEFAULT_MAX_SCENE_ASSET_BYTES,
    max_total_bytes: int = DEFAULT_MAX_SCENE_ASSET_SET_BYTES,
) -> dict[str, object]:
    """Hash an RT scene and recursively referenced XML, OBJ, MTL, and image files.

    Paths are resolved relative to each referring file, as Mitsuba does. The
    manifest hashes bytes and normalized relative names; the absolute scene
    path is returned separately for provenance. URI-backed resources that
    cannot be resolved to local files are listed and make ``complete`` false.
    """
    for name, value in (
        ("max_asset_bytes", max_asset_bytes),
        ("max_total_bytes", max_total_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    xml_path = Path(scene_xml).expanduser().resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(f"RT scene XML not found: {xml_path}")

    pending = [xml_path]
    sidecar = xml_path.with_suffix(".manifest.json")
    if sidecar.is_file():
        pending.append(sidecar)
    discovered: set[Path] = set()
    unresolved: set[str] = set()
    asset_rows: list[dict[str, object]] = []
    total_bytes = 0
    while pending:
        current = pending.pop().resolve()
        if current in discovered:
            continue
        discovered.add(current)
        if not current.is_file():
            raise FileNotFoundError(f"RT scene asset not found: {current}")
        remaining_set_bytes = max_total_bytes - total_bytes
        asset_byte_limit = min(max_asset_bytes, remaining_set_bytes)
        if remaining_set_bytes < max_asset_bytes:
            limit_error = "scene asset set exceeds the configured total byte limit"
        else:
            limit_error = "scene asset exceeds the configured per-asset byte limit"
        references, byte_count, asset_sha256 = _asset_references_and_digest(
            current, max_bytes=asset_byte_limit, limit_error=limit_error
        )
        total_bytes += byte_count

        try:
            label = current.relative_to(xml_path.parent).as_posix()
        except ValueError:
            label = f"external/{current.name}"
        asset_rows.append({"path": label, "bytes": byte_count, "sha256": asset_sha256})

        for reference in references:
            if "://" in reference or reference.startswith("data:"):
                unresolved.add(reference)
                continue
            referenced = Path(reference).expanduser()
            if not referenced.is_absolute():
                referenced = current.parent / referenced
            pending.append(referenced)

    rows = sorted(asset_rows, key=lambda row: (str(row["path"]), str(row["sha256"])))
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {
        "scene_xml_path": str(xml_path),
        "asset_count": len(rows),
        "assets": rows,
        "asset_set_sha256": hashlib.sha256(canonical).hexdigest(),
        "unresolved_references": sorted(unresolved),
        "complete": not unresolved,
    }


def sionna_scene_material_manifest(scene: object) -> dict[str, object]:
    """Snapshot active Sionna material parameters and object-to-material bindings.

    The snapshot is taken after setting the scene carrier, so frequency-updated
    material parameters reflect the values used by the subsequent RT solve.
    Unknown non-serializable state fails closed instead of producing incomplete
    provenance. Callable material updates are represented by code and closure
    hashes as well as their evaluated material values.
    """
    raw_materials = getattr(scene, "radio_materials", None)
    raw_objects = getattr(scene, "objects", None)
    if not isinstance(raw_materials, Mapping) or not isinstance(raw_objects, Mapping):
        raise TypeError("Sionna scene must expose radio_materials and objects mappings")
    materials: dict[str, object] = {}
    for name, material in sorted(raw_materials.items(), key=lambda item: str(item[0])):
        if not isinstance(name, str) or not name:
            raise ValueError("Sionna scene has an invalid radio material name")
        pattern = getattr(material, "scattering_pattern", None)
        properties: dict[str, object] = {}
        for field in (
            "relative_permittivity",
            "conductivity",
            "thickness",
            "scattering_coefficient",
            "xpd_coefficient",
        ):
            properties[field] = _material_json_value(getattr(material, field))
        itu_type = getattr(material, "itu_type", None)
        if itu_type is not None:
            properties["itu_type"] = _material_json_value(itu_type)
        callback = getattr(material, "frequency_update_callback", None)
        callback_record = None if callback is None else _callable_manifest(callback)
        materials[name] = {
            "class": f"{type(material).__module__}.{type(material).__qualname__}",
            "properties": properties,
            "scattering_pattern": {
                "class": f"{type(pattern).__module__}.{type(pattern).__qualname__}",
                "state": _material_json_value(vars(pattern)),
            },
            "frequency_update_callback": callback_record,
        }
    bindings: dict[str, str] = {}
    for object_name, scene_object in sorted(raw_objects.items(), key=lambda item: str(item[0])):
        object_name = str(object_name)
        material = getattr(scene_object, "radio_material", None)
        material_name = getattr(material, "name", None)
        if not isinstance(material_name, str) or material_name not in materials:
            raise ValueError(f"scene object {object_name!r} has an unresolved radio material")
        bindings[object_name] = material_name
    frequency_hz = _material_json_value(getattr(scene, "frequency"))
    return {
        "schema": "sionna_rt_material_state.v1",
        "frequency_hz": frequency_hz,
        "materials": materials,
        "object_material_bindings": bindings,
    }


def _material_json_value(value: object) -> object:
    numpy_method = getattr(value, "numpy", None)
    if callable(numpy_method):
        value = numpy_method()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _material_json_value(value.item())
        return [_material_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _material_json_value(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("material state contains a non-finite scalar")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("material-state mapping keys must be strings")
        return {key: _material_json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_material_json_value(item) for item in value]
    raise TypeError(f"unsupported material-state value: {type(value).__qualname__}")


def _callable_manifest(
    callback: object, active_callables: set[int] | None = None
) -> dict[str, object]:
    code = getattr(callback, "__code__", None)
    if code is None:
        raise TypeError("material frequency callback must expose Python code for provenance")
    if active_callables is None:
        active_callables = set()
    identity = id(callback)
    if identity in active_callables:
        return {
            "recursive_callable": {
                "module": getattr(callback, "__module__", ""),
                "qualname": getattr(callback, "__qualname__", type(callback).__qualname__),
            }
        }
    active_callables.add(identity)
    closure = getattr(callback, "__closure__", None) or ()
    closure_values = {
        name: _material_json_value(cell.cell_contents)
        for name, cell in zip(code.co_freevars, closure, strict=True)
    }
    normalized_code = _normalized_code(code)
    callback_globals = getattr(callback, "__globals__", {})
    referenced_globals = {
        name: _global_dependency_manifest(callback_globals[name], active_callables, set())
        for name in sorted(set(code.co_names))
        if name in callback_globals
    }
    result = {
        "module": getattr(callback, "__module__", ""),
        "qualname": getattr(callback, "__qualname__", type(callback).__qualname__),
        "code_sha256": hashlib.sha256(
            json.dumps(normalized_code, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "closure_values": closure_values,
        "referenced_globals": referenced_globals,
    }
    active_callables.remove(identity)
    return result


def _global_dependency_manifest(
    value: object, active_callables: set[int], active_containers: set[int]
) -> object:
    """Serialize callback globals without silently omitting behavior inputs."""
    if isinstance(value, ModuleType):
        return {
            "module": value.__name__,
            "version": _material_json_value(getattr(value, "__version__", None)),
        }
    if isinstance(value, FunctionType):
        return _callable_manifest(value, active_callables)
    if isinstance(value, type):
        return {"class_module": value.__module__, "class_qualname": value.__qualname__}
    if callable(value):
        return {
            "callable_module": getattr(value, "__module__", type(value).__module__),
            "callable_qualname": getattr(value, "__qualname__", type(value).__qualname__),
        }
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active_containers:
            raise ValueError("material callback references a cyclic global mapping")
        active_containers.add(identity)
        pairs = [
            [
                _global_dependency_manifest(key, active_callables, active_containers),
                _global_dependency_manifest(item, active_callables, active_containers),
            ]
            for key, item in value.items()
        ]
        active_containers.remove(identity)
        pairs.sort(key=lambda pair: json.dumps(pair[0], sort_keys=True, separators=(",", ":")))
        return {"mapping_items": pairs}
    if isinstance(value, (tuple, list)):
        identity = id(value)
        if identity in active_containers:
            raise ValueError("material callback references a cyclic global sequence")
        active_containers.add(identity)
        result = [
            _global_dependency_manifest(item, active_callables, active_containers) for item in value
        ]
        active_containers.remove(identity)
        return {"tuple" if isinstance(value, tuple) else "list": result}
    if isinstance(value, (str, bool, int, float, np.ndarray, np.generic)) or value is None:
        return _material_json_value(value)
    raise TypeError(
        "material callback references unsupported global state: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _normalized_code(code: CodeType) -> dict[str, object]:
    """Hash callback behavior without checkout paths or source line numbers."""
    return {
        "bytecode": code.co_code.hex(),
        "constants": [_code_constant(value) for value in code.co_consts],
        "names": list(code.co_names),
        "variables": list(code.co_varnames),
        "free_variables": list(code.co_freevars),
        "argument_count": code.co_argcount,
        "positional_only_count": code.co_posonlyargcount,
        "keyword_only_count": code.co_kwonlyargcount,
    }


def _code_constant(value: object) -> object:
    if isinstance(value, CodeType):
        return {"code": _normalized_code(value)}
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if value is Ellipsis:
        return {"ellipsis": True}
    if isinstance(value, (tuple, list)):
        return [_code_constant(item) for item in value]
    return _material_json_value(value)
