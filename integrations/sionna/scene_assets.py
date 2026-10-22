"""Content identities for Mitsuba XML scenes and their referenced assets."""

from __future__ import annotations

import hashlib
import json
import math
import shlex
from collections.abc import Mapping
from pathlib import Path
from types import CodeType
from xml.etree import ElementTree

import numpy as np


def scene_asset_manifest(scene_xml: str | Path) -> dict[str, object]:
    """Hash an RT scene and recursively referenced XML, OBJ, MTL, and image files.

    Paths are resolved relative to each referring file, as Mitsuba does. The
    manifest hashes bytes and normalized relative names; the absolute scene
    path is returned separately for provenance. URI-backed resources that
    cannot be resolved to local files are listed and make ``complete`` false.
    """
    xml_path = Path(scene_xml).expanduser().resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(f"RT scene XML not found: {xml_path}")

    pending = [xml_path]
    sidecar = xml_path.with_suffix(".manifest.json")
    if sidecar.is_file():
        pending.append(sidecar)
    discovered: set[Path] = set()
    unresolved: set[str] = set()
    while pending:
        current = pending.pop().resolve()
        if current in discovered:
            continue
        if not current.is_file():
            raise FileNotFoundError(f"RT scene asset not found: {current}")
        discovered.add(current)
        extension = current.suffix.lower()
        if extension == ".xml":
            root = ElementTree.parse(current).getroot()
            references = [
                element.attrib["filename"]
                for element in root.iter()
                if "filename" in element.attrib
            ]
            references.extend(
                element.attrib["value"]
                for element in root.iter("string")
                if element.attrib.get("name") == "filename" and "value" in element.attrib
            )
        elif extension == ".obj":
            references = []
            for line in current.read_text(encoding="utf-8", errors="replace").splitlines():
                tokens = shlex.split(line, comments=True)
                if tokens and tokens[0].lower() == "mtllib":
                    references.extend(tokens[1:])
        elif extension == ".mtl":
            references = []
            for line in current.read_text(encoding="utf-8", errors="replace").splitlines():
                tokens = shlex.split(line, comments=True)
                if tokens and (
                    tokens[0].lower().startswith("map_")
                    or tokens[0].lower() in {"bump", "disp", "decal", "norm", "refl"}
                ):
                    if len(tokens) > 1:
                        references.append(tokens[-1])
        else:
            references = []

        for reference in references:
            if "://" in reference or reference.startswith("data:"):
                unresolved.add(reference)
                continue
            referenced = Path(reference).expanduser()
            if not referenced.is_absolute():
                referenced = current.parent / referenced
            pending.append(referenced)

    rows = []
    for path in sorted(discovered, key=lambda value: value.as_posix()):
        try:
            label = path.relative_to(xml_path.parent).as_posix()
        except ValueError:
            label = f"external/{path.name}"
        payload = path.read_bytes()
        rows.append(
            {
                "path": label,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
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


def _callable_manifest(callback: object) -> dict[str, object]:
    code = getattr(callback, "__code__", None)
    if code is None:
        raise TypeError("material frequency callback must expose Python code for provenance")
    closure = getattr(callback, "__closure__", None) or ()
    closure_values = {
        name: _material_json_value(cell.cell_contents)
        for name, cell in zip(code.co_freevars, closure, strict=True)
    }
    normalized_code = _normalized_code(code)
    return {
        "module": getattr(callback, "__module__", ""),
        "qualname": getattr(callback, "__qualname__", type(callback).__qualname__),
        "code_sha256": hashlib.sha256(
            json.dumps(normalized_code, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "closure_values": closure_values,
    }


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
