"""Content identities for Mitsuba XML scenes and their referenced assets."""

from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path
from xml.etree import ElementTree


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
