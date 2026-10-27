"""CLI workflow for byte-auditing UMCD lineage evidence files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from artifacts.provenance import atomic_json
from datasets.episode_store import _stable_file_bytes
from datasets.lineage import audit_lineage_evidence_files
from datasets.provenance import UMCDRunProvenance

_MAX_INPUT_BYTES = 16 * 1024 * 1024
_PATH_MAP_SCHEMA = Path(__file__).resolve().parents[1] / (
    "datasets/schema/umcd-lineage-evidence-path-map.v1.schema.json"
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"input contains duplicate key {key!r}")
        result[key] = value
    return result


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = _stable_file_bytes(path, label=label, max_bytes=_MAX_INPUT_BYTES)
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def audit_provenance_file(
    provenance_path: Path,
    evidence_path_map_path: Path,
    evidence_root: Path,
    output_path: Path,
    *,
    max_total_bytes: int = 8 * 1024 * 1024 * 1024,
) -> dict[str, Any]:
    """Audit a provenance v2 claim roster and publish a new report exclusively."""
    provenance_payload = _read_json_object(provenance_path, label="UMCD provenance record")
    provenance = UMCDRunProvenance.from_dict(provenance_payload)
    if provenance.lineage_claims is None:
        raise ValueError("UMCD provenance v1 has no lineage claim roster")
    provenance_digest = provenance.to_dict()["record_sha256"]

    path_map = _read_json_object(evidence_path_map_path, label="lineage evidence path map")
    schema = json.loads(_PATH_MAP_SCHEMA.read_text(encoding="utf-8"))
    import jsonschema

    jsonschema.Draft202012Validator.check_schema(schema)
    try:
        jsonschema.Draft202012Validator(schema).validate(path_map)
    except jsonschema.ValidationError as error:
        raise ValueError("lineage evidence path map does not match its schema") from error
    report = audit_lineage_evidence_files(
        provenance.lineage_claims,
        path_map["paths"],
        evidence_root=evidence_root,
        max_total_bytes=max_total_bytes,
        provenance_record_sha256=provenance_digest,
    )
    report_schema_path = Path(__file__).resolve().parents[1] / (
        "datasets/schema/umcd-lineage-evidence-audit.v2.schema.json"
    )
    report_schema = json.loads(report_schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(report_schema)
    try:
        jsonschema.Draft202012Validator(report_schema).validate(report)
    except jsonschema.ValidationError as error:
        raise ValueError("lineage evidence audit report does not match its schema") from error
    atomic_json(output_path, report, overwrite=False)
    return report
