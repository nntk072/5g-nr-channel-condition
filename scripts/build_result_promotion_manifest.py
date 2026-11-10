"Build a fail-closed promotion manifest over audited research-result bytes."

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import jsonschema
from artifacts.provenance import atomic_json

from scripts.audit_research_result_provenance import _sha256_file, audit_tree

_POLICY_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "schema"
    / "research-result-promotion-policy.v1.schema.json"
)
_MANIFEST_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "schema"
    / "research-result-promotion-manifest.v1.schema.json"
)


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or "\\" in value:
        raise ValueError("promotion paths must be canonical relative POSIX paths")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("promotion paths must be canonical relative POSIX paths")
    if not (value.endswith(".json") or value.endswith(".json.gz")):
        raise ValueError("promotion paths must reference JSON or JSON.GZ result files")
    return value


def _load_policy(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    try:
        policy = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("promotion policy must be bounded UTF-8 JSON") from error
    schema = json.loads(_POLICY_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(policy, schema)
    requested = [_safe_relative(item) for item in policy["requested_paths"]]
    if requested != sorted(set(requested)):
        raise ValueError("requested_paths must be sorted and unique")
    return policy


def build_result_promotion_manifest(
    results_root: str | Path,
    *,
    policy_path: str | Path,
) -> dict[str, Any]:
    "Audit exact result bytes and decide whether the requested corpus may be promoted."
    root = Path(results_root).resolve()
    if not root.is_dir():
        raise ValueError("results_root must be a directory")
    policy_file = Path(policy_path).resolve()
    policy = _load_policy(policy_file)
    audit = audit_tree(root)
    by_path = {row["path"]: row["issues"] for row in audit["files"]}
    requested = list(policy["requested_paths"])
    missing = sorted(set(requested) - set(by_path))
    if missing:
        raise ValueError(f"promotion policy references absent results: {missing}")

    requested_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    requested_set = set(requested)
    for relative in sorted(by_path):
        path = root / relative
        digest = _sha256_file(path)
        size_bytes = path.stat().st_size
        issues = by_path[relative]
        row = {
            "path": relative,
            "sha256": digest,
            "size_bytes": size_bytes,
            "issue_count": len(issues),
            "issues": issues,
        }
        if relative in requested_set:
            requested_rows.append(
                {**row, "promotion_status": "eligible" if not issues else "rejected"}
            )
        elif issues:
            diagnostic_rows.append({**row, "retention_status": "diagnostic_only"})

    rejected = [row for row in requested_rows if row["promotion_status"] != "eligible"]
    core = {
        "schema": "loslab.research_result_promotion_manifest.v1",
        "status": "PROMOTION_READY" if not rejected else "PROMOTION_REJECTED",
        "qualification_status": "NOT_QUALIFIED",
        "results_root": str(root),
        "policy": {
            "path": str(policy_file),
            "sha256": hashlib.sha256(policy_file.read_bytes()).hexdigest(),
        },
        "audit_summary": {
            "scanned_files": audit["scanned_files"],
            "valid_json_files": audit["valid_json_files"],
            "malformed_files": audit["malformed_files"],
            "files_with_issues": audit["files_with_issues"],
            "issue_counts": audit["issue_counts"],
        },
        "requested": requested_rows,
        "diagnostic_legacy": diagnostic_rows,
        "promoted_file_count": sum(row["promotion_status"] == "eligible" for row in requested_rows),
        "rejected_file_count": len(rejected),
    }
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    manifest = {**core, "manifest_sha256": hashlib.sha256(canonical).hexdigest()}
    schema = json.loads(_MANIFEST_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(manifest, schema)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = build_result_promotion_manifest(args.results_root, policy_path=args.policy)
        atomic_json(args.output, manifest, overwrite=False)
    except (OSError, TypeError, ValueError, RecursionError, jsonschema.ValidationError) as error:
        parser.exit(2, f"error: {error}\n")
    print(
        json.dumps(
            {"status": manifest["status"], "manifest_sha256": manifest["manifest_sha256"]},
            sort_keys=True,
        )
    )
    return 0 if manifest["status"] == "PROMOTION_READY" else 3


if __name__ == "__main__":
    raise SystemExit(main())
