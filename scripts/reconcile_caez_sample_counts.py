"""Compare official CAEZ page counts with audited local archive metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SNAPSHOT = REPOSITORY_ROOT / "docs/research/data/caez-official-snapshot-20261024.json"
_INDOOR_AUDIT = REPOSITORY_ROOT / "docs/research/results/caez-indoor-audit.json"
_CROSS_REPORT = (
    REPOSITORY_ROOT / "docs/research/results/caez-cross-environment-summary-forecast.json"
)
_ENVIRONMENTS = {
    "caez_5g_indoor_full": "caez_5g_indoor_full.npz",
    "caez_5g_outdoor_full": "caez_5g_outdoor_full.npz",
    "caez_5g_hallway_full": "caez_5g_hallway_full.npz",
}


def _load_object(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} root must be a JSON object")
    return value, raw


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def build_reconciliation(
    *, snapshot: dict[str, Any], indoor_audit: dict[str, Any], cross_report: dict[str, Any]
) -> dict[str, Any]:
    if snapshot.get("schema") != "caez-official-count-snapshot.v1":
        raise ValueError("unsupported CAEZ official snapshot schema")
    published = snapshot.get("published_counts")
    listing = snapshot.get("preprocessed_listing")
    domains = cross_report.get("target_domains")
    if (
        not isinstance(published, dict)
        or not isinstance(listing, dict)
        or not isinstance(domains, list)
    ):
        raise ValueError("snapshot counts/listing and cross-report domains must be present")
    local_by_environment = {item.get("environment_id"): item for item in domains}
    if len(local_by_environment) != len(domains):
        raise ValueError("cross-report environment IDs must be unique")
    results: list[dict[str, Any]] = []
    for environment, filename in _ENVIRONMENTS.items():
        item = local_by_environment.get(environment)
        official_count = published.get(environment)
        remote_entry = listing.get(filename)
        if (
            not isinstance(item, dict)
            or not isinstance(official_count, int)
            or isinstance(official_count, bool)
            or not isinstance(remote_entry, dict)
        ):
            raise ValueError(f"missing official/local metadata for {environment}")
        local_count = item.get("sample_count")
        archive_digest = item.get("archive_sha256")
        archive_bytes = item.get("archive_bytes")
        if (
            not isinstance(local_count, int)
            or isinstance(local_count, bool)
            or not isinstance(archive_bytes, int)
            or not isinstance(archive_digest, str)
            or len(archive_digest) != 64
            or set(archive_digest.lower()) - set("0123456789abcdef")
            or item.get("integrity") != "passed"
        ):
            raise ValueError(f"invalid audited archive metadata for {environment}")
        if environment == "caez_5g_indoor_full":
            if (
                indoor_audit.get("sample_count") != local_count
                or indoor_audit.get("archive_sha256") != archive_digest
                or indoor_audit.get("archive_bytes") != archive_bytes
                or indoor_audit.get("source_file") != filename
            ):
                raise ValueError("indoor audit and cross-environment report disagree")
        difference = local_count - official_count
        results.append(
            {
                "environment_id": environment,
                "archive_filename": filename,
                "official_published_sample_count": official_count,
                "official_preprocessed_listing": remote_entry,
                "local_audited_sample_count": local_count,
                "local_archive_bytes": archive_bytes,
                "local_archive_sha256": archive_digest,
                "local_minus_official_sample_count": difference,
                "count_match": difference == 0,
                "reconciliation_status": (
                    "MATCH" if difference == 0 else "UNRESOLVED_COUNT_DIFFERENCE"
                ),
            }
        )
    return {
        "schema": "caez-sample-count-reconciliation.v1",
        "retrieved_date": snapshot["retrieved_date"],
        "status": (
            "ALL_COUNTS_MATCH"
            if all(item["count_match"] for item in results)
            else "UNRESOLVED_PUBLISHED_VS_PREPROCESSED_COUNT_DIFFERENCES"
        ),
        "official_sources": {
            "dataset_page": snapshot["dataset_page"],
            "license_page": snapshot["license_page"],
            "preprocessed_directory": snapshot["official_preprocessed_directory"],
            "processing_sources": snapshot["official_processing_sources"],
            "remote_checksums_available": snapshot["remote_checksums_available"],
        },
        "local_metadata_sources": {
            "indoor_audit_sha256": _sha256(_INDOOR_AUDIT.read_bytes()),
            "cross_environment_report_sha256": _sha256(_CROSS_REPORT.read_bytes()),
        },
        "results": results,
        "interpretation": [
            "Official preprocessing code documents timestamp-range filtering for all three environments and optional spatial filtering for indoor/outdoor. Those operations can explain some lower preprocessed counts, but no retained-row crosswalk or exact generator/input version binds these local archives to the official page totals.",
            "The hallway archive has more rows than the published page count; filtering alone cannot explain that difference if both counts refer to the same input version. The page/archive release relationship remains unresolved.",
            "No cause is inferred. Keep CAEZ use exploratory and do not treat these archives as G3 material/LOS qualification evidence.",
        ],
        "license_handling": {
            "raw_archives_are_not_copied_into_this_report": True,
            "raw_archives_must_remain_local_and_non_redistributed": True,
            "required_citation": "R. Wiesmayr et al., CSI-Based User Positioning, Channel Charting, and Device Classification with an NVIDIA 5G Testbed, Asilomar Conference, 2025.",
        },
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "reconciliation_source_sha256": _sha256(Path(__file__).read_bytes()),
    }


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    if not path.parent.is_dir():
        raise ValueError(f"output directory does not exist: {path.parent}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write reconciliation JSON to this path")
    args = parser.parse_args(argv)
    try:
        snapshot, _ = _load_object(_SNAPSHOT)
        indoor, _ = _load_object(_INDOOR_AUDIT)
        cross, _ = _load_object(_CROSS_REPORT)
        result = build_reconciliation(snapshot=snapshot, indoor_audit=indoor, cross_report=cross)
        encoded = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        if args.output is None:
            sys.stdout.buffer.write(encoded)
        else:
            output = args.output.resolve()
            if output in {_SNAPSHOT.resolve(), _INDOOR_AUDIT.resolve(), _CROSS_REPORT.resolve()}:
                raise ValueError("output path must not overwrite an input evidence file")
            _atomic_write(output, encoded)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
        TypeError,
    ) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
