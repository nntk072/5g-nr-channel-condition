"""Bounded semantic consistency checks for proposal-only G3.03 packages."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, cast

from evaluation.g3_joint_acceptance import (
    family_alpha_for_claim,
    validate_g3_joint_acceptance_family,
)

_SCHEMA_ROOT = Path(__file__).resolve().parents[1] / "datasets/schema"
SCHEMA_PATHS = {
    "g3.03-acquisition-package.v0.2": _SCHEMA_ROOT / "g3.03-acquisition-package.v0.2.schema.json",
    "g3.03-acquisition-package.v0.3": _SCHEMA_ROOT / "g3.03-acquisition-package.v0.3.schema.json",
    "g3.03-acquisition-package.v0.4": _SCHEMA_ROOT / "g3.03-acquisition-package.v0.4.schema.json",
}
V03_PLUS_SCHEMA_VERSIONS = frozenset(
    {
        "g3.03-acquisition-package.v0.3",
        "g3.03-acquisition-package.v0.4",
    }
)
EVALUATOR_INPUT_SCHEMA_PATHS = {
    "loslab.g3_material_calibration_predictions.v1": _SCHEMA_ROOT
    / "g3-material-calibration-predictions.v1.schema.json",
    "loslab.g3_material_calibration_predictions.v2": _SCHEMA_ROOT
    / "g3-material-calibration-predictions.v2.schema.json",
}
MATERIAL_MAPPING_SCHEMA_PATH = _SCHEMA_ROOT / "g3-material-mapping.v1.schema.json"
MAX_PACKAGE_BYTES = 16 * 1024 * 1024
MAX_SCHEMA_BYTES = 4 * 1024 * 1024
MAX_INVENTORY_FILES = 10_000
MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_TOTAL_FILE_BYTES = 1024 * 1024 * 1024
MAX_JSON_ARTIFACT_BYTES = 32 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024

LIMITATIONS = [
    "Schema-valid proposal fixtures are synthetic examples and are never measurement evidence.",
    "This consistency report cannot establish scientific validity, physical independence, "
    "calibration truth, preregistration timing, legal rights sufficiency, signature authenticity, "
    "or evaluator authorship.",
    "Only v0.3+ semantic checks validate full-sphere angular grids and their solid-angle cells; v0.2 remains a legacy structural proposal format.",
    "For the current v2 material evaluator, reported results are recomputed from the bound scorer input. Legacy v1 evaluator outputs receive structural checks only.",
    "The hash-bound material map checks package-internal specimen and evaluator IDs but cannot authenticate material identity or evidence-file contents.",
    "Acquisition-package v0.4 recomputes the declared RMS of same-grid noise-only CFR estimates from exact bounded NPY bytes, but cannot establish that the capture is representative or calibrated.",
]


def _noise_rms_summary(data: bytes) -> tuple[float, float, int, tuple[int, ...]]:
    """Summarize one NPY capture of noise-only channel estimates."""
    import numpy as np

    stream = io.BytesIO(data)
    try:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"unsupported NPY format version {version!r}; expected 1.0 or 2.0")
    except (ValueError, OSError, EOFError, OverflowError) as error:
        raise ValueError(f"noise measurement has an invalid NPY header: {error}") from error
    if dtype.kind != "c" or dtype.itemsize not in (8, 16):
        raise ValueError("noise-only CFR estimate must contain complex64 or complex128 values")
    if not shape or any(dimension <= 0 for dimension in shape):
        raise ValueError("noise-only CFR estimate must contain at least one complex value")
    sample_count = math.prod(shape)
    data_offset = stream.tell()
    expected_data_bytes = sample_count * dtype.itemsize
    if expected_data_bytes != len(data) - data_offset:
        raise ValueError("noise measurement NPY payload is truncated or has trailing bytes")
    values = np.frombuffer(data, dtype=dtype, count=sample_count, offset=data_offset)
    real = values.real.astype(np.float64, copy=False).reshape(-1)
    imag = values.imag.astype(np.float64, copy=False).reshape(-1)
    if not np.isfinite(real).all() or not np.isfinite(imag).all():
        raise ValueError("noise-only CFR estimate values must be finite")
    scale = max(float(np.max(np.abs(real))), float(np.max(np.abs(imag))))
    if scale == 0.0:
        return 0.0, 0.0, sample_count, tuple(shape)
    normalized_magnitude = np.hypot(real / scale, imag / scale)
    normalized_sum_squares = float(np.dot(normalized_magnitude, normalized_magnitude))
    if not math.isfinite(normalized_sum_squares):
        raise ValueError("noise measurement RMS accumulation is not finite")
    return scale, normalized_sum_squares, sample_count, tuple(shape)


def _canonical_sha256(value: Any) -> str:
    encoded = _canonical_json(value)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _parse_utc_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _strata_key_set(value: Any) -> set[tuple[str, str, str]] | None:
    if not isinstance(value, list):
        return None
    result: set[tuple[str, str, str]] = set()
    for item in value:
        if not isinstance(item, dict):
            return None
        key = (
            item.get("material_id"),
            item.get("polarization"),
            item.get("incidence_stratum_id"),
        )
        if not all(isinstance(part, str) for part in key):
            return None
        normalized_key = cast(tuple[str, str, str], key)
        if normalized_key in result:
            return None
        result.add(normalized_key)
    return result


def _json_bytes(data: bytes, description: str) -> Any:
    try:
        return json.loads(
            data.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise ValueError(f"{description} is not valid UTF-8 JSON: {error}") from error


def _read_bounded_regular_file(path: Path, *, limit: int, description: str) -> bytes:
    """Read a bounded regular file without following a final symlink or blocking on a FIFO."""
    if not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("platform lacks safe no-symlink file opening")
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{description} must be a regular non-symlink file")
        if info.st_size > limit:
            raise ValueError(f"{description} exceeds {limit} bytes")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(fd, min(CHUNK_BYTES, limit + 1 - size))
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                raise ValueError(f"{description} exceeds {limit} bytes")
            chunks.append(chunk)
        final_info = os.fstat(fd)
        if (
            size != info.st_size
            or final_info.st_size != info.st_size
            or final_info.st_mtime_ns != info.st_mtime_ns
        ):
            raise ValueError(f"{description} changed while being read")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _safe_relative_path(value: Any) -> tuple[str, ...]:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("inventory path must be a nonempty POSIX relative path")
    path = PurePosixPath(value)
    parts = value.split("/")
    if path.is_absolute() or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("inventory path must not be absolute or contain empty/dot segments")
    if any(":" in part for part in parts):
        raise ValueError("inventory path must not contain drive or stream syntax")
    return tuple(parts)


def _read_inventory_file(
    root_fd: int,
    parts: tuple[str, ...],
    limit: int,
    *,
    capture: bool,
) -> tuple[bytes | None, int, str]:
    """Open each path component without following symlinks, then read a bounded regular file."""
    if (
        not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
        or os.open not in os.supports_dir_fd
    ):
        raise ValueError("platform lacks safe directory-relative no-symlink file opening")
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory_flags = flags | getattr(os, "O_DIRECTORY", 0)
    opened: list[int] = []
    try:
        parent_fd = root_fd
        for component in parts[:-1]:
            child_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            opened.append(child_fd)
            parent_fd = child_fd
        file_fd = os.open(parts[-1], flags, dir_fd=parent_fd)
        opened.append(file_fd)
        info = os.fstat(file_fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("inventory path does not resolve to a regular file")
        if info.st_size > limit:
            raise ValueError(f"inventory file exceeds bounded read limit ({limit} bytes)")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(file_fd, min(CHUNK_BYTES, limit + 1 - size))
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                raise ValueError(f"inventory file exceeds bounded read limit ({limit} bytes)")
            digest.update(chunk)
            if capture:
                if size > MAX_JSON_ARTIFACT_BYTES:
                    raise ValueError(
                        f"captured inventory artifact exceeds {MAX_JSON_ARTIFACT_BYTES} bytes"
                    )
                chunks.append(chunk)
        final_info = os.fstat(file_fd)
        if (
            size != info.st_size
            or final_info.st_size != info.st_size
            or final_info.st_mtime_ns != info.st_mtime_ns
        ):
            raise ValueError("inventory file changed while being read")
        content = b"".join(chunks) if capture else None
        return content, size, digest.hexdigest()
    finally:
        for fd in reversed(opened):
            os.close(fd)


def _unique_index(
    records: Any, *, key: str, label: str, errors: list[str]
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    if not isinstance(records, list):
        errors.append(f"{label} must be an array")
        return index
    for item in records:
        if not isinstance(item, dict):
            errors.append(f"{label} contains a non-object record")
            continue
        identity = item.get(key)
        if not isinstance(identity, str) or not identity:
            errors.append(f"{label} record lacks a nonempty {key}")
            continue
        if identity in index:
            errors.append(f"duplicate {label} {key}: {identity}")
        else:
            index[identity] = item
    return index


def _pointer_item(document: Any, pointer: Any, expected_prefix: str) -> Any:
    if not isinstance(pointer, str) or not pointer.startswith(expected_prefix):
        raise ValueError(f"pointer must start with {expected_prefix}")
    suffix = pointer[len(expected_prefix) :]
    if not suffix.isdecimal():
        raise ValueError("pointer index must be a nonnegative decimal integer")
    records = document
    for part in expected_prefix.strip("/").split("/"):
        if not isinstance(records, dict) or part not in records:
            raise ValueError(f"pointer container {part!r} is missing")
        records = records[part]
    if not isinstance(records, list):
        raise ValueError("pointer target container is not an array")
    index = int(suffix)
    if index >= len(records):
        raise ValueError("pointer index is outside its array")
    return records[index]


def verify_g3_acquisition_package(
    package_path: str | Path,
    *,
    package_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate schema, exact artifact bytes, joins, and the evaluator contract.

    ``package_root`` is the directory against which inventory paths resolve. If omitted,
    it is the directory containing the package JSON.
    """
    report: dict[str, Any] = {
        "report_schema": "loslab.g3_acquisition_package_consistency.v1",
        "status": "inconsistent",
        "proposal_only": True,
        "qualification_status": "NOT_QUALIFIED",
        "package_id": None,
        "checks": [],
        "errors": [],
        "limitations": list(LIMITATIONS),
    }
    errors: list[str] = report["errors"]
    checks: list[dict[str, Any]] = report["checks"]

    def check(name: str, passed: bool, details: str) -> None:
        checks.append(
            {"name": name, "status": "passed" if passed else "failed", "details": details}
        )

    package_file = Path(package_path)
    try:
        package_bytes = _read_bounded_regular_file(
            package_file, limit=MAX_PACKAGE_BYTES, description="package JSON"
        )
        package = _json_bytes(package_bytes, "package JSON")
        if not isinstance(package, dict):
            raise ValueError("package JSON root must be an object")
        report["package_id"] = package.get("package_id")
    except (OSError, ValueError) as error:
        errors.append(str(error))
        check("package_json", False, str(error))
        report["limitations"].append("Semantic checks requiring a parsed package were not run.")
        return report
    check("package_json", True, "Package JSON is a bounded regular file and parsed successfully.")

    try:
        import jsonschema

        schema_version = package.get("schema_version")
        schema_path = SCHEMA_PATHS.get(schema_version if isinstance(schema_version, str) else "")
        if schema_path is None:
            raise ValueError(f"unsupported acquisition package schema version: {schema_version!r}")
        schema = _json_bytes(
            _read_bounded_regular_file(
                schema_path, limit=MAX_SCHEMA_BYTES, description="package schema"
            ),
            "package schema",
        )
        validator = jsonschema.Draft202012Validator(
            schema, format_checker=jsonschema.FormatChecker()
        )
        schema_errors = sorted(validator.iter_errors(package), key=lambda item: list(item.path))
        if schema_errors:
            for schema_error in schema_errors[:100]:
                location = "/" + "/".join(str(part) for part in schema_error.absolute_path)
                errors.append(f"JSON Schema {location}: {schema_error.message}")
            check("json_schema", False, f"{len(schema_errors)} schema violation(s)")
            schema_passed = False
        else:
            check("json_schema", True, f"Package conforms to the {schema_version} JSON Schema.")
            schema_passed = True
    except Exception as error:
        errors.append(f"cannot validate package schema: {error}")
        check("json_schema", False, str(error))
        schema_passed = False

    if (
        package.get("schema_version") not in SCHEMA_PATHS
        or package.get("proposal_status") != "proposal_only"
    ):
        errors.append("package must use a supported proposal schema and remain proposal_only")
    binding = package.get("evaluator_binding")
    if (
        not isinstance(binding, dict)
        or binding.get("execution_status") != "not_executed_proposal_only"
    ):
        errors.append("evaluator binding must remain not_executed_proposal_only")
    if not schema_passed:
        report["limitations"].append(
            "Semantic joins were skipped because the package failed schema validation."
        )
        return report

    root = Path(package_root) if package_root is not None else package_file.parent
    try:
        root = root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("package root must be a directory")
    except (OSError, ValueError) as error:
        errors.append(f"invalid package root: {error}")
        check("package_inventory", False, str(error))
        report["status"] = "inconsistent"
        return report

    files = _unique_index(package.get("files"), key="file_id", label="files", errors=errors)
    check(
        "inventory_ids",
        not any("files" in error for error in errors),
        "File IDs are unique and valid.",
    )
    if len(files) > MAX_INVENTORY_FILES:
        errors.append(f"inventory exceeds {MAX_INVENTORY_FILES} files")
    if len(files) > MAX_INVENTORY_FILES:
        check("package_inventory", False, "Inventory file-count bound exceeded.")
        inventory_bytes: dict[str, bytes] = {}
    else:
        inventory_bytes = {}
        verified_file_ids: set[str] = set()
        total_size = 0
        inventory_paths: dict[tuple[str, ...], str] = {}
        root_fd: int | None = None
        try:
            if not hasattr(os, "O_DIRECTORY") or os.open not in os.supports_dir_fd:
                raise ValueError("platform lacks safe directory-relative no-symlink file opening")
            root_fd = os.open(
                root,
                os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW,
            )
        except (OSError, ValueError) as error:
            errors.append(f"cannot securely open package root: {error}")
        binding_for_inventory = package.get("evaluator_binding")
        bound_artifact_names = ("scorer_input", "split_manifest", "evaluator_output")
        if package.get("schema_version") in V03_PLUS_SCHEMA_VERSIONS:
            bound_artifact_names += ("material_mapping",)
        bound_ids = {
            binding_for_inventory.get(name, {}).get("file_id")
            for name in bound_artifact_names
            if isinstance(binding_for_inventory, dict)
            and isinstance(binding_for_inventory.get(name), dict)
        }
        if package.get("schema_version") == "g3.03-acquisition-package.v0.4":
            for csi_record in package.get("csi_measurements", []):
                h_floor = csi_record.get("h_floor", {}) if isinstance(csi_record, dict) else {}
                if isinstance(h_floor, dict):
                    bound_ids.update(
                        file_id
                        for file_id in h_floor.get("noise_input_file_ids", [])
                        if isinstance(file_id, str)
                    )
        for file_id, entry in files.items():
            try:
                parts = _safe_relative_path(entry.get("path"))
                if parts in inventory_paths:
                    raise ValueError(f"path duplicates inventory entry {inventory_paths[parts]!r}")
                inventory_paths[parts] = file_id
                expected_size = entry.get("size_bytes")
                expected_hash = entry.get("sha256")
                if isinstance(expected_size, bool) or not isinstance(expected_size, int):
                    raise ValueError("size_bytes must be an integer")
                remaining_budget = MAX_TOTAL_FILE_BYTES - total_size
                if remaining_budget <= 0:
                    raise ValueError("inventory exceeds total byte-read bound")
                if root_fd is None:
                    raise ValueError("package root could not be securely opened")
                data, actual_size, actual_hash = _read_inventory_file(
                    root_fd,
                    parts,
                    min(MAX_FILE_BYTES, remaining_budget),
                    capture=file_id in bound_ids,
                )
                total_size += actual_size
                if actual_size != expected_size or actual_hash != expected_hash:
                    raise ValueError(
                        "inventory size_bytes or SHA-256 does not match exact file bytes"
                    )
                verified_file_ids.add(file_id)
                if data is not None:
                    inventory_bytes[file_id] = data
            except (OSError, ValueError) as error:
                errors.append(f"inventory {file_id}: {error}")
        check(
            "package_inventory",
            len(verified_file_ids) == len(files),
            f"Verified exact size and SHA-256 for {len(verified_file_ids)} of {len(files)} inventoried files; "
            "paths were opened without following symlinks.",
        )
        if root_fd is not None:
            os.close(root_fd)

    samples = _unique_index(package.get("samples"), key="sample_id", label="samples", errors=errors)
    specimens = _unique_index(
        package.get("specimens"), key="specimen_id", label="specimens", errors=errors
    )
    units = _unique_index(
        package.get("acquisition_units"), key="unit_id", label="acquisition_units", errors=errors
    )
    calibrations = _unique_index(
        package.get("calibrations"), key="calibration_id", label="calibrations", errors=errors
    )
    fits = _unique_index(package.get("fits"), key="fit_id", label="fits", errors=errors)
    angular = _unique_index(
        package.get("angular_measurements"),
        key="angular_id",
        label="angular_measurements",
        errors=errors,
    )
    csi = _unique_index(
        package.get("csi_measurements"), key="csi_id", label="csi_measurements", errors=errors
    )
    campaigns = _unique_index(
        package.get("campaigns"), key="campaign_id", label="campaigns", errors=errors
    )
    instruments = _unique_index(
        package.get("instruments"), key="instrument_id", label="instruments", errors=errors
    )
    protocols = _unique_index(
        package.get("protocols"), key="protocol_id", label="protocols", errors=errors
    )
    uncertainty = _unique_index(
        package.get("uncertainty_records"),
        key="uncertainty_id",
        label="uncertainty_records",
        errors=errors,
    )
    legacy_attest = package.get("attestations", {})
    legacy_attestation_roles: dict[str, str] = {
        "physical_independence": "physical_independence",
        "calibration_truth": "calibration_truth",
        "rights_sufficiency": "rights_sufficiency",
        "external_signature": "external_signature",
    }
    attestation_roles: dict[str, str] = {}
    if isinstance(legacy_attest, dict):
        for legacy_role, record in legacy_attest.items():
            if isinstance(record, dict) and isinstance(record.get("attestation_id"), str):
                attestation_id = record["attestation_id"]
                role = legacy_attestation_roles.get(legacy_role) or str(legacy_role)
                if attestation_id in attestation_roles:
                    errors.append(f"attestation ID {attestation_id!r} is not globally unique")
                else:
                    attestation_roles[attestation_id] = role
    external_attest = _unique_index(
        package.get("external_attestations"),
        key="attestation_id",
        label="external_attestations",
        errors=errors,
    )
    for attestation_id, record in external_attest.items():
        if attestation_id in attestation_roles:
            errors.append(f"attestation ID {attestation_id!r} is not globally unique")
        else:
            external_role = record.get("attestation_type")
            if isinstance(external_role, str):
                attestation_roles[attestation_id] = external_role

    def require_attestation(attestation_id: Any, expected_roles: set[str], where: str) -> None:
        actual_role = (
            attestation_roles.get(attestation_id) if isinstance(attestation_id, str) else None
        )
        if actual_role is None:
            errors.append(f"{where} references unknown attestation")
        elif actual_role not in expected_roles:
            expected = " or ".join(sorted(expected_roles))
            errors.append(
                f"{where} references attestation with type {actual_role!r}; expected {expected}"
            )

    if isinstance(legacy_attest, dict):
        for record in legacy_attest.values():
            if isinstance(record, dict):
                for file_id in record.get("evidence_file_ids", []):
                    if file_id not in files:
                        errors.append(f"legacy attestation references unknown file_id {file_id!r}")

    def require_file_refs(record: dict[str, Any], field: str, where: str) -> None:
        refs = record.get(field, [])
        if not isinstance(refs, list):
            errors.append(f"{where}.{field} must be an array")
            return
        for file_id in refs:
            if file_id not in files:
                errors.append(f"{where}.{field} references unknown file_id {file_id!r}")

    def require_ids(record: dict[str, Any], field: str, target: dict[str, Any], where: str) -> None:
        identity = record.get(field)
        if identity not in target:
            errors.append(f"{where}.{field} references unknown ID {identity!r}")

    for record in files.values():
        where = f"file {record.get('file_id')}"
        require_file_refs(record, "dependency_file_ids", where)
        require_ids_many = (
            ("sample_ids", samples),
            ("specimen_ids", specimens),
            ("calibration_ids", calibrations),
        )
        for field, index in require_ids_many:
            for identity in record.get(field, []):
                if identity not in index:
                    errors.append(f"{where}.{field} references unknown ID {identity!r}")
    for record in specimens.values():
        require_file_refs(record, "provenance_file_ids", f"specimen {record['specimen_id']}")
    for record in units.values():
        require_ids(record, "specimen_id", specimens, f"unit {record['unit_id']}")
        require_file_refs(record, "provenance_file_ids", f"unit {record['unit_id']}")
        require_attestation(
            record.get("independence_attestation_id"),
            {"physical_independence"},
            f"unit {record['unit_id']}.independence_attestation_id",
        )
    for record in calibrations.values():
        calibration_start = _parse_utc_timestamp(record.get("valid_from_utc"))
        calibration_end = _parse_utc_timestamp(record.get("valid_to_utc"))
        if (
            calibration_start is not None
            and calibration_end is not None
            and calibration_start > calibration_end
        ):
            errors.append(
                f"calibration {record['calibration_id']} validity interval starts after it ends"
            )
        for field in ("standard_file_ids", "run_file_ids", "processing_file_ids"):
            require_file_refs(record, field, f"calibration {record['calibration_id']}")
        require_attestation(
            record.get("truth_attestation_id"),
            {"calibration_truth", "calibration_traceability"},
            f"calibration {record['calibration_id']}.truth_attestation_id",
        )
    for record in samples.values():
        for field, index in (
            ("specimen_id", specimens),
            ("unit_id", units),
            ("calibration_id", calibrations),
        ):
            require_ids(record, field, index, f"sample {record['sample_id']}")
        sample_calibration_id = record.get("calibration_id")
        calibration = (
            calibrations.get(sample_calibration_id)
            if isinstance(sample_calibration_id, str)
            else None
        )
        if calibration is not None:
            acquired = _parse_utc_timestamp(record.get("acquired_utc"))
            calibration_start = _parse_utc_timestamp(calibration.get("valid_from_utc"))
            calibration_end = _parse_utc_timestamp(calibration.get("valid_to_utc"))
            if (
                acquired is not None
                and calibration_start is not None
                and calibration_end is not None
                and calibration_start <= calibration_end
                and not calibration_start <= acquired <= calibration_end
            ):
                errors.append(
                    f"sample {record['sample_id']} acquired_utc lies outside calibration validity interval"
                )
            sample_file_ids = record.get("file_ids", [])
            calibration_run_file_ids = calibration.get("run_file_ids", [])
            if isinstance(sample_file_ids, list) and isinstance(calibration_run_file_ids, list):
                if not set(calibration_run_file_ids).issubset(set(sample_file_ids)):
                    errors.append(
                        f"sample {record['sample_id']} does not reference all calibration run evidence files"
                    )
        if package.get("schema_version") in V03_PLUS_SCHEMA_VERSIONS:
            campaign_id = record.get("campaign_id")
            instrument_id = record.get("instrument_id")
            campaign = campaigns.get(campaign_id) if isinstance(campaign_id, str) else None
            instrument = instruments.get(instrument_id) if isinstance(instrument_id, str) else None
            require_ids(record, "campaign_id", campaigns, f"sample {record['sample_id']}")
            require_ids(record, "instrument_id", instruments, f"sample {record['sample_id']}")
            if campaign is not None:
                if instrument_id not in campaign.get("instrument_ids", []):
                    errors.append(
                        f"sample {record['sample_id']} instrument is not assigned to its campaign"
                    )
                campaign_start = _parse_utc_timestamp(campaign.get("start_utc"))
                campaign_end = _parse_utc_timestamp(campaign.get("end_utc"))
                acquired = _parse_utc_timestamp(record.get("acquired_utc"))
                if (
                    acquired is not None
                    and campaign_start is not None
                    and campaign_end is not None
                    and campaign_start <= campaign_end
                    and not campaign_start <= acquired <= campaign_end
                ):
                    errors.append(
                        f"sample {record['sample_id']} acquired_utc lies outside campaign interval"
                    )
                campaign_protocol = protocols.get(campaign.get("protocol_id"))
                if (
                    campaign_protocol is not None
                    and campaign_protocol.get("target_frequency_hz")
                    != record.get("frequency_hz")
                ):
                    errors.append(
                        f"sample {record['sample_id']} frequency differs from campaign protocol"
                    )
            if instrument is not None:
                configuration = instrument.get("configuration")
                instrument_frequency = (
                    configuration.get("carrier_frequency_hz")
                    if isinstance(configuration, dict)
                    else None
                )
                if instrument_frequency != record.get("frequency_hz"):
                    errors.append(
                        f"sample {record['sample_id']} frequency differs from instrument configuration"
                    )
        sample_unit = units.get(record.get("unit_id"))
        if sample_unit is not None and sample_unit.get("specimen_id") != record.get("specimen_id"):
            errors.append(
                f"sample {record['sample_id']} specimen_id differs from its acquisition unit"
            )
        require_file_refs(record, "file_ids", f"sample {record['sample_id']}")
        sample_id = record.get("sample_id")
        for file_id in record.get("file_ids", []):
            file_record = files.get(file_id) if isinstance(file_id, str) else None
            if file_record is not None and sample_id not in file_record.get("sample_ids", []):
                errors.append(
                    f"sample {sample_id} file {file_id} does not declare the sample association"
                )
    for record in fits.values():
        require_ids(record, "specimen_id", specimens, f"fit {record['fit_id']}")
        require_file_refs(record, "software_file_ids", f"fit {record['fit_id']}")
        for parameter in record.get("parameters", []):
            if "covariance_file_id" in parameter and parameter["covariance_file_id"] not in files:
                errors.append(f"fit {record['fit_id']} references unknown covariance file")
    for record in angular.values():
        require_ids(record, "sample_id", samples, f"angular {record['angular_id']}")
        theta = record.get("theta_edges_deg")
        phi = record.get("phi_edges_deg")
        expected_shape = (
            (len(theta) - 1, len(phi) - 1)
            if isinstance(theta, list)
            and isinstance(phi, list)
            and len(theta) >= 2
            and len(phi) >= 2
            else None
        )
        if expected_shape is None or any(
            isinstance(edges[index], bool)
            or not isinstance(edges[index], (int, float))
            or edges[index] >= edges[index + 1]
            for edges in (theta, phi)
            if isinstance(edges, list)
            for index in range(len(edges) - 1)
        ):
            errors.append(f"angular {record['angular_id']} edges must be strictly increasing")
        else:
            matrix_dimensions_valid = True
            for field in ("power_per_bin_w", "solid_angle_sr"):
                matrix = record.get(field)
                if (
                    not isinstance(matrix, list)
                    or len(matrix) != expected_shape[0]
                    or any(
                        not isinstance(row, list) or len(row) != expected_shape[1] for row in matrix
                    )
                ):
                    errors.append(
                        f"angular {record['angular_id']}.{field} dimensions must match theta/phi bins"
                    )
                    matrix_dimensions_valid = False
            missing = record.get("missing_bin_indices", [])
            if isinstance(missing, list):
                seen_bins: set[tuple[int, int]] = set()
                for pair in missing:
                    if (
                        not isinstance(pair, list)
                        or len(pair) != 2
                        or any(
                            isinstance(value, bool) or not isinstance(value, int) for value in pair
                        )
                        or not (
                            0 <= pair[0] < expected_shape[0] and 0 <= pair[1] < expected_shape[1]
                        )
                    ):
                        errors.append(
                            f"angular {record['angular_id']} has an invalid missing-bin index"
                        )
                        continue
                    coordinate = (pair[0], pair[1])
                    if coordinate in seen_bins:
                        errors.append(f"angular {record['angular_id']} repeats a missing-bin index")
                    seen_bins.add(coordinate)
            if package.get("schema_version") in V03_PLUS_SCHEMA_VERSIONS:
                angular_id = record["angular_id"]
                if not matrix_dimensions_valid:
                    continue
                if record.get("missing_bin_indices"):
                    errors.append(
                        f"angular {angular_id} has missing bins and cannot support a complete pattern"
                    )
                if not isinstance(theta, list) or not isinstance(phi, list):
                    continue
                if not (
                    math.isclose(float(theta[0]), 0.0, rel_tol=0.0, abs_tol=1e-12)
                    and math.isclose(float(theta[-1]), 180.0, rel_tol=0.0, abs_tol=1e-12)
                    and math.isclose(
                        float(phi[-1]) - float(phi[0]), 360.0, rel_tol=0.0, abs_tol=1e-12
                    )
                ):
                    errors.append(
                        f"angular {angular_id} must cover zenith 0–180 and a full 360-degree azimuth"
                    )
                    continue
                power = record.get("power_per_bin_w")
                solid_angle = record.get("solid_angle_sr")
                if not isinstance(power, list) or not isinstance(solid_angle, list):
                    continue
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or value < 0
                    for matrix in (power, solid_angle)
                    for matrix_row in matrix
                    for value in matrix_row
                ):
                    errors.append(
                        f"angular {angular_id} power and solid-angle cells must be finite and nonnegative"
                    )
                    continue
                if sum(sum(float(value) for value in row) for row in power) <= 0.0:
                    errors.append(f"angular {angular_id} power_per_bin_w must have a positive sum")
                theta_radians = [math.radians(float(edge)) for edge in theta]
                phi_radians = [math.radians(float(edge)) for edge in phi]
                expected_solid_angle = [
                    [
                        (phi_radians[column + 1] - phi_radians[column])
                        * (math.cos(theta_radians[row]) - math.cos(theta_radians[row + 1]))
                        for column in range(len(phi) - 1)
                    ]
                    for row in range(len(theta) - 1)
                ]
                if any(
                    not math.isclose(
                        float(solid_angle[row][column]),
                        expected_solid_angle[row][column],
                        rel_tol=1e-9,
                        abs_tol=1e-12,
                    )
                    for row in range(len(theta) - 1)
                    for column in range(len(phi) - 1)
                ):
                    errors.append(
                        f"angular {angular_id} solid_angle_sr must match the spherical bin geometry"
                    )
                elif not math.isclose(
                    sum(sum(row) for row in solid_angle),
                    4.0 * math.pi,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                ):
                    errors.append(
                        f"angular {angular_id} solid-angle cells must sum to 4*pi steradians"
                    )
    for record in csi.values():
        require_ids(record, "sample_id", samples, f"csi {record['csi_id']}")
        require_file_refs(record, "file_ids", f"csi {record['csi_id']}")
        require_file_refs(
            record.get("h_floor", {}), "noise_input_file_ids", f"csi {record['csi_id']}.h_floor"
        )
        h_floor = record.get("h_floor")
        noise_input_file_ids = (
            h_floor.get("noise_input_file_ids", []) if isinstance(h_floor, dict) else []
        )
        if package.get("schema_version") in V03_PLUS_SCHEMA_VERSIONS:
            for file_id in noise_input_file_ids:
                noise_file = files.get(file_id) if isinstance(file_id, str) else None
                if noise_file is not None and noise_file.get("role") != "noise_measurement":
                    errors.append(
                        f"csi {record['csi_id']}.h_floor noise input file {file_id!r} "
                        "must have inventory role 'noise_measurement'"
                    )
        amplitude_floor = (
            h_floor.get("amplitude_linear") if isinstance(h_floor, dict) else None
        )
        try:
            amplitude_floor_float = (
                float(amplitude_floor)
                if isinstance(amplitude_floor, (int, float))
                and not isinstance(amplitude_floor, bool)
                else math.nan
            )
        except (OverflowError, ValueError):
            amplitude_floor_float = math.nan
        if not math.isfinite(amplitude_floor_float) or amplitude_floor_float <= 0.0:
            errors.append(
                f"csi {record['csi_id']}.h_floor amplitude_linear must be finite and strictly positive"
            )
        if package.get("schema_version") == "g3.03-acquisition-package.v0.4":
            expected_method = "noise_only_cfr_estimate_rms_v1"
            if isinstance(h_floor, dict) and h_floor.get("method") != expected_method:
                errors.append(
                    f"csi {record['csi_id']}.h_floor method must be {expected_method!r}"
                )
            scale = 0.0
            normalized_sum_squares = 0.0
            sample_count = 0
            derivation_valid = bool(noise_input_file_ids)
            if not isinstance(h_floor, dict) or h_floor.get("method") != expected_method:
                derivation_valid = False
            for file_id in noise_input_file_ids:
                noise_file = files.get(file_id) if isinstance(file_id, str) else None
                if noise_file is None:
                    derivation_valid = False
                    continue
                if noise_file.get("role") != "noise_measurement":
                    derivation_valid = False
                if noise_file.get("media_type") != "application/x-npy":
                    errors.append(
                        f"csi {record['csi_id']}.h_floor noise input {file_id!r} "
                        "must declare media_type 'application/x-npy'"
                    )
                    derivation_valid = False
                if not str(noise_file.get("path", "")).endswith(".npy"):
                    errors.append(
                        f"csi {record['csi_id']}.h_floor noise input {file_id!r} "
                        "must use a .npy inventory path"
                    )
                    derivation_valid = False
                if noise_file.get("sample_ids") != [record.get("sample_id")]:
                    errors.append(
                        f"csi {record['csi_id']}.h_floor noise input {file_id!r} "
                        "must be bound only to the same sample_id"
                    )
                    derivation_valid = False
                noise_bytes = inventory_bytes.get(file_id)
                if noise_bytes is None:
                    errors.append(
                        f"csi {record['csi_id']}.h_floor noise input {file_id!r} "
                        "has no captured, hash-verified bytes"
                    )
                    derivation_valid = False
                    continue
                try:
                    (
                        local_scale,
                        local_sum_squares,
                        local_count,
                        noise_shape,
                    ) = _noise_rms_summary(noise_bytes)
                except (ValueError, OverflowError) as error:
                    errors.append(
                        f"csi {record['csi_id']}.h_floor noise input {file_id!r}: {error}"
                    )
                    derivation_valid = False
                    continue
                if noise_shape != tuple(record.get("shape", [])):
                    errors.append(
                        f"csi {record['csi_id']}.h_floor noise input {file_id!r} "
                        "must have the same CFR axis shape as the CSI record"
                    )
                    derivation_valid = False
                new_scale = max(scale, local_scale)
                if new_scale > 0.0:
                    if scale > 0.0:
                        normalized_sum_squares *= (scale / new_scale) ** 2
                    if local_scale > 0.0:
                        normalized_sum_squares += local_sum_squares * (
                            local_scale / new_scale
                        ) ** 2
                    scale = new_scale
                sample_count += local_count
            derived_floor = (
                scale * math.sqrt(normalized_sum_squares / sample_count)
                if sample_count > 0 and scale > 0.0
                else math.nan
            )
            if not math.isfinite(derived_floor) or derived_floor <= 0.0:
                errors.append(
                    f"csi {record['csi_id']}.h_floor noise measurements must derive a "
                    "finite, strictly positive complex RMS"
                )
                derivation_valid = False
            elif not math.isclose(
                amplitude_floor_float, derived_floor, rel_tol=1e-9, abs_tol=0.0
            ):
                errors.append(
                    f"csi {record['csi_id']}.h_floor amplitude_linear does not match "
                    f"derived complex-noise RMS ({derived_floor:.17g})"
                )
                derivation_valid = False
            check(
                f"csi_noise_floor:{record['csi_id']}",
                derivation_valid,
                f"method={expected_method}; samples={sample_count}; "
                f"declared={amplitude_floor_float:.17g}; derived={derived_floor:.17g}",
            )
        shape = record.get("shape")
        if (
            isinstance(shape, list)
            and len(shape) == 4
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in shape
            )
        ):
            expected_count = math.prod(shape)
            if (
                len(record.get("measured_complex", [])) != expected_count
                or len(record.get("predicted_complex", [])) != expected_count
            ):
                errors.append(f"csi {record['csi_id']} flattened complex values do not match shape")
            for field, dimension in (
                ("rx_port_ids", 0),
                ("tx_port_ids", 1),
                ("subcarrier_indices", 2),
                ("subcarrier_frequency_offsets_hz", 2),
                ("ofdm_symbol_indices", 3),
            ):
                values = record.get(field)
                if not isinstance(values, list) or len(values) != shape[dimension]:
                    errors.append(f"csi {record['csi_id']}.{field} length does not match shape")
        else:
            errors.append(f"csi {record['csi_id']} shape must contain four positive integers")
    for record in package.get("rights", []) if isinstance(package.get("rights"), list) else []:
        require_file_refs(record, "evidence_file_ids", f"rights {record.get('rights_id')}")
        require_attestation(
            record.get("attestation_id"),
            {"rights_sufficiency", "rights_review"},
            f"rights {record.get('rights_id')}.attestation_id",
        )
    for record in campaigns.values():
        if record.get("study_id") != package.get("study_id"):
            errors.append(
                f"campaign {record['campaign_id']} study_id differs from package study_id"
            )
        campaign_start = _parse_utc_timestamp(record.get("start_utc"))
        campaign_end = _parse_utc_timestamp(record.get("end_utc"))
        if (
            campaign_start is not None
            and campaign_end is not None
            and campaign_start > campaign_end
        ):
            errors.append(f"campaign {record['campaign_id']} interval starts after it ends")
        for identity in record.get("instrument_ids", []):
            if identity not in instruments:
                errors.append(
                    f"campaign {record['campaign_id']} references unknown instrument {identity!r}"
                )
        require_ids(record, "protocol_id", protocols, f"campaign {record['campaign_id']}")
        require_file_refs(record, "record_file_ids", f"campaign {record['campaign_id']}")
    for record in instruments.values():
        require_file_refs(record, "configuration_file_ids", f"instrument {record['instrument_id']}")
    for record in protocols.values():
        require_file_refs(record, "protocol_file_ids", f"protocol {record['protocol_id']}")
        if record.get("split_manifest_file_id") not in files:
            errors.append(
                f"protocol {record['protocol_id']} references unknown split manifest file"
            )
    if isinstance(binding, dict):
        require_file_refs(binding, "run_configuration_file_ids", "evaluator_binding")
    for record in uncertainty.values():
        for identity in record.get("sample_ids", []):
            if identity not in samples:
                errors.append(
                    f"uncertainty {record['uncertainty_id']} references unknown sample {identity!r}"
                )
        for field in ("repeatability_file_ids", "uncertainty_record_file_ids"):
            require_file_refs(record, field, f"uncertainty {record['uncertainty_id']}")
    for record in external_attest.values():
        for field in ("subject_file_ids", "verification_record_file_ids"):
            require_file_refs(record, field, f"attestation {record['attestation_id']}")
        if record.get("signature_file_id") is not None and record["signature_file_id"] not in files:
            errors.append(
                f"attestation {record['attestation_id']} references unknown signature file"
            )

    rights_records = package.get("per_file_rights")
    rights_index = _unique_index(
        rights_records, key="file_id", label="per_file_rights", errors=errors
    )
    if set(rights_index) != set(files):
        errors.append("per_file_rights must cover every inventoried file exactly once")
    for file_id, record in rights_index.items():
        require_attestation(
            record.get("attestation_id"),
            {"rights_sufficiency", "rights_review"},
            f"per-file rights for {file_id}.attestation_id",
        )
        require_file_refs(record, "evidence_file_ids", f"per-file rights {file_id}")
    check(
        "reference_joins",
        not errors,
        "Reference IDs, record joins, sample-to-file association, ordered campaign/calibration intervals, sample acquisition-time coverage, campaign/instrument/protocol frequency joins, attached calibration run evidence, and per-file rights coverage checked.",
    )

    semantic_errors_before_binding = len(errors)
    parsed_artifacts: dict[str, Any] = {}
    bound_file_ids: dict[str, str] = {}
    expected_roles = {
        "scorer_input": "scorer_input",
        "split_manifest": "split_manifest",
        "evaluator_output": "evaluator_output",
    }
    if package.get("schema_version") in V03_PLUS_SCHEMA_VERSIONS:
        expected_roles["material_mapping"] = "material_mapping"
    if isinstance(binding, dict):
        for field, role in expected_roles.items():
            reference = binding.get(field)
            if not isinstance(reference, dict):
                errors.append(f"evaluator_binding.{field} is missing")
                continue
            file_id_value = reference.get("file_id")
            if not isinstance(file_id_value, str):
                errors.append(f"evaluator_binding.{field}.file_id must be a string")
                continue
            file_id = file_id_value
            bound_file_ids[field] = file_id
            inventory = files.get(file_id)
            if inventory is None:
                errors.append(f"evaluator_binding.{field} references unknown file_id {file_id!r}")
                continue
            if inventory.get("role") != role:
                errors.append(f"evaluator_binding.{field} file must have inventory role {role!r}")
            bound_data = inventory_bytes.get(file_id)
            if bound_data is None:
                continue
            digest = hashlib.sha256(bound_data).hexdigest()
            if reference.get("exact_bytes_sha256") != digest or reference.get(
                "exact_bytes_size"
            ) != len(bound_data):
                errors.append(
                    f"evaluator_binding.{field} exact-byte reference disagrees with file bytes"
                )
            try:
                parsed_artifacts[field] = _json_bytes(bound_data, field)
            except ValueError as error:
                errors.append(str(error))
    check(
        "evaluator_artifact_bindings",
        len(errors) == semantic_errors_before_binding,
        "Bound evaluator artifacts resolve to exact inventoried bytes and expected roles.",
    )

    input_doc = parsed_artifacts.get("scorer_input")
    split_doc = parsed_artifacts.get("split_manifest")
    output_doc = parsed_artifacts.get("evaluator_output")
    material_mapping_doc = parsed_artifacts.get("material_mapping")
    binding_errors = len(errors)
    if isinstance(input_doc, dict):
        input_contract = binding.get("input_contract") if isinstance(binding, dict) else None
        input_schema_path = EVALUATOR_INPUT_SCHEMA_PATHS.get(
            input_contract if isinstance(input_contract, str) else ""
        )
        if input_doc.get("schema") != input_contract or input_schema_path is None:
            errors.append("scorer input schema does not match evaluator input contract")
        if input_schema_path is not None:
            try:
                import jsonschema

                input_schema = _json_bytes(
                    _read_bounded_regular_file(
                        input_schema_path,
                        limit=MAX_SCHEMA_BYTES,
                        description="evaluator input schema",
                    ),
                    "evaluator input schema",
                )
                input_validator = jsonschema.Draft202012Validator(
                    input_schema, format_checker=jsonschema.FormatChecker()
                )
                input_schema_errors = list(input_validator.iter_errors(input_doc))
                if input_schema_errors:
                    errors.append(
                        "scorer input violates evaluator input JSON Schema: "
                        + "; ".join(error.message for error in input_schema_errors[:10])
                    )
            except Exception as error:
                errors.append(f"cannot validate evaluator input schema: {error}")
        if package.get("schema_version") in V03_PLUS_SCHEMA_VERSIONS:
            try:
                import jsonschema

                mapping_schema = _json_bytes(
                    _read_bounded_regular_file(
                        MATERIAL_MAPPING_SCHEMA_PATH,
                        limit=MAX_SCHEMA_BYTES,
                        description="G3 material mapping schema",
                    ),
                    "G3 material mapping schema",
                )
                mapping_errors = list(
                    jsonschema.Draft202012Validator(
                        mapping_schema, format_checker=jsonschema.FormatChecker()
                    ).iter_errors(material_mapping_doc)
                )
                if mapping_errors:
                    errors.append(
                        "material mapping violates JSON Schema: "
                        + "; ".join(error.message for error in mapping_errors[:10])
                    )
            except Exception as error:
                errors.append(f"cannot validate G3 material mapping schema: {error}")
            if not isinstance(material_mapping_doc, dict):
                errors.append("bound material mapping must be a JSON object")
            elif material_mapping_doc.get("material_mapping_id") != input_doc.get(
                "material_mapping_id"
            ):
                errors.append("bound material mapping ID differs from scorer input")
        requires_joint_family = package.get("schema_version") in V03_PLUS_SCHEMA_VERSIONS
        if requires_joint_family and "joint_acceptance_family" not in input_doc:
            errors.append("v0.3 scorer input requires the shared joint G3 acceptance family")
        if "joint_acceptance_family" in input_doc:
            try:
                family = validate_g3_joint_acceptance_family(input_doc["joint_acceptance_family"])
                family_alpha_for_claim(
                    family,
                    evaluator="material_calibration",
                    expected_strata=[
                        (
                            stratum.get("material_id"),
                            stratum.get("polarization"),
                            stratum.get("incidence_stratum_id"),
                        )
                        for stratum in input_doc.get("expected_strata", [])
                        if isinstance(stratum, dict)
                    ],
                )
            except (TypeError, ValueError) as error:
                errors.append(f"scorer input joint G3 family is invalid: {error}")
        embedded_manifest = input_doc.get("full_split_manifest")
        embedded_hash = input_doc.get("full_split_manifest_sha256")
        if not isinstance(embedded_manifest, dict):
            errors.append("scorer input full_split_manifest must be an object")
        else:
            manifest_strata = _strata_key_set(embedded_manifest.get("expected_strata"))
            input_strata = _strata_key_set(input_doc.get("expected_strata"))
            if manifest_strata is None or manifest_strata != input_strata:
                errors.append(
                    "scorer input expected_strata differs from embedded split-manifest expected_strata"
                )
            try:
                canonical_hash = _canonical_sha256(embedded_manifest)
                if not isinstance(embedded_hash, str) or embedded_hash.lower() != canonical_hash:
                    errors.append("scorer input canonical full_split_manifest hash is incorrect")
                for protocol in protocols.values():
                    if protocol.get("split_manifest_canonical_sha256") != canonical_hash:
                        errors.append(
                            f"protocol {protocol['protocol_id']} canonical split hash differs"
                        )
                    protocol_strata = _strata_key_set(protocol.get("expected_strata"))
                    input_strata = _strata_key_set(input_doc.get("expected_strata"))
                    if protocol_strata is None or protocol_strata != input_strata:
                        errors.append(
                            f"protocol {protocol['protocol_id']} expected_strata differs from scorer input"
                        )
                    if protocol.get("target_frequency_hz") != input_doc.get("target_frequency_hz"):
                        errors.append(
                            f"protocol {protocol['protocol_id']} target frequency differs from scorer input"
                        )
            except (TypeError, ValueError) as error:
                errors.append(f"cannot canonicalize embedded split manifest: {error}")
        if bound_file_ids.get("split_manifest"):
            if not isinstance(split_doc, dict) or not isinstance(embedded_manifest, dict):
                errors.append("split-manifest file and embedded manifest must both be JSON objects")
            elif _canonical_sha256(split_doc) != _canonical_sha256(embedded_manifest):
                errors.append(
                    "split-manifest file content differs from scorer input embedded manifest"
                )
            for protocol in protocols.values():
                if protocol.get("split_manifest_file_id") != bound_file_ids["split_manifest"]:
                    errors.append(
                        f"protocol {protocol['protocol_id']} split-manifest file binding differs"
                    )
    if isinstance(output_doc, dict):
        output_contract = binding.get("output_contract") if isinstance(binding, dict) else None
        if output_doc.get("schema") != output_contract:
            errors.append("evaluator output schema does not match output contract")
        scorer_data = inventory_bytes.get(bound_file_ids.get("scorer_input", ""))
        scorer_digest = hashlib.sha256(scorer_data).hexdigest() if scorer_data is not None else None
        echoed = output_doc.get("prediction_input_sha256")
        if not isinstance(echoed, str) or echoed != scorer_digest:
            errors.append(
                "evaluator output prediction_input_sha256 does not echo exact scorer-input bytes"
            )
        if isinstance(binding, dict) and binding.get("reported_prediction_input_sha256") != echoed:
            errors.append(
                "evaluator_binding reported_prediction_input_sha256 differs from evaluator output"
            )
        # The input hash proves which bytes the output claims to describe, but it
        # does not prove that the reported metrics were calculated from them.
        # Recompute current v2 reports using the same deterministic evaluator and
        # compare the complete document, including all metric and status fields.
        if (
            output_contract == "loslab.g3_material_calibration_evaluation.v2"
            and isinstance(input_doc, dict)
            and scorer_digest is not None
        ):
            try:
                from evaluation.g3_material_calibration import evaluate_g3_material_calibration

                recomputed = evaluate_g3_material_calibration(input_doc)
                recomputed["prediction_input_sha256"] = scorer_digest
                if _canonical_json(output_doc) != _canonical_json(recomputed):
                    errors.append(
                        "evaluator output differs from deterministic recomputation from scorer input"
                    )
            except Exception as error:
                errors.append(f"cannot recompute v2 evaluator output from scorer input: {error}")
        if not isinstance(output_doc.get("strata"), list):
            errors.append("evaluator output strata must be an array")
        else:
            output_keys: list[tuple[str, str, str]] = []
            for item in output_doc["strata"]:
                if not isinstance(item, dict):
                    errors.append("evaluator output strata contain non-object entries")
                    continue
                key = (
                    item.get("material_id"),
                    item.get("polarization"),
                    item.get("incidence_stratum_id"),
                )
                if not all(isinstance(part, str) for part in key):
                    errors.append("evaluator output stratum keys must be strings")
                    continue
                output_keys.append(cast(tuple[str, str, str], key))
            if len(set(output_keys)) != len(output_keys):
                errors.append("evaluator output strata contain non-objects or duplicate keys")
            expected_strata = (
                input_doc.get("expected_strata") if isinstance(input_doc, dict) else None
            )
            if isinstance(expected_strata, list):
                expected_keys: set[tuple[str, str, str]] = set()
                for item in expected_strata:
                    if not isinstance(item, dict):
                        continue
                    key = (
                        item.get("material_id"),
                        item.get("polarization"),
                        item.get("incidence_stratum_id"),
                    )
                    if all(isinstance(part, str) for part in key):
                        expected_keys.add(cast(tuple[str, str, str], key))
                if set(output_keys) != expected_keys:
                    errors.append(
                        "evaluator output strata do not exactly match scorer input expected_strata"
                    )
    elif output_doc is not None:
        errors.append("evaluator output must be a JSON object")
    if isinstance(input_doc, dict) and isinstance(output_doc, dict):
        crosswalk_rows = package.get("row_crosswalk", [])
        seen_input: set[str] = set()
        seen_output_strata: dict[str, tuple[str, str, str]] = {}
        seen_evaluator_samples: set[str] = set()
        seen_acquisition_samples: set[str] = set()
        seen_angular_measurements: set[str] = set()
        evaluator_rows = input_doc.get("test_predictions")
        output_strata = output_doc.get("strata")
        embedded_manifest = input_doc.get("full_split_manifest")
        assignment_records = (
            embedded_manifest.get("test_sample_assignments", [])
            if isinstance(embedded_manifest, dict)
            else []
        )
        group_records = (
            embedded_manifest.get("groups", []) if isinstance(embedded_manifest, dict) else []
        )
        evaluator_group_records = input_doc.get("incidence_groups", [])
        assignments_by_sample = {
            item.get("sample_id"): item
            for item in assignment_records
            if isinstance(item, dict) and isinstance(item.get("sample_id"), str)
        }
        groups_by_id = {
            item.get("incidence_group_id"): item
            for item in group_records
            if isinstance(item, dict) and isinstance(item.get("incidence_group_id"), str)
        }
        evaluator_groups_by_id = {
            item.get("incidence_group_id"): item
            for item in evaluator_group_records
            if isinstance(item, dict) and isinstance(item.get("incidence_group_id"), str)
        }
        if len(assignments_by_sample) != len(assignment_records):
            errors.append(
                "embedded split manifest has duplicate or invalid test sample assignments"
            )
        if len(groups_by_id) != len(group_records):
            errors.append("embedded split manifest has duplicate or invalid group IDs")
        if len(evaluator_groups_by_id) != len(evaluator_group_records):
            errors.append("scorer input has duplicate or invalid incidence group IDs")
        material_id_by_specimen: dict[str, str] = {}
        if package.get("schema_version") in V03_PLUS_SCHEMA_VERSIONS and isinstance(
            material_mapping_doc, dict
        ):
            material_records = material_mapping_doc.get("materials", [])
            seen_material_ids: set[str] = set()
            for material_record in material_records if isinstance(material_records, list) else []:
                if not isinstance(material_record, dict):
                    continue
                material_id = material_record.get("material_id")
                if not isinstance(material_id, str):
                    continue
                if material_id in seen_material_ids:
                    errors.append("material mapping contains duplicate material_id entries")
                seen_material_ids.add(material_id)
                for specimen_id in material_record.get("specimen_ids", []):
                    if not isinstance(specimen_id, str):
                        continue
                    specimen = specimens.get(specimen_id)
                    if specimen is None:
                        errors.append("material mapping references an unknown specimen_id")
                        continue
                    if specimen.get("material_claim") not in material_record.get(
                        "material_claims", []
                    ):
                        errors.append(
                            "material mapping does not include the mapped specimen material_claim"
                        )
                    if specimen_id in material_id_by_specimen:
                        errors.append(
                            "material mapping assigns one specimen to multiple material IDs"
                        )
                    material_id_by_specimen[specimen_id] = material_id
                for file_id in material_record.get("evidence_file_ids", []):
                    if file_id not in files:
                        errors.append("material mapping references an unknown evidence file")
            if set(material_id_by_specimen) != set(specimens):
                errors.append("material mapping must assign every package specimen exactly once")
            input_material_ids = {
                item.get("material_id")
                for item in input_doc.get("expected_strata", [])
                if isinstance(item, dict) and isinstance(item.get("material_id"), str)
            }
            if not input_material_ids.issubset(seen_material_ids):
                errors.append("material mapping omits a scorer input expected material ID")
        if not isinstance(evaluator_rows, list) or not isinstance(output_strata, list):
            errors.append("evaluator input test_predictions and output strata must be arrays")
        else:
            for row in crosswalk_rows if isinstance(crosswalk_rows, list) else []:
                try:
                    input_row = _pointer_item(
                        input_doc, row.get("input_row_pointer"), "/test_predictions/"
                    )
                    output_stratum = _pointer_item(
                        output_doc, row.get("output_stratum_pointer"), "/strata/"
                    )
                    if not isinstance(input_row, dict) or not isinstance(output_stratum, dict):
                        raise ValueError("crosswalk pointers must resolve to objects")
                    if row.get("evaluator_sample_id") != input_row.get("sample_id"):
                        errors.append("crosswalk evaluator_sample_id differs from scorer input row")
                    acquisition_sample_id = row.get("acquisition_sample_id")
                    if isinstance(acquisition_sample_id, str):
                        if acquisition_sample_id in seen_acquisition_samples:
                            errors.append("row_crosswalk repeats acquisition_sample_id")
                        seen_acquisition_samples.add(acquisition_sample_id)
                    sample = (
                        samples.get(acquisition_sample_id)
                        if isinstance(acquisition_sample_id, str)
                        else None
                    )
                    if sample is None:
                        errors.append("crosswalk acquisition_sample_id is not in package samples")
                    else:
                        sample_specimen_id = sample.get("specimen_id")
                        if package.get("schema_version") in V03_PLUS_SCHEMA_VERSIONS:
                            angular_id = row.get("angular_measurement_id")
                            if isinstance(angular_id, str):
                                if angular_id in seen_angular_measurements:
                                    errors.append(
                                        "row_crosswalk repeats angular_measurement_id"
                                    )
                                seen_angular_measurements.add(angular_id)
                            angular_record = (
                                angular.get(angular_id) if isinstance(angular_id, str) else None
                            )
                            if angular_record is None:
                                errors.append(
                                    "crosswalk angular_measurement_id is not in package angular_measurements"
                                )
                            else:
                                if angular_record.get("sample_id") != acquisition_sample_id:
                                    errors.append(
                                        "crosswalk angular measurement belongs to a different acquisition sample"
                                    )
                                angular_joins = (
                                    ("coordinate_frame_id", "angular_coordinate_frame_id"),
                                    ("theta_edges_deg", "theta_edges_deg"),
                                    ("phi_edges_deg", "phi_edges_deg"),
                                    ("array_order", "angular_array_order"),
                                    ("power_semantics", "angular_power_semantics"),
                                    ("power_per_bin_w", "measured_angular_power_per_bin_w"),
                                    ("solid_angle_sr", "solid_angle_sr"),
                                    ("missing_bin_indices", "missing_angular_bin_indices"),
                                )
                                for acquisition_field, evaluator_field in angular_joins:
                                    if angular_record.get(acquisition_field) != input_row.get(
                                        evaluator_field
                                    ):
                                        errors.append(
                                            "acquisition angular measurement "
                                            f"{acquisition_field} differs from scorer input "
                                            f"{evaluator_field}"
                                        )
                        if package.get("schema_version") in V03_PLUS_SCHEMA_VERSIONS:
                            mapped_material_id = (
                                material_id_by_specimen.get(sample_specimen_id)
                                if isinstance(sample_specimen_id, str)
                                else None
                            )
                            if mapped_material_id != row.get("material_id"):
                                errors.append(
                                    "crosswalk material_id differs from bound specimen material mapping"
                                )
                        if sample.get("polarization") != input_row.get("polarization"):
                            errors.append(
                                "acquisition sample polarization differs from scorer input row"
                            )
                        if sample.get("incidence_zenith_deg") != input_row.get(
                            "incidence_angle_deg"
                        ):
                            errors.append(
                                "acquisition sample incidence angle differs from scorer input row"
                            )
                        if sample.get("frequency_hz") != input_doc.get("target_frequency_hz"):
                            errors.append(
                                "acquisition sample frequency differs from scorer target frequency"
                            )
                        if sample.get("calibration_id") != input_doc.get(
                            "measurement_calibration_id"
                        ):
                            errors.append(
                                "acquisition sample calibration differs from scorer measurement calibration"
                            )
                        sample_unit_id = sample.get("unit_id")
                        unit = (
                            units.get(sample_unit_id) if isinstance(sample_unit_id, str) else None
                        )
                        if sample_unit_id != row.get("measurement_unit_id"):
                            errors.append(
                                "crosswalk measurement_unit_id differs from acquisition sample unit"
                            )
                        if unit is None:
                            errors.append(
                                "crosswalk acquisition sample references an unknown measurement unit"
                            )
                        elif not set(row.get("source_file_ids", [])).issubset(
                            set(sample.get("file_ids", []))
                        ):
                            errors.append(
                                "crosswalk source_file_ids are not attached to the acquisition sample"
                            )
                    for field in (
                        "incidence_group_id",
                        "split",
                        "material_id",
                        "polarization",
                        "incidence_stratum_id",
                    ):
                        if row.get(field) != input_row.get(field):
                            errors.append(f"crosswalk {field} differs from scorer input row")
                    stratum_key = (
                        row.get("material_id"),
                        row.get("polarization"),
                        row.get("incidence_stratum_id"),
                    )
                    evaluator_sample_id = row.get("evaluator_sample_id")
                    if evaluator_sample_id in seen_evaluator_samples:
                        errors.append("row_crosswalk repeats evaluator_sample_id")
                    seen_evaluator_samples.add(evaluator_sample_id)
                    if stratum_key != (
                        output_stratum.get("material_id"),
                        output_stratum.get("polarization"),
                        output_stratum.get("incidence_stratum_id"),
                    ):
                        errors.append(
                            "crosswalk output stratum pointer resolves to a different stratum"
                        )
                    assignment = assignments_by_sample.get(input_row.get("sample_id"))
                    if assignment is None:
                        errors.append(
                            "crosswalk sample is absent from embedded split-manifest assignments"
                        )
                    else:
                        if input_row.get("incidence_angle_deg") != assignment.get(
                            "incidence_angle_deg"
                        ):
                            errors.append("scorer input row angle differs from split assignment")
                        for field in (
                            "incidence_group_id",
                            "material_id",
                            "polarization",
                            "incidence_stratum_id",
                            "measurement_unit_id",
                            "measurement_unit_provenance_id",
                        ):
                            if row.get(field) != assignment.get(field):
                                errors.append(
                                    f"crosswalk {field} differs from embedded split assignment"
                                )
                    manifest_group = groups_by_id.get(row.get("incidence_group_id"))
                    if (
                        manifest_group is None
                        or manifest_group.get("split") != row.get("split")
                        or input_row.get("sample_id") not in manifest_group.get("sample_ids", [])
                    ):
                        errors.append("crosswalk row does not join to its split-manifest group")
                    evaluator_group = evaluator_groups_by_id.get(row.get("incidence_group_id"))
                    if evaluator_group is None:
                        errors.append(
                            "crosswalk incidence_group_id is absent from scorer input groups"
                        )
                    else:
                        for field in (
                            "measurement_unit_id",
                            "measurement_unit_provenance_id",
                            "incidence_stratum_id",
                        ):
                            if row.get(field) != evaluator_group.get(field):
                                errors.append(
                                    f"crosswalk {field} differs from scorer input incidence group"
                                )
                        if input_row.get("incidence_angle_deg") != evaluator_group.get(
                            "incidence_angle_deg"
                        ):
                            errors.append("scorer input row angle differs from incidence group")
                        if input_row.get("sample_id") not in evaluator_group.get("sample_ids", []):
                            errors.append("crosswalk row sample is absent from scorer input group")
                    if row.get("input_row_pointer") in seen_input:
                        errors.append("row_crosswalk repeats an input_row_pointer")
                    seen_input.add(row.get("input_row_pointer"))
                    if (
                        evaluator_sample_id in seen_output_strata
                        and seen_output_strata[evaluator_sample_id] != stratum_key
                    ):
                        errors.append("one evaluator sample is mapped to multiple output strata")
                    seen_output_strata[evaluator_sample_id] = stratum_key
                    for file_id in row.get("source_file_ids", []):
                        if file_id not in files:
                            errors.append(f"crosswalk references unknown source file {file_id!r}")
                except (AttributeError, TypeError, ValueError) as error:
                    errors.append(f"row_crosswalk entry cannot be resolved: {error}")
            expected_pointers = {
                f"/test_predictions/{index}" for index in range(len(evaluator_rows))
            }
            if seen_input != expected_pointers:
                errors.append(
                    "row_crosswalk input pointers must cover every scorer input prediction row exactly once"
                )
            if isinstance(assignment_records, list):
                expected_samples = {
                    item.get("sample_id")
                    for item in assignment_records
                    if isinstance(item, dict) and isinstance(item.get("sample_id"), str)
                }
                if seen_evaluator_samples != expected_samples:
                    errors.append(
                        "row_crosswalk evaluator samples must cover every held-out split assignment exactly once"
                    )
    check(
        "evaluator_contract_and_crosswalk",
        len(errors) == binding_errors,
        "Evaluator schema IDs, deterministic v2 report recomputation, canonical embedded split hash, input-hash echo, one-to-one held-out sample and angular-measurement crosswalk, exact acquisition/evaluator angular-grid joins, bound specimen-to-material mapping, acquisition specimen/unit consistency, sample polarization/angle/frequency/calibration joins, and available row/stratum joins checked.",
    )
    report["status"] = "consistent_proposal_only" if not errors else "inconsistent"
    return report
