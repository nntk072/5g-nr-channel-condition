"""Bounded semantic consistency checks for proposal-only G3.03 packages."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any, cast

SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "datasets/schema/g3.03-acquisition-package.v0.2.schema.json"
)
EVALUATOR_INPUT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "datasets/schema/g3-material-calibration-predictions.v1.schema.json"
)
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
]


def _canonical_sha256(value: Any) -> str:
    encoded = _canonical_json(value)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


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
                        f"bound evaluator JSON artifact exceeds {MAX_JSON_ARTIFACT_BYTES} bytes"
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

        schema = _json_bytes(
            _read_bounded_regular_file(
                SCHEMA_PATH, limit=MAX_SCHEMA_BYTES, description="package schema"
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
            check("json_schema", True, "Package conforms to the v0.2 JSON Schema.")
            schema_passed = True
    except Exception as error:
        errors.append(f"cannot validate package schema: {error}")
        check("json_schema", False, str(error))
        schema_passed = False

    if (
        package.get("schema_version") != "g3.03-acquisition-package.v0.2"
        or package.get("proposal_status") != "proposal_only"
    ):
        errors.append("package must remain g3.03-acquisition-package.v0.2 and proposal_only")
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
        bound_ids = {
            binding_for_inventory.get(name, {}).get("file_id")
            for name in ("scorer_input", "split_manifest", "evaluator_output")
            if isinstance(binding_for_inventory, dict)
            and isinstance(binding_for_inventory.get(name), dict)
        }
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
    attestation_ids = (
        {
            record.get("attestation_id")
            for record in legacy_attest.values()
            if isinstance(record, dict) and isinstance(record.get("attestation_id"), str)
        }
        if isinstance(legacy_attest, dict)
        else set()
    )
    external_attest = _unique_index(
        package.get("external_attestations"),
        key="attestation_id",
        label="external_attestations",
        errors=errors,
    )
    attestation_ids.update(external_attest)
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
        if record.get("independence_attestation_id") not in attestation_ids:
            errors.append(f"unit {record['unit_id']} references unknown attestation")
    for record in calibrations.values():
        for field in ("standard_file_ids", "run_file_ids", "processing_file_ids"):
            require_file_refs(record, field, f"calibration {record['calibration_id']}")
        if record.get("truth_attestation_id") not in attestation_ids:
            errors.append(f"calibration {record['calibration_id']} references unknown attestation")
    for record in samples.values():
        for field, index in (
            ("specimen_id", specimens),
            ("unit_id", units),
            ("calibration_id", calibrations),
        ):
            require_ids(record, field, index, f"sample {record['sample_id']}")
        require_file_refs(record, "file_ids", f"sample {record['sample_id']}")
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
    for record in csi.values():
        require_ids(record, "sample_id", samples, f"csi {record['csi_id']}")
        require_file_refs(record, "file_ids", f"csi {record['csi_id']}")
        require_file_refs(
            record.get("h_floor", {}), "noise_input_file_ids", f"csi {record['csi_id']}.h_floor"
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
        if record.get("attestation_id") not in attestation_ids:
            errors.append(f"rights {record.get('rights_id')} references unknown attestation")
    for record in campaigns.values():
        if record.get("study_id") != package.get("study_id"):
            errors.append(
                f"campaign {record['campaign_id']} study_id differs from package study_id"
            )
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
        if record.get("attestation_id") not in attestation_ids:
            errors.append(f"per-file rights for {file_id} references unknown attestation")
        require_file_refs(record, "evidence_file_ids", f"per-file rights {file_id}")
    check(
        "reference_joins",
        not errors,
        "Reference IDs, record joins, and per-file rights coverage checked.",
    )

    semantic_errors_before_binding = len(errors)
    parsed_artifacts: dict[str, Any] = {}
    bound_file_ids: dict[str, str] = {}
    expected_roles = {
        "scorer_input": "scorer_input",
        "split_manifest": "split_manifest",
        "evaluator_output": "evaluator_output",
    }
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
        "Scorer input, split manifest, and evaluator output references resolve to exact inventoried bytes and expected roles.",
    )

    input_doc = parsed_artifacts.get("scorer_input")
    split_doc = parsed_artifacts.get("split_manifest")
    output_doc = parsed_artifacts.get("evaluator_output")
    binding_errors = len(errors)
    if isinstance(input_doc, dict):
        if input_doc.get("schema") != "loslab.g3_material_calibration_predictions.v1":
            errors.append("scorer input schema does not match evaluator input contract")
        try:
            import jsonschema

            input_schema = _json_bytes(
                _read_bounded_regular_file(
                    EVALUATOR_INPUT_SCHEMA_PATH,
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
        if output_doc.get("schema") != "loslab.g3_material_calibration_evaluation.v1":
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
                    sample = (
                        samples.get(acquisition_sample_id)
                        if isinstance(acquisition_sample_id, str)
                        else None
                    )
                    if sample is None:
                        errors.append("crosswalk acquisition_sample_id is not in package samples")
                    else:
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
        "Evaluator schema IDs, canonical embedded split hash, input-hash echo, and available row/stratum joins checked.",
    )
    report["status"] = "consistent_proposal_only" if not errors else "inconsistent"
    return report
