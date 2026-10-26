"""File-bound execution and replay for the five-branch G3 numeric bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
from pathlib import Path
from typing import Any

from evaluation.g3_joint_evaluation import (
    evaluate_g3_joint_evidence,
)

ARTIFACT_SCHEMA = "loslab.g3_joint_evaluator_artifact.v2"
LEGACY_ARTIFACT_SCHEMA = "loslab.g3_joint_evaluator_artifact.v1"
MAX_INPUT_BYTES = 256 * 1024 * 1024
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _stat_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _decode_json_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_json_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"$complex"}:
        pair = value["$complex"]
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in pair)
        ):
            raise ValueError("$complex must contain exactly two real numeric components")
        real, imaginary = float(pair[0]), float(pair[1])
        if not math.isfinite(real) or not math.isfinite(imaginary):
            raise ValueError("$complex components must be finite")
        return complex(real, imaginary)
    return {key: _decode_json_value(item) for key, item in value.items()}


def _read_regular_file(path: Path, *, limit: int, label: str) -> bytes:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("safe artifact reading requires O_NOFOLLOW support")
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError(f"{label} must be a single-link regular file")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | os.O_NOFOLLOW
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if _stat_signature(opened) != _stat_signature(before):
            raise ValueError(f"{label} changed while opening")
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise ValueError(f"{label} must be a single-link regular file")
        if opened.st_size > limit:
            raise ValueError(f"{label} exceeds {limit} bytes")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, limit + 1 - size))
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                raise ValueError(f"{label} exceeds {limit} bytes")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        path_after = os.lstat(path)
        if (
            size != opened.st_size
            or _stat_signature(after) != _stat_signature(opened)
            or _stat_signature(path_after) != _stat_signature(opened)
        ):
            raise ValueError(f"{label} changed while being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _parse_json(raw: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON: {error}") from error


def _json_tree(value: Any) -> Any:
    """Convert evaluator values into deterministic JSON-compatible values."""
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - required by the project runtime
        raise RuntimeError("NumPy is required to serialize G3 evidence") from error
    if isinstance(value, np.ndarray):
        return _json_tree(value.tolist())
    if isinstance(value, np.generic):
        return _json_tree(value.item())
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("G3 artifact object keys must be strings")
        return {key: _json_tree(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_json_tree(item) for item in value]
    if isinstance(value, complex):
        return {"$complex": [float(value.real), float(value.imag)]}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("G3 artifact cannot contain non-finite numbers")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported G3 artifact value: {type(value).__name__}")


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _json_tree(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _evaluate_g3_joint_evidence_bytes(
    raw: bytes, *, coefficient_relative_tolerance: float | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed = _parse_json(raw, label="G3 joint input")
    payload = _decode_json_value(parsed)
    if not isinstance(payload, dict):
        raise ValueError("G3 joint input root must be an object")
    if _canonical_json_bytes(payload) != raw:
        raise ValueError("G3 joint input must use canonical JSON bytes")
    bundle = _json_tree(
        evaluate_g3_joint_evidence(
            payload, coefficient_relative_tolerance=coefficient_relative_tolerance
        )
    )
    report = {
        "schema": ARTIFACT_SCHEMA,
        "status": "NUMERICAL_EVALUATION_ONLY_NOT_QUALIFIED",
        "qualification_status": "NOT_QUALIFIED",
        "input_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "bundle_sha256": hashlib.sha256(_canonical_json_bytes(bundle)).hexdigest(),
        "bundle": bundle,
    }
    return payload, report


def evaluate_g3_joint_evidence_file(
    input_path: str | Path, *, coefficient_relative_tolerance: float | None = None
) -> dict[str, Any]:
    """Recompute all five scorers and bind the report to exact source bytes."""
    raw = _read_regular_file(Path(input_path), limit=MAX_INPUT_BYTES, label="G3 joint input")
    _, report = _evaluate_g3_joint_evidence_bytes(
        raw, coefficient_relative_tolerance=coefficient_relative_tolerance
    )
    return report


def run_g3_frozen_qualification_evaluation(
    input_path: str | Path,
    qualification_contract_path: str | Path,
    artifact_path: str | Path,
    run_manifest_path: str | Path,
    *,
    source_root: str | Path | None = None,
) -> dict[str, Any]:
    """Preflight a frozen contract, evaluate exact input bytes, and write its receipt.

    This is still a numerical, not scientifically qualified, run. The contract
    and current source bundle are checked before any evaluator branch executes.
    """
    from evaluation import g3_joint_evidence_run_manifest as run_manifest

    input_file = Path(input_path)
    contract_file = Path(qualification_contract_path)
    artifact_file = Path(artifact_path)
    receipt_file = Path(run_manifest_path)
    root = input_file.resolve(strict=True).parent
    source_dir = (
        Path(source_root).resolve(strict=True)
        if source_root is not None
        else Path(__file__).resolve().parents[1]
    )
    if not root.is_dir() or not source_dir.is_dir():
        raise ValueError("input and source roots must be directories")
    if receipt_file.resolve().parent != root or artifact_file.resolve().parent != root:
        raise ValueError("artifact and run receipt must be written beside the evaluator input")
    resolved_inputs = {input_file.resolve(), contract_file.resolve()}
    resolved_outputs = {artifact_file.resolve(), receipt_file.resolve()}
    if len(resolved_outputs) != 2 or resolved_inputs & resolved_outputs:
        raise ValueError("input, contract, artifact, and receipt paths must be distinct")
    if artifact_file.exists() or receipt_file.exists():
        raise FileExistsError("artifact or run receipt already exists")

    input_raw = _read_regular_file(input_file, limit=MAX_INPUT_BYTES, label="G3 joint input")
    contract_raw = _read_regular_file(
        contract_file, limit=MAX_ARTIFACT_BYTES, label="G3 qualification contract"
    )
    contract = _parse_json(contract_raw, label="G3 qualification contract")
    if not isinstance(contract, dict) or _canonical_json_bytes(contract) != contract_raw:
        raise ValueError("qualification contract must be canonical compact JSON")
    import jsonschema

    contract_schema_path = {
        "loslab.g3_qualification_run_contract.v3": (
            run_manifest.QUALIFICATION_CONTRACT_SCHEMA_RELATIVE_PATH
        ),
        "loslab.g3_qualification_run_contract.v4": (
            run_manifest.QUALIFICATION_CONTRACT_SCHEMA_V4_RELATIVE_PATH
        ),
    }.get(contract.get("schema"))
    if contract_schema_path is None:
        raise ValueError("new contract-bound G3 evaluation requires contract v3 or v4")
    schema_raw = run_manifest._read_rooted_source_file(
        source_dir, contract_schema_path, limit=run_manifest.MAX_SCHEMA_BYTES
    )
    schema = _parse_json(schema_raw, label="G3 qualification contract schema")
    schema_errors = list(jsonschema.Draft202012Validator(schema).iter_errors(contract))
    if schema_errors:
        raise ValueError(
            "G3 qualification contract violates schema: "
            + "; ".join(error.message for error in schema_errors[:10])
        )
    contract_result = run_manifest.verify_g3_qualification_run_contract(contract)
    source_identity = run_manifest.g3_evaluator_source_bundle_identity(source_root=source_dir)
    if contract["source_identity"]["source_bundle_sha256"] != source_identity[
        "evaluator_source_bundle_sha256"
    ]:
        raise ValueError("qualification contract source bundle differs from current evaluator sources")
    input_payload = _parse_json(input_raw, label="G3 joint evaluator input")
    input_schema_path = run_manifest._validate_joint_evaluator_input_schema(
        input_payload, source_root=source_dir
    )
    expected_input_schema = {
        "loslab.g3_joint_evaluator_inputs.v4": run_manifest.INPUT_SCHEMA_V4_RELATIVE_PATH,
        "loslab.g3_joint_evaluator_inputs.v5": run_manifest.INPUT_SCHEMA_V5_RELATIVE_PATH,
        "loslab.g3_joint_evaluator_inputs.v6": run_manifest.INPUT_SCHEMA_V6_RELATIVE_PATH,
    }.get(input_payload.get("schema"))
    if input_schema_path != expected_input_schema:
        raise ValueError(
            "new contract-bound G3 evaluation requires joint evaluator input v4, v5, or v6"
        )
    run_manifest._verify_qualification_contract_bindings(
        contract, input_raw, require_powered_design=True
    )

    runtime_identity = run_manifest._runtime_identity()
    coefficient_tolerance = (
        contract["path_matching"]["coefficient_relative_tolerance"]
        if contract["schema"] == "loslab.g3_qualification_run_contract.v4"
        else None
    )
    _, report = _evaluate_g3_joint_evidence_bytes(
        input_raw, coefficient_relative_tolerance=coefficient_tolerance
    )
    _write_new_artifact(artifact_file, report)
    manifest = run_manifest.build_g3_joint_evidence_run_manifest(
        contract["run_id"],
        input_file,
        artifact_file,
        package_root=root,
        source_root=source_dir,
        qualification_contract=contract,
        runtime_identity=runtime_identity,
    )
    artifact_digest = hashlib.sha256(_canonical_json_bytes(report)).hexdigest()
    if (
        manifest["input"]["sha256"] != report["input_bytes_sha256"]
        or manifest["artifact"]["sha256"] != artifact_digest
    ):
        raise ValueError("input or artifact changed while creating the frozen run receipt")
    _write_new_artifact(receipt_file, manifest)
    return {
        "schema": "loslab.g3_preregistered_numerical_run.v1",
        "status": "CONTRACT_BOUND_NUMERICAL_EVALUATION_NOT_QUALIFIED",
        "qualification_status": "NOT_QUALIFIED",
        "run_id": contract["run_id"],
        "qualification_contract_sha256": contract_result["contract_sha256"],
        "evaluator_source_bundle_sha256": source_identity[
            "evaluator_source_bundle_sha256"
        ],
        "input_sha256": report["input_bytes_sha256"],
        "artifact_sha256": artifact_digest,
        "run_manifest_sha256": hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest(),
    }


def verify_g3_joint_evidence_file(
    input_path: str | Path,
    artifact_path: str | Path,
    *,
    coefficient_relative_tolerance: float | None = None,
) -> dict[str, Any]:
    """Verify exact input-byte binding and deterministic five-branch replay."""
    artifact_raw = _read_regular_file(
        Path(artifact_path), limit=MAX_ARTIFACT_BYTES, label="G3 joint artifact"
    )
    input_raw = _read_regular_file(
        Path(input_path), limit=MAX_INPUT_BYTES, label="G3 joint input"
    )
    return verify_g3_joint_evidence_bytes(
        input_raw,
        artifact_raw,
        coefficient_relative_tolerance=coefficient_relative_tolerance,
    )


def verify_g3_joint_evidence_bytes(
    input_raw: bytes,
    artifact_raw: bytes,
    *,
    coefficient_relative_tolerance: float | None = None,
) -> dict[str, Any]:
    """Verify exact input-byte binding and replay for already authenticated file bytes."""
    if len(input_raw) > MAX_INPUT_BYTES:
        raise ValueError(f"G3 joint input exceeds {MAX_INPUT_BYTES} bytes")
    if len(artifact_raw) > MAX_ARTIFACT_BYTES:
        raise ValueError(f"G3 joint artifact exceeds {MAX_ARTIFACT_BYTES} bytes")
    artifact = _parse_json(artifact_raw, label="G3 joint artifact")
    if not isinstance(artifact, dict) or set(artifact) != {
        "schema",
        "status",
        "qualification_status",
        "input_bytes_sha256",
        "bundle_sha256",
        "bundle",
    }:
        raise ValueError("G3 joint artifact fields do not match the v2 contract")
    if _canonical_json_bytes(artifact) != artifact_raw:
        raise ValueError("G3 joint artifact must use canonical JSON bytes")
    _, expected = _evaluate_g3_joint_evidence_bytes(
        input_raw, coefficient_relative_tolerance=coefficient_relative_tolerance
    )
    if artifact.get("schema") == LEGACY_ARTIFACT_SCHEMA:
        legacy_bundle = dict(expected["bundle"])
        legacy_bundle.pop("numerical_criterion_rollup")
        legacy_bundle["schema"] = "loslab.g3_joint_evaluator_bundle.v1"
        expected = {
            "schema": LEGACY_ARTIFACT_SCHEMA,
            "status": expected["status"],
            "qualification_status": expected["qualification_status"],
            "input_bytes_sha256": expected["input_bytes_sha256"],
            "bundle_sha256": hashlib.sha256(_canonical_json_bytes(legacy_bundle)).hexdigest(),
            "bundle": legacy_bundle,
        }
    if _canonical_json_bytes(artifact) != _canonical_json_bytes(expected):
        raise ValueError("G3 joint artifact differs from exact-byte-bound recomputation")
    return {
        "schema": "loslab.g3_joint_evaluator_artifact_verification.v1",
        "status": "PASS_INPUT_SCOPED_RECOMPUTATION",
        "qualification_status": "NOT_QUALIFIED",
        "input_bytes_sha256": expected["input_bytes_sha256"],
        "artifact_sha256": hashlib.sha256(artifact_raw).hexdigest(),
        "bundle_sha256": expected["bundle_sha256"],
        "branch_count": len(expected["bundle"]["branch_inputs_sha256"]),
    }


def _write_new_artifact(path: Path, value: dict[str, Any]) -> None:
    encoded = _canonical_json_bytes(value)
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("evaluate", "verify", "evaluate-preregistered"))
    parser.add_argument("input", type=Path, help="canonical G3 five-branch input JSON")
    parser.add_argument("artifact", type=Path, help="output or saved evidence artifact JSON")
    parser.add_argument("--qualification-contract", type=Path)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "evaluate-preregistered":
            if args.qualification_contract is None or args.run_manifest is None:
                parser.error(
                    "evaluate-preregistered requires --qualification-contract and --run-manifest"
                )
            report = run_g3_frozen_qualification_evaluation(
                args.input,
                args.qualification_contract,
                args.artifact,
                args.run_manifest,
                source_root=args.source_root,
            )
            print(json.dumps(report, sort_keys=True))
        elif args.qualification_contract is not None or args.run_manifest is not None:
            parser.error("contract-bound options require evaluate-preregistered")
        elif args.action == "evaluate":
            report = evaluate_g3_joint_evidence_file(args.input)
            _write_new_artifact(args.artifact, report)
            print(json.dumps({key: report[key] for key in ("schema", "status", "bundle_sha256")}))
        else:
            report = verify_g3_joint_evidence_file(args.input, args.artifact)
            print(json.dumps(report, sort_keys=True))
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"G3 joint evidence artifact error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
