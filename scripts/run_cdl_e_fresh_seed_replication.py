"""Run the preregistered, bounded CDL-E covariance replication."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import jsonschema
from evaluation.cdl_covariance.four_element import (
    audit_cdl_spatial_covariance_claims,
    cdl_spatial_covariance_claims,
    cdl_spatial_covariance_reference,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/research/CDL_E_FRESH_SEED_REPLICATION_PROTOCOL_2026-11-12.md"
SEED_AUDIT = ROOT / "docs/research/CDL_E_FRESH_SEED_AUDIT_2026-11-12.json"
SCHEMA = ROOT / "datasets/schema/cdl-e-covariance-replication.v1.schema.json"
NATIVE_TEST = ROOT / "tests/test_integrations/test_cdl_spatial_covariance_4element_native.py"
EVALUATOR = Path(__file__).resolve().parents[1] / "evaluation/cdl_covariance/four_element.py"
RESERVED_SEED = 2_823_000
DRAW_COUNT = 8_192
PROFILE_TABLE_SHA256 = "04328f33d8d79b24aa3d0303fecb58b465ed7e13874930b05e76be0d97743480"
SEED_AUDIT_SHA256 = "4eb74446ac3e1f50b1fc9644c43f8974bc899ab30b1a3897e570a8dacf38ae86"
PROTOCOL_SHA256 = "784750be5b1b7ddc8b95cdd3cd006f14be9b04bdabd772b0067f63f9f40ac73e"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sionna_version() -> str:
    for distribution in ("sionna", "sionna-no-rt"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unavailable"


def _publish_no_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_preregistration() -> tuple[dict[str, Any], dict[str, Any]]:
    audit_bytes = SEED_AUDIT.read_bytes()
    if _sha256_bytes(audit_bytes) != SEED_AUDIT_SHA256:
        raise ValueError("seed audit bytes differ from the preregistered digest")
    protocol_bytes = PROTOCOL.read_bytes()
    if _sha256_bytes(protocol_bytes) != PROTOCOL_SHA256:
        raise ValueError("protocol bytes differ from the preregistered digest")
    audit = json.loads(audit_bytes)
    if (
        audit.get("schema") != "loslab.seed_roster_collision_audit.v2"
        or audit.get("status") != "CLEAR_WITHIN_SCANNED_JSON_CORPUS"
        or audit.get("reserved_seeds") != [RESERVED_SEED]
        or audit.get("collisions")
        or audit.get("malformed_seed_fields")
        or audit.get("incomplete_files")
        or audit.get("global_exclusivity_established") is not False
    ):
        raise ValueError("the frozen seed audit does not clear the exact replication seed")
    return audit, {
        "protocol_sha256": _sha256_bytes(protocol_bytes),
        "seed_audit_sha256": _sha256_bytes(audit_bytes),
    }


def _generate_claims() -> tuple[list[dict[str, Any]], dict[str, str]]:
    if _sionna_version() != "2.2.0":
        raise RuntimeError("the frozen replication requires Sionna 2.2.0")
    import torch
    from integrations.sionna.cdl import CDLArrayConfig, generate_cdl_frequency_response
    from sionna.phy.channel.tr38901 import models as profile_resources

    table_path = Path(profile_resources.parameter_file("CDL-E.json", "19.2"))
    table_sha256 = _sha256(table_path)
    if table_sha256 != PROFILE_TABLE_SHA256:
        raise ValueError("CDL-E parameter-table bytes differ from the frozen digest")
    reference, reference_sha256 = cdl_spatial_covariance_reference("E")
    channel = generate_cdl_frequency_response(
        model="E",
        seed=RESERVED_SEED,
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=DRAW_COUNT,
        delay_spread_s=100e-9,
        ut_array=CDLArrayConfig(
            num_cols_per_panel=4,
            num_cols=1,
            element_horizontal_spacing=0.5,
        ),
        ut_orientation_rad=(0.0, 0.0, 0.0),
        spec_version="19.2",
        device="cpu",
    )
    path_values = channel.h_path[:, 0, :, 0, 0, :, 0]
    if tuple(path_values.shape) != (DRAW_COUNT, 4, 14):
        raise ValueError(f"unexpected path coefficient shape: {tuple(path_values.shape)}")
    samples = path_values.sum(dim=-1).to(torch.complex128).detach().cpu().numpy()
    claims = cdl_spatial_covariance_claims(samples, reference)
    if len(claims) != 16:
        raise AssertionError("the frozen covariance claim family must contain 16 coordinates")
    return claims, {
        "profile_table_sha256": table_sha256,
        "reference_sha256": reference_sha256,
    }


def _report(
    claims: list[dict[str, Any]],
    *,
    audit: dict[str, Any],
    preregistration_digests: dict[str, str],
    source_digests: dict[str, str],
    command: list[str],
    revision: str,
    error: str | None = None,
    profile_table_verified: bool = True,
) -> dict[str, Any]:
    complete = len(claims) == 16
    passed = complete and all(claim["passed"] for claim in claims) and error is None
    status = (
        "PASS_WITHIN_BOUNDED_PROTOCOL_SCOPE"
        if passed
        else "FAIL"
        if complete and error is None
        else "INCOMPLETE"
    )
    sionna_version = _sionna_version()
    try:
        torch_version = importlib.metadata.version("torch")
        torch_cuda_build = str(__import__("torch").version.cuda or "cpu-only")
    except (importlib.metadata.PackageNotFoundError, ImportError):
        torch_version = "unavailable"
        torch_cuda_build = "unavailable"
    return {
        "schema": "loslab.g4_cdl_e_covariance_replication.v1",
        "protocol": PROTOCOL.name,
        "date": dt.datetime.now(dt.timezone.utc).date().isoformat(),
        "status": status,
        "error": error,
        "qualification_claim": "none",
        "criteria_changed": False,
        "design": {
            "profile": "E",
            "seed": RESERVED_SEED,
            "draw_count": DRAW_COUNT,
            "carrier_hz": 3_500_000_000.0,
            "subcarrier_spacing_hz": 30_000.0,
            "subcarrier_count": 8,
            "delay_spread_s": 1e-7,
            "rx_element_count": 4,
            "rx_element_spacing_wavelengths": 0.5,
            "rx_orientation_rad": [0.0, 0.0, 0.0],
            "familywise_alpha": 0.01,
            "claim_count": 16,
            "maximum_half_width": 0.05,
        },
        "seed_audit": {
            "path": SEED_AUDIT.relative_to(ROOT).as_posix(),
            "sha256": preregistration_digests["seed_audit_sha256"],
            "status": audit["status"],
            "scanned_file_count": audit["scanned_file_count"],
            "global_exclusivity_established": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "sionna": sionna_version,
            "torch": torch_version,
            "torch_cuda_build": torch_cuda_build,
            "device": "cpu",
        },
        "provenance": {
            "git_revision": revision,
            "clean_worktree_at_run_start": True,
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "evaluator_sha256": source_digests["evaluator_sha256"],
            "native_test_sha256": source_digests["native_test_sha256"],
            "protocol_sha256": preregistration_digests["protocol_sha256"],
            "schema_sha256": _sha256(SCHEMA),
            "reference_sha256": source_digests["reference_sha256"],
            "profile_table_sha256": source_digests["profile_table_sha256"],
            "profile_table_verified": profile_table_verified,
        },
        "command": command,
        "claims": claims,
        "summary": {
            "claims_passed": sum(bool(claim["passed"]) for claim in claims),
            "claims_failed": sum(not claim["passed"] for claim in claims),
            "maximum_absolute_error": max(
                (claim["absolute_error"] for claim in claims), default=None
            ),
            "maximum_half_width": max((claim["half_width"] for claim in claims), default=None),
        },
        "scope": (
            "One fresh-seed CDL-E four-element ULA replication only. This cannot replace the "
            "retained seed-6714 failure, establish global seed exclusivity, qualify G4, or "
            "authorize production RT or advanced training."
        ),
    }


def audit_replication_report(report: dict[str, Any]) -> dict[str, Any]:
    """Check complete or incomplete report status against retained claims."""
    claims = report.get("claims")
    if not isinstance(claims, list) or len(claims) > 16:
        raise ValueError("replication report has an invalid claim roster")
    error = report.get("error")
    status = report.get("status")
    if len(claims) == 16:
        if error is not None:
            raise ValueError("a complete claim roster cannot carry an execution error")
        audited = audit_cdl_spatial_covariance_claims(claims)
        if status != audited["status"]:
            raise ValueError("report status differs from independent claim recomputation")
        expected_summary = {
            key: audited[key]
            for key in (
                "claims_passed",
                "claims_failed",
                "maximum_absolute_error",
                "maximum_half_width",
            )
        }
        if report.get("summary") != expected_summary:
            raise ValueError("report summary differs from independent claim recomputation")
        return audited
    if status != "INCOMPLETE" or not isinstance(error, str) or not error:
        raise ValueError("an incomplete claim roster must retain an explicit execution error")
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("incomplete report summary must be an object")
    if summary.get("claims_passed") != sum(claim.get("passed") is True for claim in claims):
        raise ValueError("incomplete report pass count differs from retained claims")
    if summary.get("claims_failed") != sum(claim.get("passed") is False for claim in claims):
        raise ValueError("incomplete report failure count differs from retained claims")
    return {"status": "INCOMPLETE", "claim_count": len(claims)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new immutable result JSON path")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    sidecar = Path(f"{output}.sha256")
    if output.exists() or sidecar.exists():
        parser.error("refusing to overwrite a result or checksum")
    worktree_status = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"], text=True
    )
    if worktree_status:
        parser.error("refusing to run from a dirty worktree; commit the frozen protocol first")
    audit, preregistration_digests = _verify_preregistration()
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    command = [sys.executable, "-m", "scripts.run_cdl_e_fresh_seed_replication", *sys.argv[1:]]
    try:
        claims, source_digests = _generate_claims()
    except Exception as exc:
        result = _report(
            [],
            audit=audit,
            preregistration_digests=preregistration_digests,
            source_digests={
                "profile_table_sha256": PROFILE_TABLE_SHA256,
                "reference_sha256": "24ee2f8f73f26e27522a296a4a0860bdaa7f284e9138855666e09636bc97b090",
                "evaluator_sha256": _sha256(EVALUATOR),
                "native_test_sha256": _sha256(NATIVE_TEST),
            },
            command=command,
            revision=revision,
            error=f"{type(exc).__name__}: {exc}",
            profile_table_verified=False,
        )
    else:
        source_digests.update(
            {
                "evaluator_sha256": _sha256(EVALUATOR),
                "native_test_sha256": _sha256(NATIVE_TEST),
            }
        )
        summary = audit_cdl_spatial_covariance_claims(claims)
        result = _report(
            claims,
            audit=audit,
            preregistration_digests=preregistration_digests,
            source_digests=source_digests,
            command=command,
            revision=revision,
        )
        if result["summary"] != summary or result["status"] != summary["status"]:
            raise ValueError(
                "the retained report summary differs from independent claim recomputation"
            )
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(result)
    audit_replication_report(result)
    payload = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    checksum = f"{_sha256_bytes(payload)}  {output.name}\n".encode()
    _publish_no_replace(output, payload)
    try:
        _publish_no_replace(sidecar, checksum)
    except FileExistsError:
        parser.error("result was published but the checksum path already exists")
    print(f"{result['status']}: {output}")
    return (
        0
        if result["status"] == "PASS_WITHIN_BOUNDED_PROTOCOL_SCOPE"
        else 1
        if result["status"] == "FAIL"
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
