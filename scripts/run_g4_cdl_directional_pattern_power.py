"""Run and retain the frozen G4 CDL directional-pattern power protocol."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "tests/test_integrations/test_g4_cdl_antenna_pattern_power_native.py"
PROTOCOL = ROOT / "docs/research/G4_07_CDL_DIRECTIONAL_PATTERN_VECTOR_FIELD_PROTOCOL_2026-11-01.md"
SEED_AUDIT = ROOT / "docs/research/G4_CDL_ANTENNA_PATTERN_VECTOR_FIELD_SEED_AUDIT_2026-11-01.json"
SCHEMA_PATH = ROOT / "datasets/schema/g4-cdl-directional-pattern-power-result.v1.schema.json"
REFERENCE_PATHS = (
    ROOT / "docs/research/G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json",
    ROOT / "docs/research/G4_CDL_PROFILE_SCALAR_REFERENCE_2026-10-26.json",
)
TABLE_HASHES = {
    "A": "b96f8a179646ee468717c615e8061103d1e66eb1ff1ea78301de243740254cb7",
    "B": "7c87c092cb2bb44891d301ee3c1991cdda1c6b64094a03f18746ec5fa14ee2a2",
    "C": "f2044ecc3b72a1967221805ba70b1f0dab9634bb831478c1337b737ced6e4a28",
    "D": "a8960f176347de457c0cb90fbc8cf137ac98bb1c5295b096f004d8d5c6e40176",
    "E": "04328f33d8d79b24aa3d0303fecb58b465ed7e13874930b05e76be0d97743480",
}
CLUSTER_COUNTS = {"A": 23, "B": 23, "C": 24, "D": 13, "E": 14}
_PATTERNS = ("38.901", "38.901-handheld")
_ORIENTATIONS = ((0.0, 0.0, 0.0), (0.2, -0.1, 0.3), (-0.7, 0.35, 1.1))
EXPECTED_CASES = (
    {
        ("one_end_zero", profile, 2_700_000 + 6 * profile_index + 3 * pattern_index)
        for profile_index, profile in enumerate("ABCDE")
        for pattern_index in range(2)
    }
    | {
        (
            "one_end_tilted",
            profile,
            2_700_000 + 6 * profile_index + 3 * pattern_index + orientation_index,
        )
        for profile_index, profile in enumerate("ABCDE")
        for pattern_index in range(2)
        for orientation_index in (1, 2)
    }
    | {
        ("both_ends_zero", profile, 2_701_000 + 8 * profile_index + 2 * pair_index)
        for profile_index, profile in enumerate("ABCDE")
        for pair_index in range(4)
    }
    | {
        ("both_ends_tilted", profile, 2_701_000 + 8 * profile_index + 2 * pair_index + 1)
        for profile_index, profile in enumerate("ABCDE")
        for pair_index in range(4)
    }
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sionna_version() -> str:
    for distribution in ("sionna", "sionna-no-rt"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    raise RuntimeError("neither sionna nor sionna-no-rt distribution is installed")


def expected_case_configuration(
    cohort: str, profile: str, seed: int
) -> tuple[list[str], list[list[float]], int]:
    """Return the frozen patterns, orientations, and family size for one seed."""
    profile_index = "ABCDE".index(profile)
    zero = [0.0, 0.0, 0.0]
    if cohort.startswith("one_end_"):
        pattern_index, orientation_index = divmod(seed - (2_700_000 + 6 * profile_index), 3)
        if pattern_index not in (0, 1) or orientation_index not in (0, 1, 2):
            raise ValueError(f"invalid one-end seed for {cohort}/{profile}: {seed}")
        expected_cohort = "one_end_zero" if orientation_index == 0 else "one_end_tilted"
        if cohort != expected_cohort:
            raise ValueError(f"cohort/seed mismatch for {cohort}/{profile}/{seed}")
        return (
            [_PATTERNS[pattern_index], "omni"],
            [list(_ORIENTATIONS[orientation_index]), zero],
            194 if cohort == "one_end_zero" else 388,
        )
    if cohort.startswith("both_ends_"):
        pair_index, orientation_index = divmod(seed - (2_701_000 + 8 * profile_index), 2)
        if pair_index not in range(4) or orientation_index not in (0, 1):
            raise ValueError(f"invalid dual-end seed for {cohort}/{profile}: {seed}")
        expected_cohort = "both_ends_zero" if orientation_index == 0 else "both_ends_tilted"
        if cohort != expected_cohort:
            raise ValueError(f"cohort/seed mismatch for {cohort}/{profile}/{seed}")
        patterns = [_PATTERNS[pair_index // 2], _PATTERNS[pair_index % 2]]
        orientations = (
            [zero, zero]
            if orientation_index == 0
            else [list(_ORIENTATIONS[1]), list(_ORIENTATIONS[2])]
        )
        return patterns, orientations, 776
    raise ValueError(f"unknown cohort: {cohort}")


def audit_result(result: dict[str, Any]) -> dict[str, int | float | str]:
    """Validate report structure and independently recompute all frozen claims."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(
        result
    )
    cases = result["cases"]
    seen: set[tuple[str, str, int]] = set()
    total_claims = 0
    failed_claims = 0
    maximum_width_fraction = 0.0
    for case in cases:
        key = (case["cohort"], case["profile"], case["seed"])
        if key in seen:
            raise ValueError(f"duplicate case identity: {key}")
        seen.add(key)
        if case["parameter_table_sha256"] != TABLE_HASHES[case["profile"]]:
            raise ValueError(f"profile table digest mismatch in {key}")
        patterns, orientations, family_claim_count = expected_case_configuration(*key)
        if case["patterns"] != patterns or case["orientations_rad"] != orientations:
            raise ValueError(f"pattern/orientation binding mismatch in {key}")
        if case["family_claim_count"] != family_claim_count:
            raise ValueError(f"familywise claim count mismatch in {key}")
        expected_cluster_count = CLUSTER_COUNTS[case["profile"]]
        if len(case["claims"]) != expected_cluster_count:
            raise ValueError(f"unexpected cluster count in {key}")
        if case["sample_count"] != 16_384 or case["familywise_alpha"] != 0.01:
            raise ValueError(f"design mismatch in {key}")
        cluster_widths = []
        references = []
        for cluster_index, claim in enumerate(case["claims"]):
            if claim["cluster_index"] != cluster_index:
                raise ValueError(f"cluster roster mismatch in {key}")
            lower, upper = claim["interval"]
            expected_lower = claim["estimate"] - claim["half_width"]
            expected_upper = claim["estimate"] + claim["half_width"]
            if lower != expected_lower or upper != expected_upper:
                raise ValueError(f"interval arithmetic mismatch in {key}/{cluster_index}")
            passed = lower <= claim["reference"] <= upper
            if claim["passed"] is not passed:
                raise ValueError(f"claim status mismatch in {key}/{cluster_index}")
            failed_claims += not passed
            cluster_widths.append(claim["half_width"])
            references.append(claim["reference"])
        if case["maximum_half_width"] != max(cluster_widths):
            raise ValueError(f"reported maximum interval width mismatch in {key}")
        if not math.isclose(
            case["total_expected_power"], math.fsum(references), rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ValueError(f"reported total expected power mismatch in {key}")
        if not math.isclose(
            case["maximum_allowed_half_width"],
            0.05 * case["total_expected_power"],
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError(f"declared interval-width limit mismatch in {key}")
        width_fraction = case["maximum_half_width"] / case["total_expected_power"]
        maximum_width_fraction = max(maximum_width_fraction, width_fraction)
        if case["maximum_half_width"] > case["maximum_allowed_half_width"]:
            raise ValueError(f"simultaneous interval-width criterion failed in {key}")
        if not case["passed"]:
            raise ValueError(f"case recorded as failed: {key}")
        if case["test_outcome"] != "passed":
            raise ValueError(f"native case did not pass: {key}")
        total_claims += len(case["claims"])
    if seen != EXPECTED_CASES:
        missing = sorted(EXPECTED_CASES - seen)
        extra = sorted(seen - EXPECTED_CASES)
        raise ValueError(f"case roster mismatch: missing={missing}, extra={extra}")
    if failed_claims:
        raise ValueError(f"{failed_claims} per-cluster claims failed")
    if total_claims != 1_358:
        raise ValueError(f"expected 1,358 claims, observed {total_claims}")
    outcome = result["outcome"]
    observed_passed = sum(case["test_outcome"] == "passed" and case["passed"] for case in cases)
    if (
        outcome["pytest_exit_code"] != 0
        or outcome["cases_observed"] != len(cases)
        or outcome["cases_passed"] != observed_passed
        or outcome["claims_passed"] != total_claims
        or outcome["claims_failed"] != 0
    ):
        raise ValueError("top-level result summary does not match the retained case roster")
    if result["status"] != "PASS_WITHIN_BOUNDED_PROTOCOL_SCOPE":
        raise ValueError("a nonpassing report cannot be audited as a bounded pass")
    return {
        "status": result["status"],
        "case_count": len(cases),
        "claim_count": total_claims,
        "failed_claim_count": failed_claims,
        "maximum_half_width_fraction": maximum_width_fraction,
    }


class _ResultCollector:
    def __init__(self) -> None:
        self.cases: list[dict[str, Any]] = []

    def pytest_runtest_logreport(self, report: Any) -> None:
        if report.when != "call":
            return
        for key, value in report.user_properties:
            if key == "g4_cdl_directional_pattern_case":
                case = json.loads(value)
                case["test_nodeid"] = report.nodeid
                case["test_outcome"] = report.outcome
                self.cases.append(case)


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


def _build_report(
    cases: list[dict[str, Any]], command: str, pytest_exit_code: int
) -> dict[str, Any]:
    cases.sort(key=lambda row: (row["cohort"], row["profile"], row["seed"]))
    passed = pytest_exit_code == 0 and len(cases) == len(EXPECTED_CASES)
    return {
        "schema": "loslab.g4_cdl_directional_pattern_power_result.v1",
        "protocol": "G4_07_CDL_DIRECTIONAL_PATTERN_VECTOR_FIELD_PROTOCOL_2026-11-01",
        "run_date": dt.datetime.now(dt.timezone.utc).date().isoformat(),
        "status": "PASS_WITHIN_BOUNDED_PROTOCOL_SCOPE" if passed else "FAILED_OR_INCOMPLETE",
        "seed_collision_audit": {
            "path": SEED_AUDIT.name,
            "sha256": _sha256(SEED_AUDIT),
            "status": "CLEAR_WITHIN_SCANNED_JSON_CORPUS",
            "scanned_file_count": 2276,
        },
        "design": {
            "profiles": list("ABCDE"),
            "patterns": ["38.901", "38.901-handheld"],
            "samples_per_case": 16384,
            "one_end_seed_range": [2700000, 2700029],
            "both_ends_seed_range": [2701000, 2701039],
            "familywise_alpha": 0.01,
            "maximum_half_width_fraction_of_total_power": 0.05,
            "case_count": 70,
            "cluster_claim_count": 1358,
        },
        "runtime": {
            "sionna": _sionna_version(),
            "torch": importlib.metadata.version("torch"),
            "torch_cuda": str(__import__("torch").version.cuda or "cpu-only"),
            "device": "cpu",
        },
        "command": command,
        "provenance": {
            "git_revision": subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            "clean_worktree_at_run_start": True,
            "runner_source_sha256": _sha256(Path(__file__).resolve()),
            "test_source_sha256": _sha256(TEST),
            "protocol_sha256": _sha256(PROTOCOL),
            "schema_sha256": _sha256(SCHEMA_PATH),
            "reference_sha256": {path.name: _sha256(path) for path in REFERENCE_PATHS},
        },
        "outcome": {
            "pytest_exit_code": pytest_exit_code,
            "cases_observed": len(cases),
            "cases_passed": sum(
                case["test_outcome"] == "passed" and case["passed"] for case in cases
            ),
            "claims_passed": sum(claim["passed"] for case in cases for claim in case["claims"]),
            "claims_failed": sum(not claim["passed"] for case in cases for claim in case["claims"]),
        },
        "cases": cases,
        "scope": "Bounded Sionna 2.2.0 CDL-A-E single-element pattern consistency only; not full G4 qualification.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new result JSON path")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"refusing to overwrite result: {output}")
    worktree_status = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    )
    if worktree_status:
        parser.error("refusing to run from a dirty worktree; commit all source changes first")
    collector = _ResultCollector()
    command = (
        f"{sys.executable} -m scripts.run_g4_cdl_directional_pattern_power "
        f"--output {output} --device cpu"
    )
    pytest_exit_code = int(
        pytest.main(
            [
                "--no-cov",
                "-o",
                "addopts=",
                "-q",
                "-m",
                "native_sionna",
                str(TEST),
            ],
            plugins=[collector],
        )
    )
    result = _build_report(collector.cases, command, pytest_exit_code)
    if result["status"] == "PASS_WITHIN_BOUNDED_PROTOCOL_SCOPE":
        summary = audit_result(result)
        result["audit"] = summary
    else:
        result["audit"] = {"status": "NOT_AUDITED_FAILED_OR_INCOMPLETE"}
    jsonschema.Draft202012Validator(
        json.loads(SCHEMA_PATH.read_text(encoding="utf-8")),
        format_checker=jsonschema.FormatChecker(),
    ).validate(result)
    payload = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    try:
        _publish_no_replace(output, payload)
    except FileExistsError:
        parser.error(f"refusing to overwrite result: {output}")
    print(f"{result['status']}: {output}")
    return (
        pytest_exit_code
        if pytest_exit_code
        else (0 if result["status"] == "PASS_WITHIN_BOUNDED_PROTOCOL_SCOPE" else 2)
    )


if __name__ == "__main__":
    raise SystemExit(main())
