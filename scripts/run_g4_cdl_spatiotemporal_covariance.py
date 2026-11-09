"""Run and retain the frozen G4 CDL joint space-time covariance protocol."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "tests/test_integrations/test_g4_cdl_spatiotemporal_covariance_native.py"
PROTOCOL = ROOT / "docs/research/G4_07_CDL_SPATIOTEMPORAL_COVARIANCE_PROTOCOL_2026-11-09.md"
SEED_AUDIT = ROOT / "docs/research/G4_CDL_SPATIOTEMPORAL_COVARIANCE_SEED_AUDIT_2026-11-09.json"
REFERENCE = ROOT / "docs/research/G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json"
EVALUATION = ROOT / "evaluation/g4_cdl_spatiotemporal_covariance.py"
MODELS = tuple("ABCDE")
VELOCITIES = ("x", "oblique_122")
SEED_START = 2_810_000
CASE_COUNT = 10
CLAIM_COUNT = 80
DRAW_COUNT = 16_384
FAMILYWISE_ALPHA = 0.01
MAXIMUM_HALF_WIDTH = 0.05


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sionna_version() -> str:
    for distribution in ("sionna", "sionna-no-rt"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    raise RuntimeError("neither sionna nor sionna-no-rt distribution is installed")


class _Collector:
    def __init__(self) -> None:
        self.cases: list[dict[str, Any]] = []

    def pytest_runtest_logreport(self, report: Any) -> None:
        if report.when != "call":
            return
        for key, value in report.user_properties:
            if key == "g4_cdl_spatiotemporal_covariance_case":
                case = json.loads(value)
                case["test_nodeid"] = report.nodeid
                case["test_outcome"] = report.outcome
                self.cases.append(case)


def _expected_seed(model: str, velocity_label: str) -> int:
    return SEED_START + 2 * MODELS.index(model) + VELOCITIES.index(velocity_label)


def _validate_cases(cases: list[dict[str, Any]]) -> dict[str, float | int]:
    if len(cases) != CASE_COUNT:
        raise ValueError(f"expected {CASE_COUNT} cases, observed {len(cases)}")
    seen: set[tuple[str, str]] = set()
    claim_count = 0
    maximum_half_width = 0.0
    maximum_absolute_error = 0.0
    for case in cases:
        key = (case["model"], case["velocity_label"])
        if key in seen or key[0] not in MODELS or key[1] not in VELOCITIES:
            raise ValueError(f"invalid or duplicate case identity: {key}")
        seen.add(key)
        if case["seed"] != _expected_seed(*key):
            raise ValueError(f"seed mismatch for {key}")
        if (
            case["draw_count"] != DRAW_COUNT
            or case["family_claim_count"] != CLAIM_COUNT
            or case["familywise_alpha"] != FAMILYWISE_ALPHA
            or case["test_outcome"] != "passed"
            or case["batch_axis_semantics"] != "independent_stochastic_channel_draws"
        ):
            raise ValueError(f"frozen design/outcome mismatch for {key}")
        if len(case["claims"]) != 8:
            raise ValueError(f"expected eight matrix-component claims for {key}")
        for claim in case["claims"]:
            claim_count += 1
            maximum_half_width = max(maximum_half_width, float(claim["half_width"]))
            maximum_absolute_error = max(maximum_absolute_error, float(claim["absolute_error"]))
            if not claim["passed"] or claim["half_width"] > MAXIMUM_HALF_WIDTH:
                raise ValueError(f"frozen confidence criterion failed for {key}")
    expected = {(model, velocity) for model in MODELS for velocity in VELOCITIES}
    if seen != expected or claim_count != CLAIM_COUNT:
        raise ValueError("case or claim roster differs from the frozen protocol")
    return {
        "case_count": len(cases),
        "claim_count": claim_count,
        "maximum_half_width": maximum_half_width,
        "maximum_absolute_error": maximum_absolute_error,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    sidecar = Path(str(output) + ".sha256")
    if output.exists() or sidecar.exists():
        parser.error("refusing to overwrite retained result or checksum")
    if subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"], text=True
    ):
        parser.error("refusing to run from a dirty worktree")

    seed_audit = json.loads(SEED_AUDIT.read_text(encoding="utf-8"))
    expected_seeds = list(range(SEED_START, SEED_START + CASE_COUNT))
    if (
        seed_audit.get("status") != "CLEAR_WITHIN_SCANNED_JSON_CORPUS"
        or seed_audit.get("reserved_seeds") != expected_seeds
        or seed_audit.get("collisions")
        or seed_audit.get("malformed_seed_fields")
        or seed_audit.get("incomplete_files")
    ):
        parser.error("fresh seed audit does not clear the frozen roster")

    collector = _Collector()
    exit_code = int(
        pytest.main(
            ["--no-cov", "-o", "addopts=", "-q", "-m", "native_sionna", str(TEST)],
            plugins=[collector],
        )
    )
    if exit_code:
        parser.error(f"native pytest execution failed with exit code {exit_code}")
    summary = _validate_cases(collector.cases)
    cases = sorted(
        collector.cases,
        key=lambda case: (MODELS.index(case["model"]), VELOCITIES.index(case["velocity_label"])),
    )
    report = {
        "schema": "loslab.g4_cdl_spatiotemporal_covariance_native_validation.v1",
        "date": "2026-11-09",
        "base_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "all_passed": True,
        "criteria_changed": False,
        "profiles": list(MODELS),
        "velocity_labels": list(VELOCITIES),
        "draws_per_case": DRAW_COUNT,
        "case_count": CASE_COUNT,
        "simultaneous_claim_count": CLAIM_COUNT,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "maximum_interval_half_width": MAXIMUM_HALF_WIDTH,
        "maximum_observed_half_width": summary["maximum_half_width"],
        "maximum_observed_absolute_error": summary["maximum_absolute_error"],
        "seed_audit": {
            "path": str(SEED_AUDIT.relative_to(ROOT)),
            "sha256": _sha256(SEED_AUDIT),
            "status": seed_audit["status"],
            "scanned_file_count": seed_audit["scanned_file_count"],
            "global_exclusivity_established": False,
        },
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": importlib.metadata.version("torch"),
            "sionna": _sionna_version(),
            "device": "cpu",
        },
        "source_hashes": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (
                EVALUATION,
                TEST,
                PROTOCOL,
                SEED_AUDIT,
                REFERENCE,
                Path(__file__).resolve(),
            )
        },
        "cases": cases,
        "qualification_claim": "none",
        "limitations": [
            "Bounded Sionna PHY 2.2.0 CPU model-consistency evidence for the frozen two-element CDL joint space-time covariance family only.",
            "The family covers CDL-A through E, two velocity vectors, one 3.5 GHz carrier, one 30 m/s speed, one 0.7 ms lag, a zero-oriented half-wavelength two-element single-V omni UT array, and a single-V omni BS port.",
            "It does not establish continuous orientation or velocity coverage, arbitrary arrays or patterns, mutual coupling, multiport transmit covariance, measured-channel truth, RT/material truth, receiver validity, or overall G4 qualification.",
            "The seed audit establishes no global seed exclusivity.",
            "Production D1 generation and advanced training remain gated.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": digest, **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
