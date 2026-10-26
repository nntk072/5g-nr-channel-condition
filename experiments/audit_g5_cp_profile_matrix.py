"""Build a family-wise audit of the saved five-profile G5 CP curves.

This audit reuses paired independent channel parents from existing reports. It
does not regenerate channels and remains exploratory evidence, not a G5 pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import numpy as np

PROFILES = ("A", "B", "C", "D", "E")
EXPECTED_SNR_DB = (60.0, 80.0, 100.0)
FAMILY_SIZE = len(PROFILES) * len(EXPECTED_SNR_DB)
SCHEMA = "umcd.g5_cp_profile_matrix_familywise_audit.v1"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _sha256(encoded.encode("utf-8"))


def _wilson_interval(failures: int, trials: int) -> tuple[float, float]:
    """Independently recompute the report's two-sided 95% Wilson interval."""
    z = 1.959963984540054
    proportion = failures / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = (proportion + z_squared / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z_squared / (4.0 * trials * trials))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _validate_outcome_summary(
    summary: Any, values: list[bool], *, name: str, expected_count: int
) -> None:
    if not isinstance(summary, dict):
        raise ValueError(f"invalid {name} outcome summary")
    failures = expected_count - sum(values)
    bler = summary.get("bler")
    if (
        summary.get("independent_channel_draws") != expected_count
        or isinstance(summary.get("independent_channel_draws"), bool)
        or summary.get("block_errors") != failures
        or isinstance(summary.get("block_errors"), bool)
        or isinstance(bler, bool)
        or not isinstance(bler, (int, float))
        or not math.isfinite(bler)
        or bler != failures / expected_count
    ):
        raise ValueError(f"inconsistent {name} count or BLER summary")
    interval = summary.get("wilson_95_percent_interval")
    expected_interval = _wilson_interval(failures, expected_count)
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or any(
            isinstance(bound, bool)
            or not isinstance(bound, (int, float))
            or not math.isfinite(bound)
            for bound in interval
        )
        or any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15)
            for actual, expected in zip(interval, expected_interval, strict=True)
        )
    ):
        raise ValueError(f"inconsistent {name} Wilson interval")


def paired_parent_bootstrap_interval(
    control_pass: Sequence[bool],
    treatment_pass: Sequence[bool],
    *,
    alpha: float,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Return a percentile interval for paired BLER change by independent parent.

    The contrast is treatment error rate minus control error rate. Each parent
    contributes one paired binary outcome. A multinomial draw is exactly the
    ordinary nonparametric bootstrap over these parent-level contrasts.
    """
    if len(control_pass) != len(treatment_pass) or not control_pass:
        raise ValueError("paired outcomes must have equal nonzero lengths")
    if any(not isinstance(value, (bool, np.bool_)) for value in control_pass):
        raise ValueError("control outcomes must be booleans")
    if any(not isinstance(value, (bool, np.bool_)) for value in treatment_pass):
        raise ValueError("treatment outcomes must be booleans")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and between zero and one")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1000:
        raise ValueError("draws must be an integer >= 1000")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    # Error treatment minus error control equals control-pass minus
    # treatment-pass for each paired parent.
    parent_deltas = np.asarray(control_pass, dtype=np.int8) - np.asarray(
        treatment_pass, dtype=np.int8
    )
    minus_count = int(np.count_nonzero(parent_deltas == -1))
    zero_count = int(np.count_nonzero(parent_deltas == 0))
    plus_count = int(np.count_nonzero(parent_deltas == 1))
    parent_count = len(parent_deltas)
    probabilities = np.asarray([minus_count, zero_count, plus_count], dtype=np.float64)
    probabilities /= parent_count
    sampled_counts = np.random.default_rng(seed).multinomial(
        parent_count, probabilities, size=draws
    )
    sampled_delta = (sampled_counts[:, 2] - sampled_counts[:, 0]) / parent_count
    tail = alpha / 2.0
    low, high = np.quantile(sampled_delta, [tail, 1.0 - tail], method="linear")
    return {
        "independent_parents": parent_count,
        "paired_treatment_minus_control_bler": float(parent_deltas.mean()),
        "parent_delta_counts_minus_one_zero_plus_one": [minus_count, zero_count, plus_count],
        "bootstrap_interval": [float(low), float(high)],
        "interval_confidence": 1.0 - alpha,
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
    }


def _read_curve(path: Path, expected_profile: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    report = json.loads(raw)
    if not isinstance(report, dict):
        raise ValueError(f"curve report must be an object: {path}")
    if report.get("schema") != "umcd.g5_cp_exceedance_snr_curve.v4":
        raise ValueError(f"unexpected report schema in {path}")
    config = report.get("configuration")
    if not isinstance(config, dict) or _canonical_sha256(config) != report.get(
        "configuration_sha256"
    ):
        raise ValueError(f"configuration hash mismatch in {path}")
    channel_config = config.get("channel_and_receiver_configuration")
    if not isinstance(channel_config, dict):
        raise ValueError(f"missing channel/receiver configuration in {path}")
    channel = channel_config.get("channel_family")
    if channel != f"Sionna TDL-{expected_profile} stochastic channel draws":
        raise ValueError(f"profile mismatch in {path}: {channel!r}")
    source_hashes = report.get("source_files_sha256")
    if (
        not isinstance(source_hashes, dict)
        or not source_hashes
        or source_hashes != report.get("source_sha256")
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for name, digest in source_hashes.items()
        )
    ):
        raise ValueError(f"invalid or inconsistent source hashes in {path}")
    source_root = Path(__file__).resolve().parents[1]
    for name, expected_digest in source_hashes.items():
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or relative.as_posix() != name
            or any(part in {"", ".", ".."} or ":" in part for part in name.split("/"))
            or "\\" in name
        ):
            raise ValueError(f"noncanonical source path in {path}: {name!r}")
        source_path = source_root.joinpath(*relative.parts)
        if not source_path.is_file() or _sha256(source_path.read_bytes()) != expected_digest:
            raise ValueError(f"source snapshot mismatch for {name} in {path}")
    if tuple(config.get("snr_db", ())) != EXPECTED_SNR_DB:
        raise ValueError(f"unexpected SNR grid in {path}")
    lineage = report.get("parent_lineage")
    if not isinstance(lineage, list) or any(not isinstance(item, dict) for item in lineage):
        raise ValueError(f"invalid parent lineage in {path}")
    parent_ids = [item.get("parent_subject_id") for item in lineage]
    expected_count = config.get("independent_parent_count")
    if (
        not isinstance(expected_count, int)
        or isinstance(expected_count, bool)
        or expected_count != 128
        or len(parent_ids) != expected_count
        or any(not isinstance(parent_id, str) or not parent_id.strip() for parent_id in parent_ids)
        or len(set(parent_ids)) != expected_count
        or any(
            not isinstance(item.get("channel_truth_sha256"), str)
            or len(item["channel_truth_sha256"]) != 64
            or any(
                character not in "0123456789abcdef" for character in item["channel_truth_sha256"]
            )
            for item in lineage
        )
    ):
        raise ValueError(f"expected 128 unique independent parents in {path}")
    points = report.get("points")
    hashes = report.get("point_report_sha256")
    if not isinstance(points, list) or len(points) != len(EXPECTED_SNR_DB):
        raise ValueError(f"expected three SNR points in {path}")
    if not isinstance(hashes, list) or len(hashes) != len(points):
        raise ValueError(f"point hash count mismatch in {path}")
    for point, expected_snr, expected_hash in zip(points, EXPECTED_SNR_DB, hashes, strict=True):
        if not isinstance(point, dict):
            raise ValueError(f"invalid point in {path} at {expected_snr:g} dB")
        if point.get("snr_db") != expected_snr or _canonical_sha256(point) != expected_hash:
            raise ValueError(f"point content/hash mismatch in {path} at {expected_snr:g} dB")
        outcomes = point.get("outcomes")
        if not isinstance(outcomes, dict):
            raise ValueError(f"invalid outcomes in {path} at {expected_snr:g} dB")
        for mode in ("time_domain", "ideal_cfr"):
            for estimate in ("dmrs_ls", "diagonal_cfr_oracle"):
                key = f"{mode}.{estimate}"
                summary = outcomes.get(key)
                values = summary.get("crc_pass_by_draw") if isinstance(summary, dict) else None
                if (
                    not isinstance(values, list)
                    or len(values) != expected_count
                    or any(not isinstance(value, bool) for value in values)
                ):
                    raise ValueError(f"invalid {key} outcomes in {path} at {expected_snr:g} dB")
                _validate_outcome_summary(summary, values, name=key, expected_count=expected_count)
        paired_table = point.get("paired_pass_table")
        if not isinstance(paired_table, dict):
            raise ValueError(f"invalid paired pass table in {path} at {expected_snr:g} dB")
        for estimate in ("dmrs_ls", "diagonal_cfr_oracle"):
            left = outcomes[f"time_domain.{estimate}"]["crc_pass_by_draw"]
            right = outcomes[f"ideal_cfr.{estimate}"]["crc_pass_by_draw"]
            actual_table = paired_table.get(estimate)
            expected_table = {
                "both_pass": sum(a and b for a, b in zip(left, right, strict=True)),
                "left_only_pass": sum(a and not b for a, b in zip(left, right, strict=True)),
                "right_only_pass": sum(not a and b for a, b in zip(left, right, strict=True)),
                "both_fail": sum(not a and not b for a, b in zip(left, right, strict=True)),
            }
            if (
                not isinstance(actual_table, dict)
                or set(actual_table) != set(expected_table)
                or any(type(value) is not int for value in actual_table.values())
                or actual_table != expected_table
            ):
                raise ValueError(f"inconsistent {estimate} paired pass table in {path}")
    return report, _sha256(raw)


def audit_profile_matrix(
    results_dir: Path,
    *,
    alpha: float = 0.05,
    draws: int = 100_000,
    seed: int = 20261024,
) -> dict[str, Any]:
    """Verify five source reports and calculate 15 Bonferroni intervals."""
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("family-wise alpha must be finite and between zero and one")
    report_rows: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    claim_index = 0
    shared_source_hashes: dict[str, str] | None = None
    for profile in PROFILES:
        path = results_dir / (
            f"g5-standardized-pusch-cp-exceedance-tdl-{profile.lower()}-"
            f"snr-curve-128-"
            + {"A": "v2", "B": "v1", "C": "v3", "D": "v1", "E": "v1"}[profile]
            + "-20261024.json"
        )
        report, report_sha = _read_curve(path, profile)
        source_hashes = report["source_files_sha256"]
        if shared_source_hashes is None:
            shared_source_hashes = source_hashes
        elif source_hashes != shared_source_hashes:
            raise ValueError("profile reports were generated from different source snapshots")
        report_rows.append(
            {
                "profile": profile,
                "report": path.name,
                "report_sha256": report_sha,
                "configuration_sha256": report["configuration_sha256"],
                "source_files_sha256": source_hashes,
                "point_report_sha256": report["point_report_sha256"],
                "independent_parent_count": report["configuration"]["independent_parent_count"],
            }
        )
        for point in report["points"]:
            claim_index += 1
            outcomes = point["outcomes"]
            control = outcomes["ideal_cfr.dmrs_ls"]["crc_pass_by_draw"]
            treatment = outcomes["time_domain.dmrs_ls"]["crc_pass_by_draw"]
            result = paired_parent_bootstrap_interval(
                control,
                treatment,
                alpha=alpha / FAMILY_SIZE,
                draws=draws,
                seed=seed + claim_index,
            )
            claims.append(
                {
                    "claim_id": f"TDL-{profile}@{point['snr_db']:g}dB",
                    "profile": profile,
                    "snr_db": point["snr_db"],
                    "control": "ideal-CFR DMRS-LS",
                    "treatment": "time-domain DMRS-LS",
                    **result,
                }
            )
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=False, capture_output=True, text=True
    )
    source_sha = _sha256(Path(__file__).read_bytes())
    family_confidence = 1.0 - alpha
    return {
        "schema": SCHEMA,
        "status": "exploratory; does not pass G5 or qualify G3",
        "run_date": date.today().isoformat(),
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "repository_commit_at_run": revision.stdout.strip()
        if revision.returncode == 0
        else "unavailable",
        "analysis_source_sha256": source_sha,
        "family": {
            "profile_count": len(PROFILES),
            "snr_points_per_profile": list(EXPECTED_SNR_DB),
            "claim_count": FAMILY_SIZE,
            "family_wise_confidence": family_confidence,
            "method": "two-sided percentile bootstrap with Bonferroni alpha/FAMILY_SIZE per claim",
            "resampling_unit": "independent channel parent; paired receiver outcomes retained",
            "primary_view": "DMRS-LS; oracle outcomes are excluded from this inferential family",
            "bootstrap_draws_per_claim": draws,
            "bootstrap_seed_base": seed,
        },
        "input_reports": report_rows,
        "claims": claims,
        "limitations": [
            "Intervals are conditional on one standardized Sionna configuration and the sampled channel parents.",
            "The profile/SNR family correction does not support physical-site generalization or deployment BLER claims.",
            "Sparse failures in TDL-D/E limit resolution; this audit does not close G5.09 or authorize production RT data.",
        ],
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    """Write a report immutably, allowing only byte-identical replays."""
    encoded = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(encoded)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise FileExistsError(f"refusing to replace a different report: {path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--results-dir", type=Path, default=root / "docs/research/results")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--draws", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args(argv)
    output = args.output or (
        root / "docs/research/results/g5-standardized-pusch-cp-profile-matrix-familywise-"
        "20261024.json"
    )
    report = audit_profile_matrix(
        args.results_dir, alpha=args.alpha, draws=args.draws, seed=args.seed
    )
    write_report(report, output)
    print(json.dumps({"output": str(output), "claims": report["claims"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
