"""Tests for the family-wise audit of saved CP-exceedance curves."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from experiments.audit_g5_cp_profile_matrix import (
    _read_curve,
    paired_parent_bootstrap_interval,
)
from experiments.standardized_pusch_cp_exceedance import wilson_interval


def test_paired_parent_bootstrap_interval_preserves_paired_delta() -> None:
    result = paired_parent_bootstrap_interval(
        [True, True, False, False],
        [True, False, False, False],
        alpha=0.05,
        draws=20_000,
        seed=37,
    )

    assert result["independent_parents"] == 4
    assert result["paired_treatment_minus_control_bler"] == pytest.approx(0.25)
    assert result["parent_delta_counts_minus_one_zero_plus_one"] == [0, 3, 1]
    low, high = result["bootstrap_interval"]
    assert low <= 0.25 <= high
    assert result["interval_confidence"] == pytest.approx(0.95)


def test_paired_parent_bootstrap_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="equal nonzero"):
        paired_parent_bootstrap_interval([], [], alpha=0.05, draws=1000, seed=1)
    with pytest.raises(ValueError, match="booleans"):
        paired_parent_bootstrap_interval([1], [True], alpha=0.05, draws=1000, seed=1)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="alpha"):
        paired_parent_bootstrap_interval([True], [False], alpha=1.0, draws=1000, seed=1)


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _source_hashes() -> dict[str, str]:
    relative = "experiments/standardized_pusch_cp_exceedance.py"
    source = Path(__file__).resolve().parents[2] / relative
    return {relative: hashlib.sha256(source.read_bytes()).hexdigest()}


def test_curve_reader_checks_profile_lineage_and_point_hash(tmp_path) -> None:
    lineage = [
        {"parent_subject_id": f"parent-{index}", "channel_truth_sha256": "a" * 64}
        for index in range(128)
    ]
    points = []
    point_hashes = []
    for snr_db in (60.0, 80.0, 100.0):
        outcomes = {}
        for key in (
            "time_domain.dmrs_ls",
            "ideal_cfr.dmrs_ls",
            "time_domain.diagonal_cfr_oracle",
            "ideal_cfr.diagonal_cfr_oracle",
        ):
            passed = [True] * 128
            if key == "time_domain.dmrs_ls":
                passed[-1] = False
            failures = 128 - sum(passed)
            outcomes[key] = {
                "independent_channel_draws": 128,
                "block_errors": failures,
                "bler": failures / 128,
                "wilson_95_percent_interval": list(wilson_interval(failures, 128)),
                "crc_pass_by_draw": passed,
            }
        point = {
            "snr_db": snr_db,
            "outcomes": outcomes,
            "paired_pass_table": {
                "dmrs_ls": {
                    "both_pass": 127,
                    "left_only_pass": 0,
                    "right_only_pass": 1,
                    "both_fail": 0,
                },
                "diagonal_cfr_oracle": {
                    "both_pass": 128,
                    "left_only_pass": 0,
                    "right_only_pass": 0,
                    "both_fail": 0,
                },
            },
        }
        points.append(point)
        point_hashes.append(_canonical_hash(point))
    configuration = {
        "snr_db": [60.0, 80.0, 100.0],
        "independent_parent_count": 128,
        "channel_and_receiver_configuration": {
            "channel_family": "Sionna TDL-A stochastic channel draws"
        },
    }
    report: dict[str, Any] = {
        "schema": "umcd.g5_cp_exceedance_snr_curve.v4",
        "source_files_sha256": _source_hashes(),
        "source_sha256": _source_hashes(),
        "configuration": configuration,
        "configuration_sha256": _canonical_hash(configuration),
        "parent_lineage": lineage,
        "points": points,
        "point_report_sha256": point_hashes,
    }
    path = tmp_path / "curve.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    loaded, digest = _read_curve(path, "A")

    assert loaded["configuration"] == configuration
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    source_name = next(iter(report["source_files_sha256"]))
    source_digest = report["source_files_sha256"][source_name]
    report["source_files_sha256"][source_name] = "c" * 64
    report["source_sha256"] = dict(report["source_files_sha256"])
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="source snapshot mismatch"):
        _read_curve(path, "A")
    report["source_files_sha256"][source_name] = source_digest
    report["source_sha256"] = dict(report["source_files_sha256"])
    report["point_report_sha256"][1] = "0" * 64
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="point content/hash mismatch"):
        _read_curve(path, "A")


def test_curve_reader_recomputes_summaries_even_when_point_hash_is_updated(tmp_path) -> None:
    lineage = [
        {"parent_subject_id": f"parent-{index}", "channel_truth_sha256": "a" * 64}
        for index in range(128)
    ]
    outcomes = {}
    for key in (
        "time_domain.dmrs_ls",
        "ideal_cfr.dmrs_ls",
        "time_domain.diagonal_cfr_oracle",
        "ideal_cfr.diagonal_cfr_oracle",
    ):
        passed = [True] * 128
        failures = 0
        outcomes[key] = {
            "independent_channel_draws": 128,
            "block_errors": failures,
            "bler": 0.0,
            "wilson_95_percent_interval": list(wilson_interval(failures, 128)),
            "crc_pass_by_draw": passed,
        }
    point = {
        "snr_db": 60.0,
        "outcomes": outcomes,
        "paired_pass_table": {
            "dmrs_ls": {
                "both_pass": 128,
                "left_only_pass": 0,
                "right_only_pass": 0,
                "both_fail": 0,
            },
            "diagonal_cfr_oracle": {
                "both_pass": 128,
                "left_only_pass": 0,
                "right_only_pass": 0,
                "both_fail": 0,
            },
        },
    }
    points = [point]
    for snr_db in (80.0, 100.0):
        points.append({**point, "snr_db": snr_db})
    configuration = {
        "snr_db": [60.0, 80.0, 100.0],
        "independent_parent_count": 128,
        "channel_and_receiver_configuration": {
            "channel_family": "Sionna TDL-A stochastic channel draws"
        },
    }
    report: dict[str, Any] = {
        "schema": "umcd.g5_cp_exceedance_snr_curve.v4",
        "source_files_sha256": _source_hashes(),
        "source_sha256": _source_hashes(),
        "configuration": configuration,
        "configuration_sha256": _canonical_hash(configuration),
        "parent_lineage": lineage,
        "points": points,
        "point_report_sha256": [_canonical_hash(item) for item in points],
    }
    report["points"][0]["outcomes"]["time_domain.dmrs_ls"]["bler"] = 0.25
    report["point_report_sha256"][0] = _canonical_hash(report["points"][0])
    path = tmp_path / "curve.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="inconsistent time_domain.dmrs_ls count or BLER"):
        _read_curve(path, "A")

    report["points"][0]["outcomes"]["time_domain.dmrs_ls"]["bler"] = 0.0
    report["points"][0]["paired_pass_table"]["dmrs_ls"]["both_pass"] = 127
    report["point_report_sha256"][0] = _canonical_hash(report["points"][0])
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent dmrs_ls paired pass table"):
        _read_curve(path, "A")
