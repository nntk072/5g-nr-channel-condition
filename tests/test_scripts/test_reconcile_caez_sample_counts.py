"""Tests for official-versus-local CAEZ count reconciliation metadata."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.reconcile_caez_sample_counts import build_reconciliation

ROOT = Path(__file__).resolve().parents[2]


def test_current_caez_snapshot_keeps_all_three_count_differences_unresolved() -> None:
    snapshot = json.loads(
        (ROOT / "docs/research/data/caez-official-snapshot-20261024.json").read_text()
    )
    indoor = json.loads((ROOT / "docs/research/results/caez-indoor-audit.json").read_text())
    cross = json.loads(
        (ROOT / "docs/research/results/caez-cross-environment-summary-forecast.json").read_text()
    )

    report = build_reconciliation(snapshot=snapshot, indoor_audit=indoor, cross_report=cross)

    assert report["status"] == "UNRESOLVED_PUBLISHED_VS_PREPROCESSED_COUNT_DIFFERENCES"
    assert [item["local_minus_official_sample_count"] for item in report["results"]] == [
        -1,
        -17,
        2000,
    ]
    assert all(
        item["reconciliation_status"] == "UNRESOLVED_COUNT_DIFFERENCE" for item in report["results"]
    )
    assert all(item["local_archive_sha256"] for item in report["results"])
    assert report["license_handling"]["raw_archives_must_remain_local_and_non_redistributed"]


def test_reconciliation_rejects_disagreement_between_indoor_audit_and_cross_report() -> None:
    snapshot = json.loads(
        (ROOT / "docs/research/data/caez-official-snapshot-20261024.json").read_text()
    )
    indoor = json.loads((ROOT / "docs/research/results/caez-indoor-audit.json").read_text())
    cross = json.loads(
        (ROOT / "docs/research/results/caez-cross-environment-summary-forecast.json").read_text()
    )
    cross["target_domains"][0]["sample_count"] += 1

    try:
        build_reconciliation(snapshot=snapshot, indoor_audit=indoor, cross_report=cross)
    except ValueError as error:
        assert "indoor audit and cross-environment report disagree" in str(error)
    else:
        raise AssertionError("inconsistent local evidence should be rejected")
