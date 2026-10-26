from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from experiments.audit_g5_tdl_c_dmrs_density import DEFAULT_REPORT, audit_report


def _saved_report() -> dict:
    root = Path(__file__).resolve().parents[2]
    return json.loads((root / DEFAULT_REPORT).read_text(encoding="utf-8"))


def test_saved_tdl_c_confirmation_reports_source_freshness_truthfully() -> None:
    report = _saved_report()

    audited = audit_report(report)
    root = Path(__file__).resolve().parents[2]
    source_hashes = report["source_sha256"]
    current_sources_match = all(
        (root / path).is_file()
        and hashlib.sha256((root / path).read_bytes()).hexdigest() == expected_hash
        for path, expected_hash in source_hashes.items()
    )

    assert audited["source_matches_worktree"] is current_sources_match
    if current_sources_match:
        assert audited["status"] == "PASS", audited["issues"]
    else:
        assert audited["status"] == "FAIL", audited["issues"]
        assert any("source hash mismatch" in issue for issue in audited["issues"])


def test_report_audit_rejects_changed_raw_outcome_even_with_rehashed_envelope() -> None:
    report = copy.deepcopy(_saved_report())
    report["paired_channel_outcomes"][0]["mobile_position_3_crc_error"] = True

    audited = audit_report(report)

    assert audited["status"] == "FAIL"
    assert any("result_sha256" in issue for issue in audited["issues"])
    assert any("saved mobile_position_3_crc_error differs" in issue for issue in audited["issues"])


def test_report_audit_rejects_protocol_drift() -> None:
    report = copy.deepcopy(_saved_report())
    report["protocol_sha256"] = "0" * 64

    audited = audit_report(report)

    assert audited["status"] == "FAIL"
    assert "protocol hash differs from the checked-in protocol" in audited["issues"]


def test_report_audit_rejects_source_drift() -> None:
    report = copy.deepcopy(_saved_report())
    report["source_sha256"]["experiments/qualify_g5_tdl_c_dmrs_density.py"] = "0" * 64

    audited = audit_report(report)

    assert audited["status"] == "FAIL"
    assert any("source hash mismatch" in issue for issue in audited["issues"])
    assert audited["source_matches_worktree"] is False


def test_report_audit_recomputes_bler_summary_from_retained_crc_mask() -> None:
    report = copy.deepcopy(_saved_report())
    report["conditions"]["mobile_position_0"]["point"]["bler"] = 0.0
    unhashed = dict(report)
    unhashed.pop("result_sha256")
    report["result_sha256"] = hashlib.sha256(
        json.dumps(unhashed, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()

    audited = audit_report(report)

    assert audited["status"] == "FAIL"
    assert "mobile_position_0 BLER does not match raw outcomes" in audited["issues"]
