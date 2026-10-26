from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from experiments.audit_g5_tdl_dft_profile_matrix import (
    DEFAULT_REPORT,
    audit_family_report,
)


def _saved_report() -> dict:
    root = Path(__file__).resolve().parents[2]
    return json.loads((root / DEFAULT_REPORT).read_text(encoding="utf-8"))


def _rehash(report: dict) -> None:
    report.pop("result_sha256", None)
    report["result_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def test_saved_signed_lag_dft_profile_family_audits_all_profile_results() -> None:
    report = _saved_report()

    audited = audit_family_report(report)

    assert audited["status"] == "FAIL", audited["issues"]
    assert audited["source_matches_worktree"] is False
    assert all("source hash mismatch" in issue for issue in audited["issues"])
    assert audited["profile_statuses"] == {
        f"TDL-{profile}": f"PASS_BOUNDED_TDL_{profile}_FIXTURE" for profile in "ABCDE"
    }


def test_family_auditor_rejects_changed_estimator_configuration() -> None:
    report = copy.deepcopy(_saved_report())
    report["channel_estimator"]["max_delay_samples"] = 8
    _rehash(report)

    audited = audit_family_report(report)

    assert audited["status"] == "FAIL"
    assert "family estimator differs from protocol" in audited["issues"]
