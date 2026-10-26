from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from experiments.audit_g5_tdl_dft_density import DEFAULT_REPORT, audit_dft_report


def _saved_report() -> dict:
    root = Path(__file__).resolve().parents[2]
    return json.loads((root / DEFAULT_REPORT).read_text(encoding="utf-8"))


def _rehash(report: dict) -> None:
    report["configuration_sha256"] = hashlib.sha256(
        json.dumps(
            report["configuration"], sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    unhashed = dict(report)
    unhashed.pop("result_sha256", None)
    report["result_sha256"] = hashlib.sha256(
        json.dumps(unhashed, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def test_saved_signed_lag_dft_confirmation_is_rejected_after_source_drift() -> None:
    report = _saved_report()

    audited = audit_dft_report(report)

    assert audited["status"] == "FAIL"
    assert set(audited["issues"]) == {
        "source hash mismatch: experiments/g5_cp_bounded_dft_estimator.py",
        "source hash mismatch: integrations/sionna/nr_pusch.py",
    }
    assert audited["source_matches_worktree"] is False


def test_audit_rejects_changed_estimator_even_when_report_hashes_are_recomputed() -> None:
    report = copy.deepcopy(_saved_report())
    report["configuration"]["channel_estimator"]["max_delay_samples"] = 8
    _rehash(report)

    audited = audit_dft_report(report)

    assert audited["status"] == "FAIL"
    assert "channel estimator differs from the frozen protocol" in audited["issues"]
