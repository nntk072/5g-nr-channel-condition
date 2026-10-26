from __future__ import annotations

import json
from pathlib import Path

import pytest
from experiments.audit_g5_delay_domain_dft import audit_report


def test_historical_delay_domain_dft_result_is_rejected_after_source_drift() -> None:
    root = Path(__file__).resolve().parents[2]
    result_path = root / "docs/research/results/g5-delay-domain-dft-confirm-20261024.json"
    report = json.loads(result_path.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="source fingerprint mismatch"):
        audit_report(report, root=root)

    # Keep the historical outcomes, but do not count this report as current-
    # source confirmation after its runner/estimator implementation changed.
    assert report["g5_09_closed"] is False
    assert report["status"] == "PASS_NAMED_SYNTHETIC_CONFIGURATION_ONLY"
