from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "docs/research/results/g5-estimator-choice-posthoc-20261024.json"


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def test_selected_failure_estimator_artifact_preserves_outcomes_and_flags_source_drift() -> None:
    report = json.loads(RESULT.read_text(encoding="utf-8"))
    digest = report.pop("result_sha256")

    assert report["schema"] == "umcd.g5_estimator_choice_posthoc_diagnostic.v1"
    assert report["status"] == "POST_HOC_DIAGNOSTIC_NOT_QUALIFICATION"
    assert digest == _canonical_sha256(report)
    assert report["selection_is_outcome_dependent"] is True
    assert report["independent_channel_count"] == 7
    assert report["blocks_per_channel"] == 8
    assert report["totals"]["nn"]["crc_passes"] == 7
    assert report["totals"]["lin"]["crc_passes"] == 22
    assert report["totals"]["lin_time_avg"]["crc_passes"] == 22
    assert report["interpretation"].endswith("disjoint channel seeds.")
    assert "runtime_versions" not in report
    assert "sionna_version" not in report
    assert "torch_version" not in report

    current_sources = {
        path
        for path, digest in report["source_sha256"].items()
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest
    }
    assert current_sources == {
        "integrations/sionna/nr_pusch.py",
        "integrations/sionna/ofdm_impairments.py",
    }
