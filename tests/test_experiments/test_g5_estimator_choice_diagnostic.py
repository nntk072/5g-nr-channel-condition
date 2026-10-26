from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from experiments.diagnose_g5_estimator_choice import diagnose_estimator_choice

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "docs/research/results/g5-estimator-choice-posthoc-20261024.json"


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@pytest.mark.native_sionna
def test_selected_failure_estimator_diagnostic_replays_hash_bound_outcomes() -> None:
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

    replay = diagnose_estimator_choice()
    assert replay["observations"] == report["observations"]
    assert replay["totals"] == report["totals"]
    assert replay["confirmation_artifact_sha256"] == report["confirmation_artifact_sha256"]
    assert replay["source_sha256"] != report["source_sha256"]
    assert {
        path
        for path, digest in replay["source_sha256"].items()
        if report["source_sha256"].get(path) != digest
    } == {"integrations/sionna/nr_pusch.py"}
