from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from experiments.scan_g5_doppler_operating_point import (
    DOPPLER_CONDITIONS,
    NOISE_VARIANCES,
    _validate_design,
)
from tests.test_experiments.frozen_g5_source_manifest import assert_frozen_source_manifest


@pytest.mark.parametrize(
    ("seeds", "variances", "message"),
    [
        ((), (0.01,), "channel_seeds"),
        ((1, 1), (0.01,), "channel_seeds"),
        ((1,), (), "noise_variances"),
        ((1,), (0.01, 0.01), "noise_variances"),
        ((1,), (0.0,), "finite positive"),
        ((1,), (True,), "finite positive"),
    ],
)
def test_operating_point_scan_rejects_invalid_design(seeds, variances, message):
    with pytest.raises(ValueError, match=message):
        _validate_design(seeds, variances)


def test_saved_doppler_operating_point_scan_is_hash_bound_and_development_only():
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/research/results/g5-doppler-operating-point-scan-20261024.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    digest = report.pop("result_sha256")
    actual = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    assert report["schema"] == "umcd.g5_doppler_operating_point_scan.v1"
    assert report["status"] == "DEVELOPMENT_ONLY_NOT_QUALIFIED"
    assert digest == actual
    assert_frozen_source_manifest(
        report,
        expected_paths=(
            "experiments/g5_independent_channel_receiver.py",
            "experiments/scan_g5_doppler_operating_point.py",
            "integrations/sionna/nr_pusch.py",
            "integrations/sionna/ofdm_impairments.py",
            "receivers/nr/pusch.py",
        ),
    )
    config = report["configuration"]
    assert config["channel_seeds"] == list(range(83_001, 83_017))
    assert set(config["channel_seeds"]).isdisjoint(range(73_001, 73_065))
    assert set(config["channel_seeds"]).isdisjoint(range(99_1001, 99_1065))
    assert config["noise_variances"] == list(NOISE_VARIANCES)
    assert report["summary_by_noise_variance"].keys() == {str(value) for value in NOISE_VARIANCES}
    assert "acceptance" not in report

    records = report["channel_draw_records"]
    assert [record["channel_seed"] for record in records] == config["channel_seeds"]
    condition_names = {f"{int(doppler)}hz_{model}" for doppler, model in DOPPLER_CONDITIONS}
    for record in records:
        assert set(record["outcomes_by_noise_variance"]) == {
            str(value) for value in NOISE_VARIANCES
        }
        for conditions in record["outcomes_by_noise_variance"].values():
            assert set(conditions) == condition_names
            assert all(
                len(passed) == config["blocks_per_channel"] for passed in conditions.values()
            )

    for variance in NOISE_VARIANCES:
        summaries = report["summary_by_noise_variance"][str(variance)]["summary_by_condition"]
        for condition in condition_names:
            paired = [
                record["outcomes_by_noise_variance"][str(variance)][condition] for record in records
            ]
            failures = sum(sum(not passed for passed in blocks) for blocks in paired)
            block_count = sum(map(len, paired))
            summary = summaries[condition]
            assert summary["crc_failures"] == failures
            assert summary["blocks"] == block_count
            assert summary["pooled_bler_descriptive"] == pytest.approx(failures / block_count)
