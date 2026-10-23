import hashlib
import json
from pathlib import Path

import pytest
from experiments.g5_doppler_awgn_receiver import (
    _source_fingerprints,
    exact_paired_binomial_p_value,
    run_doppler_awgn_probe,
    wilson_interval,
)


def test_wilson_interval_bounds_zero_failures_and_validates_counts():
    low, high = wilson_interval(0, 48)
    assert low == pytest.approx(0.0, abs=1e-15)
    assert high == pytest.approx(0.0741, abs=1e-4)
    with pytest.raises(ValueError, match="require"):
        wilson_interval(49, 48)


@pytest.mark.parametrize(
    ("reference", "comparison", "expected"),
    [
        ([True] * 10, [True] * 10, 1.0),
        ([True] * 8, [False] * 8, 2.0 / 256.0),
        ([False] * 8, [True] * 8, 2.0 / 256.0),
    ],
)
def test_paired_exact_binomial_test_uses_discordant_payloads(reference, comparison, expected):
    assert exact_paired_binomial_p_value(reference, comparison) == pytest.approx(expected)


@pytest.mark.parametrize("seeds", [(), (1, 1), (-1,), (True,)])
def test_doppler_awgn_probe_rejects_invalid_batch_seed_design(seeds):
    with pytest.raises(ValueError, match="batch_seeds"):
        run_doppler_awgn_probe(batch_seeds=seeds)


@pytest.mark.native_sionna
def test_saved_doppler_awgn_report_recomputes_paired_outcomes_and_source_hashes():
    report_path = (
        Path(__file__).resolve().parents[2]
        / "docs/research/results/g5-time-varying-multipath-pusch-awgn-20261023.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    recorded_hash = report.pop("result_sha256")
    actual_hash = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    assert report["schema"] == "umcd.g5_doppler_awgn_receiver.v1"
    assert recorded_hash == actual_hash
    assert report["source_sha256"] == _source_fingerprints()
    assert report["command"]
    assert len(report["config_sha256"]) == 64
    assert (
        report["config_sha256"]
        == hashlib.sha256(
            json.dumps(report["configuration"], sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
    )
    assert report["runtime"]["device"] == "cpu"
    assert report["configuration"]["independent_channel_draws"] == 0
    assert report["configuration"]["awgn_model"].startswith("independent proper-complex Gaussian")
    assert report["configuration"]["batch_seeds"]
    assert len(report["paired_batch_outcomes"]) == len(report["configuration"]["batch_seeds"])

    def outcomes_for(condition: str) -> list[bool]:
        return [
            passed
            for record in report["paired_batch_outcomes"]
            for passed in record["conditions"][condition]["crc_pass_by_block"]
        ]

    for condition, summary in report["summary_by_condition"].items():
        passed = outcomes_for(condition)
        failures = len(passed) - sum(passed)
        bit_errors = sum(
            error
            for record in report["paired_batch_outcomes"]
            for error in record["conditions"][condition]["bit_errors_by_block"]
        )
        assert summary["blocks"] == len(passed)
        assert summary["crc_failures"] == failures
        assert summary["bler"] == pytest.approx(failures / len(passed))
        assert summary["bit_errors"] == bit_errors
        assert summary["bit_error_rate"] == pytest.approx(bit_errors / summary["total_bits"])
        assert summary["bler_wilson_95_percent_interval"] == pytest.approx(
            wilson_interval(failures, len(passed))
        )

    static = outcomes_for("0hz_continuous")
    mobile = outcomes_for("750hz_continuous")
    symbol_static = outcomes_for("750hz_symbol_static")
    assert report["acceptance"]["observed_paired_exact_p_value"] == pytest.approx(
        exact_paired_binomial_p_value(static, mobile)
    )
    assert report["intra_symbol_acceptance"]["observed_paired_exact_p_value"] == pytest.approx(
        exact_paired_binomial_p_value(symbol_static, mobile)
    )

    for record in report["paired_batch_outcomes"]:
        noise = record["conditions"]["0hz_continuous"]["mean_realized_awgn_power"]
        for condition in report["summary_by_condition"]:
            assert record["conditions"][condition]["mean_realized_awgn_power"] == noise
