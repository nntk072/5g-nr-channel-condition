import hashlib
import json
from pathlib import Path

import pytest
from experiments.g5_time_varying_multipath_receiver import (
    _source_fingerprints,
    exact_paired_binomial_p_value,
    wilson_interval,
)


def test_wilson_interval_handles_extreme_and_balanced_outcomes():
    assert wilson_interval(0, 48) == pytest.approx((0.0, 0.0741), abs=1e-4)
    assert wilson_interval(48, 48) == pytest.approx((0.9259, 1.0), abs=1e-4)
    low, high = wilson_interval(24, 48)
    assert low < 0.5 < high


@pytest.mark.parametrize(
    ("reference", "comparison", "expected"),
    [
        ([True] * 10, [True] * 10, 1.0),
        ([True] * 8, [False] * 8, 2.0 / 256.0),
        ([False] * 8, [True] * 8, 2.0 / 256.0),
    ],
)
def test_paired_exact_binomial_test_uses_discordant_pairs(reference, comparison, expected):
    assert exact_paired_binomial_p_value(reference, comparison) == pytest.approx(expected)


def test_saved_mobility_receiver_report_is_hash_bound_and_meets_frozen_fixture_criteria():
    report_path = (
        Path(__file__).resolve().parents[2]
        / "docs/research/results/g5-time-varying-multipath-pusch-symbol-control-20261023.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    recorded_hash = report.pop("result_sha256")
    actual_hash = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    assert report["schema"] == "umcd.g5_time_varying_multipath_receiver.v2"
    assert recorded_hash == actual_hash
    assert report["source_sha256"] == _source_fingerprints()
    assert report["runtime"]["device"] == "cpu"
    assert report["configuration"]["independent_channel_draws"] == 0
    assert report["configuration"]["waveform_awgn_injected"] is False
    assert report["acceptance"]["pass"] is True
    assert report["acceptance"]["observed_absolute_bler_increase"] >= 0.20
    assert report["acceptance"]["observed_paired_exact_p_value"] <= 0.01
    assert report["intra_symbol_acceptance"]["pass"] is False
    assert len(report["paired_batch_outcomes"]) == 12 * 5

    def outcomes_for(condition: str) -> list[bool]:
        doppler_text, channel_model = condition.split("_", maxsplit=1)
        doppler_hz = float(doppler_text.removesuffix("hz"))
        return [
            passed
            for record in report["paired_batch_outcomes"]
            if record["echo_doppler_hz"] == doppler_hz and record["channel_model"] == channel_model
            for passed in record["crc_pass_by_block"]
        ]

    for condition, summary in report["summary_by_condition"].items():
        passes = outcomes_for(condition)
        failures = len(passes) - sum(passes)
        total_bits = len(passes) * report["configuration"]["bits_per_block"]
        bit_errors = sum(
            error
            for record in report["paired_batch_outcomes"]
            if record["echo_doppler_hz"] == summary["echo_doppler_hz"]
            and record["channel_model"] == summary["channel_model"]
            for error in record["bit_errors_by_block"]
        )
        assert summary["blocks"] == len(passes)
        assert summary["crc_failures"] == failures
        assert summary["bler"] == pytest.approx(failures / len(passes))
        assert summary["bit_error_rate"] == pytest.approx(bit_errors / total_bits)

    continuous_zero = outcomes_for("0hz_continuous")
    continuous_750 = outcomes_for("750hz_continuous")
    symbol_static_750 = outcomes_for("750hz_symbol_static")
    assert report["summary_by_condition"]["0hz_continuous"]["bler"] <= 0.05
    assert report["acceptance"]["observed_paired_exact_p_value"] == pytest.approx(
        exact_paired_binomial_p_value(continuous_zero, continuous_750)
    )
    assert report["intra_symbol_acceptance"]["observed_paired_exact_p_value"] == pytest.approx(
        exact_paired_binomial_p_value(symbol_static_750, continuous_750)
    )
    assert report["intra_symbol_acceptance"]["observed_absolute_bler_increase"] < 0.10
    assert report["intra_symbol_acceptance"]["observed_paired_exact_p_value"] > 0.01
