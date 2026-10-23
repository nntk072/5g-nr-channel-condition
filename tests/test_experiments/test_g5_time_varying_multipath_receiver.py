import hashlib
import json
from pathlib import Path

import pytest
from experiments.g5_time_varying_multipath_receiver import (
    _source_fingerprints,
    exact_paired_binomial_p_value,
    wilson_interval,
)
from scipy.stats import binomtest

# Frozen in Unified Plan editions 3.241 and 3.244. Keep these values
# independent of the experiment module so producer changes cannot redefine
# the criteria used to audit its saved report.
MAXIMUM_REFERENCE_BLER = 0.05
MINIMUM_TOTAL_DOPPLER_BLER_INCREASE = 0.20
MAXIMUM_TOTAL_PAIRED_P_VALUE = 0.01
MINIMUM_INTRA_SYMBOL_BLER_INCREASE = 0.10
MAXIMUM_INTRA_SYMBOL_PAIRED_P_VALUE = 0.01


def _paired_binomial_reference(reference: list[bool], comparison: list[bool]) -> float:
    """Independent exact two-sided McNemar/binomial reference from discordant pairs."""
    assert len(reference) == len(comparison)
    reference_only = sum(a and not b for a, b in zip(reference, comparison, strict=True))
    comparison_only = sum(not a and b for a, b in zip(reference, comparison, strict=True))
    discordant = reference_only + comparison_only
    if discordant == 0:
        return 1.0
    return float(binomtest(reference_only, discordant, p=0.5, alternative="two-sided").pvalue)


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
    assert len(report["paired_batch_outcomes"]) == 12 * 5

    # Build each observation by its saved payload-batch seed and within-batch
    # block index. The report's record order must not determine the pairing.
    by_condition: dict[str, dict[tuple[int, int], bool]] = {}
    for record in report["paired_batch_outcomes"]:
        condition = f"{record['echo_doppler_hz']:g}hz_{record['channel_model']}"
        observations = by_condition.setdefault(condition, {})
        for block_index, passed in enumerate(record["crc_pass_by_block"]):
            key = (record["batch_seed"], block_index)
            assert key not in observations
            observations[key] = passed

    expected_batch_seeds = set(report["configuration"]["batch_seeds"])
    assert len(expected_batch_seeds) == len(report["configuration"]["batch_seeds"])
    assert set(by_condition) == set(report["summary_by_condition"])
    for condition, observations in by_condition.items():
        assert {seed for seed, _ in observations} == expected_batch_seeds
        assert (
            len(observations)
            == len(expected_batch_seeds) * report["configuration"]["blocks_per_seed"]
        )

    def outcomes_for(condition: str) -> dict[tuple[int, int], bool]:
        return by_condition[condition]

    for condition, summary in report["summary_by_condition"].items():
        keyed_passes = outcomes_for(condition)
        passes = list(keyed_passes.values())
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
    paired_keys = set(continuous_zero)
    assert paired_keys == set(continuous_750) == set(symbol_static_750)
    ordered_keys = sorted(paired_keys)
    zero_outcomes = [continuous_zero[key] for key in ordered_keys]
    mobile_outcomes = [continuous_750[key] for key in ordered_keys]
    symbol_static_outcomes = [symbol_static_750[key] for key in ordered_keys]
    baseline_bler = report["summary_by_condition"]["0hz_continuous"]["bler"]
    continuous_bler = report["summary_by_condition"]["750hz_continuous"]["bler"]
    symbol_static_bler = report["summary_by_condition"]["750hz_symbol_static"]["bler"]
    acceptance = report["acceptance"]
    observed_effect = continuous_bler - baseline_bler
    observed_p = _paired_binomial_reference(zero_outcomes, mobile_outcomes)
    assert acceptance["maximum_reference_bler"] == MAXIMUM_REFERENCE_BLER
    assert acceptance["minimum_absolute_bler_increase"] == MINIMUM_TOTAL_DOPPLER_BLER_INCREASE
    assert acceptance["maximum_two_sided_exact_p_value"] == MAXIMUM_TOTAL_PAIRED_P_VALUE
    assert acceptance["observed_absolute_bler_increase"] == pytest.approx(observed_effect)
    assert acceptance["observed_paired_exact_p_value"] == pytest.approx(observed_p)
    assert acceptance["pass"] is (
        baseline_bler <= MAXIMUM_REFERENCE_BLER
        and observed_effect >= MINIMUM_TOTAL_DOPPLER_BLER_INCREASE
        and observed_p <= MAXIMUM_TOTAL_PAIRED_P_VALUE
    )

    intra = report["intra_symbol_acceptance"]
    intra_effect = continuous_bler - symbol_static_bler
    intra_p = _paired_binomial_reference(symbol_static_outcomes, mobile_outcomes)
    assert intra["minimum_absolute_bler_increase"] == MINIMUM_INTRA_SYMBOL_BLER_INCREASE
    assert intra["maximum_two_sided_exact_p_value"] == MAXIMUM_INTRA_SYMBOL_PAIRED_P_VALUE
    assert intra["observed_absolute_bler_increase"] == pytest.approx(intra_effect)
    assert intra["observed_paired_exact_p_value"] == pytest.approx(intra_p)
    assert intra["pass"] is (
        intra_effect >= MINIMUM_INTRA_SYMBOL_BLER_INCREASE
        and intra_p <= MAXIMUM_INTRA_SYMBOL_PAIRED_P_VALUE
    )
