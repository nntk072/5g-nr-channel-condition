import hashlib
import json
from pathlib import Path

import pytest
from experiments.g5_doppler_awgn_receiver import (
    exact_paired_binomial_p_value,
    run_doppler_awgn_probe,
    wilson_interval,
)
from scipy.stats import binomtest
from tests.test_experiments.frozen_g5_source_manifest import assert_frozen_source_manifest

# Frozen in Unified Plan edition 3.246. Keep these values independent of the
# experiment module so producer changes cannot redefine the saved-report audit.
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
    assert_frozen_source_manifest(
        report,
        expected_paths=(
            "experiments/g5_doppler_awgn_receiver.py",
            "integrations/sionna/nr_pusch.py",
            "integrations/sionna/ofdm_impairments.py",
            "receivers/nr/pusch.py",
        ),
    )
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
    expected_batch_seeds = report["configuration"]["batch_seeds"]
    assert len(set(expected_batch_seeds)) == len(expected_batch_seeds)
    records_by_seed = {record["batch_seed"]: record for record in report["paired_batch_outcomes"]}
    assert len(records_by_seed) == len(report["paired_batch_outcomes"])
    assert set(records_by_seed) == set(expected_batch_seeds)
    expected_noise_seed_by_batch = dict(
        zip(expected_batch_seeds, report["configuration"]["waveform_noise_seeds"], strict=True)
    )
    for seed, record in records_by_seed.items():
        assert record["waveform_noise_seed"] == expected_noise_seed_by_batch[seed]
        assert set(record["conditions"]) == set(report["summary_by_condition"])
        lengths = {
            len(condition["crc_pass_by_block"]) for condition in record["conditions"].values()
        }
        assert lengths == {report["configuration"]["blocks_per_seed"]}

    def outcomes_for(condition: str) -> dict[tuple[int, int], bool]:
        return {
            (seed, block_index): passed
            for seed, record in records_by_seed.items()
            for block_index, passed in enumerate(
                record["conditions"][condition]["crc_pass_by_block"]
            )
        }

    for condition, summary in report["summary_by_condition"].items():
        keyed_passes = outcomes_for(condition)
        passed = list(keyed_passes.values())
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
    paired_keys = set(static)
    assert paired_keys == set(mobile) == set(symbol_static)
    ordered_keys = sorted(paired_keys)
    static_outcomes = [static[key] for key in ordered_keys]
    mobile_outcomes = [mobile[key] for key in ordered_keys]
    symbol_static_outcomes = [symbol_static[key] for key in ordered_keys]
    baseline_bler = report["summary_by_condition"]["0hz_continuous"]["bler"]
    mobile_bler = report["summary_by_condition"]["750hz_continuous"]["bler"]
    symbol_static_bler = report["summary_by_condition"]["750hz_symbol_static"]["bler"]
    acceptance = report["acceptance"]
    total_effect = mobile_bler - baseline_bler
    total_p = _paired_binomial_reference(static_outcomes, mobile_outcomes)
    assert acceptance["maximum_static_reference_bler"] == MAXIMUM_REFERENCE_BLER
    assert acceptance["minimum_absolute_bler_increase"] == MINIMUM_TOTAL_DOPPLER_BLER_INCREASE
    assert acceptance["maximum_two_sided_exact_p_value"] == MAXIMUM_TOTAL_PAIRED_P_VALUE
    assert acceptance["observed_absolute_bler_increase"] == pytest.approx(total_effect)
    assert acceptance["observed_paired_exact_p_value"] == pytest.approx(total_p)
    assert acceptance["pass"] is (
        baseline_bler <= MAXIMUM_REFERENCE_BLER
        and total_effect >= MINIMUM_TOTAL_DOPPLER_BLER_INCREASE
        and total_p <= MAXIMUM_TOTAL_PAIRED_P_VALUE
    )

    intra = report["intra_symbol_acceptance"]
    intra_effect = mobile_bler - symbol_static_bler
    intra_p = _paired_binomial_reference(symbol_static_outcomes, mobile_outcomes)
    assert intra["minimum_absolute_bler_increase"] == MINIMUM_INTRA_SYMBOL_BLER_INCREASE
    assert intra["maximum_two_sided_exact_p_value"] == MAXIMUM_INTRA_SYMBOL_PAIRED_P_VALUE
    assert intra["observed_absolute_bler_increase"] == pytest.approx(intra_effect)
    assert intra["observed_paired_exact_p_value"] == pytest.approx(intra_p)
    assert intra["pass"] is (
        intra_effect >= MINIMUM_INTRA_SYMBOL_BLER_INCREASE
        and intra_p <= MAXIMUM_INTRA_SYMBOL_PAIRED_P_VALUE
    )

    for record in report["paired_batch_outcomes"]:
        noise = record["conditions"]["0hz_continuous"]["mean_realized_awgn_power"]
        for condition in report["summary_by_condition"]:
            assert record["conditions"][condition]["mean_realized_awgn_power"] == noise
