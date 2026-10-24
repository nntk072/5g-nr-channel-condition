import hashlib
import json
import math
import random
from pathlib import Path

import pytest
from experiments.g5_independent_channel_receiver import (
    _source_fingerprints,
    clopper_pearson_upper_bound,
    cluster_bootstrap_interval,
)

MAXIMUM_REFERENCE_BLER = 0.05
MAXIMUM_REFERENCE_UPPER_BOUND = 0.10
MINIMUM_MOBILE_BLER_INCREASE = 0.10
MINIMUM_INTRA_SYMBOL_BLER_INCREASE = 0.10
FAMILYWISE_ALPHA = 0.01
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 73_999


def _independent_cluster_interval(
    values: list[float], *, seed: int, alpha: float
) -> tuple[float, float]:
    """Recompute the percentile interval without calling the experiment helper."""
    rng = random.Random(seed)
    count = len(values)
    estimates = sorted(
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(BOOTSTRAP_REPLICATES)
    )
    tail = alpha / 2
    return (
        estimates[math.floor(tail * BOOTSTRAP_REPLICATES)],
        estimates[math.ceil((1 - tail) * BOOTSTRAP_REPLICATES) - 1],
    )


def test_cluster_bootstrap_is_seeded_and_resamples_channel_units():
    values = [0.0, 0.0, 0.5, 1.0]
    interval = cluster_bootstrap_interval(values, seed=41, alpha=0.05, replicates=2_000)
    assert interval == cluster_bootstrap_interval(values, seed=41, alpha=0.05, replicates=2_000)
    assert interval[0] == 0.0
    assert interval[1] == 0.75


def test_cluster_bootstrap_rejects_invalid_design_parameters():
    with pytest.raises(ValueError, match="nonempty"):
        cluster_bootstrap_interval([], seed=1, alpha=0.05)
    with pytest.raises(ValueError, match="finite"):
        cluster_bootstrap_interval([math.nan], seed=1, alpha=0.05)
    with pytest.raises(ValueError, match="alpha"):
        cluster_bootstrap_interval([0.0], seed=1, alpha=1.0)
    with pytest.raises(ValueError, match="at least 100"):
        cluster_bootstrap_interval([0.0], seed=1, alpha=0.05, replicates=99)


def test_exact_channel_prevalence_bound_is_nonzero_for_zero_failures():
    upper = clopper_pearson_upper_bound(0, 64, alpha=0.005)
    assert upper == pytest.approx(1 - 0.005 ** (1 / 64))
    assert upper < 0.10
    assert clopper_pearson_upper_bound(64, 64, alpha=0.005) == 1.0
    with pytest.raises(ValueError, match="require"):
        clopper_pearson_upper_bound(2, 1, alpha=0.05)


def test_saved_independent_channel_report_has_cluster_level_reproducible_decision():
    report_path = (
        Path(__file__).resolve().parents[2]
        / "docs/research/results/g5-independent-channel-pusch-doppler-stress-20261024.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    recorded_hash = report.pop("result_sha256")
    actual_hash = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    assert report["schema"] == "umcd.g5_independent_channel_receiver.v3"
    assert recorded_hash == actual_hash
    assert report["source_sha256"] == _source_fingerprints()
    assert report["runtime"]["device"] == "cpu"
    configuration = report["configuration"]
    assert configuration["channel_draws"] == len(configuration["channel_seeds"]) == 64
    assert len(set(configuration["channel_seeds"])) == 64
    assert configuration["blocks_per_channel"] == 8
    assert configuration["bootstrap"]["resampling_unit"] == "independent_channel_draw"
    assert configuration["channel_distribution"]["echo_delay_samples_inclusive"] == [2, 8]
    assert configuration["channel_distribution"]["echo_doppler_hz_conditions"] == [750.0, 1500.0]

    records = report["channel_draw_records"]
    assert [record["channel_seed"] for record in records] == configuration["channel_seeds"]
    reference_rates: list[float] = []
    condition_names = (
        "750hz_symbol_static",
        "750hz_continuous",
        "1500hz_symbol_static",
        "1500hz_continuous",
    )
    rates = {condition: [] for condition in condition_names}
    for record in records:
        results = record["crc_pass_by_condition"]
        assert set(results) == {"static", *condition_names}
        assert all(len(blocks) == 8 for blocks in results.values())
        reference_rates.append(1 - sum(results["static"]) / 8)
        for condition in condition_names:
            rates[condition].append(1 - sum(results[condition]) / 8)

    per_interval_alpha = FAMILYWISE_ALPHA / 5
    channels_with_failures = sum(rate > 0 for rate in reference_rates)
    assert channels_with_failures == 0
    reference_upper_bound = 1 - per_interval_alpha ** (1 / len(reference_rates))
    result = report["acceptance"]
    reference_mean = sum(reference_rates) / len(reference_rates)
    assert result["reference_bler"] == pytest.approx(reference_mean)
    assert result["reference_channels_with_any_crc_failure"] == channels_with_failures
    assert result["reference_bler_upper_bound"] == pytest.approx(reference_upper_bound)
    assert result["familywise_confidence"] == 0.99
    assert result["interval_count"] == 5
    mobile_effect_pass = True
    intra_symbol_effect_pass = True
    for index, doppler in enumerate((750, 1500)):
        key = str(doppler)
        mobile_rates = rates[f"{key}hz_continuous"]
        symbol_static_rates = rates[f"{key}hz_symbol_static"]
        differences = [mobile - ref for ref, mobile in zip(reference_rates, mobile_rates)]
        difference_mean = sum(differences) / len(differences)
        difference_interval = _independent_cluster_interval(
            differences,
            seed=BOOTSTRAP_SEED + 1 + 2 * index,
            alpha=per_interval_alpha,
        )
        mobile = result["mobile_effect_by_doppler_hz"][key]
        assert mobile["bler"] == pytest.approx(sum(mobile_rates) / len(mobile_rates))
        assert mobile["paired_mean_bler_increase"] == pytest.approx(difference_mean)
        assert mobile["paired_increase_cluster_bootstrap_interval"] == pytest.approx(
            difference_interval
        )
        mobile_effect_pass &= (
            difference_mean >= MINIMUM_MOBILE_BLER_INCREASE and difference_interval[0] > 0
        )

        ici_differences = [
            continuous - symbol_static
            for symbol_static, continuous in zip(symbol_static_rates, mobile_rates)
        ]
        ici_mean = sum(ici_differences) / len(ici_differences)
        ici_interval = _independent_cluster_interval(
            ici_differences,
            seed=BOOTSTRAP_SEED + 2 + 2 * index,
            alpha=per_interval_alpha,
        )
        ici = result["intra_symbol_effect_by_doppler_hz"][key]
        assert ici["symbol_static_bler"] == pytest.approx(
            sum(symbol_static_rates) / len(symbol_static_rates)
        )
        assert ici["bler_increase"] == pytest.approx(ici_mean)
        assert ici["increase_cluster_bootstrap_interval"] == pytest.approx(ici_interval)
        intra_symbol_effect_pass &= (
            ici_mean >= MINIMUM_INTRA_SYMBOL_BLER_INCREASE and ici_interval[0] > 0
        )

    mobile_effect_pass &= (
        reference_mean <= MAXIMUM_REFERENCE_BLER
        and reference_upper_bound <= MAXIMUM_REFERENCE_UPPER_BOUND
    )
    assert result["mobile_effect_pass"] is mobile_effect_pass
    assert result["intra_symbol_effect_pass"] is intra_symbol_effect_pass
    assert result["pass"] is (mobile_effect_pass and intra_symbol_effect_pass)
