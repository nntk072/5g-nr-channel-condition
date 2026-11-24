import pytest
from experiments.doppler_awgn_receiver import (
    exact_paired_binomial_p_value,
    run_doppler_awgn_probe,
    wilson_interval,
)
from scipy.stats import binomtest

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
