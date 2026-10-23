"""Tests for the bounded standardized PUSCH CP-exceedance diagnostic."""

from __future__ import annotations

import pytest
from experiments.standardized_pusch_cp_exceedance import (
    _paired_table,
    run_cp_exceedance_probe,
    wilson_interval,
)


def test_wilson_interval_has_valid_bounds_at_extreme_outcomes() -> None:
    assert wilson_interval(0, 16) == pytest.approx((0.0, 0.1936076805))
    assert wilson_interval(16, 16) == pytest.approx((0.8063923195, 1.0))
    with pytest.raises(ValueError, match="0 <= successes <= trials"):
        wilson_interval(17, 16)


def test_paired_pass_table_accounts_for_every_trial() -> None:
    table = _paired_table([True, True, False, False], [True, False, True, False])
    assert table == {
        "both_pass": 1,
        "left_only_pass": 1,
        "right_only_pass": 1,
        "both_fail": 1,
    }


@pytest.mark.native_sionna
def test_repeated_cp_exceedance_probe_keeps_parent_and_oracle_semantics() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("sionna.phy")

    report = run_cp_exceedance_probe(samples=4)

    assert report["status"].startswith("exploratory")
    assert report["configuration"]["trial_unit"].startswith("independent TDL draw")
    assert (
        report["configuration"]["max_excess_delay_s"]
        > report["configuration"]["cyclic_prefix_duration_s"]
    )
    assert len(report["configuration_sha256"]) == 64
    for condition in report["outcomes"].values():
        assert condition["independent_channel_draws"] == 4
        assert len(condition["crc_pass_by_draw"]) == 4
        assert condition["payload_mismatches"] == condition["block_errors"]
        assert 0.0 <= condition["wilson_95_percent_interval"][0]
        assert condition["wilson_95_percent_interval"][1] <= 1.0
    for paired in report["paired_pass_table"].values():
        assert sum(paired.values()) == 4
