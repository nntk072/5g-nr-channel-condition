from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import pytest
from scipy.stats import beta
from tests.test_experiments.frozen_g5_source_manifest import assert_frozen_source_manifest


def _bootstrap_interval(values: list[float], *, seed: int, alpha: float) -> tuple[float, float]:
    rng = random.Random(seed)
    count = len(values)
    replicates = 20_000
    estimates = sorted(
        sum(values[rng.randrange(count)] for _ in range(count)) / count for _ in range(replicates)
    )
    tail = alpha / 2.0
    return (
        estimates[math.floor(tail * replicates)],
        estimates[min(replicates - 1, math.ceil((1.0 - tail) * replicates) - 1)],
    )


def test_confirmatory_dmrs_density_result_is_source_and_protocol_bound():
    root = Path(__file__).resolve().parents[2]
    result_path = root / "docs/research/results/g5-doppler-dmrs-density-confirm-20261024.json"
    report = json.loads(result_path.read_text(encoding="utf-8"))
    digest = report.pop("result_sha256")
    actual_digest = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    protocol = root / report["protocol_path"]
    development = root / "docs/research/results/g5-doppler-dmrs-density-scan-20261024.json"
    development_report = json.loads(development.read_text(encoding="utf-8"))

    assert report["schema"] == "umcd.g5_doppler_dmrs_density_confirmatory.v1"
    assert report["status"] == "NOT_MET"
    assert report["g5_09_closed"] is False
    assert digest == actual_digest
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == report["protocol_sha256"]
    assert_frozen_source_manifest(
        report,
        expected_paths=(
            "experiments/g5_independent_channel_receiver.py",
            "experiments/qualify_g5_doppler_dmrs_density.py",
            "experiments/scan_g5_doppler_dmrs_density.py",
            "integrations/sionna/nr_pusch.py",
            "integrations/sionna/ofdm_impairments.py",
            "receivers/nr/pusch.py",
        ),
    )
    assert report["development_scan_result_sha256"] == development_report["result_sha256"]
    assert (
        report["configuration_sha256"]
        == hashlib.sha256(
            json.dumps(report["configuration"], sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
    )


def test_confirmatory_dmrs_density_claims_recompute_from_channel_records():
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/research/results/g5-doppler-dmrs-density-confirm-20261024.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    records = report["confirmatory_channel_draw_records"]
    config = report["configuration"]
    acceptance = report["acceptance"]
    alpha = acceptance["per_claim_alpha"]
    block_count = config["blocks_per_channel"]

    assert [record["channel_seed"] for record in records] == list(range(85_001, 85_065))
    assert len(records) == config["channel_draws"] == 64
    assert set(records[0]["configuration_results"]) == {"0", "3"}
    for record in records:
        result_0 = record["configuration_results"]["0"]
        result_3 = record["configuration_results"]["3"]
        assert record["channel_seed"] not in range(84_001, 84_065)
        assert result_0["payload_seed"] == result_3["payload_seed"]
        assert result_0["waveform_noise_sha256"] == result_3["waveform_noise_sha256"]
        assert set(result_0["crc_pass_by_condition"]) == {"static", "750hz_continuous"}
        assert all(
            len(outcomes) == block_count
            for result in (result_0, result_3)
            for outcomes in result["crc_pass_by_condition"].values()
        )

    bler_0_static = [
        1.0
        - sum(record["configuration_results"]["0"]["crc_pass_by_condition"]["static"]) / block_count
        for record in records
    ]
    bler_0_mobile = [
        1.0
        - sum(record["configuration_results"]["0"]["crc_pass_by_condition"]["750hz_continuous"])
        / block_count
        for record in records
    ]
    bler_3_mobile = [
        1.0
        - sum(record["configuration_results"]["3"]["crc_pass_by_condition"]["750hz_continuous"])
        / block_count
        for record in records
    ]
    delta_bler = [low - high for low, high in zip(bler_0_mobile, bler_3_mobile, strict=True)]
    tbs_0 = records[0]["configuration_results"]["0"]["transport_block_bits"]
    tbs_3 = records[0]["configuration_results"]["3"]["transport_block_bits"]
    delta_goodput = [
        sum(record["configuration_results"]["3"]["crc_pass_by_condition"]["750hz_continuous"])
        * tbs_3
        / block_count
        - sum(record["configuration_results"]["0"]["crc_pass_by_condition"]["750hz_continuous"])
        * tbs_0
        / block_count
        for record in records
    ]

    static_failures = sum(rate > 0 for rate in bler_0_static)
    expected_upper = float(
        beta.ppf(1.0 - alpha, static_failures + 1, len(records) - static_failures)
    )
    assert acceptance["static_reference"]["additional_position_0_mean_bler"] == pytest.approx(
        sum(bler_0_static) / len(records)
    )
    assert acceptance["static_reference"]["channels_with_any_failure"] == static_failures
    assert acceptance["static_reference"][
        "one_sided_exact_family_adjusted_channel_failure_upper_bound"
    ] == pytest.approx(expected_upper)
    assert acceptance["mobile_bler_reduction_position_0_minus_3"][
        "equal_channel_mean"
    ] == pytest.approx(sum(delta_bler) / len(records))
    assert acceptance["mobile_bler_reduction_position_0_minus_3"][
        "family_adjusted_cluster_bootstrap_interval"
    ] == pytest.approx(_bootstrap_interval(delta_bler, seed=85_099, alpha=alpha))
    assert acceptance["mobile_goodput_increase_position_3_minus_0_bits_per_slot"][
        "family_adjusted_cluster_bootstrap_interval"
    ] == pytest.approx(_bootstrap_interval(delta_goodput, seed=85_100, alpha=alpha))
    assert acceptance["static_reference"]["pass"] is False
    assert acceptance["mobile_bler_reduction_position_0_minus_3"]["pass"] is True
    assert acceptance["mobile_goodput_increase_position_3_minus_0_bits_per_slot"]["pass"] is True
