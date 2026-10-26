from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from experiments.scan_g5_doppler_dmrs_density import (
    CHANNEL_CONDITIONS,
    _validate_design,
)
from tests.test_experiments.frozen_g5_source_manifest import assert_frozen_source_manifest


@pytest.mark.parametrize(
    ("seeds", "blocks", "error", "message"),
    [
        ((), 1, ValueError, "channel_seeds"),
        ((1, 1), 1, ValueError, "channel_seeds"),
        ((True,), 1, ValueError, "channel_seeds"),
        ((1,), 0, ValueError, "positive"),
        ((1,), True, TypeError, "integer"),
    ],
)
def test_dmrs_density_scan_rejects_invalid_design(seeds, blocks, error, message):
    with pytest.raises(error, match=message):
        _validate_design(seeds, blocks)


def test_saved_dmrs_density_scan_preserves_pairing_and_goodput_accounting():
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/research/results/g5-doppler-dmrs-density-scan-20261024.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    digest = report.pop("result_sha256")
    actual = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert report["schema"] == "umcd.g5_doppler_dmrs_density_scan.v1"
    assert report["status"] == "DEVELOPMENT_ONLY_NOT_QUALIFIED"
    assert digest == actual
    assert_frozen_source_manifest(
        report,
        expected_paths=(
            "experiments/scan_g5_doppler_dmrs_density.py",
            "integrations/sionna/nr_pusch.py",
            "integrations/sionna/ofdm_impairments.py",
            "receivers/nr/pusch.py",
        ),
    )

    config = report["configuration"]
    seeds = config["channel_seeds"]
    assert seeds == list(range(84_001, 84_065))
    assert len(set(seeds)) == config["channel_draws"]
    assert config["noise_seed_offset"] == 2_000_000
    assert config["waveform_awgn_complex_variance"] == 0.001

    records = report["channel_draw_records"]
    assert [record["channel_seed"] for record in records] == seeds
    assert set(CHANNEL_CONDITIONS) == {"static", "750hz_continuous"}
    for record in records:
        config_results = record["configuration_results"]
        assert set(config_results) == {"0", "3"}
        assert config_results["0"]["payload_seed"] == config_results["3"]["payload_seed"]
        assert (
            config_results["0"]["waveform_noise_sha256"]
            == config_results["3"]["waveform_noise_sha256"]
        )
        assert all(
            len(outcomes) == config["blocks_per_channel"]
            for result in config_results.values()
            for outcomes in result["crc_pass_by_condition"].values()
        )

    for position in ("0", "3"):
        for condition in CHANNEL_CONDITIONS:
            passed = [
                value
                for record in records
                for value in record["configuration_results"][position]["crc_pass_by_condition"][
                    condition
                ]
            ]
            summary = report["summary_by_dmrs_additional_position"][position][condition]
            failures = len(passed) - sum(passed)
            tbs_values = {
                record["configuration_results"][position]["transport_block_bits"]
                for record in records
            }
            assert len(tbs_values) == 1
            tbs = tbs_values.pop()
            assert summary["crc_failures"] == failures
            assert summary["blocks"] == len(passed)
            assert summary["transport_block_bits_per_payload"] == tbs
            assert summary["pooled_bler_descriptive"] == pytest.approx(failures / len(passed))
            assert summary["successful_transport_block_bits_per_slot_descriptive"] == pytest.approx(
                (len(passed) - failures) * tbs / len(passed)
            )
