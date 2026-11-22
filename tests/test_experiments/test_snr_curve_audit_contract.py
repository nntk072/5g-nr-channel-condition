"""Torch-independent checks for retained G5 SNR-curve inference contracts."""

import hashlib
import json

import pytest
from experiments.audit_ordered_impairments_result import (
    _g5_execution_context_matches,
    _inference_seed_roster_consistent,
)


@pytest.mark.parametrize("version", (7, 8))
def test_acceptance_curve_audit_requires_the_exact_ordered_seed_roster(version: int) -> None:
    schema = f"umcd.g5_ordered_impairments_independent_channels_snr_curve.v{version}"
    assert _inference_seed_roster_consistent(
        schema,
        {"channel_seeds": [101, 103, 107]},
        [101, 103, 107],
    )


@pytest.mark.parametrize(
    "inference_seeds",
    (
        [103, 101, 107],
        [101, 103, 109],
        [101, True, 107],
        [101, 103, 103],
        [101, 103, -1],
        "101,103,107",
    ),
)
@pytest.mark.parametrize("version", (7, 8))
def test_acceptance_curve_audit_rejects_malformed_or_different_roster(
    version: int, inference_seeds: object
) -> None:
    schema = f"umcd.g5_ordered_impairments_independent_channels_snr_curve.v{version}"
    assert not _inference_seed_roster_consistent(
        schema,
        {"channel_seeds": inference_seeds},
        [101, 103, 107],
    )


@pytest.mark.parametrize("version", range(1, 7))
def test_historical_curve_audit_keeps_inference_shape_unchanged(version: int) -> None:
    schema = f"umcd.g5_ordered_impairments_independent_channels_snr_curve.v{version}"
    assert _inference_seed_roster_consistent(schema, {"cluster_count": 3}, [101, 103, 107])
    assert not _inference_seed_roster_consistent(
        schema,
        {"cluster_count": 3, "channel_seeds": [101, 103, 107]},
        [101, 103, 107],
    )


def _execution_bound_fixture() -> tuple[dict[str, object], list[dict[str, object]]]:
    config: dict[str, object] = {
        "receiver_configuration": {"receiver": "fixture"},
        "decoder_configuration": {"decoder": "fixture"},
        "sample_rate_hz": 30_720_000,
        "fft_size": 1024,
        "cyclic_prefix_samples": 72,
        "echo_delay_samples_range": [2, 6],
        "echo_amplitude_range": [0.2, 0.5],
        "direct_coefficient_real_imag": [0.9, -0.1],
        "cfo_hz": 250.0,
        "timing_offset_samples": 0,
        "tx_phase_noise_linewidth_hz": 100.0,
        "rx_phase_noise_linewidth_hz": 100.0,
        "tx_iq_gain_imbalance_db": 0.0,
        "tx_iq_phase_imbalance_rad": 0.0,
        "rx_iq_gain_imbalance_db": 0.0,
        "rx_iq_phase_imbalance_rad": 0.0,
    }
    waveform_keys = tuple(
        key
        for key in config
        if key
        not in {
            "receiver_configuration",
            "decoder_configuration",
        }
    )
    context = {
        "receiver_configuration": config["receiver_configuration"],
        "decoder_configuration": config["decoder_configuration"],
        "waveform_configuration": {
            "channel_family": "independently_seeded_static_two_path",
            **{key: config[key] for key in waveform_keys},
            "noise_variance_per_complex_sample": "10**(-snr_db/10)",
            "noise_reference": "unit_complex_waveform_power",
        },
    }
    contract = {
        "execution_context": context,
        "execution_context_sha256": hashlib.sha256(
            json.dumps(context, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    return contract, [{"experiment": {"config": config}}]


def test_v3_audit_binds_the_digest_and_each_point_execution_context() -> None:
    contract, points = _execution_bound_fixture()

    assert _g5_execution_context_matches(contract, points)

    altered_point = json.loads(json.dumps(points))
    altered_point[0]["experiment"]["config"]["cfo_hz"] = 251.0
    assert not _g5_execution_context_matches(contract, altered_point)

    altered_contract = json.loads(json.dumps(contract))
    altered_contract["execution_context_sha256"] = "0" * 64
    assert not _g5_execution_context_matches(altered_contract, points)
