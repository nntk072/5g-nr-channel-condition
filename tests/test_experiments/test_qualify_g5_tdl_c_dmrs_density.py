from __future__ import annotations

import copy

import pytest
from experiments.qualify_g5_tdl_c_dmrs_density import (
    CHANNEL_SEEDS,
    FRAME_COUNT,
    _goodput_by_channel,
    _validate_run_set,
)


def _run(position: int, *, mobile: bool, bit_hash_suffix: str = "same") -> dict:
    checks = [
        {
            "seed": seed,
            "channel_truth_sha256": f"channel-{seed}",
            "transmit_bits_sha256": f"bits-{seed}-{bit_hash_suffix}",
            "noise_real_seed": seed + 2_000,
            "noise_imag_seed": seed + 3_000,
        }
        for seed in CHANNEL_SEEDS
    ]
    return {
        "configuration": {
            "profile": "TDL-C",
            "dmrs_additional_position": position,
            "allow_cp_overrun": False,
            "cp_covers_channel_memory": True,
            "matched_realization_checks": checks,
        },
        "points": [
            {
                "snr_db": 18.0,
                "frame_error_mask": [False] * FRAME_COUNT,
            }
        ],
    }


def _matched_runs() -> tuple[dict, dict, dict]:
    static = _run(0, mobile=False)
    mobile_zero = _run(0, mobile=True)
    mobile_three = _run(3, mobile=True, bit_hash_suffix="different-tbs")
    return static, mobile_zero, mobile_three


def test_validate_run_set_accepts_seed_paired_different_tbs_payloads() -> None:
    static, mobile_zero, mobile_three = _matched_runs()
    runs = {
        "static_position_0": static,
        "mobile_position_0": mobile_zero,
        "mobile_position_3": mobile_three,
    }
    points = {name: run["points"][0] for name, run in runs.items()}

    checks = _validate_run_set(runs, points)

    assert len(checks["mobile_position_3"]) == FRAME_COUNT
    assert (
        checks["mobile_position_0"][0]["transmit_bits_sha256"]
        != checks["mobile_position_3"][0]["transmit_bits_sha256"]
    )


def test_validate_run_set_rejects_changed_mobile_channel_truth() -> None:
    static, mobile_zero, mobile_three = _matched_runs()
    mobile_three["configuration"]["matched_realization_checks"][0]["channel_truth_sha256"] = (
        "different-channel"
    )
    runs = {
        "static_position_0": static,
        "mobile_position_0": mobile_zero,
        "mobile_position_3": mobile_three,
    }
    points = {name: run["points"][0] for name, run in runs.items()}

    with pytest.raises(ValueError, match="share channel truth"):
        _validate_run_set(runs, points)


def test_validate_run_set_rejects_cp_overrun() -> None:
    static, mobile_zero, mobile_three = _matched_runs()
    static["configuration"]["cp_covers_channel_memory"] = False
    runs = {
        "static_position_0": static,
        "mobile_position_0": mobile_zero,
        "mobile_position_3": mobile_three,
    }
    points = {name: run["points"][0] for name, run in runs.items()}

    with pytest.raises(ValueError, match="not covered by the CP"):
        _validate_run_set(runs, points)


def test_validate_run_set_rejects_different_static_mobile_payload() -> None:
    static, mobile_zero, mobile_three = _matched_runs()
    static["configuration"]["matched_realization_checks"][0]["transmit_bits_sha256"] = (
        "different-payload"
    )
    runs = {
        "static_position_0": static,
        "mobile_position_0": mobile_zero,
        "mobile_position_3": mobile_three,
    }
    points = {name: run["points"][0] for name, run in runs.items()}

    with pytest.raises(ValueError, match="do not share payload bits"):
        _validate_run_set(runs, points)


def test_goodput_counts_failed_transport_blocks_as_zero_bits() -> None:
    point = {
        "frame_error_mask": [False, True, False],
        "mean_transmitted_tb_bits_per_codeword": 1_024,
    }

    assert _goodput_by_channel(point) == [1_024.0, 0.0, 1_024.0]


def test_validate_run_set_rejects_misaligned_seed_sequence() -> None:
    static, mobile_zero, mobile_three = _matched_runs()
    mobile_zero["configuration"]["matched_realization_checks"] = copy.deepcopy(
        mobile_zero["configuration"]["matched_realization_checks"]
    )
    mobile_zero["configuration"]["matched_realization_checks"][0]["seed"] += 1
    runs = {
        "static_position_0": static,
        "mobile_position_0": mobile_zero,
        "mobile_position_3": mobile_three,
    }
    points = {name: run["points"][0] for name, run in runs.items()}

    with pytest.raises(ValueError, match="frozen sequence"):
        _validate_run_set(runs, points)
