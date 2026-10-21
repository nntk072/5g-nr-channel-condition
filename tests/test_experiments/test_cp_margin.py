"""CPU tests for paired CP-margin result assembly."""

from __future__ import annotations

import pytest
from experiments import cp_margin


def test_holm_adjust_p_values_applies_step_down_and_preserves_names():
    adjusted = cp_margin.holm_adjust_p_values(
        {"largest": 0.04, "small": 0.01, "middle": 0.03, "tiny": 0.002}
    )

    assert adjusted == {"tiny": 0.008, "small": 0.03, "middle": 0.06, "largest": 0.06}


def test_holm_adjust_p_values_rejects_invalid_probability():
    try:
        cp_margin.holm_adjust_p_values({"invalid": 1.01})
    except ValueError as error:
        assert "finite values in [0, 1]" in str(error)
    else:
        raise AssertionError("invalid p-value must be rejected")


def test_cp_margin_wrapper_summarizes_matched_conditions(monkeypatch):
    native_result = {
        "configuration": {
            "seed": 7,
            "noise_variance_per_complex_sample": 0.001,
        },
        "receive_waveform_mean_power_by_batch": {
            "inside_cp": [0.99],
            "beyond_cp": [1.01],
        },
        "points": {
            "inside_cp": {
                "frames": 4,
                "bler_errors": 1,
                "bit_errors": 2,
                "bits": 400,
                "frame_error_mask": [False, False, False, True],
            },
            "beyond_cp": {
                "frames": 4,
                "bler_errors": 4,
                "bit_errors": 120,
                "bits": 400,
                "frame_error_mask": [True, True, True, True],
            },
        },
        "limitations": ["controlled test fixture"],
    }
    monkeypatch.setattr(cp_margin, "_run_native", lambda **_: native_result)

    result = cp_margin.run_static_cp_margin_experiment()

    assert result["schema"] == "loslab.pusch_static_cp_margin.v1"
    assert result["points"]["inside_cp"]["bler"] == 0.25
    assert result["points"]["beyond_cp"]["bler"] == 1.0
    assert result["points"]["inside_cp"]["measured_receive_snr_db_by_batch"]
    assert result["paired_comparison"]["treatment_minus_control_bler"] == 0.75
    assert result["paired_comparison"]["mcnemar_exact_two_sided_p_value_holm"] == pytest.approx(
        0.25
    )
    assert result["configuration"]["package_source_sha256"]
