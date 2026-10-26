from __future__ import annotations

from typing import Any

import pytest
from experiments import audit_g5_tdl_dmrs_density_development as audit


def _fake_result(
    *, seed: int, frames: int, speed: float, position: int, mismatch: bool = False
) -> dict[str, Any]:
    records = [
        {
            "seed": seed + index,
            "noise_real_seed": seed + index + 2,
            "noise_imag_seed": seed + index + 3,
            "channel_truth_sha256": f"channel-{index + int(mismatch)}",
        }
        for index in range(frames)
    ]
    return {
        "configuration": {"device": "cpu", "matched_realization_checks": records},
        "points": [
            {
                "snr_db": audit.SNR_DB,
                "frame_error_mask": [False] * frames,
                "frames": frames,
                "bler_errors": 0,
            }
        ],
        "test_inputs": (speed, position),
    }


def test_development_audit_records_exploratory_conditions(monkeypatch: pytest.MonkeyPatch) -> None:
    seed = 932001

    def run(**kwargs: Any) -> dict[str, Any]:
        return _fake_result(
            seed=kwargs["seed"],
            frames=kwargs["frames_per_snr"],
            speed=kwargs["speed_mps"],
            position=kwargs["dmrs_additional_position"],
        )

    monkeypatch.setattr(audit, "run_tdl_pusch_time_bler_curve", run)
    report = audit.run_development_audit(frames=4, seed=seed, draws=20)

    assert report["status"] == "DEVELOPMENT_ONLY_NOT_QUALIFIED"
    assert report["protocol"]["channel_seeds"] == list(range(seed, seed + 4))
    assert set(report["conditions"]) == {
        "static_position_0",
        "mobile_position_0",
        "mobile_position_3",
    }
    assert set(report["paired_comparisons"]) == {
        "mobile_position_0_minus_static_position_0",
        "mobile_position_3_minus_mobile_position_0",
    }


def test_development_audit_rejects_unmatched_mobile_channel_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def run(**kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _fake_result(
            seed=kwargs["seed"],
            frames=kwargs["frames_per_snr"],
            speed=kwargs["speed_mps"],
            position=kwargs["dmrs_additional_position"],
            mismatch=calls == 3,
        )

    monkeypatch.setattr(audit, "run_tdl_pusch_time_bler_curve", run)
    with pytest.raises(ValueError, match="did not share exact channel truth"):
        audit.run_development_audit(frames=4, seed=932001, draws=20)
