"""NR timing and low-confidence audit inputs: typed case, numerology, slot and threshold rules."""

from __future__ import annotations

import numpy as np
import pytest
from evaluation.scoring.low_confidence import build_low_confidence_audit
from integrations.sionna.nr_timing import nr_normal_cp_symbol_start_times
from integrations.sionna.ssb_timing import ssb_candidate_symbols


@pytest.mark.parametrize(
    ("case", "message"),
    [
        pytest.param(3, "case must be one of A through G", id="case-type"),
        pytest.param("Z", "case must be one of A through G", id="case-unknown"),
    ],
)
def test_ssb_case_must_be_a_known_letter(case, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        ssb_candidate_symbols(case, 3.5e9, subcarrier_spacing_hz=30e3, frequency_range="FR1")


def test_ssb_paired_spectrum_flag_must_be_boolean() -> None:
    with pytest.raises(TypeError, match="paired_spectrum must be a bool"):
        ssb_candidate_symbols(
            "A", 3.5e9, subcarrier_spacing_hz=15e3, frequency_range="FR1", paired_spectrum="yes"
        )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        pytest.param(
            {"num_symbols": 2.0}, TypeError, "num_symbols must be an integer", id="symbols-type"
        ),
        pytest.param({"num_symbols": 15}, ValueError, "1 to 14 symbols", id="symbols-range"),
        pytest.param(
            {"subcarrier_spacing_hz": 20e3}, ValueError, "15 kHz times 2\\*\\*mu", id="numerology"
        ),
        pytest.param(
            {"slot_index_in_subframe": 1.5},
            TypeError,
            "slot_index_in_subframe must be an integer",
            id="slot-type",
        ),
    ],
)
def test_symbol_schedule_rejects_invalid_numerology_and_slots(kwargs, error, message) -> None:
    arguments = {"subcarrier_spacing_hz": 15e3, "num_symbols": 14}
    arguments.update(kwargs)
    with pytest.raises(error, match=message):
        nr_normal_cp_symbol_start_times(**arguments)


def test_symbol_schedule_is_monotone_within_a_slot() -> None:
    starts = nr_normal_cp_symbol_start_times(30e3, 14)
    assert len(starts) == 14
    assert all(later > earlier for earlier, later in zip(starts, starts[1:], strict=False))


def test_low_confidence_audit_validates_ids_and_threshold() -> None:
    probabilities = np.asarray([0.2, 0.8])
    ids = ["g0", "g1"]
    with pytest.raises(ValueError, match="canonical nonempty strings"):
        build_low_confidence_audit(
            probabilities, [" g0", "g1"], ["f", "f"], ["d", "d"], threshold=0.5, margin=0.1
        )
    with pytest.raises(ValueError, match="threshold must be finite and in"):
        build_low_confidence_audit(
            probabilities, ids, ["f", "f"], ["d", "d"], threshold=1.5, margin=0.1
        )
