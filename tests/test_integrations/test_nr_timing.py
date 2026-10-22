"""3GPP normal-CP NR symbol-timing contract tests."""

import pytest
from integrations.sionna.nr_timing import nr_normal_cp_symbol_start_times


def test_15_khz_slot_has_two_long_cp_symbol_intervals() -> None:
    times = nr_normal_cp_symbol_start_times(15_000.0, 14)
    useful_s = 1.0 / 15_000.0
    normal_cp_s = 144.0 / (15_000.0 * 2048.0)
    long_cp_s = 160.0 / (15_000.0 * 2048.0)

    assert times[0] == 0.0
    assert times[1] - times[0] == pytest.approx(useful_s + long_cp_s)
    assert times[7] - times[6] == pytest.approx(useful_s + normal_cp_s)
    assert times[8] - times[7] == pytest.approx(useful_s + long_cp_s)
    assert times[-1] < 1e-3


def test_long_cp_position_depends_on_slot_within_subframe() -> None:
    first_slot = nr_normal_cp_symbol_start_times(60_000.0, 14, slot_index_in_subframe=0)
    second_slot = nr_normal_cp_symbol_start_times(60_000.0, 14, slot_index_in_subframe=1)
    third_slot = nr_normal_cp_symbol_start_times(60_000.0, 14, slot_index_in_subframe=2)
    useful_s = 1.0 / 60_000.0
    normal_cp_s = 144.0 / (4.0 * 15_000.0 * 2048.0)
    long_cp_s = (144.0 / 4.0 + 16.0) / (15_000.0 * 2048.0)

    assert first_slot[1] - first_slot[0] == pytest.approx(useful_s + long_cp_s)
    assert second_slot[1] - second_slot[0] == pytest.approx(useful_s + normal_cp_s)
    assert third_slot[1] - third_slot[0] == pytest.approx(useful_s + long_cp_s)


@pytest.mark.parametrize(
    ("spacing_hz", "num_symbols", "slot_index"),
    [
        (0.0, 14, 0),
        (float("nan"), 14, 0),
        (30_000.0, 0, 0),
        (30_000.0, 15, 0),
        (30_000.0, 14, 2),
        (45_000.0, 14, 0),
    ],
)
def test_nr_normal_cp_timing_rejects_invalid_schedule(
    spacing_hz: float, num_symbols: int, slot_index: int
) -> None:
    with pytest.raises(ValueError):
        nr_normal_cp_symbol_start_times(spacing_hz, num_symbols, slot_index_in_subframe=slot_index)
