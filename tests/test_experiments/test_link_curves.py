import pytest
from experiments.link_curves import (
    run_tdl_pusch_bler_curve,
    run_tdl_pusch_rx_diversity_curve,
    run_tdl_pusch_time_bler_curve,
    wilson_score_interval_95,
)


def test_wilson_interval_has_finite_upper_bound_for_zero_observed_errors():
    lower, upper = wilson_score_interval_95(0, 100)

    assert lower == 0.0
    assert 0.03 < upper < 0.04


def test_wilson_interval_is_symmetric_at_half_error_rate():
    lower, upper = wilson_score_interval_95(50, 100)

    assert lower == pytest.approx(1.0 - upper)
    assert lower < 0.5 < upper


@pytest.mark.parametrize(
    ("errors", "total", "error"),
    [
        (True, 10, TypeError),
        (1.0, 10, TypeError),
        (-1, 10, ValueError),
        (11, 10, ValueError),
        (0, 0, ValueError),
    ],
)
def test_wilson_interval_rejects_invalid_counts(errors, total, error):
    with pytest.raises(error):
        wilson_score_interval_95(errors, total)


def test_link_curve_rejects_empty_snr_grid_before_loading_native_dependencies():
    with pytest.raises(ValueError, match="snr_db"):
        run_tdl_pusch_bler_curve(
            snr_db=[],
            frames_per_snr=1,
            batch_size=1,
            seed=1,
        )


def test_sample_domain_link_curve_rejects_empty_snr_grid_before_loading_dependencies():
    with pytest.raises(ValueError, match="snr_db"):
        run_tdl_pusch_time_bler_curve(
            snr_db=[],
            frames_per_snr=1,
            batch_size=1,
            seed=1,
        )


def test_rx_diversity_curve_rejects_empty_snr_grid_before_loading_dependencies():
    with pytest.raises(ValueError, match="snr_db"):
        run_tdl_pusch_rx_diversity_curve(
            snr_db=[],
            frames_per_snr=1,
            batch_size=1,
            seed=1,
        )


@pytest.mark.parametrize("position", [-1, 4, True])
def test_sample_domain_link_curve_rejects_invalid_dmrs_position(position):
    expected = TypeError if isinstance(position, bool) else ValueError
    with pytest.raises(expected, match="dmrs_additional_position"):
        run_tdl_pusch_time_bler_curve(
            snr_db=(10.0,),
            frames_per_snr=1,
            batch_size=1,
            seed=1,
            dmrs_additional_position=position,
        )


@pytest.mark.parametrize(
    ("num_rx_ant", "error"),
    [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)],
)
def test_sample_domain_link_curve_rejects_invalid_receive_antenna_count(num_rx_ant, error):
    with pytest.raises(error, match="num_rx_ant"):
        run_tdl_pusch_time_bler_curve(
            snr_db=(10.0,),
            frames_per_snr=1,
            batch_size=1,
            seed=1,
            num_rx_ant=num_rx_ant,
        )
