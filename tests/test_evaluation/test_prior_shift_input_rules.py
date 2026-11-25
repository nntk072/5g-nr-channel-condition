"""Prior-shift correction: probabilities, prevalences, labels and numeric array parsing are validated."""

from __future__ import annotations

import numpy as np
import pytest
from evaluation.scoring.numeric import finite_real_array
from evaluation.scoring.prior_shift import adjust_probabilities_between_priors


@pytest.mark.parametrize(
    "scores",
    [
        np.asarray([[0.2, 0.4]]),
        np.asarray([0.2, np.nan]),
        np.asarray([0.2, 1.5]),
    ],
    ids=["two-dimensional", "nonfinite", "out-of-range"],
)
def test_probabilities_must_be_a_finite_unit_vector(scores) -> None:
    with pytest.raises(ValueError, match="probabilities must be a finite one-dimensional vector"):
        adjust_probabilities_between_priors(
            scores, source_positive_prevalence=0.3, target_positive_prevalence=0.5
        )


@pytest.mark.parametrize("prior", [0.0, 1.0, True, float("nan")])
def test_prevalences_must_be_finite_and_strictly_inside_the_unit_interval(prior) -> None:
    with pytest.raises(ValueError, match="must be finite and in"):
        adjust_probabilities_between_priors(
            np.asarray([0.2, 0.6]), source_positive_prevalence=prior, target_positive_prevalence=0.5
        )


def test_valid_prevalence_shift_moves_scores_toward_the_target_prior() -> None:
    scores = np.asarray([0.2, 0.6])
    adjusted = adjust_probabilities_between_priors(
        scores, source_positive_prevalence=0.3, target_positive_prevalence=0.5
    )
    if isinstance(adjusted, tuple):
        adjusted = adjusted[0]
    assert np.all(np.asarray(adjusted) > scores)


def test_numeric_array_parser_refuses_strings_and_wrong_rank() -> None:
    with pytest.raises(ValueError, match="must be a finite real array"):
        finite_real_array("0.1,0.2", "scores")
    with pytest.raises(ValueError, match="must be a 2-dimensional"):
        finite_real_array([0.1, 0.2], "scores", ndim=2)
    with pytest.raises(ValueError, match="values must be a finite real number"):
        finite_real_array([[0.1], [0.2, 0.3]], "scores")
