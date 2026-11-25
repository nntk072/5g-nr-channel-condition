"""SNR curve inference input: intervals, grids, radius and paired-contrast ordering rules."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from experiments.acceptance import evaluate_snr_curve_acceptance
from tests.test_experiments.test_acceptance import _freeze, _inference


def _evaluate(inference: dict[str, Any]) -> dict[str, Any]:
    return evaluate_snr_curve_acceptance(_freeze(), curve_inference=inference)


def test_curve_inference_must_be_an_object() -> None:
    with pytest.raises(TypeError, match="curve_inference must be an object"):
        evaluate_snr_curve_acceptance(_freeze(), curve_inference=["not", "object"])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(
            lambda i: i["by_condition"]["clean"].__setitem__(0, "row"),
            "clean SNR ordering differs",
            id="row-type",
        ),
        pytest.param(
            lambda i: i["by_condition"]["clean"][0].update(simultaneous_95_percent_interval=[0.1]),
            "two-endpoint",
            id="interval-short",
        ),
        pytest.param(
            lambda i: i.update(simultaneous_radius=2.0),
            "simultaneous radius must be finite",
            id="radius",
        ),
        pytest.param(
            lambda i: i["by_condition"]["clean"].pop(),
            "rows do not match the frozen SNR grid",
            id="condition-rows",
        ),
        pytest.param(
            lambda i: i["by_paired_contrast"].pop(),
            "paired curve inference rows do not match",
            id="paired-rows",
        ),
        pytest.param(
            lambda i: i["by_paired_contrast"].reverse(),
            "paired SNR ordering differs",
            id="paired-order",
        ),
    ],
)
def test_curve_inference_rows_are_checked(mutate, message) -> None:
    inference = copy.deepcopy(_inference())
    mutate(inference)
    with pytest.raises(ValueError, match=message):
        _evaluate(inference)


def test_curve_inference_snr_order_must_match_the_contract() -> None:
    inference = copy.deepcopy(_inference())
    for condition in inference["by_condition"].values():
        condition.reverse()
    with pytest.raises(ValueError, match="SNR ordering differs"):
        _evaluate(inference)
