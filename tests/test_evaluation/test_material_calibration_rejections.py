"""Material calibration evaluator: each rejected split, incidence, prediction and angular input."""

from __future__ import annotations

import copy

import pytest

from evaluation.acceptance.material_calibration import evaluate_material_calibration
from tests.test_evaluation.test_material_calibration import (
    _assessable_payload,
    _refresh_split_manifest_hash,
)


def _mutated(mutate, *, refresh: bool = True):
    payload = copy.deepcopy(_assessable_payload())
    mutate(payload)
    if refresh:
        _refresh_split_manifest_hash(payload)
    return payload


_UNHASHED = {"refresh": False}


def _strip_prediction(payload, index=0, **changes):
    payload["test_predictions"][index].update(changes)


@pytest.mark.parametrize(
    ("mutate", "message", "refresh"),
    [
        (lambda p: p.update(uncertainty_description="  "), "uncertainty_description must document", True),
        (lambda p: p.update(incidence_groups=[]), "incidence_groups must be a nonempty complete group manifest", False),
        (lambda p: p.update(test_predictions=[]), "test_predictions must be a nonempty list", False),
        (lambda p: p["expected_strata"].append("bogus"), "each expected_strata entry must be an object", False),
        (lambda p: p["expected_strata"].append(copy.deepcopy(p["expected_strata"][0])), "duplicate expected stratum", False),
        (lambda p: p["full_split_manifest"].update(schema="other.schema"), "full_split_manifest must use schema", True),
        (lambda p: p["full_split_manifest"]["groups"].pop(0), "outside its declared test group", True),
        (lambda p: p["full_split_manifest"]["groups"][0].update(split="validation"), "outside its declared test group", True),
        (lambda p: p["full_split_manifest"]["groups"][0].update(sample_ids=[]), "requires nonempty sample_ids", True),
        (lambda p: p["test_predictions"][0].update(polarization="X"), "polarization must be V or H", True),
    ],
)
def test_split_and_prediction_rejections(mutate, message, refresh) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_material_calibration(_mutated(mutate, refresh=refresh))


def test_each_prediction_must_be_an_object_and_unique() -> None:
    with pytest.raises(ValueError, match="each test prediction must be an object"):
        evaluate_material_calibration(_mutated(lambda p: p["test_predictions"].append("not-a-row"), refresh=False))
    with pytest.raises(ValueError, match="duplicate sample_id"):
        evaluate_material_calibration(
            _mutated(lambda p: p["test_predictions"].append(copy.deepcopy(p["test_predictions"][0])), refresh=False)
        )


def test_prediction_interval_must_be_an_ordered_pair() -> None:
    with pytest.raises(ValueError, match="prediction_interval_95_db must be"):
        evaluate_material_calibration(
            _mutated(lambda p: _strip_prediction(p, prediction_interval_95_db=[1.0]))
        )
    with pytest.raises(ValueError, match="lower bound exceeds upper bound"):
        evaluate_material_calibration(
            _mutated(lambda p: _strip_prediction(p, prediction_interval_95_db=[5.0, 1.0]))
        )


def test_angular_semantics_are_fixed_by_the_contract() -> None:
    with pytest.raises(ValueError, match="angular_power_semantics must be integrated_power_per_bin"):
        evaluate_material_calibration(
            _mutated(lambda p: _strip_prediction(p, angular_power_semantics="density"))
        )
