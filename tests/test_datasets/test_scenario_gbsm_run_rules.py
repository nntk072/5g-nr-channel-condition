"""Scenario GBSM records and runs: identity, shape, LOS and run-consistency rules."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from datasets.scenario_gbsm import (
    ScenarioGBSMRecord,
    ScenarioGBSMRun,
    scenario_gbsm_run_from_channel,
)
from tests.test_datasets.test_scenario_gbsm_record_rules import _base
from tests.test_datasets.test_scenario_gbsm_store import _native_channel


def _record(**changes: Any) -> ScenarioGBSMRecord:
    kwargs = _base()
    kwargs.update(changes)
    return ScenarioGBSMRecord(**kwargs)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        pytest.param({"sample_index": 1.0}, "sample_index must be an integer", id="index-type"),
        pytest.param({"scenario": "XYZ"}, "scenario must be one of", id="scenario"),
        pytest.param({"scenario_los": 1}, "scenario_los must be Boolean", id="los-type"),
        pytest.param({"sample_id": ""}, "non-empty canonical string", id="sample-empty"),
    ],
)
def test_record_identity_fields_are_typed(changes, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _record(**changes)


def test_record_rejects_a_single_stream_tau_shape_and_short_offsets() -> None:
    base = _base()
    with pytest.raises(ValueError, match="tau_s must be nonnegative"):
        ScenarioGBSMRecord(**{**base, "tau_s": np.zeros((2, 1, 2))})
    with pytest.raises(ValueError, match="at least 8 entries"):
        ScenarioGBSMRecord(**{**base, "subcarrier_offsets_hz": np.arange(4) * 1.0})
    with pytest.raises(ValueError, match="increase strictly"):
        ScenarioGBSMRecord(**{**base, "subcarrier_offsets_hz": np.zeros(8)})


def test_run_requires_identity_and_one_run_id_per_record() -> None:
    record = _record()
    with pytest.raises(ValueError, match="run_id must be a non-empty string"):
        ScenarioGBSMRun(run_id="", records=(record,))
    other = replace(record, run_id="other-run")
    with pytest.raises(ValueError, match="must all use the run ID"):
        ScenarioGBSMRun(run_id=record.run_id, records=(other,))


def test_run_rejects_duplicate_samples() -> None:
    record = _record()
    with pytest.raises(ValueError, match="unique"):
        ScenarioGBSMRun(run_id=record.run_id, records=(record, record))


def test_run_from_channel_requires_a_run_identity() -> None:
    with pytest.raises(ValueError):
        scenario_gbsm_run_from_channel(_native_channel(), run_id="", sample_ids=["d0"])
