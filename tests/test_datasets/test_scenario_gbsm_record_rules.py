"""Scenario GBSM record construction: each declared axis, digest and metadata rule is enforced."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
from datasets.scenario_gbsm import (
    ScenarioGBSMRecord,
    _digest_json,
    scenario_gbsm_run_from_channel,
)
from tests.test_datasets.test_scenario_gbsm_store import _native_channel


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _base() -> dict[str, Any]:
    record = scenario_gbsm_run_from_channel(
        _native_channel(), run_id="pilot-umix", sample_ids=["drop-000"]
    ).records[0]
    return {
        "sample_id": record.sample_id,
        "run_id": record.run_id,
        "sample_index": record.sample_index,
        "scenario": record.scenario,
        "scenario_los": record.scenario_los,
        "h_path": np.array(record.h_path),
        "tau_s": np.array(record.tau_s),
        "cfr": np.array(record.cfr),
        "subcarrier_offsets_hz": np.array(record.subcarrier_offsets_hz),
        "metadata": _thaw(record.metadata),
    }


def _rebind(metadata: dict[str, Any]) -> dict[str, Any]:
    """Recompute the configuration and topology digests after an edit."""
    metadata["configuration_sha256"] = _digest_json(
        metadata["generator_configuration"], label="generator configuration"
    )
    metadata["topology_row_sha256"] = _digest_json(metadata["topology"], label="topology row")
    return metadata


def _meta(change) -> dict[str, Any]:
    kwargs = _base()
    metadata = kwargs["metadata"]
    change(metadata)
    if not getattr(change, "keep_digests", False):
        _rebind(metadata)
    kwargs["metadata"] = metadata
    return kwargs


def _configure(key: str, value: Any):
    def change(metadata: dict[str, Any]) -> None:
        metadata["generator_configuration"][key] = value

    return change


def _topology(key: str, value: Any):
    def change(metadata: dict[str, Any]) -> None:
        metadata["topology"][key] = value

    return change


def _top(key: str, value: Any):
    def change(metadata: dict[str, Any]) -> None:
        metadata[key] = value

    return change


def test_base_record_is_valid() -> None:
    assert isinstance(ScenarioGBSMRecord(**_base()), ScenarioGBSMRecord)


@pytest.mark.parametrize(
    ("kwargs_change", "error", "message"),
    [
        pytest.param(
            {"sample_id": " drop "}, ValueError, "non-empty canonical string", id="sample-spaces"
        ),
        pytest.param({"run_id": ""}, ValueError, "non-empty canonical string", id="run-empty"),
        pytest.param(
            {"sample_index": True}, TypeError, "sample_index must be an integer", id="index-bool"
        ),
        pytest.param(
            {"sample_index": -1},
            ValueError,
            "sample_index must be nonnegative",
            id="index-negative",
        ),
        pytest.param(
            {"scenario": "XYZ"}, ValueError, "scenario must be one of", id="unknown-scenario"
        ),
        pytest.param(
            {"scenario_los": "yes"}, TypeError, "scenario_los must be Boolean", id="los-type"
        ),
        pytest.param(
            {"h_path": np.ones((1, 1, 1, 1, 2), dtype=np.complex64)},
            ValueError,
            "h_path must have non-empty",
            id="h-rank",
        ),
        pytest.param(
            {"h_path": np.ones((2, 1, 1, 1, 2, 2), dtype=np.complex64)},
            ValueError,
            "one BS and one UT link",
            id="two-links",
        ),
        pytest.param(
            {"tau_s": np.zeros((1, 1, 3))}, ValueError, "tau_s must be nonnegative", id="tau-shape"
        ),
        pytest.param(
            {"tau_s": np.full((1, 1, 2), -1.0)},
            ValueError,
            "tau_s must be nonnegative",
            id="tau-negative",
        ),
        pytest.param(
            {"subcarrier_offsets_hz": np.arange(4) * 1.0},
            ValueError,
            "at least 8 entries",
            id="offsets-short",
        ),
        pytest.param(
            {"subcarrier_offsets_hz": np.zeros((2, 8))},
            ValueError,
            "at least 8 entries",
            id="offsets-rank",
        ),
        pytest.param(
            {"subcarrier_offsets_hz": np.zeros(8)},
            ValueError,
            "increase strictly",
            id="offsets-flat",
        ),
        pytest.param(
            {"cfr": np.zeros((1, 1, 1, 1, 2, 4), dtype=np.complex64)},
            ValueError,
            "cfr must have shape",
            id="cfr-shape",
        ),
        pytest.param({"cfr": None}, TypeError, "cfr", id="cfr-missing"),
    ],
)
def test_record_rejects_malformed_axes_and_identity(kwargs_change, error, message) -> None:
    kwargs = _base()
    kwargs.update(kwargs_change)
    with pytest.raises(error, match=message):
        ScenarioGBSMRecord(**kwargs)


def test_record_cfr_must_be_the_path_sum_of_its_delays() -> None:
    kwargs = _base()
    kwargs["cfr"] = np.asarray(kwargs["cfr"]) + 0.5
    with pytest.raises(ValueError, match="does not match stored path coefficients"):
        ScenarioGBSMRecord(**kwargs)


def _digest_change(key: str, value: str):
    """Set a digest field and keep it, so the digest check itself is what fails."""

    def change(metadata: dict[str, Any]) -> None:
        metadata[key] = value

    change.keep_digests = True  # type: ignore[attr-defined]
    return change


@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param(
            _top("generator_configuration", None),
            "generator_configuration metadata is required",
            id="no-configuration",
        ),
        pytest.param(
            _digest_change("configuration_sha256", "0" * 64),
            "generator configuration digest is invalid",
            id="configuration-digest",
        ),
        pytest.param(
            _digest_change("topology_row_sha256", "0" * 64),
            "topology row digest is invalid",
            id="topology-digest",
        ),
        pytest.param(
            _top("scenario", "UMa"), "scenario metadata disagrees", id="scenario-metadata"
        ),
        pytest.param(
            _configure("scenario", "UMa"),
            "generator configuration scenario disagrees",
            id="configuration-scenario",
        ),
        pytest.param(
            _top("sionna_version", "0.0"), "Sionna version metadata disagrees", id="sionna-version"
        ),
        pytest.param(
            _top("scenario_los", True), "scenario LOS metadata disagrees", id="scenario-los"
        ),
        pytest.param(
            _top("geometric_los_label", "LOS"),
            "must not be recorded as geometric LOS",
            id="geometric-label",
        ),
        pytest.param(
            _top("split_subject_id", "other"),
            "split subject must equal its sample ID",
            id="split-subject",
        ),
        pytest.param(
            _top("split_parent_type", "episode"),
            "must declare its standardized-draw parent type",
            id="parent-type",
        ),
        pytest.param(
            _top("split_source_family", "scenario_gbsm:UMa"),
            "split source family disagrees",
            id="source-family",
        ),
        pytest.param(
            _top("split_leakage_relations", "none"),
            "must be explicitly listed",
            id="leakage-relations",
        ),
        pytest.param(_topology("los", [[True]]), "scenario LOS state disagrees", id="topology-los"),
        pytest.param(
            _configure("subcarrier_spacing_hz", 0.0), "positive subcarrier spacing", id="spacing"
        ),
        pytest.param(
            _configure("subcarrier_spacing_hz", 15_000.0),
            "subcarrier offsets disagree",
            id="spacing-mismatch",
        ),
        pytest.param(
            _configure("num_subcarriers", 99), "subcarrier offsets disagree", id="subcarrier-count"
        ),
        pytest.param(
            _configure("carrier_frequency_hz", 0.0), "positive carrier frequency", id="carrier-zero"
        ),
        pytest.param(
            _configure("carrier_frequency_hz", 1.0),
            "positive RF frequencies",
            id="carrier-below-offsets",
        ),
    ],
)
def test_record_metadata_must_agree_with_the_declared_drop(change, message) -> None:
    kwargs = _meta(change)
    with pytest.raises(ValueError, match=message):
        ScenarioGBSMRecord(**kwargs)
