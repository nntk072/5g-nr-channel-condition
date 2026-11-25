"""Paired link-curve admission: schema, evidence grade, configuration and matched-realization rules."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from artifacts.provenance import atomic_json
from evaluation.scoring.paired_link_curves import (
    compare_curves,
    compare_rx_correlation_curves,
    load_curve,
)

SCHEMA = "loslab.tdl_pusch_time_bler_curve.v1"


def _curve(**changes: Any) -> dict[str, Any]:
    curve: dict[str, Any] = {
        "schema": SCHEMA,
        "evidence_grade": "SIMULATION_ONLY_UNVALIDATED",
        "configuration": {
            "speed_mps": 3.0,
            "matched_realization_checks": [
                {
                    "seed": 4,
                    "noise_seed": 6,
                    "transmit_bits_sha256": "bits",
                    "channel_truth_sha256": "truth",
                }
            ],
        },
        "points": [{"snr_db": 12.0, "frames": 1, "bits": 100, "frame_error_mask": [False]}],
    }
    curve.update(changes)
    return curve


def _write(path: Path, document: dict[str, Any]) -> Path:
    atomic_json(path, document)
    return path


def test_loader_requires_a_supported_schema_and_simulation_only_grade(tmp_path: Path) -> None:
    good = _write(tmp_path / "good.json", _curve())
    assert load_curve(good)["schema"] == SCHEMA
    bad_schema = _write(tmp_path / "schema.json", _curve(schema="other.v1"))
    with pytest.raises(ValueError, match="unsupported sample-domain curve schema"):
        load_curve(bad_schema)
    bad_grade = _write(tmp_path / "grade.json", _curve(evidence_grade="VALIDATED"))
    with pytest.raises(ValueError, match="unexpected evidence grade"):
        load_curve(bad_grade)


def _pair(tmp_path: Path, control: dict[str, Any], treatment: dict[str, Any]):
    return (
        _write(tmp_path / "control.json", control),
        _write(tmp_path / "treatment.json", treatment),
    )


def _estimator_pair_documents() -> tuple[dict[str, Any], dict[str, Any]]:
    control = _curve(schema="loslab.tdl_pusch_time_bler_curve.v2")
    treatment = _curve(schema="loslab.tdl_pusch_time_bler_curve.v2")
    for curve in (control, treatment):
        check = curve["configuration"]["matched_realization_checks"][0]
        check.update(
            noise_real_seed=7,
            noise_imag_seed=8,
        )
        curve["configuration"]["channel_estimator"] = {
            "kind": "sionna_ls",
            "min_delay_samples": None,
            "max_delay_samples": None,
        }
    treatment["configuration"]["channel_estimator"] = {
        "kind": "finite_delay_dft",
        "min_delay_samples": 0,
        "max_delay_samples": 3,
    }
    return control, treatment


def _rx_curve_pair_documents() -> tuple[dict[str, Any], dict[str, Any]]:
    control_check = {
        "seed": 4,
        "noise_real_seed": 7,
        "noise_imag_seed": 8,
        "transmit_bits_sha256": "bits",
        "first_rx_channel_truth_sha256": "first-rx",
        "all_rx_channel_truth_sha256": "all-rx",
    }
    treatment_check = {
        **control_check,
        "latent_uncorrelated_channel_truth_sha256": "all-rx",
    }
    control = {
        "schema": "loslab.tdl_pusch_rx_diversity.v1",
        "evidence_grade": "SIMULATION_ONLY_UNVALIDATED",
        "configuration": {
            "rx_correlation_coefficient": 0.0,
            "matched_realization_checks": [control_check],
        },
        "points": [
            {
                "snr_db": 5.0,
                "frames": 1,
                "single_rx": {"frame_error_mask": [False]},
                "two_rx": {"frame_error_mask": [False]},
            }
        ],
    }
    treatment = deepcopy(control)
    treatment["configuration"] = {
        "rx_correlation_coefficient": 0.5,
        "matched_realization_checks": [treatment_check],
    }
    return control, treatment


@pytest.mark.parametrize(
    ("treatment_change", "message"),
    [
        pytest.param(
            {
                "configuration": {
                    "speed_mps": 3.0,
                    "matched_realization_checks": [
                        {
                            "seed": 4,
                            "noise_seed": 6,
                            "transmit_bits_sha256": "bits",
                            "channel_truth_sha256": "truth",
                        }
                    ],
                    "extra": 1,
                }
            },
            "configuration differs unexpectedly",
            id="extra-config",
        ),
        pytest.param(
            {"points": [{"snr_db": 99.0, "frames": 1, "bits": 100, "frame_error_mask": [False]}]},
            "SNR values or codeword counts differ|SNR grid|must differ",
            id="snr-grid",
        ),
        pytest.param({"points": []}, "SNR point counts differ|must differ", id="point-count"),
        pytest.param(
            {"configuration": {"speed_mps": 3.0, "matched_realization_checks": []}},
            "matched realization batch counts differ|must differ",
            id="batch-count",
        ),
    ],
)
def test_matched_pairs_must_differ_only_in_the_declared_field(
    tmp_path: Path, treatment_change, message
) -> None:
    treatment = _curve(**treatment_change)
    treatment["configuration"]["speed_mps"] = 30.0  # the declared varying field
    control_path, treatment_path = _pair(tmp_path, _curve(), treatment)
    with pytest.raises(ValueError, match=message):
        compare_curves(control_path, treatment_path)


def test_varying_field_must_be_declared_and_must_change(tmp_path: Path) -> None:
    control_path, treatment_path = _pair(tmp_path, _curve(), _curve())
    with pytest.raises(ValueError, match="varying_config_field must be"):
        compare_curves(control_path, treatment_path, varying_config_field="snr")
    with pytest.raises(ValueError, match="must differ at speed_mps"):
        compare_curves(control_path, treatment_path)


def test_matched_realizations_must_share_seeds_and_noise(tmp_path: Path) -> None:
    treatment = deepcopy(_curve())
    treatment["configuration"]["speed_mps"] = 30.0
    treatment["configuration"]["matched_realization_checks"][0]["seed"] = 5
    control_path, treatment_path = _pair(tmp_path, _curve(), treatment)
    with pytest.raises(ValueError, match="batch seeds differ"):
        compare_curves(control_path, treatment_path)

    treatment = deepcopy(_curve())
    treatment["configuration"]["speed_mps"] = 30.0
    treatment["configuration"]["matched_realization_checks"][0]["noise_seed"] = 7
    control_path, treatment_path = _pair(tmp_path, _curve(), treatment)
    with pytest.raises(ValueError, match="noise seeds differ"):
        compare_curves(control_path, treatment_path)


@pytest.mark.parametrize(
    ("target", "change", "message"),
    [
        pytest.param("control", "missing", "requires one Sionna LS", id="missing-estimator"),
        pytest.param("control", "extra", "unsupported field roster", id="extra-field"),
        pytest.param("control", "ls-bounds", "must not declare DFT delay bounds", id="ls-bounds"),
        pytest.param("treatment", "dft-bounds", "ordered integer delay bounds", id="dft-bounds"),
        pytest.param("treatment", "bits", "transmitted bits differ", id="payload-mismatch"),
        pytest.param(
            "treatment", "noise-component", "noise_real_seed differs", id="noise-mismatch"
        ),
        pytest.param("control", "hash", "requires payload/channel hashes", id="missing-hash"),
    ],
)
def test_channel_estimator_comparison_enforces_metadata_and_matched_inputs(
    tmp_path: Path, target: str, change: str, message: str
) -> None:
    control, treatment = _estimator_pair_documents()
    selected = control if target == "control" else treatment
    estimator = selected["configuration"]["channel_estimator"]
    if change == "missing":
        del selected["configuration"]["channel_estimator"]
    elif change == "extra":
        estimator["unexpected"] = True
    elif change == "ls-bounds":
        estimator["min_delay_samples"] = 0
    elif change == "dft-bounds":
        estimator["min_delay_samples"] = 4
    elif change == "bits":
        selected["configuration"]["matched_realization_checks"][0]["transmit_bits_sha256"] = (
            "other-bits"
        )
    elif change == "noise-component":
        selected["configuration"]["matched_realization_checks"][0]["noise_real_seed"] = 9
    elif change == "hash":
        for curve in (control, treatment):
            curve["configuration"]["matched_realization_checks"][0]["channel_truth_sha256"] = ""

    control_path, treatment_path = _pair(tmp_path, control, treatment)
    with pytest.raises(ValueError, match=message):
        compare_curves(control_path, treatment_path, varying_config_field="channel_estimator")


@pytest.mark.parametrize(
    ("experiment", "failure", "message"),
    [
        pytest.param("profile", "payload", "transmitted bits differ", id="profile-payload"),
        pytest.param("profile", "same-channel", "channel truth is identical", id="profile-truth"),
        pytest.param("dmrs", "channel", "channel truth differs", id="dmrs-truth"),
        pytest.param("delay", "payload", "transmitted bits differ", id="delay-payload"),
        pytest.param(
            "delay",
            "missing-seeds",
            "separate real/imaginary noise seeds",
            id="delay-missing-seeds",
        ),
        pytest.param("delay", "same-noise", "must use distinct seeds", id="delay-same-noise"),
        pytest.param("delay", "same-channel", "channel truth is identical", id="delay-same-truth"),
        pytest.param(
            "delay", "control-cp", "control curve must have a CP-contained", id="delay-control-cp"
        ),
        pytest.param(
            "delay", "treatment-cp", "must explicitly record CP overrun", id="delay-treatment-cp"
        ),
        pytest.param("delay", "order", "treatment delay spread must exceed", id="delay-order"),
    ],
)
def test_profile_dmrs_and_delay_spread_pairs_enforce_channel_contracts(
    tmp_path: Path, experiment: str, failure: str, message: str
) -> None:
    control, treatment = _curve(), _curve()
    control_config = control["configuration"]
    treatment_config = treatment["configuration"]
    if experiment == "profile":
        varying = "profile"
        control_config.update(profile="TDL-A", channel_model="TDL-A")
        treatment_config.update(profile="TDL-B", channel_model="TDL-B")
    elif experiment == "dmrs":
        varying = "dmrs_additional_position"
        control_config[varying] = 0
        treatment_config[varying] = 1
    else:
        varying = "delay_spread_s"
        control_config.update(
            delay_spread_s=1e-8,
            allow_cp_overrun=False,
            cp_covers_channel_memory=True,
        )
        treatment_config.update(
            delay_spread_s=2e-8,
            allow_cp_overrun=True,
            cp_covers_channel_memory=False,
        )
        treatment_config["matched_realization_checks"][0]["channel_truth_sha256"] = "other-truth"
        if failure != "missing-seeds":
            for curve in (control, treatment):
                curve["configuration"]["matched_realization_checks"][0].update(
                    noise_real_seed=7,
                    noise_imag_seed=8,
                )

    if failure == "payload":
        treatment_config["matched_realization_checks"][0]["transmit_bits_sha256"] = "other"
    elif failure == "same-channel" and experiment == "delay":
        treatment_config["matched_realization_checks"][0]["channel_truth_sha256"] = "truth"
    elif failure == "channel":
        treatment_config["matched_realization_checks"][0]["channel_truth_sha256"] = "other"
    elif failure == "same-noise":
        for curve in (control, treatment):
            check = curve["configuration"]["matched_realization_checks"][0]
            check.update(noise_real_seed=7, noise_imag_seed=7)
    elif failure == "control-cp":
        control_config["allow_cp_overrun"] = True
    elif failure == "treatment-cp":
        treatment_config["allow_cp_overrun"] = False
    elif failure == "order":
        treatment_config["delay_spread_s"] = 0.5e-8

    control_path, treatment_path = _pair(tmp_path, control, treatment)
    with pytest.raises(ValueError, match=message):
        compare_curves(control_path, treatment_path, varying_config_field=varying)


@pytest.mark.parametrize(
    ("schema", "cluster_ids", "message"),
    [
        pytest.param("v1", None, "requires v2 curves", id="requires-v2"),
        pytest.param("v2", ["4", "4"], "at least two declared batch clusters", id="one-cluster"),
    ],
)
def test_estimator_comparison_requires_v2_curves_with_declared_clusters(
    tmp_path: Path, schema: str, cluster_ids: list[str] | None, message: str
) -> None:
    control, treatment = _estimator_pair_documents()
    for curve in (control, treatment):
        curve["points"][0].update(
            frames=2,
            frame_error_mask=[False, True],
        )
        checks = curve["configuration"]["matched_realization_checks"]
        checks.extend(
            [
                {
                    **checks[0],
                    "seed": 5,
                }
            ]
        )
        curve["schema"] = f"loslab.tdl_pusch_time_bler_curve.{schema}"
        if schema == "v2":
            curve["points"][0]["frame_cluster_ids"] = cluster_ids or ["4", "5"]

    control_path, treatment_path = _pair(tmp_path, control, treatment)
    with pytest.raises(ValueError, match=message):
        compare_curves(control_path, treatment_path, varying_config_field="channel_estimator")


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        pytest.param("schema", "unsupported receive-diversity schema", id="schema"),
        pytest.param("grade", "unexpected evidence grade", id="grade"),
        pytest.param("correlation", "control must be uncorrelated", id="correlation"),
        pytest.param("configuration", "configuration differs unexpectedly", id="configuration"),
        pytest.param("batch-count", "matched realization batch counts differ", id="batch-count"),
        pytest.param("noise", "noise_real_seed differs", id="noise"),
        pytest.param("first-rx", "first receive-antenna channel truth differs", id="first-rx"),
        pytest.param("latent", "does not reference the control's latent channel", id="latent"),
        pytest.param("point-count", "SNR point counts differ", id="point-count"),
        pytest.param("snr", "SNR values or codeword counts differ", id="snr"),
        pytest.param("frames", "SNR values or codeword counts differ", id="frames"),
        pytest.param("single-rx", "single-RX outcomes differ", id="single-rx"),
    ],
)
def test_receive_diversity_comparison_rejects_unpaired_evidence(
    tmp_path: Path, failure: str, message: str
) -> None:
    control, treatment = _rx_curve_pair_documents()
    if failure == "schema":
        treatment["schema"] = "other"
    elif failure == "grade":
        treatment["evidence_grade"] = "VALIDATED"
    elif failure == "correlation":
        control["configuration"]["rx_correlation_coefficient"] = 0.1
    elif failure == "configuration":
        treatment["configuration"]["speed_mps"] = 3.0
    elif failure == "batch-count":
        treatment["configuration"]["matched_realization_checks"] = []
    elif failure == "noise":
        treatment["configuration"]["matched_realization_checks"][0]["noise_real_seed"] = 9
    elif failure == "first-rx":
        treatment["configuration"]["matched_realization_checks"][0][
            "first_rx_channel_truth_sha256"
        ] = "other"
    elif failure == "latent":
        treatment["configuration"]["matched_realization_checks"][0][
            "latent_uncorrelated_channel_truth_sha256"
        ] = "other"
    elif failure == "point-count":
        treatment["points"] = []
    elif failure == "snr":
        treatment["points"][0]["snr_db"] = 6.0
    elif failure == "frames":
        treatment["points"][0]["frames"] = 2
    elif failure == "single-rx":
        treatment["points"][0]["single_rx"]["frame_error_mask"] = [True]

    control_path, treatment_path = _pair(tmp_path, control, treatment)
    with pytest.raises(ValueError, match=message):
        compare_rx_correlation_curves(control_path, treatment_path)
