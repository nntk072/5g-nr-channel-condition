"""Independent retained-data audit for the G5.09 shared-sync result."""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import experiments.audit_g5_ordered_impairments_result as auditor
import numpy as np
import pytest
from experiments.audit_g5_ordered_impairments_result import (
    _audit_result,
    _canonical_json_bytes,
    _estimator_source_checks,
    _g5_declared_preregistration_order_matches,
    _preflight_waveform_archive,
    _read_bounded_evidence_file,
    _recompute_waveform_diagnostics,
    _replay_experiment,
    audit_result_file,
)


def test_estimator_source_binding_accepts_exact_dft_source_and_flags_drift() -> None:
    source = b"frozen DFT implementation"
    digest = hashlib.sha256(source).hexdigest()
    identity = {
        "name": "finite_delay_dft",
        "min_delay_samples": 0,
        "max_delay_samples": 7,
        "implementation_path": "experiments/g5_cp_bounded_dft_estimator.py",
        "implementation_sha256": digest,
    }

    assert _estimator_source_checks(identity, source_bytes=source, committed_bytes=source) == {
        "estimator_source_worktree_digest": True,
        "estimator_committed_source_digest": True,
    }
    drifted = _estimator_source_checks(identity, source_bytes=b"changed", committed_bytes=None)
    assert drifted == {
        "estimator_source_worktree_digest": False,
        "estimator_committed_source_digest": None,
    }


def test_estimator_source_binding_rejects_unfrozen_estimator_span() -> None:
    assert _estimator_source_checks(
        {
            "name": "finite_delay_dft",
            "min_delay_samples": 0,
            "max_delay_samples": 8,
            "implementation_path": "experiments/g5_cp_bounded_dft_estimator.py",
            "implementation_sha256": "0" * 64,
        },
        source_bytes=b"source",
        committed_bytes=None,
    ) == {
        "estimator_source_worktree_digest": False,
        "estimator_committed_source_digest": None,
    }


def test_deterministic_replay_forwards_the_recorded_estimator(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_probe(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"replayed": True}

    monkeypatch.setattr(
        auditor,
        "_run_receiver_probe",
        fake_probe,
    )
    replay = auditor._replay_experiment(
        {
            "config": {
                "channel_seeds": [180001],
                "blocks_per_channel": 1,
                "channel_estimator": {"name": "finite_delay_dft"},
                "waveform_evidence": None,
            }
        }
    )

    assert replay == {"replayed": True}
    assert captured["channel_estimator"] == "finite_delay_dft"


ROOT = Path(__file__).resolve().parents[2]
RESULT = (
    ROOT
    / "docs/research/results/g5-ordered-impairments-independent-channels-cpu-64x8-shared-sync-diagnostic-retention-v3-20261025.json"
)
LEGACY_V6_RESULT = (
    ROOT
    / "docs/research/results/g5-ordered-impairments-independent-channels-cpu-64x8-shared-sync-replication-v2-20261025.json"
)
LEGACY_V4_RESULT = (
    ROOT
    / "docs/research/results/g5-ordered-impairments-independent-channels-cpu-64x8-20261024.json"
)


def test_v10_declared_preregistration_order_is_recomputed_by_auditor() -> None:
    criteria = {
        "schema": "umcd.g5_ordered_impairments_snr_acceptance.v2",
        "snr_db": [0.0],
        "channel_seeds": list(range(20)),
        "channel_estimator": "sionna_default_ls",
        "blocks_per_channel": 1,
        "minimum_independent_channels": 20,
        "maximum_corrected_macro_bler": [0.1],
        "minimum_uncorrected_minus_corrected_macro_bler": [0.02],
        "preregistered_utc": "2026-10-25T09:00:00Z",
    }
    signature = {
        "preregistered_utc": criteria["preregistered_utc"],
        "receiver_run_started_utc": "2026-10-26T09:00:00Z",
        "declared_temporal_order_status": "preregistration_precedes_receiver_run",
        "timestamp_authority": "not_established_by_this_verifier",
    }

    assert _g5_declared_preregistration_order_matches(criteria, signature)
    signature["receiver_run_started_utc"] = "2026-10-24T09:00:00Z"
    assert not _g5_declared_preregistration_order_matches(criteria, signature)
    signature["receiver_run_started_utc"] = "2026-10-25T09:00:00Z"
    assert not _g5_declared_preregistration_order_matches(criteria, signature)


def _current_source_bound_result() -> tuple[dict[str, Any], bytes]:
    """Build a synthetic current-source receipt without rewriting archived evidence."""
    result: dict[str, Any] = json.loads(RESULT.read_text(encoding="utf-8"))
    source_path = ROOT / result["provenance"]["source_path"]
    source_bytes = source_path.read_bytes()
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    result["source_sha256"] = source_digest
    result["provenance"]["source_sha256"] = source_digest
    _refresh_result_digest(result)
    return result, source_bytes


def _refresh_result_digest(result: dict[str, Any]) -> None:
    unhashed = dict(result)
    unhashed.pop("result_sha256", None)
    result["result_sha256"] = hashlib.sha256(_canonical_json_bytes(unhashed)).hexdigest()


def _write_current_source_bound_result(tmp_path: Path) -> Path:
    result, _ = _current_source_bound_result()
    path = tmp_path / RESULT.name
    path.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    return path


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="requires no-follow file opening")
def test_waveform_evidence_reader_is_bounded_and_rejects_link_aliases(tmp_path: Path) -> None:
    evidence = tmp_path / "capture.npz"
    evidence.write_bytes(b"bounded evidence")
    assert _read_bounded_evidence_file(evidence, maximum_bytes=32) == b"bounded evidence"
    with pytest.raises(ValueError, match="size limit"):
        _read_bounded_evidence_file(evidence, maximum_bytes=8)

    symlink = tmp_path / "capture-symlink.npz"
    symlink.symlink_to(evidence)
    with pytest.raises(OSError):
        _read_bounded_evidence_file(symlink, maximum_bytes=32)

    hard_link = tmp_path / "capture-hardlink.npz"
    hard_link.hardlink_to(evidence)
    with pytest.raises(ValueError, match="single-link regular file"):
        _read_bounded_evidence_file(evidence, maximum_bytes=32)


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="requires no-follow file opening")
def test_waveform_evidence_reader_rejects_a_file_changed_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "changing-capture.npz"
    evidence.write_bytes(b"original")
    original_read = os.read
    changed = False

    def append_after_read(descriptor: int, count: int) -> bytes:
        nonlocal changed
        result = original_read(descriptor, count)
        if not changed:
            changed = True
            with evidence.open("ab") as stream:
                stream.write(b"-changed")
        return result

    monkeypatch.setattr(auditor.os, "read", append_after_read)
    with pytest.raises(ValueError, match="changed while being read"):
        _read_bounded_evidence_file(evidence, maximum_bytes=32)


def test_shared_sync_artifact_recomputes_retained_cluster_summaries_and_acceptance(
    tmp_path: Path,
) -> None:
    report = audit_result_file(_write_current_source_bound_result(tmp_path))

    assert report["status"] == "VERIFIED_RESULT_AND_WORKTREE_SOURCE_PENDING_COMMIT"
    assert report["independent_channel_count"] == 64
    assert report["blocks_per_channel"] == 8
    assert report["checks"]["result_digest"] is True
    assert report["checks"]["source_digest_bindings"] is True
    assert report["checks"]["source_worktree_digest"] is True
    assert report["checks"]["committed_source_digest"] is None
    assert report["checks"]["flattened_block_arrays"] is True
    assert report["checks"]["crc_and_bler_summaries"] is True
    assert report["checks"]["paired_cluster_inference"] is True
    assert report["checks"]["cfo_cluster_tail_bounds"] is True
    assert report["checks"]["crc_channel_cluster_tail_bounds"] is None
    assert report["checks"]["retained_observed_statistics"] is True
    assert report["checks"]["acceptance_verdict"] is True


def test_legacy_report_without_per_channel_waveform_diagnostics_remains_auditable() -> None:
    result, source_bytes = _current_source_bound_result()
    experiment = result["experiment"]
    experiment["schema"] = "umcd.g5_ordered_impairments_independent_channels.v6"
    for record in experiment["config"]["per_channel_draws"]:
        record.pop("samplewise_reference_max_absolute_error")
        record.pop("wrong_order_waveform_rmse")
    config = experiment["config"]
    result["configuration_sha256"] = hashlib.sha256(_canonical_json_bytes(config)).hexdigest()
    protocol = result["provenance"]["protocol"]
    protocol["experiment_schema"] = experiment["schema"]
    protocol["config"] = config
    result["provenance"]["protocol_sha256"] = hashlib.sha256(
        _canonical_json_bytes(protocol)
    ).hexdigest()
    _refresh_result_digest(result)

    report = auditor._audit_result(result, source_bytes, None)

    assert report["checks"]["retained_observed_statistics"] is True
    assert report["checks"]["waveform_diagnostic_extrema"] is None
    assert report["checks"]["acceptance_verdict"] is True
    assert report["status"] == "VERIFIED_RESULT_AND_WORKTREE_SOURCE_PENDING_COMMIT"
    assert any("predates per-channel waveform-diagnostic retention" in item for item in report["limitations"])


def test_legacy_report_rejects_partial_per_channel_waveform_diagnostics() -> None:
    result, source_bytes = _current_source_bound_result()
    experiment = result["experiment"]
    experiment["schema"] = "umcd.g5_ordered_impairments_independent_channels.v6"
    experiment["config"]["per_channel_draws"][0].pop("wrong_order_waveform_rmse")

    with pytest.raises(ValueError, match="complete roster"):
        auditor._audit_result(result, source_bytes, None)


def test_v7_schema_requires_per_channel_waveform_diagnostics() -> None:
    result, source_bytes = _current_source_bound_result()
    assert result["experiment"]["schema"] == "umcd.g5_ordered_impairments_independent_channels.v7"
    for record in result["experiment"]["config"]["per_channel_draws"]:
        record.pop("samplewise_reference_max_absolute_error")
        record.pop("wrong_order_waveform_rmse")

    with pytest.raises(ValueError, match="schema requires per-channel waveform diagnostics"):
        auditor._audit_result(result, source_bytes, None)


def test_archived_v6_replication_audits_retained_data_without_rewriting_result() -> None:
    original_bytes = LEGACY_V6_RESULT.read_bytes()

    report = audit_result_file(LEGACY_V6_RESULT)

    assert LEGACY_V6_RESULT.read_bytes() == original_bytes
    assert report["checks"]["result_digest"] is True
    assert report["checks"]["crc_and_bler_summaries"] is True
    assert report["checks"]["paired_cluster_inference"] is True
    assert report["checks"]["retained_observed_statistics"] is True
    assert report["checks"]["acceptance_verdict"] is True
    assert report["checks"]["waveform_diagnostic_extrema"] is None
    assert report["checks"]["source_worktree_digest"] is False
    assert report["status"] == "AUDIT_FAILED"


def test_archived_v4_result_marks_later_cfo_tail_diagnostic_unavailable() -> None:
    report = audit_result_file(LEGACY_V4_RESULT)

    assert report["checks"]["result_digest"] is True
    assert report["checks"]["crc_and_bler_summaries"] is True
    assert report["checks"]["paired_cluster_inference"] is True
    assert report["checks"]["cfo_cluster_tail_bounds"] is None
    assert report["checks"]["source_worktree_digest"] is False
    assert report["status"] == "AUDIT_FAILED"


def test_v6_schema_requires_its_declared_cfo_channel_tail_diagnostics() -> None:
    result, source_bytes = _current_source_bound_result()
    experiment = result["experiment"]
    experiment["schema"] = "umcd.g5_ordered_impairments_independent_channels.v6"
    inference = experiment["acceptance"]["inference"]
    inference.pop("cfo_channel_tail_diagnostics")
    protocol = result["provenance"]["protocol"]
    protocol["experiment_schema"] = experiment["schema"]
    protocol["config"] = experiment["config"]
    result["configuration_sha256"] = hashlib.sha256(
        _canonical_json_bytes(experiment["config"])
    ).hexdigest()
    result["provenance"]["protocol_sha256"] = hashlib.sha256(
        _canonical_json_bytes(protocol)
    ).hexdigest()
    _refresh_result_digest(result)

    report = auditor._audit_result(result, source_bytes, None)

    assert report["checks"]["cfo_cluster_tail_bounds"] is False
    assert report["status"] == "AUDIT_FAILED"


def test_new_schema_recomputes_exact_crc_failure_channel_bounds() -> None:
    result, source_bytes = _current_source_bound_result()
    experiment = result["experiment"]
    experiment["schema"] = "umcd.g5_ordered_impairments_independent_channels.v11"
    records = experiment["config"]["per_channel_draws"]
    inference = experiment["acceptance"]["inference"]
    diagnostics = []
    for condition in ("clean", "corrected", "uncorrected"):
        failed = sum(not all(record["crc_pass_by_condition"][condition]) for record in records)
        diagnostics.append(
            {
                "condition": condition,
                "channels_with_any_block_failure": failed,
                "independent_channels": len(records),
                "blocks_per_channel": experiment["config"]["blocks_per_channel"],
                "exact_one_sided_95_percent_channel_prevalence_upper_bound": (
                    1.0
                    if failed == len(records)
                    else float(auditor.beta.ppf(0.95, failed + 1, len(records) - failed))
                ),
            }
        )
    inference["crc_channel_failure_diagnostics"] = diagnostics

    _refresh_result_digest(result)
    report = _audit_result(result, source_bytes, None)
    assert report["checks"]["result_digest"] is True
    assert report["checks"]["crc_channel_cluster_tail_bounds"] is True

    diagnostics[0]["exact_one_sided_95_percent_channel_prevalence_upper_bound"] += 0.01
    _refresh_result_digest(result)
    tampered = _audit_result(result, source_bytes, None)
    assert tampered["checks"]["result_digest"] is True
    assert tampered["checks"]["crc_channel_cluster_tail_bounds"] is False


def test_block_bound_schema_requires_complete_receiver_and_transport_block_rosters() -> None:
    result, source_bytes = _current_source_bound_result()
    experiment = result["experiment"]
    config = experiment["config"]
    experiment["schema"] = "umcd.g5_ordered_impairments_independent_channels.v15"
    config["channel_estimator"] = {"name": "sionna_default_ls"}
    config["decoder_configuration"] = {
        "api": "SionnaPUSCHLSReceiver.decode_pusch",
        "channel_estimator": config["channel_estimator"],
        "crc_status_semantics": "true_means_transport_block_crc_passed",
    }
    config["receiver_configuration"] = {
        "n_size_bwp": 6,
        "subcarrier_spacing_khz": 30,
        "num_antenna_ports": 1,
        "num_layers": 1,
        "n_cell_id": 1,
        "dmrs_n_id": None,
        "dmrs_n_scid": 0,
        "dmrs_config_type": 1,
        "dmrs_length": 1,
        "dmrs_additional_position": 0,
        "num_cdm_groups_without_data": 1,
        "mapping_type": "A",
        "symbol_allocation": [0, 14],
        "precoding": "non-codebook",
        "frequency_hopping": "neither",
        "transform_precoding": False,
        "interpolation_type": "lin",
        "n_size_grid": 6,
        "n_start_grid": 0,
        "n_start_bwp": 0,
        "sionna_version": result["provenance"]["runtime"]["packages"]["sionna"],
        "receiver": "PUSCHLSChannelEstimator",
        "error_variance_semantics": (
            "sionna_linear_interpolated_marginal_variance_not_exact_covariance"
        ),
        "device": "cpu",
        "resource_grid": {
            "num_ofdm_symbols": 14,
            "fft_size": 72,
            "subcarrier_spacing_hz": 30_000,
        },
        "carrier_grid": {
            "n_size_grid": 6,
            "n_start_grid": 0,
            "n_start_bwp": 0,
            "bwp_start_rb_in_grid": 0,
            "fft_size": 72,
        },
    }
    for record in config["per_channel_draws"]:
        seed = record["channel_seed"]
        payload_seed = seed + 1_000_000
        record["payload_seed"] = payload_seed
        record["transport_blocks"] = [
            {
                "transport_block_id": (f"channel-{seed}:payload-{payload_seed}:block-{index}"),
                "block_index": index,
                "payload_seed": payload_seed,
                "crc_pass_by_condition": {
                    condition: record["crc_pass_by_condition"][condition][index]
                    for condition in ("clean", "corrected", "uncorrected")
                },
                "timing_error_samples": record["timing_error_samples"][index],
                "cfo_error_hz": record["cfo_error_hz"][index],
            }
            for index in range(config["blocks_per_channel"])
        ]

    _refresh_result_digest(result)
    report = auditor._audit_result(result, source_bytes=source_bytes, committed_source=None)
    assert report["checks"]["receiver_configuration_binding"] is True
    assert report["checks"]["transport_block_roster_binding"] is True

    legacy_defaults = {
        key: config["receiver_configuration"].pop(key)
        for key in ("frequency_hopping", "transform_precoding")
    }
    _refresh_result_digest(result)
    legacy_report = auditor._audit_result(result, source_bytes=source_bytes, committed_source=None)
    assert legacy_report["checks"]["receiver_configuration_binding"] is True
    config["receiver_configuration"].update(legacy_defaults)

    for key, unsupported_value in (
        ("frequency_hopping", "intraSlot"),
        ("transform_precoding", True),
    ):
        original_value = config["receiver_configuration"][key]
        config["receiver_configuration"][key] = unsupported_value
        _refresh_result_digest(result)
        with pytest.raises(ValueError, match="unsupported PUSCH mode"):
            auditor._audit_result(result, source_bytes=source_bytes, committed_source=None)
        config["receiver_configuration"][key] = original_value
        _refresh_result_digest(result)

    config["per_channel_draws"][0]["transport_blocks"][0]["transport_block_id"] = "wrong-id"
    _refresh_result_digest(result)
    with pytest.raises(ValueError, match="transport-block identity"):
        auditor._audit_result(result, source_bytes=b"", committed_source=None)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("non_object_record", "per-channel records must be objects"),
        ("boolean_seed", "invalid independent-channel and block dimensions"),
        ("non_boolean_crc", "per-channel raw records do not match blocks_per_channel"),
        ("non_finite_cfo", "per-channel raw records do not match blocks_per_channel"),
        ("overflow_cfo", "per-channel raw records do not match blocks_per_channel"),
    ],
)
def test_auditor_rejects_malformed_retained_channel_records(mutation: str, message: str) -> None:
    result, source_bytes = _current_source_bound_result()
    config = result["experiment"]["config"]
    record = config["per_channel_draws"][0]
    if mutation == "non_object_record":
        config["per_channel_draws"][0] = None
    elif mutation == "boolean_seed":
        config["channel_seeds"][0] = True
    elif mutation == "non_boolean_crc":
        record["crc_pass_by_condition"]["clean"][0] = 1
    elif mutation == "non_finite_cfo":
        record["cfo_error_hz"][0] = float("nan")
    else:
        record["cfo_error_hz"][0] = 10**1000

    expected_message = None if mutation in {"non_finite_cfo", "overflow_cfo"} else message
    with pytest.raises(ValueError, match=expected_message):
        auditor._audit_result(result, source_bytes=source_bytes, committed_source=None)


def test_auditor_recomputes_waveform_diagnostic_extrema_from_channel_records() -> None:
    result: dict[str, Any] = json.loads(RESULT.read_text(encoding="utf-8"))
    records = result["experiment"]["config"]["per_channel_draws"]
    assert (
        max(row["samplewise_reference_max_absolute_error"] for row in records)
        == result["experiment"]["acceptance"]["observed"]["max_samplewise_reference_absolute_error"]
    )
    assert (
        min(row["wrong_order_waveform_rmse"] for row in records)
        == result["experiment"]["acceptance"]["observed"]["min_wrong_order_rmse"]
    )

    # Keep the result digest valid while corrupting the retained evidence.
    record = max(records, key=lambda row: row["samplewise_reference_max_absolute_error"])
    record["samplewise_reference_max_absolute_error"] += 1.0
    unhashed = dict(result)
    unhashed.pop("result_sha256")
    result["result_sha256"] = hashlib.sha256(_canonical_json_bytes(unhashed)).hexdigest()
    source_path = ROOT / result["provenance"]["source_path"]

    report = _audit_result(result, source_path.read_bytes(), None)

    assert report["checks"]["result_digest"] is True
    assert report["checks"]["retained_observed_statistics"] is False
    assert report["checks"]["acceptance_verdict"] is False
    assert report["status"] == "AUDIT_FAILED"


def test_auditor_detects_recomputed_hash_with_inconsistent_retained_arrays() -> None:
    result: dict[str, Any] = json.loads(RESULT.read_text(encoding="utf-8"))
    result["experiment"]["cfo_error_hz_by_block"][0] += 1.0
    unhashed = dict(result)
    unhashed.pop("result_sha256")
    result["result_sha256"] = hashlib.sha256(_canonical_json_bytes(unhashed)).hexdigest()
    source_path = ROOT / result["provenance"]["source_path"]

    report = _audit_result(result, source_path.read_bytes(), None)

    assert report["checks"]["result_digest"] is True
    assert report["checks"]["flattened_block_arrays"] is False
    assert report["status"] == "AUDIT_FAILED"


def test_deterministic_replay_reuses_exact_saved_seed_roster(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result, _ = _current_source_bound_result()
    experiment = result["experiment"]
    calls: list[tuple[tuple[int, ...], int, str]] = []

    def replay(
        *, channel_seeds: tuple[int, ...], blocks_per_channel: int, channel_estimator: str
    ) -> dict[str, Any]:
        calls.append((channel_seeds, blocks_per_channel, channel_estimator))
        return deepcopy(experiment)

    monkeypatch.setattr(auditor, "_run_receiver_probe", replay)

    report = audit_result_file(
        _write_current_source_bound_result(tmp_path), deterministic_replay=True
    )

    assert calls == [(tuple(experiment["config"]["channel_seeds"]), 8, "sionna_default_ls")]
    assert report["checks"]["deterministic_replay"] is True
    assert report["status"] == "VERIFIED_RESULT_AND_WORKTREE_SOURCE_PENDING_COMMIT"
    assert "not an independently implemented validation" in report["limitations"][0]


def test_deterministic_replay_rejects_a_mismatched_recomputed_experiment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result, _ = _current_source_bound_result()
    replayed = deepcopy(result["experiment"])
    replayed["acceptance"]["observed"]["max_samplewise_reference_absolute_error"] += 1.0
    monkeypatch.setattr(
        auditor,
        "_run_receiver_probe",
        lambda **_: replayed,
    )

    report = audit_result_file(
        _write_current_source_bound_result(tmp_path), deterministic_replay=True
    )

    assert report["checks"]["deterministic_replay"] is False
    assert report["status"] == "AUDIT_FAILED"


def test_deterministic_replay_enforces_the_bounded_replay_budget() -> None:
    result: dict[str, Any] = json.loads(RESULT.read_text(encoding="utf-8"))
    experiment = deepcopy(result["experiment"])
    experiment["config"]["blocks_per_channel"] = 9

    with pytest.raises(ValueError, match="64 channels × 8 blocks"):
        _replay_experiment(experiment)


def test_deterministic_replay_recreates_waveform_sidecar_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = {
        "schema": "umcd.g5_ordered_impairments_independent_channels.v8",
        "config": {
            "channel_seeds": [74_001],
            "blocks_per_channel": 2,
            "waveform_evidence": {
                "file_name": "saved.waveforms.npz",
            },
        },
    }
    calls: list[dict[str, Any]] = []

    def replay(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        kwargs["waveform_evidence_output"].write_bytes(b"synthetic evidence")
        return experiment

    monkeypatch.setattr(auditor, "_run_receiver_probe", replay)

    assert _replay_experiment(experiment) == experiment
    assert calls[0]["channel_seeds"] == (74_001,)
    assert calls[0]["blocks_per_channel"] == 2
    assert calls[0]["waveform_evidence_output"].name == "saved.waveforms.npz"
    assert calls[0]["waveform_evidence_output"].exists() is False


def test_waveform_equations_are_recomputed_from_hash_bound_raw_samples() -> None:
    tx = np.asarray([[[[1 + 1j, 2 - 1j, 3 + 0j, 4 + 2j]]]], dtype=np.complex64)
    tx_phase = np.zeros(tx.shape, dtype=np.float32)
    rx_phase = np.linspace(0.0, 0.7, tx.shape[-1], dtype=np.float32).reshape(tx.shape)
    propagated = tx[0].copy()
    propagated[..., 1:] += 0.5 * tx[0][..., :-1]
    unshifted_expected = propagated * np.exp(1j * rx_phase[0])
    actual = np.zeros_like(tx)
    actual[..., 1:] = unshifted_expected[..., :-1]
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        channel_seeds=np.asarray([74_001], dtype=np.int64),
        tx_waveform=tx,
        tx_phase_noise_rad=tx_phase,
        rx_phase_noise_rad=rx_phase,
        actual_capture=actual,
    )
    evidence = buffer.getvalue()
    config: dict[str, Any] = {
        "sample_rate_hz": 1_000_000.0,
        "cfo_hz": 0.0,
        "timing_offset_samples": 1,
        "tx_iq_gain_imbalance_db": 0.0,
        "tx_iq_phase_imbalance_rad": 0.0,
        "rx_iq_gain_imbalance_db": 0.0,
        "rx_iq_phase_imbalance_rad": 0.0,
        "waveform_evidence": {
            "schema": "umcd.g5_ordered_impairments_waveform_evidence.v1",
            "file_name": "fixture.waveforms.npz",
            "sha256": hashlib.sha256(evidence).hexdigest(),
            "byte_count": len(evidence),
            "channel_count": 1,
            "arrays": sorted(
                [
                    "channel_seeds",
                    "tx_waveform",
                    "tx_phase_noise_rad",
                    "rx_phase_noise_rad",
                    "actual_capture",
                ]
            ),
        },
    }
    records = [
        {
            "channel_seed": 74_001,
            "path_coefficients_real_imag": [[1.0, 0.0], [0.5, 0.0]],
            "path_delays_samples": [0, 1],
        }
    ]

    reference_error, wrong_order_error = _recompute_waveform_diagnostics(config, records, evidence)
    assert reference_error < 1e-6
    assert wrong_order_error > 1e-3

    tampered = bytearray(evidence)
    tampered[-1] ^= 0x01
    with pytest.raises(ValueError, match="hash-bound manifest"):
        _recompute_waveform_diagnostics(config, records, bytes(tampered))


@pytest.mark.parametrize("include_condition_inputs", [False, True])
def test_waveform_evidence_reconstructs_and_binds_receiver_inputs(
    include_condition_inputs: bool,
) -> None:
    tx = np.asarray([[[[1 + 1j, 2 - 1j, 3 + 0j, 4 + 2j]]]], dtype=np.complex64)
    tx_phase = np.zeros(tx.shape, dtype=np.float32)
    rx_phase = np.linspace(0.0, 0.7, tx.shape[-1], dtype=np.float32).reshape(tx.shape)
    propagated = tx[0].copy()
    propagated[..., 1:] += 0.5 * tx[0][..., :-1]
    expected = propagated * np.exp(1j * rx_phase[0])
    actual = np.zeros_like(tx)
    actual[..., 1:] = expected[..., :-1]
    awgn_real = np.full(tx.shape, 0.25, dtype=np.float32)
    awgn_imag = np.full(tx.shape, -0.5, dtype=np.float32)
    variance = 0.4
    scale = np.float32(np.sqrt(variance / 2.0))
    noise = np.empty(tx.shape, dtype=np.complex64)
    noise.real = awgn_real * scale
    noise.imag = awgn_imag * scale
    receiver_input = actual + noise
    receiver_input_digest = hashlib.sha256(
        np.ascontiguousarray(receiver_input[0]).tobytes()
    ).hexdigest()
    clean_capture = np.zeros_like(propagated)
    clean_capture[..., 1:] = propagated[..., :-1]
    clean_capture += noise[0]
    clean_input = np.zeros_like(clean_capture)
    clean_input[..., :-1] = clean_capture[..., 1:]
    corrected_input = np.zeros_like(receiver_input[0])
    corrected_input[..., :-1] = receiver_input[0, ..., 1:]
    condition_inputs = {
        "clean": clean_input[None, ...],
        "uncorrected": receiver_input,
        "corrected": corrected_input[None, ...],
    }
    condition_hashes = {
        condition: [hashlib.sha256(np.ascontiguousarray(value[0]).tobytes()).hexdigest()]
        for condition, value in condition_inputs.items()
    }

    buffer = io.BytesIO()
    archive_arrays: dict[str, np.ndarray] = {
        "channel_seeds": np.asarray([74_001], dtype=np.int64),
        "tx_waveform": tx,
        "tx_phase_noise_rad": tx_phase,
        "rx_phase_noise_rad": rx_phase,
        "actual_capture": actual,
        "awgn_standard_normal_real": awgn_real,
        "awgn_standard_normal_imag": awgn_imag,
    }
    if include_condition_inputs:
        archive_arrays.update(
            {
                "clean_receiver_input": condition_inputs["clean"],
                "uncorrected_receiver_input": condition_inputs["uncorrected"],
                "corrected_receiver_input": condition_inputs["corrected"],
            }
        )
    np.savez_compressed(buffer, **archive_arrays)
    evidence = buffer.getvalue()
    arrays = sorted(archive_arrays)
    config: dict[str, Any] = {
        "sample_rate_hz": 1_000_000.0,
        "cfo_hz": 0.0,
        "timing_offset_samples": 1,
        "tx_iq_gain_imbalance_db": 0.0,
        "tx_iq_phase_imbalance_rad": 0.0,
        "rx_iq_gain_imbalance_db": 0.0,
        "rx_iq_phase_imbalance_rad": 0.0,
        "waveform_awgn_complex_variance": variance,
        "noisy_receiver_input_sha256_by_channel": [receiver_input_digest],
        **(
            {"receiver_input_sha256_by_condition": condition_hashes}
            if include_condition_inputs
            else {}
        ),
        "waveform_evidence": {
            "schema": (
                "umcd.g5_ordered_impairments_waveform_evidence.v3"
                if include_condition_inputs
                else "umcd.g5_ordered_impairments_waveform_evidence.v2"
            ),
            "file_name": "fixture.waveforms.npz",
            "sha256": hashlib.sha256(evidence).hexdigest(),
            "byte_count": len(evidence),
            "channel_count": 1,
            "arrays": arrays,
        },
    }
    records = [
        {
            "channel_seed": 74_001,
            "path_coefficients_real_imag": [[1.0, 0.0], [0.5, 0.0]],
            "path_delays_samples": [0, 1],
            "timing_error_samples": [0],
            "cfo_error_hz": [0.0],
        }
    ]

    reference_error, wrong_order_error = _recompute_waveform_diagnostics(config, records, evidence)
    assert reference_error < 1e-6
    assert wrong_order_error > 1e-3

    if include_condition_inputs:
        config["receiver_input_sha256_by_condition"]["clean"] = ["0" * 64]
    else:
        config["noisy_receiver_input_sha256_by_channel"] = ["0" * 64]
    with pytest.raises(ValueError, match="receiver inputs differ"):
        _recompute_waveform_diagnostics(config, records, evidence)


def test_waveform_archive_preflight_rejects_forged_npy_shape_before_load() -> None:
    npy_header = io.BytesIO()
    np.lib.format.write_array_header_2_0(
        npy_header,
        {"descr": "<c8", "fortran_order": False, "shape": (100_000_000,)},
    )
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("tx_waveform.npy", npy_header.getvalue())

    with pytest.raises(ValueError, match="shape disagrees with its member size"):
        _preflight_waveform_archive(archive_buffer.getvalue(), required_arrays={"tx_waveform"})
