"""Audit of independent-channel and SNR-curve ordered-impairment results."""

from __future__ import annotations

import math
import zipfile
from typing import Any

from artifacts.rebinding import recorded_digest_matches
from scipy.stats import beta

from .constants import (
    _BLOCK_BOUND_REPORT_SCHEMAS,
    _CFO_CHANNEL_TAIL_SCHEMAS,
    _CUDA_DEVICE_REPORT_SCHEMAS,
    _PER_CHANNEL_WAVEFORM_DIAGNOSTIC_SCHEMAS,
    _RESULT_SCHEMA,
    REPOSITORY_ROOT,
)
from .evidence_files import (
    _estimator_source_checks,
)
from .hashing import (
    _canonical_json_bytes,
    _sha256,
)
from .references import (
    _is_finite_nonnegative,
    _is_finite_number,
    _same_float,
)
from .waveform import (
    _cluster_bootstrap_interval,
    _recompute_waveform_diagnostics,
)


def _audit_result(
    result: dict[str, Any],
    source_bytes: bytes,
    committed_source: bytes | None,
    replayed_experiment: dict[str, Any] | None = None,
    waveform_evidence_bytes: bytes | None = None,
    estimator_source_bytes: bytes | None = None,
    committed_estimator_source: bytes | None = None,
) -> dict[str, Any]:
    checks: dict[str, bool | None] = {}
    checks["schema"] = result.get("schema") == _RESULT_SCHEMA
    provenance = result.get("provenance")
    experiment = result.get("experiment")
    if not isinstance(provenance, dict) or not isinstance(experiment, dict):
        raise ValueError("result must contain provenance and experiment objects")
    config = experiment.get("config")
    acceptance = experiment.get("acceptance")
    if not isinstance(config, dict) or not isinstance(acceptance, dict):
        raise ValueError("experiment config and acceptance must be objects")

    unhashed = dict(result)
    supplied_result_digest = unhashed.pop("result_sha256", None)
    checks["result_digest"] = supplied_result_digest == _sha256(_canonical_json_bytes(unhashed))
    checks["source_digest_bindings"] = result.get("source_sha256") == provenance.get(
        "source_sha256"
    )
    checks["source_worktree_digest"] = recorded_digest_matches(
        provenance.get("source_sha256"), source_bytes, REPOSITORY_ROOT
    )
    estimator = config.get("channel_estimator", {"name": "sionna_default_ls"})
    if not isinstance(estimator, dict):
        raise ValueError("channel_estimator must be a provenance object")
    checks.update(
        _estimator_source_checks(
            estimator,
            source_bytes=estimator_source_bytes,
            committed_bytes=committed_estimator_source,
        )
    )
    checks["configuration_digest"] = result.get("configuration_sha256") == _sha256(
        _canonical_json_bytes(config)
    )
    protocol = provenance.get("protocol")
    expected_protocol = {
        "experiment_schema": experiment.get("schema"),
        "config": config,
        "acceptance_criteria": acceptance.get("criteria_frozen_before_run"),
    }
    checks["protocol_digest"] = protocol == expected_protocol and provenance.get(
        "protocol_sha256"
    ) == _sha256(_canonical_json_bytes(expected_protocol))
    checks["committed_source_digest"] = (
        None
        if committed_source is None
        else provenance.get("source_sha256") == _sha256(committed_source)
    )
    checks["deterministic_replay"] = (
        None if replayed_experiment is None else replayed_experiment == experiment
    )

    blocks_per_channel = config.get("blocks_per_channel")
    seeds = config.get("channel_seeds")
    records = config.get("per_channel_draws")
    if (
        isinstance(blocks_per_channel, bool)
        or not isinstance(blocks_per_channel, int)
        or blocks_per_channel < 1
        or not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(seeds) != len(set(seeds))
        or not isinstance(records, list)
        or len(records) != len(seeds)
    ):
        raise ValueError("invalid independent-channel and block dimensions")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("per-channel records must be objects")
    if [record.get("channel_seed") for record in records] != seeds:
        raise ValueError("per-channel records do not follow the declared unique seed roster")
    per_channel_diagnostic_fields = (
        "samplewise_reference_max_absolute_error",
        "wrong_order_waveform_rmse",
    )
    diagnostic_field_presence = [
        field in record for record in records for field in per_channel_diagnostic_fields
    ]
    if any(diagnostic_field_presence) and not all(diagnostic_field_presence):
        raise ValueError("per-channel waveform diagnostics must be present as a complete roster")
    has_per_channel_waveform_diagnostics = all(diagnostic_field_presence)
    report_schema = experiment.get("schema")
    if (
        report_schema in _PER_CHANNEL_WAVEFORM_DIAGNOSTIC_SCHEMAS
        and not has_per_channel_waveform_diagnostics
    ):
        raise ValueError("report schema requires per-channel waveform diagnostics")

    if any(
        not isinstance(record.get("crc_pass_by_condition"), dict)
        or any(
            not isinstance(record["crc_pass_by_condition"].get(condition), list)
            or len(record["crc_pass_by_condition"][condition]) != blocks_per_channel
            or any(
                not isinstance(value, bool) for value in record["crc_pass_by_condition"][condition]
            )
            for condition in ("clean", "corrected", "uncorrected")
        )
        or not isinstance(record.get("cfo_error_hz"), list)
        or len(record["cfo_error_hz"]) != blocks_per_channel
        or any(not _is_finite_number(value) for value in record.get("cfo_error_hz", []))
        or not isinstance(record.get("timing_error_samples"), list)
        or len(record["timing_error_samples"]) != blocks_per_channel
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in record.get("timing_error_samples", [])
        )
        or (
            has_per_channel_waveform_diagnostics
            and (
                not _is_finite_nonnegative(record.get("samplewise_reference_max_absolute_error"))
                or not _is_finite_nonnegative(record.get("wrong_order_waveform_rmse"))
            )
        )
        for record in records
    ):
        raise ValueError("per-channel raw records do not match blocks_per_channel")

    if experiment.get("schema") in _BLOCK_BOUND_REPORT_SCHEMAS:
        decoder = config.get("decoder_configuration")
        if decoder != {
            "api": "SionnaPUSCHLSReceiver.decode_pusch",
            "channel_estimator": estimator,
            "crc_status_semantics": "true_means_transport_block_crc_passed",
        }:
            raise ValueError("decoder_configuration disagrees with the selected estimator contract")
        receiver = config.get("receiver_configuration")
        if not isinstance(receiver, dict):
            raise ValueError("new receiver reports require a complete receiver_configuration")
        execution_device = (
            config.get("execution_device", "cpu")
            if experiment.get("schema") in _CUDA_DEVICE_REPORT_SCHEMAS
            else "cpu"
        )
        expected_receiver_values = {
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
            "interpolation_type": "lin",
            "n_size_grid": 6,
            "n_start_grid": 0,
            "n_start_bwp": 0,
            "receiver": "PUSCHLSChannelEstimator",
            "device": execution_device,
            "error_variance_semantics": (
                "sionna_linear_interpolated_marginal_variance_not_exact_covariance"
            ),
            "resource_grid": {
                "num_ofdm_symbols": 14,
                "fft_size": 72,
                "cyclic_prefix_length_samples": 7,
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
        if any(receiver.get(key) != value for key, value in expected_receiver_values.items()):
            raise ValueError("receiver_configuration disagrees with the frozen CPU PUSCH fixture")
        # Older report versions predate these explicit capability fields. Their
        # pinned Sionna 2.2.0 receiver could only run with these same defaults.
        # Newer reports retain both fields, and neither may claim an unsupported
        # hopping or transform-precoding mode.
        if (
            receiver.get("frequency_hopping", "neither") != "neither"
            or receiver.get("transform_precoding", False) is not False
        ):
            raise ValueError("receiver_configuration declares an unsupported PUSCH mode")
        runtime = provenance.get("runtime")
        runtime_packages = runtime.get("packages") if isinstance(runtime, dict) else None
        runtime_sionna_version = (
            runtime_packages.get("sionna-no-rt") or runtime_packages.get("sionna")
            if isinstance(runtime_packages, dict)
            else None
        )
        if (
            not isinstance(runtime_packages, dict)
            or not isinstance(receiver.get("sionna_version"), str)
            or not receiver.get("sionna_version")
            or receiver.get("sionna_version") != runtime_sionna_version
        ):
            raise ValueError("receiver Sionna version disagrees with runtime provenance")
        if experiment.get("schema") in _CUDA_DEVICE_REPORT_SCHEMAS:
            runtime = provenance.get("runtime")
            if (
                not isinstance(execution_device, str)
                or not execution_device.startswith("cuda:")
                or not isinstance(runtime, dict)
                or runtime.get("torch_device") != execution_device
            ):
                raise ValueError("CUDA execution device disagrees with receiver/runtime provenance")

        for seed, record in zip(seeds, records, strict=True):
            blocks = record.get("transport_blocks")
            payload_seed = record.get("payload_seed")
            if (
                isinstance(payload_seed, bool)
                or not isinstance(payload_seed, int)
                or payload_seed != seed + 1_000_000
                or not isinstance(blocks, list)
                or len(blocks) != blocks_per_channel
            ):
                raise ValueError("transport-block roster does not match channel and payload seeds")
            for index, block in enumerate(blocks):
                expected_id = f"channel-{seed}:payload-{payload_seed}:block-{index}"
                if (
                    not isinstance(block, dict)
                    or isinstance(block.get("block_index"), bool)
                    or not isinstance(block.get("block_index"), int)
                    or block.get("transport_block_id") != expected_id
                    or block.get("block_index") != index
                    or isinstance(block.get("payload_seed"), bool)
                    or not isinstance(block.get("payload_seed"), int)
                    or block.get("payload_seed") != payload_seed
                    or not isinstance(block.get("crc_pass_by_condition"), dict)
                    or any(
                        not isinstance(block["crc_pass_by_condition"].get(condition), bool)
                        for condition in ("clean", "corrected", "uncorrected")
                    )
                    or block.get("crc_pass_by_condition")
                    != {
                        condition: record["crc_pass_by_condition"][condition][index]
                        for condition in ("clean", "corrected", "uncorrected")
                    }
                    or block.get("timing_error_samples") != record["timing_error_samples"][index]
                    or isinstance(block.get("timing_error_samples"), bool)
                    or not isinstance(block.get("timing_error_samples"), int)
                    or isinstance(block.get("cfo_error_hz"), bool)
                    or not _is_finite_number(block.get("cfo_error_hz"))
                    or not _same_float(block.get("cfo_error_hz"), record["cfo_error_hz"][index])
                ):
                    raise ValueError(
                        "transport-block identity or outcome disagrees with channel record"
                    )
        checks["receiver_configuration_binding"] = True
        checks["transport_block_roster_binding"] = True
    else:
        checks["receiver_configuration_binding"] = None
        checks["transport_block_roster_binding"] = None

    flat_cfo = [float(value) for record in records for value in record["cfo_error_hz"]]
    flat_timing = [int(value) for record in records for value in record["timing_error_samples"]]
    checks["flattened_block_arrays"] = (
        experiment.get("cfo_error_hz_by_block") == flat_cfo
        and experiment.get("timing_error_samples_by_block") == flat_timing
    )
    outcomes = experiment.get("paired_outcomes")
    if not isinstance(outcomes, dict):
        raise ValueError("paired_outcomes must be an object")
    condition_rates: dict[str, list[float]] = {}
    total_blocks = len(seeds) * blocks_per_channel
    outcomes_consistent = True
    for condition in ("clean", "corrected", "uncorrected"):
        pass_vectors = [record["crc_pass_by_condition"][condition] for record in records]
        failures_by_channel = [blocks_per_channel - sum(values) for values in pass_vectors]
        rates = [failures / blocks_per_channel for failures in failures_by_channel]
        condition_rates[condition] = rates
        failed = sum(failures_by_channel)
        summary = outcomes.get(condition)
        if not isinstance(summary, dict):
            outcomes_consistent = False
            continue
        outcomes_consistent &= (
            summary.get("blocks") == total_blocks
            and summary.get("crc_failures") == failed
            and _same_float(summary.get("bler"), failed / total_blocks)
            and _same_float(summary.get("channel_macro_bler"), sum(rates) / len(rates))
        )
    checks["crc_and_bler_summaries"] = outcomes_consistent

    timing_limit = acceptance.get("criteria_frozen_before_run", {})
    observed = acceptance.get("observed", {})
    absolute_cfo = sorted(abs(value) for value in flat_cfo)
    p95_index = math.ceil(0.95 * len(absolute_cfo)) - 1
    recomputed = {
        "max_abs_timing_error_samples": max(abs(value) for value in flat_timing),
        "p95_abs_cfo_error_hz": absolute_cfo[p95_index],
        "max_abs_cfo_error_hz": absolute_cfo[-1],
        "clean_bler": outcomes["clean"]["bler"],
        "impaired_corrected_bler": outcomes["corrected"]["bler"],
    }
    if has_per_channel_waveform_diagnostics:
        recomputed["max_samplewise_reference_absolute_error"] = max(
            float(record["samplewise_reference_max_absolute_error"]) for record in records
        )
        recomputed["min_wrong_order_rmse"] = min(
            float(record["wrong_order_waveform_rmse"]) for record in records
        )
        checks["waveform_diagnostic_extrema"] = all(
            observed.get(key) == recomputed[key]
            for key in (
                "max_samplewise_reference_absolute_error",
                "min_wrong_order_rmse",
            )
        )
    else:
        for key in (
            "max_samplewise_reference_absolute_error",
            "min_wrong_order_rmse",
        ):
            if not _is_finite_nonnegative(observed.get(key)):
                raise ValueError(
                    "legacy reported waveform diagnostics must be finite and nonnegative"
                )
            recomputed[key] = observed[key]
        checks["waveform_diagnostic_extrema"] = None
    checks["retained_observed_statistics"] = all(
        observed.get(key) == value
        for key, value in recomputed.items()
        if has_per_channel_waveform_diagnostics
        or key not in {"max_samplewise_reference_absolute_error", "min_wrong_order_rmse"}
    )
    if config.get("waveform_evidence") is None:
        checks["waveform_equation_recomputation"] = (
            None
            if experiment.get("schema")
            not in {
                "umcd.g5_ordered_impairments_independent_channels.v8",
                "umcd.g5_ordered_impairments_independent_channels.v10",
                "umcd.g5_ordered_impairments_independent_channels.v12",
                "umcd.g5_ordered_impairments_independent_channels.v14",
                "umcd.g5_ordered_impairments_independent_channels.v16",
                "umcd.g5_ordered_impairments_independent_channels.v18",
                "umcd.g5_ordered_impairments_independent_channels.v19",
                "umcd.g5_ordered_impairments_independent_channels.v20",
                "umcd.g5_ordered_impairments_independent_channels.v21",
                "umcd.g5_ordered_impairments_independent_channels.v22",
                "umcd.g5_ordered_impairments_independent_channels.v23",
                "umcd.g5_ordered_impairments_independent_channels.v24",
            }
            else False
        )
    elif waveform_evidence_bytes is None:
        checks["waveform_equation_recomputation"] = False
    else:
        try:
            waveform_maximum, wrong_order_minimum = _recompute_waveform_diagnostics(
                config, records, waveform_evidence_bytes
            )
            checks["waveform_equation_recomputation"] = (
                abs(
                    waveform_maximum
                    - float(observed.get("max_samplewise_reference_absolute_error", math.inf))
                )
                <= 5e-6
                and abs(
                    wrong_order_minimum - float(observed.get("min_wrong_order_rmse", -math.inf))
                )
                <= 5e-6
                and waveform_maximum
                < float(timing_limit.get("max_samplewise_reference_absolute_error", -math.inf))
                and wrong_order_minimum > float(timing_limit.get("min_wrong_order_rmse", math.inf))
            )
        except (
            OSError,
            EOFError,
            ValueError,
            TypeError,
            KeyError,
            OverflowError,
            zipfile.BadZipFile,
        ):
            checks["waveform_equation_recomputation"] = False

    inference = acceptance.get("inference", {})
    paired_deltas = [
        corrected - clean
        for clean, corrected in zip(
            condition_rates["clean"], condition_rates["corrected"], strict=True
        )
    ]
    bootstrap_interval = _cluster_bootstrap_interval(
        paired_deltas,
        seed=inference.get("bootstrap_seed"),
        draws=inference.get("bootstrap_draws"),
    )
    expected_mean_delta = sum(paired_deltas) / len(paired_deltas)
    interval = inference.get("paired_corrected_minus_clean_channel_bootstrap_95_percent_interval")
    checks["paired_cluster_inference"] = (
        _same_float(inference.get("paired_corrected_minus_clean_mean_bler"), expected_mean_delta)
        and isinstance(interval, list)
        and len(interval) == 2
        and _same_float(interval[0], bootstrap_interval[0])
        and _same_float(interval[1], bootstrap_interval[1])
    )

    if experiment.get("schema") in _CFO_CHANNEL_TAIL_SCHEMAS:
        tails = inference.get("cfo_channel_tail_diagnostics")
        expected_tails = []
        for threshold in (
            timing_limit["p95_abs_cfo_error_hz"],
            timing_limit["max_abs_cfo_error_hz"],
        ):
            exceedances = sum(
                any(abs(float(value)) >= threshold for value in record["cfo_error_hz"])
                for record in records
            )
            upper = (
                1.0
                if exceedances == len(records)
                else float(beta.ppf(0.95, exceedances + 1, len(records) - exceedances))
            )
            expected_tails.append((threshold, exceedances, upper))
        checks["cfo_cluster_tail_bounds"] = (
            isinstance(tails, list)
            and len(tails) == len(expected_tails)
            and all(
                row.get("absolute_error_threshold_hz") == threshold
                and row.get("channels_with_any_block_exceedance") == exceedances
                and row.get("independent_channels") == len(records)
                and _same_float(
                    row.get("exact_one_sided_95_percent_channel_prevalence_upper_bound"), upper
                )
                for row, (threshold, exceedances, upper) in zip(tails, expected_tails, strict=True)
            )
        )
    else:
        checks["cfo_cluster_tail_bounds"] = None

    crc_channel_tail_diagnostics = []
    for condition in ("clean", "corrected", "uncorrected"):
        failed_channels = sum(
            not all(record["crc_pass_by_condition"][condition]) for record in records
        )
        upper = (
            1.0
            if failed_channels == len(records)
            else float(beta.ppf(0.95, failed_channels + 1, len(records) - failed_channels))
        )
        crc_channel_tail_diagnostics.append(
            {
                "condition": condition,
                "channels_with_any_block_failure": failed_channels,
                "independent_channels": len(records),
                "blocks_per_channel": blocks_per_channel,
                "exact_one_sided_95_percent_channel_prevalence_upper_bound": upper,
            }
        )
    if experiment.get("schema") in {
        "umcd.g5_ordered_impairments_independent_channels.v11",
        "umcd.g5_ordered_impairments_independent_channels.v12",
        "umcd.g5_ordered_impairments_independent_channels.v13",
        "umcd.g5_ordered_impairments_independent_channels.v14",
        "umcd.g5_ordered_impairments_independent_channels.v15",
        "umcd.g5_ordered_impairments_independent_channels.v16",
        "umcd.g5_ordered_impairments_independent_channels.v17",
        "umcd.g5_ordered_impairments_independent_channels.v18",
        "umcd.g5_ordered_impairments_independent_channels.v19",
        "umcd.g5_ordered_impairments_independent_channels.v20",
        "umcd.g5_ordered_impairments_independent_channels.v21",
        "umcd.g5_ordered_impairments_independent_channels.v22",
        "umcd.g5_ordered_impairments_independent_channels.v23",
        "umcd.g5_ordered_impairments_independent_channels.v24",
    }:
        retained_crc_tails = inference.get("crc_channel_failure_diagnostics")
        checks["crc_channel_cluster_tail_bounds"] = (
            isinstance(retained_crc_tails, list)
            and len(retained_crc_tails) == len(crc_channel_tail_diagnostics)
            and all(
                isinstance(row, dict)
                and row.get("condition") == expected["condition"]
                and row.get("channels_with_any_block_failure")
                == expected["channels_with_any_block_failure"]
                and row.get("independent_channels") == expected["independent_channels"]
                and row.get("blocks_per_channel") == expected["blocks_per_channel"]
                and _same_float(
                    row.get("exact_one_sided_95_percent_channel_prevalence_upper_bound"),
                    expected["exact_one_sided_95_percent_channel_prevalence_upper_bound"],
                )
                for row, expected in zip(
                    retained_crc_tails, crc_channel_tail_diagnostics, strict=True
                )
            )
        )
    else:
        checks["crc_channel_cluster_tail_bounds"] = None

    checks["acceptance_verdict"] = acceptance.get("pass") == (
        recomputed["max_abs_timing_error_samples"] <= timing_limit["max_abs_timing_error_samples"]
        and recomputed["p95_abs_cfo_error_hz"] < timing_limit["p95_abs_cfo_error_hz"]
        and recomputed["max_abs_cfo_error_hz"] < timing_limit["max_abs_cfo_error_hz"]
        and recomputed["max_samplewise_reference_absolute_error"]
        < timing_limit["max_samplewise_reference_absolute_error"]
        and recomputed["min_wrong_order_rmse"] > timing_limit["min_wrong_order_rmse"]
        and recomputed["clean_bler"] <= timing_limit["max_clean_reference_bler"]
        and recomputed["impaired_corrected_bler"] <= timing_limit["max_impaired_corrected_bler"]
    )

    optional_checks = {
        "committed_source_digest",
        "estimator_committed_source_digest",
        "source_worktree_digest",
        "deterministic_replay",
        "waveform_equation_recomputation",
        "crc_channel_cluster_tail_bounds",
        "receiver_configuration_binding",
        "transport_block_roster_binding",
        "waveform_diagnostic_extrema",
    }
    all_pass = all(value is True for key, value in checks.items() if key not in optional_checks)
    all_pass &= checks["deterministic_replay"] is not False
    if experiment.get("schema") in {
        "umcd.g5_ordered_impairments_independent_channels.v8",
        "umcd.g5_ordered_impairments_independent_channels.v10",
        "umcd.g5_ordered_impairments_independent_channels.v12",
        "umcd.g5_ordered_impairments_independent_channels.v14",
        "umcd.g5_ordered_impairments_independent_channels.v16",
        "umcd.g5_ordered_impairments_independent_channels.v18",
        "umcd.g5_ordered_impairments_independent_channels.v19",
        "umcd.g5_ordered_impairments_independent_channels.v20",
        "umcd.g5_ordered_impairments_independent_channels.v21",
        "umcd.g5_ordered_impairments_independent_channels.v22",
        "umcd.g5_ordered_impairments_independent_channels.v23",
        "umcd.g5_ordered_impairments_independent_channels.v24",
    }:
        all_pass &= checks["waveform_equation_recomputation"] is True
    if experiment.get("schema") in {
        "umcd.g5_ordered_impairments_independent_channels.v11",
        "umcd.g5_ordered_impairments_independent_channels.v12",
        "umcd.g5_ordered_impairments_independent_channels.v13",
        "umcd.g5_ordered_impairments_independent_channels.v14",
        "umcd.g5_ordered_impairments_independent_channels.v15",
        "umcd.g5_ordered_impairments_independent_channels.v16",
        "umcd.g5_ordered_impairments_independent_channels.v17",
        "umcd.g5_ordered_impairments_independent_channels.v18",
        "umcd.g5_ordered_impairments_independent_channels.v19",
        "umcd.g5_ordered_impairments_independent_channels.v20",
    }:
        all_pass &= checks["crc_channel_cluster_tail_bounds"] is True
    source_worktree_matches = checks["source_worktree_digest"]
    committed_matches = checks["committed_source_digest"]
    status = (
        "AUDIT_FAILED"
        if committed_matches is False
        else "VERIFIED_AGAINST_COMMITTED_SOURCE"
        if all_pass and committed_matches is True
        else "VERIFIED_RESULT_AND_WORKTREE_SOURCE_PENDING_COMMIT"
        if all_pass and source_worktree_matches
        else "AUDIT_FAILED"
        if not source_worktree_matches
        else "AUDIT_FAILED"
    )
    limitations = [
        "A passing audit verifies this declared synthetic fixture only; it does not establish a field channel population, G5 closure, or production RT authorization."
    ]
    if replayed_experiment is None:
        if checks["waveform_equation_recomputation"] is not True:
            limitations.insert(
                0,
                "The artifact does not provide hash-verified raw waveform evidence for an independent recomputation of the impairment equations.",
            )
        else:
            limitations.insert(
                0,
                "The auditor independently recomputes samplewise-reference and wrong-order waveform diagnostics from the hash-verified raw samples using NumPy equations.",
            )
    if checks["waveform_diagnostic_extrema"] is None:
        limitations.insert(
            0,
            "This legacy report schema predates per-channel waveform-diagnostic retention; those two frozen scalar criteria are compared with the result's reported acceptance values but are not independently recomputed.",
        )
    else:
        limitations.insert(
            0,
            "Deterministic replay checks reproducibility with the source-bound implementation; it is not an independently implemented validation of the reference mathematics.",
        )
    return {
        "schema": "umcd.g5_ordered_impairments_result_audit.v1",
        "status": status,
        "checks": checks,
        "independent_channel_count": len(records),
        "blocks_per_channel": blocks_per_channel,
        "limitations": limitations,
    }
