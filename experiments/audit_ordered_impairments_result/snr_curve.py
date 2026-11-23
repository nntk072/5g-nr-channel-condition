"""Audit of independent-channel and SNR-curve ordered-impairment results."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from artifacts.rebinding import recorded_digest_matches

from experiments.acceptance import (
    MIN_INDEPENDENT_ACCEPTANCE_CHANNELS,
    SNR_CURVE_BOOTSTRAP_BATCH_SIZE,
    SNR_CURVE_BOOTSTRAP_DRAWS,
    SNR_CURVE_BOOTSTRAP_SEED,
    SNR_EXECUTION_ACCEPTANCE_SCHEMA,
    SNR_REVIEWER_POLICY_PURPOSE,
    SNR_REVIEWER_POLICY_SIGNATURE_NAMESPACE,
    SNR_TIMESTAMPED_ACCEPTANCE_SCHEMA,
    _verify_reviewer_policy,
)

from .constants import (
    _CUDA_CURVE_BASE_VERSIONS,
    _HEX,
    _SNR_ACCEPTANCE_SIGNATURE_NAMESPACE,
    REPOSITORY_ROOT,
)
from .evidence_files import (
    _curve_report_schema_status,
)
from .hashing import (
    _canonical_json_bytes,
    _sha256,
)
from .preregistration import (
    _declared_preregistration_order_matches,
    _execution_context_matches,
    _inference_seed_roster_consistent,
    _is_sha256,
    _verify_preregistration_signature,
)
from .references import (
    _is_finite_number,
    _same_float,
)
from .waveform import (
    _recompute_waveform_diagnostics,
)


def _audit_independent_snr_curve(
    result: dict[str, Any],
    source_bytes: bytes,
    committed_source: bytes | None,
    waveform_evidence_bytes: bytes | None = None,
    acceptance_signature_path: Path | None = None,
    trusted_allowed_signers_path: Path | None = None,
    reviewer_policy_path: Path | None = None,
    reviewer_policy_signature_path: Path | None = None,
    trusted_reviewer_allowed_signers_path: Path | None = None,
    reviewer_policy_signer_id: str | None = None,
) -> dict[str, Any]:
    """Audit curve identity, shared roster, per-point noise, and retained CRC counts."""
    experiment = result.get("experiment")
    provenance = result.get("provenance")
    if not isinstance(experiment, dict) or not isinstance(provenance, dict):
        raise ValueError("SNR curve result must contain experiment and provenance objects")
    config = experiment.get("config")
    acceptance = experiment.get("acceptance")
    points = experiment.get("points")
    if (
        not isinstance(config, dict)
        or not isinstance(acceptance, dict)
        or not isinstance(points, list)
    ):
        raise ValueError("SNR curve config, acceptance, and points have invalid types")
    curve_schema = experiment.get("schema")
    if curve_schema not in {
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v1",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v2",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v3",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v4",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v5",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v6",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v7",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v8",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v9",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v10",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v11",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v12",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v13",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v14",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v15",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v16",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v17",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v25",
    }:
        raise ValueError("unsupported independent-channel SNR curve schema")

    checks: dict[str, bool | None] = {}
    unhashed = dict(result)
    supplied_digest = unhashed.pop("result_sha256", None)
    checks["result_digest"] = supplied_digest == _sha256(_canonical_json_bytes(unhashed))
    checks["source_worktree_digest"] = recorded_digest_matches(
        provenance.get("source_sha256"), source_bytes, REPOSITORY_ROOT
    )
    checks["source_digest_bindings"] = result.get("source_sha256") == provenance.get(
        "source_sha256"
    )
    checks["configuration_digest"] = result.get("configuration_sha256") == _sha256(
        _canonical_json_bytes(config)
    )
    expected_protocol = {
        "experiment_schema": experiment.get("schema"),
        "config": config,
        "acceptance_criteria": acceptance.get("criteria_frozen_before_run"),
    }
    checks["protocol_digest"] = provenance.get("protocol") == expected_protocol and provenance.get(
        "protocol_sha256"
    ) == _sha256(_canonical_json_bytes(expected_protocol))
    curve_schema_version = curve_schema.rsplit("v", maxsplit=1)[1]
    schema_path = (
        REPOSITORY_ROOT
        / "datasets"
        / "schema"
        / f"ordered-impairments-snr-curve.v{curve_schema_version}.schema.json"
    )
    schema_digest, schema_valid = _curve_report_schema_status(experiment, schema_path)
    checks["schema_digest"] = config.get("schema_sha256") == schema_digest
    checks["curve_report_schema"] = schema_valid
    checks["committed_source_digest"] = (
        None
        if committed_source is None
        else provenance.get("source_sha256") == _sha256(committed_source)
    )

    actual_curve_version = int(curve_schema.rsplit("v", maxsplit=1)[1])
    execution_device = config.get("execution_device", "cpu")
    runtime = provenance.get("runtime")
    if actual_curve_version in _CUDA_CURVE_BASE_VERSIONS:
        checks["execution_device_binding"] = (
            isinstance(execution_device, str)
            and execution_device.startswith("cuda:")
            and execution_device[5:].isdigit()
            and str(int(execution_device[5:])) == execution_device[5:]
            and isinstance(runtime, dict)
            and runtime.get("torch_device") == execution_device
        )
        curve_schema = (
            "umcd.g5_ordered_impairments_independent_channels_snr_curve.v"
            f"{_CUDA_CURVE_BASE_VERSIONS[actual_curve_version]}"
        )
    else:
        checks["execution_device_binding"] = (
            "execution_device" not in config
            and isinstance(runtime, dict)
            and runtime.get("torch_device") == "cpu"
        )

    seeds = config.get("channel_seeds")
    blocks = config.get("blocks_per_channel")
    snr_grid = config.get("snr_db")
    frozen_acceptance = acceptance.get("criteria_frozen_before_run")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
        or len(seeds) != len(set(seeds))
        or isinstance(blocks, bool)
        or not isinstance(blocks, int)
        or blocks < 1
        or not isinstance(snr_grid, list)
        or len(snr_grid) != len(points)
        or (
            curve_schema
            in {
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v2",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v3",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v4",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v5",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v6",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v7",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v8",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v9",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v10",
            }
            and (len(snr_grid) > 64 or config.get("maximum_snr_points") != 64)
        )
        or any(not _is_finite_number(value) for value in snr_grid)
        or any(right <= left for left, right in zip(snr_grid, snr_grid[1:]))
        or not isinstance(frozen_acceptance, dict)
        or (
            curve_schema
            not in {
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v5",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v6",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v7",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v8",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v10",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v11",
            }
            and (
                acceptance.get("pass") is not False
                or frozen_acceptance.get("status") != "descriptive_only"
            )
        )
        or (
            curve_schema
            in {
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v5",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v6",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v7",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v8",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v10",
            }
            and frozen_acceptance.get("status") != "contract_supplied_before_run"
        )
    ):
        raise ValueError("invalid SNR curve seed, grid, or diagnostic-only acceptance contract")
    expected_total = len(snr_grid) * len(seeds) * blocks
    checks["bounded_snr_grid"] = (
        None
        if curve_schema == "umcd.g5_ordered_impairments_independent_channels_snr_curve.v1"
        else len(snr_grid) <= 64 and config.get("maximum_snr_points") == 64
    )
    checks["bounded_total_work"] = (
        config.get("transport_blocks_per_snr_point") == len(seeds) * blocks
        and config.get("transport_blocks_across_curve") == expected_total
        and expected_total <= config.get("maximum_transport_blocks_across_curve", 0)
        and config.get("maximum_transport_blocks_across_curve") == 16_384
    )

    points_consistent = True
    shared_path_signatures: list[tuple[Any, Any]] | None = None
    for snr_index, (snr, point) in enumerate(zip(snr_grid, points, strict=True)):
        if not isinstance(point, dict) or point.get("snr_db") != snr:
            points_consistent = False
            continue
        report = point.get("experiment")
        point_config = report.get("config") if isinstance(report, dict) else None
        estimator = (
            point_config.get("channel_estimator") if isinstance(point_config, dict) else None
        )
        point_evidence = (
            point_config.get("waveform_evidence") if isinstance(point_config, dict) else None
        )
        receiver_configuration = (
            point_config.get("receiver_configuration") if isinstance(point_config, dict) else None
        )
        cuda_execution = execution_device != "cpu"
        if cuda_execution:
            expected_point_version = (
                24
                if config.get("channel_estimator") == "finite_delay_dft"
                and point_evidence is not None
                else 23
                if config.get("channel_estimator") == "finite_delay_dft"
                else 22
                if point_evidence is not None
                else 21
            )
        else:
            expected_point_version = (
                20
                if config.get("channel_estimator") == "finite_delay_dft"
                and point_evidence is not None
                else 17
                if config.get("channel_estimator") == "finite_delay_dft"
                else 19
                if point_evidence is not None
                else 15
            )
        expected_point_schema = (
            f"umcd.g5_ordered_impairments_independent_channels.v{expected_point_version}"
        )
        if (
            not isinstance(report, dict)
            or report.get("schema") != expected_point_schema
            or not isinstance(point_config, dict)
            or point_config.get("channel_seeds") != seeds
            or point_config.get("blocks_per_channel") != blocks
            or point_config.get("execution_device", "cpu") != execution_device
            or (
                cuda_execution
                and (
                    not isinstance(receiver_configuration, dict)
                    or receiver_configuration.get("device") != execution_device
                )
            )
            or not isinstance(estimator, dict)
            or estimator.get("name") != config.get("channel_estimator")
        ):
            points_consistent = False
            continue
        try:
            variance = math.pow(10.0, -float(snr) / 10.0)
        except OverflowError:
            points_consistent = False
            continue
        if not _same_float(
            point_config.get("waveform_awgn_complex_variance"), variance
        ) or not _same_float(point_config.get("nominal_snr_db"), float(snr)):
            points_consistent = False
            continue
        powers = point_config.get("transmit_waveform_power_by_channel")
        receiver_input_hashes = point_config.get("noisy_receiver_input_sha256_by_channel")
        if receiver_input_hashes is not None and (
            not isinstance(receiver_input_hashes, list)
            or len(receiver_input_hashes) != len(seeds)
            or any(
                not isinstance(digest, str) or len(digest) != 64 or not set(digest) <= _HEX
                for digest in receiver_input_hashes
            )
        ):
            points_consistent = False
        if (
            not isinstance(powers, list)
            or len(powers) != len(seeds)
            or any(not _is_finite_number(power) or power <= 0 for power in powers)
            or not _same_float(
                point_config.get("mean_transmit_waveform_power"), sum(powers) / len(powers)
            )
        ):
            points_consistent = False
            continue
        records = point_config.get("per_channel_draws")
        outcomes = report.get("paired_outcomes")
        if (
            not isinstance(records, list)
            or len(records) != len(seeds)
            or not isinstance(outcomes, dict)
        ):
            points_consistent = False
            continue
        if [row.get("channel_seed") if isinstance(row, dict) else None for row in records] != seeds:
            points_consistent = False
            continue
        if (
            point_config.get("payload_seed_offset") != 1_000_000
            or point_config.get("noise_seed_offset") != 1_200_000
            or point_config.get("tx_phase_noise_seed_offset") != 1_300_000
            or point_config.get("rx_phase_noise_seed_offset") != 1_400_000
        ):
            points_consistent = False
            continue
        path_signatures: list[tuple[Any, Any]] = []
        for seed, row in zip(seeds, records, strict=True):
            expected_seeds = {
                "payload_seed": seed + 1_000_000,
                "noise_seed": seed + 1_200_000,
                "tx_phase_noise_seed": seed + 1_300_000,
                "rx_phase_noise_seed": seed + 1_400_000,
            }
            if any(row.get(key) != value for key, value in expected_seeds.items()):
                points_consistent = False
            delays = row.get("path_delays_samples")
            coefficients = row.get("path_coefficients_real_imag")
            if not isinstance(delays, list) or not isinstance(coefficients, list):
                points_consistent = False
            path_signatures.append((delays, coefficients))
            transport_blocks = row.get("transport_blocks")
            if (
                not isinstance(transport_blocks, list)
                or len(transport_blocks) != blocks
                or any(not isinstance(block, dict) for block in transport_blocks)
            ):
                points_consistent = False
            else:
                for block_index, block in enumerate(transport_blocks):
                    if (
                        not isinstance(block, dict)
                        or block.get("transport_block_id")
                        != f"channel-{seed}:payload-{seed + 1_000_000}:block-{block_index}"
                        or block.get("block_index") != block_index
                        or block.get("payload_seed") != seed + 1_000_000
                    ):
                        points_consistent = False
        if shared_path_signatures is None:
            shared_path_signatures = path_signatures
        elif path_signatures != shared_path_signatures:
            points_consistent = False
        point_blocks = len(seeds) * blocks
        for condition in ("clean", "uncorrected", "corrected"):
            vectors = [
                crc.get(condition)
                if isinstance(crc := row.get("crc_pass_by_condition"), dict)
                else None
                for row in records
            ]
            validated_vectors: list[list[bool]] = []
            vectors_valid = True
            for vector in vectors:
                if (
                    not isinstance(vector, list)
                    or len(vector) != blocks
                    or any(not isinstance(value, bool) for value in vector)
                ):
                    vectors_valid = False
                    break
                validated_vectors.append([value for value in vector if isinstance(value, bool)])
            if not vectors_valid:
                points_consistent = False
                continue
            for channel_index, vector in enumerate(validated_vectors):
                blocks_for_channel = records[channel_index]["transport_blocks"]
                for block_index, block in enumerate(blocks_for_channel):
                    block_crc = (
                        block.get("crc_pass_by_condition") if isinstance(block, dict) else None
                    )
                    if (
                        not isinstance(block_crc, dict)
                        or block_crc.get(condition) != vector[block_index]
                    ):
                        points_consistent = False
                        break
            failures = sum(int(not value) for vector in validated_vectors for value in vector)
            channel_rates = [
                sum(int(not value) for value in vector) / blocks for vector in validated_vectors
            ]
            summary = outcomes.get(condition)
            if not isinstance(summary, dict) or not (
                summary.get("blocks") == point_blocks
                and summary.get("crc_failures") == failures
                and _same_float(summary.get("bler"), failures / point_blocks)
                and _same_float(summary.get("channel_macro_bler"), sum(channel_rates) / len(seeds))
            ):
                points_consistent = False
    checks["paired_roster_noise_and_crc_summaries"] = points_consistent
    curve_inference = experiment.get("curve_inference")
    inference_consistent = isinstance(curve_inference, dict)
    if inference_consistent:
        assert isinstance(curve_inference, dict)
        conditions = ("clean", "uncorrected", "corrected")
        by_condition = curve_inference.get("by_condition")
        seed_roster_consistent = _inference_seed_roster_consistent(
            curve_schema,
            curve_inference,
            seeds,
        )
        has_paired_contrast = curve_schema in {
            "umcd.g5_ordered_impairments_independent_channels_snr_curve.v4",
            "umcd.g5_ordered_impairments_independent_channels_snr_curve.v5",
            "umcd.g5_ordered_impairments_independent_channels_snr_curve.v6",
            "umcd.g5_ordered_impairments_independent_channels_snr_curve.v7",
            "umcd.g5_ordered_impairments_independent_channels_snr_curve.v8",
            "umcd.g5_ordered_impairments_independent_channels_snr_curve.v9",
            "umcd.g5_ordered_impairments_independent_channels_snr_curve.v10",
            "umcd.g5_ordered_impairments_independent_channels_snr_curve.v11",
        }
        expected_inference_method = (
            "channel-cluster percentile intervals with simultaneous max-deviation band over "
            "conditions and paired correction effects"
            if has_paired_contrast
            else "channel-cluster percentile intervals with simultaneous max-deviation band"
        )
        by_paired_contrast = curve_inference.get("by_paired_contrast")
        inference_consistent = (
            curve_inference.get("cluster_count") == len(seeds)
            and seed_roster_consistent
            and curve_inference.get("family_size")
            == len(snr_grid) * (len(conditions) + int(has_paired_contrast))
            and curve_inference.get("bootstrap_seed") == SNR_CURVE_BOOTSTRAP_SEED
            and curve_inference.get("bootstrap_draws") == SNR_CURVE_BOOTSTRAP_DRAWS
            and curve_inference.get("unit")
            == "independent synthetic channel; transport blocks are nested"
            and curve_inference.get("method") == expected_inference_method
            and (
                curve_inference.get("paired_contrast_definition")
                == "uncorrected macro-BLER minus corrected macro-BLER; positive values favor correction"
                if has_paired_contrast
                else curve_inference.get("paired_contrast_definition") is None
            )
            and isinstance(by_condition, dict)
            and set(by_condition) == set(conditions)
            and (
                isinstance(by_paired_contrast, list)
                if has_paired_contrast
                else by_paired_contrast is None
            )
        )
        if inference_consistent and len(seeds) < 2:
            assert isinstance(by_condition, dict)
            inference_consistent = (
                curve_inference.get("status") == "unavailable"
                and isinstance(curve_inference.get("reason"), str)
                and all(by_condition[condition] == [] for condition in conditions)
                and (not has_paired_contrast or by_paired_contrast == [])
            )
        elif inference_consistent:
            assert isinstance(by_condition, dict)
            inference_consistent = (
                points_consistent and curve_inference.get("status") == "available"
            )
            if inference_consistent:
                reports_for_rates = [point["experiment"] for point in points]
                rates: np.ndarray = np.empty(
                    (len(points), len(conditions), len(seeds)), dtype=np.float64
                )
                for snr_index, report in enumerate(reports_for_rates):
                    for condition_index, condition in enumerate(conditions):
                        for channel_index, row in enumerate(report["config"]["per_channel_draws"]):
                            vector = row["crc_pass_by_condition"][condition]
                            rates[snr_index, condition_index, channel_index] = 1.0 - (
                                sum(vector) / blocks
                            )
                observed = rates.mean(axis=2)
                rng = np.random.default_rng(SNR_CURVE_BOOTSTRAP_SEED)
                bootstrap_batches: list[np.ndarray] = []
                for start in range(0, SNR_CURVE_BOOTSTRAP_DRAWS, SNR_CURVE_BOOTSTRAP_BATCH_SIZE):
                    batch_size = min(
                        SNR_CURVE_BOOTSTRAP_BATCH_SIZE,
                        SNR_CURVE_BOOTSTRAP_DRAWS - start,
                    )
                    sampled = rng.integers(0, len(seeds), size=(batch_size, len(seeds)))
                    sampled_means = rates[:, :, sampled].mean(axis=-1)
                    bootstrap_batches.append(np.moveaxis(sampled_means, -1, 0))
                bootstrap = np.concatenate(bootstrap_batches, axis=0)
                deviations = np.max(np.abs(bootstrap - observed[None, :, :]), axis=(1, 2))
                if has_paired_contrast:
                    paired_improvement_rates = rates[:, 1, :] - rates[:, 2, :]
                    observed_paired_improvement = paired_improvement_rates.mean(axis=1)
                    paired_bootstrap_batches: list[np.ndarray] = []
                    rng = np.random.default_rng(SNR_CURVE_BOOTSTRAP_SEED)
                    for start in range(
                        0, SNR_CURVE_BOOTSTRAP_DRAWS, SNR_CURVE_BOOTSTRAP_BATCH_SIZE
                    ):
                        batch_size = min(
                            SNR_CURVE_BOOTSTRAP_BATCH_SIZE,
                            SNR_CURVE_BOOTSTRAP_DRAWS - start,
                        )
                        sampled = rng.integers(0, len(seeds), size=(batch_size, len(seeds)))
                        sampled_improvements = paired_improvement_rates[:, sampled].mean(axis=-1)
                        paired_bootstrap_batches.append(np.moveaxis(sampled_improvements, -1, 0))
                    paired_bootstrap = np.concatenate(paired_bootstrap_batches, axis=0)
                    paired_deviations = np.max(
                        np.abs(paired_bootstrap - observed_paired_improvement[None, :]),
                        axis=1,
                    )
                    deviations = np.maximum(deviations, paired_deviations)
                simultaneous_radius = float(np.quantile(deviations, 0.95, method="higher"))
                inference_consistent &= _same_float(
                    curve_inference.get("simultaneous_radius"), simultaneous_radius
                )
                for condition_index, condition in enumerate(conditions):
                    rows = by_condition.get(condition)
                    if not isinstance(rows, list) or len(rows) != len(snr_grid):
                        inference_consistent = False
                        continue
                    for snr_index, row in enumerate(rows):
                        if not isinstance(row, dict):
                            inference_consistent = False
                            continue
                        ordered = np.sort(bootstrap[:, snr_index, condition_index])
                        expected_intervals = {
                            "pointwise_95_percent_interval": [
                                float(ordered[math.floor(0.025 * SNR_CURVE_BOOTSTRAP_DRAWS)]),
                                float(ordered[math.ceil(0.975 * SNR_CURVE_BOOTSTRAP_DRAWS) - 1]),
                            ],
                            "simultaneous_95_percent_interval": [
                                max(
                                    0.0,
                                    float(observed[snr_index, condition_index])
                                    - simultaneous_radius,
                                ),
                                min(
                                    1.0,
                                    float(observed[snr_index, condition_index])
                                    + simultaneous_radius,
                                ),
                            ],
                        }
                        inference_consistent &= (
                            row.get("snr_db") == snr_grid[snr_index]
                            and _same_float(
                                row.get("mean_channel_macro_bler"),
                                float(observed[snr_index, condition_index]),
                            )
                            and all(
                                isinstance(row.get(key), list)
                                and len(row[key]) == 2
                                and all(
                                    _same_float(actual, expected)
                                    for actual, expected in zip(
                                        row[key], expected_values, strict=True
                                    )
                                )
                                for key, expected_values in expected_intervals.items()
                            )
                        )
                if inference_consistent and has_paired_contrast:
                    if not isinstance(by_paired_contrast, list) or len(by_paired_contrast) != len(
                        snr_grid
                    ):
                        inference_consistent = False
                    else:
                        for snr_index, row in enumerate(by_paired_contrast):
                            if not isinstance(row, dict):
                                inference_consistent = False
                                continue
                            ordered = np.sort(paired_bootstrap[:, snr_index])
                            estimate = float(observed_paired_improvement[snr_index])
                            expected_intervals = {
                                "pointwise_95_percent_interval": [
                                    float(ordered[math.floor(0.025 * SNR_CURVE_BOOTSTRAP_DRAWS)]),
                                    float(
                                        ordered[math.ceil(0.975 * SNR_CURVE_BOOTSTRAP_DRAWS) - 1]
                                    ),
                                ],
                                "simultaneous_95_percent_interval": [
                                    max(-1.0, estimate - simultaneous_radius),
                                    min(1.0, estimate + simultaneous_radius),
                                ],
                            }
                            inference_consistent &= (
                                row.get("snr_db") == snr_grid[snr_index]
                                and _same_float(
                                    row.get("mean_channel_macro_bler_improvement"), estimate
                                )
                                and all(
                                    isinstance(row.get(key), list)
                                    and len(row[key]) == 2
                                    and all(
                                        _same_float(actual, expected)
                                        for actual, expected in zip(
                                            row[key], expected_values, strict=True
                                        )
                                    )
                                    for key, expected_values in expected_intervals.items()
                                )
                            )
    checks["channel_cluster_curve_inference"] = inference_consistent
    if curve_schema in {
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v5",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v6",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v7",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v8",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v10",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v11",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v12",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v13",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v14",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v15",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v16",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v17",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v25",
    }:
        frozen_contract = frozen_acceptance.get("criteria")
        contract_consistent = isinstance(frozen_contract, dict)
        if contract_consistent:
            assert isinstance(frozen_contract, dict)
            base_contract_keys = {
                "schema",
                "snr_db",
                "channel_seeds",
                "channel_estimator",
                "blocks_per_channel",
                "minimum_independent_channels",
                "maximum_corrected_macro_bler",
                "minimum_uncorrected_minus_corrected_macro_bler",
            }
            timestamped_contract = frozen_contract.get("schema") in {
                SNR_TIMESTAMPED_ACCEPTANCE_SCHEMA,
                SNR_EXECUTION_ACCEPTANCE_SCHEMA,
            }
            execution_bound_contract = (
                frozen_contract.get("schema") == SNR_EXECUTION_ACCEPTANCE_SCHEMA
            )
            expected_contract_keys = base_contract_keys | (
                {"preregistered_utc"} if timestamped_contract else set()
            )
            if execution_bound_contract:
                expected_contract_keys |= {"execution_context", "execution_context_sha256"}
            contract_consistent = (
                set(frozen_contract) == expected_contract_keys
                and frozen_contract.get("schema")
                in {
                    "umcd.g5_ordered_impairments_snr_acceptance.v1",
                    SNR_TIMESTAMPED_ACCEPTANCE_SCHEMA,
                    SNR_EXECUTION_ACCEPTANCE_SCHEMA,
                }
                and timestamped_contract
                == (
                    curve_schema
                    in {
                        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v10",
                        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v11",
                        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v17",
                        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v25",
                    }
                )
                and execution_bound_contract
                == (
                    curve_schema
                    in {
                        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v17",
                        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v25",
                    }
                )
                and frozen_contract.get("snr_db") == snr_grid
                and frozen_contract.get("channel_seeds") == seeds
                and frozen_contract.get("channel_estimator") == config.get("channel_estimator")
                and frozen_contract.get("blocks_per_channel") == blocks
                and isinstance(frozen_contract.get("minimum_independent_channels"), int)
                and not isinstance(frozen_contract.get("minimum_independent_channels"), bool)
                and MIN_INDEPENDENT_ACCEPTANCE_CHANNELS
                <= frozen_contract["minimum_independent_channels"]
                <= len(seeds)
                and isinstance(frozen_contract.get("maximum_corrected_macro_bler"), list)
                and len(frozen_contract["maximum_corrected_macro_bler"]) == len(snr_grid)
                and isinstance(
                    frozen_contract.get("minimum_uncorrected_minus_corrected_macro_bler"),
                    list,
                )
                and len(frozen_contract["minimum_uncorrected_minus_corrected_macro_bler"])
                == len(snr_grid)
                and frozen_acceptance.get("criteria_sha256")
                == _sha256(_canonical_json_bytes(frozen_contract))
                and frozen_acceptance.get("confidence_level") == 0.95
                and frozen_acceptance.get("interval_scope")
                == "simultaneous across all three conditions and paired improvements"
                and frozen_acceptance.get("decision_unit")
                == "independent synthetic channel; transport blocks are nested"
            )
            if execution_bound_contract:
                contract_consistent &= _execution_context_matches(frozen_contract, points)
            if curve_schema in {
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v6",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v8",
            }:
                signature = frozen_acceptance.get("preregistration_signature")
                contract_consistent &= (
                    isinstance(signature, dict)
                    and signature.get("status") == "verified_before_receiver_run"
                    and signature.get("namespace") == "umcd-g5-snr-acceptance-v1"
                    and isinstance(signature.get("signer_id"), str)
                    and bool(signature["signer_id"].strip())
                    and signature.get("contract_payload_sha256")
                    == _sha256(_canonical_json_bytes(frozen_contract))
                    and _is_sha256(signature.get("signature_sha256"))
                    and _is_sha256(signature.get("trusted_allowed_signers_sha256"))
                    and signature.get("trust_file_custody") == "not_established_by_this_verifier"
                )
                signature_reverified = False
                if (
                    acceptance_signature_path is not None
                    and trusted_allowed_signers_path is not None
                    and isinstance(signature, dict)
                    and isinstance(frozen_contract, dict)
                ):
                    signature_reverified = _verify_preregistration_signature(
                        frozen_contract,
                        signature,
                        signature_path=acceptance_signature_path,
                        trusted_allowed_signers_path=trusted_allowed_signers_path,
                    )
                checks["preregistration_signature_reverified"] = signature_reverified
            elif curve_schema in {
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v10",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v11",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v17",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v25",
            }:
                signature = frozen_acceptance.get("preregistration_signature")
                temporal_order_consistent = _declared_preregistration_order_matches(
                    frozen_contract, signature
                )
                contract_consistent &= (
                    isinstance(signature, dict)
                    and signature.get("status") == "verified_before_receiver_run"
                    and signature.get("namespace") == _SNR_ACCEPTANCE_SIGNATURE_NAMESPACE
                    and isinstance(signature.get("signer_id"), str)
                    and bool(signature["signer_id"].strip())
                    and signature.get("contract_payload_sha256")
                    == _sha256(_canonical_json_bytes(frozen_contract))
                    and _is_sha256(signature.get("signature_sha256"))
                    and _is_sha256(signature.get("trusted_allowed_signers_sha256"))
                    and signature.get("trust_file_custody") == "not_established_by_this_verifier"
                    and temporal_order_consistent
                )
                signature_reverified = False
                if (
                    acceptance_signature_path is not None
                    and trusted_allowed_signers_path is not None
                    and isinstance(signature, dict)
                    and isinstance(frozen_contract, dict)
                ):
                    signature_reverified = _verify_preregistration_signature(
                        frozen_contract,
                        signature,
                        signature_path=acceptance_signature_path,
                        trusted_allowed_signers_path=trusted_allowed_signers_path,
                    )
                checks["preregistration_signature_reverified"] = signature_reverified
                policy_identity = (
                    signature.get("reviewer_policy") if isinstance(signature, dict) else None
                )
                policy_reverified = False
                reviewer_policy_required = curve_schema in {
                    "umcd.g5_ordered_impairments_independent_channels_snr_curve.v11",
                    "umcd.g5_ordered_impairments_independent_channels_snr_curve.v17",
                    "umcd.g5_ordered_impairments_independent_channels_snr_curve.v25",
                }
                policy_consistent = not reviewer_policy_required
                if reviewer_policy_required:
                    signature_dict = signature if isinstance(signature, dict) else {}
                    policy_consistent = (
                        isinstance(policy_identity, dict)
                        and policy_identity.get("status") == "verified"
                        and policy_identity.get("reviewer_principal")
                        == signature_dict.get("signer_id")
                        and "independent_reviewer"
                        in str(policy_identity.get("reviewer_roles", "")).split(",")
                        and policy_identity.get("purpose") == SNR_REVIEWER_POLICY_PURPOSE
                        and policy_identity.get("signature_namespace")
                        == SNR_REVIEWER_POLICY_SIGNATURE_NAMESPACE
                        and policy_identity.get("custodian_principal")
                        != signature_dict.get("signer_id")
                        and policy_identity.get("trust_file_custody")
                        == "not_established_by_this_verifier"
                    )
                    if (
                        policy_consistent
                        and reviewer_policy_path is not None
                        and reviewer_policy_signature_path is not None
                        and trusted_reviewer_allowed_signers_path is not None
                        and reviewer_policy_signer_id is not None
                        and isinstance(signature, dict)
                    ):
                        try:
                            verified_policy = _verify_reviewer_policy(
                                reviewer_policy_path,
                                reviewer_policy_signature_path,
                                trusted_reviewer_allowed_signers_path,
                                reviewer_policy_signer_id,
                                acceptance_signer_id=signature.get("signer_id"),
                                acceptance_allowed_signers_sha256=signature.get(
                                    "trusted_allowed_signers_sha256", ""
                                ),
                            )
                        except (OSError, RuntimeError, TypeError, ValueError):
                            verified_policy = None
                        policy_reverified = verified_policy == policy_identity
                    policy_consistent &= bool(policy_reverified)
                    contract_consistent &= policy_consistent
                checks["reviewer_policy_reverified"] = (
                    policy_reverified if reviewer_policy_required else None
                )
            else:
                contract_consistent &= "preregistration_signature" not in frozen_acceptance
                checks["preregistration_signature_reverified"] = None
            for key, lower, upper in (
                ("maximum_corrected_macro_bler", 0.0, 1.0),
                ("minimum_uncorrected_minus_corrected_macro_bler", 0.0, 1.0),
            ):
                values = frozen_contract.get(key)
                contract_consistent &= isinstance(values, list) and all(
                    _is_finite_number(value) and lower <= value <= upper for value in values
                )
        decision_consistent = contract_consistent and inference_consistent
        expected_points: list[dict[str, Any]] = []
        if decision_consistent:
            assert isinstance(frozen_contract, dict)
            assert isinstance(curve_inference, dict)
            corrected_rows = curve_inference["by_condition"]["corrected"]
            improvement_rows = curve_inference["by_paired_contrast"]
            all_pass = True
            for index, snr in enumerate(snr_grid):
                corrected_upper = corrected_rows[index]["simultaneous_95_percent_interval"][1]
                improvement_lower = improvement_rows[index]["simultaneous_95_percent_interval"][0]
                corrected_pass = (
                    corrected_upper <= frozen_contract["maximum_corrected_macro_bler"][index]
                )
                improvement_pass = (
                    improvement_lower
                    >= frozen_contract["minimum_uncorrected_minus_corrected_macro_bler"][index]
                )
                point_pass = corrected_pass and improvement_pass
                all_pass &= point_pass
                expected_points.append(
                    {
                        "snr_db": snr,
                        "corrected_macro_bler_simultaneous_upper": corrected_upper,
                        "maximum_corrected_macro_bler": frozen_contract[
                            "maximum_corrected_macro_bler"
                        ][index],
                        "corrected_bler_criterion_pass": corrected_pass,
                        "uncorrected_minus_corrected_simultaneous_lower": improvement_lower,
                        "minimum_uncorrected_minus_corrected_macro_bler": frozen_contract[
                            "minimum_uncorrected_minus_corrected_macro_bler"
                        ][index],
                        "minimum_improvement_criterion_pass": improvement_pass,
                        "pass": point_pass,
                    }
                )
            decision_consistent &= (
                acceptance.get("status") == ("PASS" if all_pass else "FAIL")
                and acceptance.get("pass") is all_pass
                and acceptance.get("cluster_count") == len(seeds)
                and acceptance.get("point_decisions") == expected_points
            )
        checks["acceptance_contract_decision"] = decision_consistent
        checks["acceptance_is_diagnostic_only"] = None
    else:
        checks["acceptance_contract_decision"] = None
        checks["acceptance_is_diagnostic_only"] = (
            isinstance(frozen_acceptance, dict)
            and frozen_acceptance.get("status") == "descriptive_only"
            and acceptance.get("pass") is False
        )
    evidence_manifest = config.get("waveform_evidence")
    if evidence_manifest is None:
        point_manifests = [
            (
                point.get("experiment", {}).get("config", {}).get("waveform_evidence")
                if isinstance(point, dict)
                and isinstance(point.get("experiment"), dict)
                and isinstance(point["experiment"].get("config"), dict)
                else None
            )
            for point in points
        ]
        checks["waveform_equation_recomputation"] = (
            None if all(manifest is None for manifest in point_manifests) else False
        )
    elif waveform_evidence_bytes is None or not isinstance(evidence_manifest, dict):
        checks["waveform_equation_recomputation"] = False
    else:
        evidence_checks: list[bool] = []
        manifest_grid_matches = not (
            curve_schema
            in {
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v3",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v4",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v5",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v6",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v7",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v8",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v9",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v10",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v11",
            }
            and (
                evidence_manifest.get("schema")
                != "umcd.g5_ordered_impairments_waveform_evidence.v4"
                or evidence_manifest.get("snr_point_count") != len(snr_grid)
                or evidence_manifest.get("snr_db") != snr_grid
            )
        )
        for snr_index, point in enumerate(points):
            if not manifest_grid_matches:
                evidence_checks.append(False)
                continue
            report = point.get("experiment") if isinstance(point, dict) else None
            point_config = report.get("config") if isinstance(report, dict) else None
            if not isinstance(point_config, dict):
                evidence_checks.append(False)
                continue
            point_manifest = point_config.get("waveform_evidence")
            records = point_config.get("per_channel_draws")
            acceptance = report.get("acceptance") if isinstance(report, dict) else None
            observed_acceptance = (
                acceptance.get("observed") if isinstance(acceptance, dict) else None
            )
            criteria = (
                acceptance.get("criteria_frozen_before_run")
                if isinstance(acceptance, dict)
                else None
            )
            if (
                point_manifest != evidence_manifest
                or not isinstance(records, list)
                or not isinstance(observed_acceptance, dict)
                or not isinstance(criteria, dict)
            ):
                evidence_checks.append(False)
                continue
            try:
                maximum_error, minimum_wrong_order = _recompute_waveform_diagnostics(
                    point_config,
                    records,
                    waveform_evidence_bytes,
                    snr_index=snr_index,
                )
                evidence_checks.append(
                    abs(
                        maximum_error
                        - float(
                            observed_acceptance.get(
                                "max_samplewise_reference_absolute_error", math.inf
                            )
                        )
                    )
                    <= 5e-6
                    and abs(
                        minimum_wrong_order
                        - float(observed_acceptance.get("min_wrong_order_rmse", -math.inf))
                    )
                    <= 5e-6
                    and maximum_error
                    < float(criteria.get("max_samplewise_reference_absolute_error", -math.inf))
                    and minimum_wrong_order > float(criteria.get("min_wrong_order_rmse", math.inf))
                )
            except (OSError, EOFError, ValueError, TypeError, KeyError, OverflowError):
                evidence_checks.append(False)
        checks["waveform_equation_recomputation"] = bool(evidence_checks) and all(evidence_checks)
    required_checks = [value for value in checks.values() if value is not None]
    return {
        "schema": "umcd.g5_ordered_impairments_result_audit.v1",
        "status": "AUDIT_PARTIAL" if required_checks and all(required_checks) else "AUDIT_FAILED",
        "scope": (
            "integrity, retained-count, and bounded fixture acceptance replay; no deployment qualification"
            if curve_schema
            in {
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v5",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v6",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v7",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v8",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v10",
            }
            else "integrity and retained-count audit; no familywise scientific acceptance"
        ),
        "checks": checks,
    }
