"""Independent snr curve for ordered impairments receiver."""

from __future__ import annotations

import math
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from integrations.sionna.nr_pusch import SionnaPUSCHLSReceiver
from receivers.nr import NRPUSCHConfig

from experiments.acceptance import (
    SNR_ACCEPTANCE_SIGNATURE_NAMESPACE,
    SNR_CURVE_BOOTSTRAP_BATCH_SIZE,
    SNR_EXECUTION_ACCEPTANCE_SCHEMA,
    SNR_REVIEWER_POLICY_PURPOSE,
    SNR_REVIEWER_POLICY_SIGNATURE_NAMESPACE,
    SNR_TIMESTAMPED_ACCEPTANCE_SCHEMA,
    validate_preregistration_order,
)
from experiments.acceptance import (
    evaluate_snr_curve_acceptance as _evaluate_independent_snr_curve_acceptance,
)
from experiments.acceptance import (
    freeze_snr_acceptance_criteria as _freeze_independent_snr_acceptance_criteria,
)
from experiments.acceptance import (
    snr_execution_context_from_config as _snr_execution_context_from_config,
)

from .constants import (
    _MAX_INDEPENDENT_SNR_POINTS,
    _MAX_INDEPENDENT_TRANSPORT_BLOCKS,
    _MAX_WAVEFORM_EVIDENCE_BYTES,
    CFO_HZ,
    DIRECT_COEFFICIENT,
    INDEPENDENT_CHANNEL_SEEDS,
    INDEPENDENT_SNR_CURVE_BOOTSTRAP_DRAWS,
    INDEPENDENT_SNR_CURVE_BOOTSTRAP_SEED,
    RX_IQ_GAIN_IMBALANCE_DB,
    RX_IQ_PHASE_IMBALANCE_RAD,
    RX_LINEWIDTH_HZ,
    TIMING_OFFSET_SAMPLES,
    TX_IQ_GAIN_IMBALANCE_DB,
    TX_IQ_PHASE_IMBALANCE_RAD,
    TX_LINEWIDTH_HZ,
)
from .hashing import (
    _canonical_json_bytes,
    _sha256,
)
from .independent_probe import (
    _validate_execution_device,
    run_independent_ordered_impairment_probe,
)
from .paired_probe import (
    _configure_probe_decoder,
)
from .waveform_evidence import (
    _combine_snr_curve_waveform_evidence,
)


def _snr_execution_context(channel_estimator: str, device: str) -> dict[str, Any]:
    """Construct the exact receiver/decoder/waveform identity before trials."""
    config = NRPUSCHConfig(n_size_bwp=6, subcarrier_spacing_khz=30)
    receiver = SionnaPUSCHLSReceiver(config, device=device)
    _, estimator_config = _configure_probe_decoder(receiver, channel_estimator)
    grid = receiver.resource_grid
    sample_rate_hz = float(grid.fft_size * config.subcarrier_spacing_khz * 1_000)
    cp_length = int(grid.cyclic_prefix_length)
    max_echo_delay = min(6, cp_length - 1)
    if max_echo_delay < 2:
        raise RuntimeError("cyclic prefix is too short for the declared channel family")
    return _snr_execution_context_from_config(
        {
            "receiver_configuration": receiver.metadata,
            "decoder_configuration": {
                "api": "SionnaPUSCHLSReceiver.decode_pusch",
                "channel_estimator": estimator_config,
                "crc_status_semantics": "true_means_transport_block_crc_passed",
            },
            "sample_rate_hz": sample_rate_hz,
            "fft_size": int(grid.fft_size),
            "cyclic_prefix_samples": cp_length,
            "echo_delay_samples_range": [2, max_echo_delay],
            "echo_amplitude_range": [0.20, 0.50],
            "direct_coefficient_real_imag": [
                DIRECT_COEFFICIENT.real,
                DIRECT_COEFFICIENT.imag,
            ],
            "cfo_hz": CFO_HZ,
            "timing_offset_samples": TIMING_OFFSET_SAMPLES,
            "tx_phase_noise_linewidth_hz": TX_LINEWIDTH_HZ,
            "rx_phase_noise_linewidth_hz": RX_LINEWIDTH_HZ,
            "tx_iq_gain_imbalance_db": TX_IQ_GAIN_IMBALANCE_DB,
            "tx_iq_phase_imbalance_rad": TX_IQ_PHASE_IMBALANCE_RAD,
            "rx_iq_gain_imbalance_db": RX_IQ_GAIN_IMBALANCE_DB,
            "rx_iq_phase_imbalance_rad": RX_IQ_PHASE_IMBALANCE_RAD,
        }
    )


def run_independent_ordered_impairment_snr_curve(
    *,
    snr_db: Sequence[float],
    channel_seeds: tuple[int, ...] = INDEPENDENT_CHANNEL_SEEDS,
    blocks_per_channel: int = 8,
    channel_estimator: str = "sionna_default_ls",
    waveform_evidence_output: Path | None = None,
    acceptance_criteria: dict[str, Any] | None = None,
    preregistration_signature: dict[str, str] | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run a bounded paired diagnostic curve over common channel/payload seeds.

    Each SNR point uses the same channel, payload, oscillator, and standardized
    AWGN random-number streams. Only AWGN amplitude changes. The convention is
    ``noise_variance = 10**(-snr_db/10)`` for unit-reference complex waveform
    power; the realized transmit waveform power is retained at every point.
    Optional evidence retains shared TX, oscillator, pre-AWGN capture, and
    standardized AWGN draws once, then stores the exact clean, uncorrected,
    and corrected receiver inputs for every SNR point. The auditor binds each
    point to its evidence index and noise realization. This curve is
    descriptive unless an exact, bounded acceptance contract is supplied before
    the run. A contract decision applies only to this synthetic fixture; it is
    not field or deployment qualification.
    """
    device = _validate_execution_device(device)
    if isinstance(snr_db, (str, bytes)) or not (
        isinstance(snr_db, Sequence) or isinstance(snr_db, np.ndarray)
    ):
        raise TypeError("snr_db must be a bounded sequence of real scalars")
    if isinstance(snr_db, np.ndarray) and snr_db.ndim != 1:
        raise ValueError("snr_db array must be one-dimensional")
    if len(snr_db) > _MAX_INDEPENDENT_SNR_POINTS:
        raise ValueError(
            f"independent-channel SNR curve has {len(snr_db)} points, exceeding the "
            f"bounded limit of {_MAX_INDEPENDENT_SNR_POINTS}"
        )
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in snr_db):
        raise TypeError("snr_db values must be real scalars")
    snr_values = tuple(float(value) for value in snr_db)
    if not snr_values or any(not math.isfinite(value) for value in snr_values):
        raise ValueError("snr_db must contain finite values")
    if any(right <= left for left, right in zip(snr_values, snr_values[1:])):
        raise ValueError("snr_db must be strictly increasing without duplicates")
    if isinstance(blocks_per_channel, bool) or not isinstance(blocks_per_channel, int):
        raise TypeError("blocks_per_channel must be an integer")
    if blocks_per_channel < 1:
        raise ValueError("blocks_per_channel must be positive")
    if isinstance(channel_seeds, (str, bytes)) or not isinstance(channel_seeds, Sequence):
        raise TypeError("channel_seeds must be a bounded sequence of integers")
    if not channel_seeds:
        raise ValueError("channel_seeds must be nonempty and unique")
    if len(channel_seeds) > _MAX_INDEPENDENT_TRANSPORT_BLOCKS // (
        len(snr_values) * blocks_per_channel
    ):
        requested = len(snr_values) * len(channel_seeds) * blocks_per_channel
        raise ValueError(
            "independent-channel SNR curve requests "
            f"{requested} transport blocks, exceeding the bounded limit of "
            f"{_MAX_INDEPENDENT_TRANSPORT_BLOCKS}"
        )
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in channel_seeds
    ):
        raise ValueError("channel_seeds must contain nonnegative integers")
    if len(set(channel_seeds)) != len(channel_seeds):
        raise ValueError("channel_seeds must be nonempty and unique")
    if not isinstance(channel_estimator, str):
        raise TypeError("channel_estimator must be a string")
    if channel_estimator not in {"sionna_default_ls", "finite_delay_dft"}:
        raise ValueError("channel_estimator must be sionna_default_ls or finite_delay_dft")
    execution_context = (
        _snr_execution_context(channel_estimator, device)
        if isinstance(acceptance_criteria, dict)
        and acceptance_criteria.get("schema") == SNR_EXECUTION_ACCEPTANCE_SCHEMA
        else None
    )
    frozen_acceptance = _freeze_independent_snr_acceptance_criteria(
        acceptance_criteria,
        snr_values=snr_values,
        channel_seeds=channel_seeds,
        channel_estimator=channel_estimator,
        blocks_per_channel=blocks_per_channel,
        execution_context=execution_context,
    )
    if preregistration_signature is not None and frozen_acceptance is None:
        raise ValueError("a preregistration signature requires an acceptance contract")
    if (
        frozen_acceptance is not None
        and frozen_acceptance.get("schema")
        in {SNR_TIMESTAMPED_ACCEPTANCE_SCHEMA, SNR_EXECUTION_ACCEPTANCE_SCHEMA}
        and preregistration_signature is None
    ):
        raise ValueError("timestamped acceptance contract v2/v3 requires a verified signature")
    receiver_run_started_utc = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    preregistration_order: dict[str, str] | None = None
    if preregistration_signature is not None:
        required_signature_fields = {
            "status",
            "namespace",
            "signer_id",
            "contract_payload_sha256",
            "signature_sha256",
            "trusted_allowed_signers_sha256",
            "trust_file_custody",
        }
        if frozen_acceptance is not None and frozen_acceptance.get("schema") in {
            SNR_TIMESTAMPED_ACCEPTANCE_SCHEMA,
            SNR_EXECUTION_ACCEPTANCE_SCHEMA,
        }:
            required_signature_fields.add("preregistered_utc")
        reviewer_policy_identity = (
            preregistration_signature.get("reviewer_policy")
            if isinstance(preregistration_signature, dict)
            else None
        )
        if reviewer_policy_identity is not None:
            required_signature_fields.add("reviewer_policy")
        if (
            not isinstance(preregistration_signature, dict)
            or set(preregistration_signature) != required_signature_fields
            or preregistration_signature.get("status") != "verified_before_receiver_run"
            or preregistration_signature.get("namespace") != SNR_ACCEPTANCE_SIGNATURE_NAMESPACE
            or not isinstance(preregistration_signature.get("signer_id"), str)
            or not preregistration_signature["signer_id"].strip()
            or preregistration_signature.get("contract_payload_sha256")
            != _sha256(_canonical_json_bytes(frozen_acceptance))
            or preregistration_signature.get("trust_file_custody")
            != "not_established_by_this_verifier"
            or any(
                not isinstance(preregistration_signature.get(field), str)
                or len(preregistration_signature[field]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in preregistration_signature[field]
                )
                for field in ("signature_sha256", "trusted_allowed_signers_sha256")
            )
        ):
            raise ValueError("preregistration signature identity does not bind the frozen contract")
        if reviewer_policy_identity is not None:
            assert isinstance(frozen_acceptance, dict)
            assert isinstance(preregistration_signature, dict)
            expected_policy_fields = {
                "status",
                "policy_id",
                "policy_sha256",
                "reviewer_principal",
                "reviewer_organization",
                "reviewer_roles",
                "purpose",
                "custodian_principal",
                "signature_namespace",
                "signature_sha256",
                "trusted_allowed_signers_sha256",
                "trust_file_custody",
            }
            if (
                frozen_acceptance.get("schema")
                not in {SNR_TIMESTAMPED_ACCEPTANCE_SCHEMA, SNR_EXECUTION_ACCEPTANCE_SCHEMA}
                or not isinstance(reviewer_policy_identity, dict)
                or set(reviewer_policy_identity) != expected_policy_fields
                or reviewer_policy_identity.get("status") != "verified"
                or reviewer_policy_identity.get("reviewer_principal")
                != preregistration_signature.get("signer_id")
                or "independent_reviewer"
                not in str(reviewer_policy_identity.get("reviewer_roles", "")).split(",")
                or reviewer_policy_identity.get("purpose") != SNR_REVIEWER_POLICY_PURPOSE
                or reviewer_policy_identity.get("signature_namespace")
                != SNR_REVIEWER_POLICY_SIGNATURE_NAMESPACE
                or reviewer_policy_identity.get("custodian_principal")
                == preregistration_signature.get("signer_id")
                or reviewer_policy_identity.get("trust_file_custody")
                != "not_established_by_this_verifier"
                or any(
                    not isinstance(reviewer_policy_identity.get(field), str)
                    or len(reviewer_policy_identity[field]) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in reviewer_policy_identity[field]
                    )
                    for field in (
                        "policy_sha256",
                        "signature_sha256",
                        "trusted_allowed_signers_sha256",
                    )
                )
            ):
                raise ValueError("G5 independent reviewer policy identity is invalid")
        assert isinstance(frozen_acceptance, dict)
        assert isinstance(preregistration_signature, dict)
        if frozen_acceptance.get("schema") in {
            SNR_TIMESTAMPED_ACCEPTANCE_SCHEMA,
            SNR_EXECUTION_ACCEPTANCE_SCHEMA,
        }:
            if preregistration_signature.get("preregistered_utc") != frozen_acceptance.get(
                "preregistered_utc"
            ):
                raise ValueError(
                    "preregistration signature timestamp differs from the frozen contract"
                )
            preregistration_order = validate_preregistration_order(
                frozen_acceptance, receiver_run_started_utc=receiver_run_started_utc
            )
            preregistration_signature = {
                **preregistration_signature,
                **preregistration_order,
            }
    cpu_curve_schema_version = (
        17
        if frozen_acceptance is not None
        and frozen_acceptance.get("schema") == SNR_EXECUTION_ACCEPTANCE_SCHEMA
        else 11
        if isinstance(preregistration_signature, dict)
        and "reviewer_policy" in preregistration_signature
        else 10
        if preregistration_order is not None
        else 8
        if preregistration_signature is not None
        else 7
        if frozen_acceptance is not None
        else 9
    )
    cuda_curve_schema_versions = {9: 12, 7: 13, 8: 14, 10: 15, 11: 16, 17: 25}
    curve_schema_version = (
        cuda_curve_schema_versions[cpu_curve_schema_version]
        if device != "cpu"
        else cpu_curve_schema_version
    )
    evidence_path = Path(waveform_evidence_output) if waveform_evidence_output is not None else None
    if evidence_path is not None:
        if "\\" in evidence_path.name or not evidence_path.name.endswith(".npz"):
            raise ValueError("waveform evidence output must be a sibling-safe .npz filename")
        if evidence_path.exists() or evidence_path.is_symlink():
            raise FileExistsError(f"refusing to overwrite waveform evidence: {evidence_path}")
    requested_transport_blocks = len(snr_values) * len(channel_seeds) * blocks_per_channel
    if requested_transport_blocks > _MAX_INDEPENDENT_TRANSPORT_BLOCKS:
        raise ValueError(
            "independent-channel SNR curve requests "
            f"{requested_transport_blocks} transport blocks, exceeding the bounded limit of "
            f"{_MAX_INDEPENDENT_TRANSPORT_BLOCKS}"
        )

    variances: list[float] = []
    for snr in snr_values:
        try:
            variance = math.pow(10.0, -snr / 10.0)
        except OverflowError as error:
            raise ValueError(
                f"SNR {snr} dB does not yield a finite positive noise variance"
            ) from error
        if not math.isfinite(variance) or variance <= 0:
            raise ValueError(f"SNR {snr} dB does not yield a finite positive noise variance")
        if not np.isfinite(np.float32(variance)) or np.float32(variance) <= 0:
            raise ValueError(f"SNR {snr} dB noise variance is not representable as float32")
        variances.append(variance)

    points: list[dict[str, Any]] = []
    evidence_manifest: dict[str, Any] | None = None
    staging_directory: str | None = None
    staging_evidence_path: Path | None = None
    staging_point_evidence_paths: list[Path] = []
    staging_point_evidence_bytes = 0
    try:
        if evidence_path is not None:
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            staging_directory = tempfile.mkdtemp(
                prefix=f".{evidence_path.stem}-", dir=evidence_path.parent
            )
            staging_evidence_path = Path(staging_directory) / evidence_path.name
        for index, (snr, variance) in enumerate(zip(snr_values, variances, strict=True)):
            point_evidence_path = None
            if staging_directory is not None:
                assert evidence_path is not None
                point_evidence_path = (
                    Path(staging_directory) / f"{evidence_path.stem}.point-{index:04d}.npz"
                )
                staging_point_evidence_paths.append(point_evidence_path)
            report = run_independent_ordered_impairment_probe(
                channel_seeds=channel_seeds,
                blocks_per_channel=blocks_per_channel,
                channel_estimator=channel_estimator,
                noise_variance=variance,
                waveform_evidence_output=point_evidence_path,
                include_condition_inputs=point_evidence_path is not None,
                device=device,
            )
            config = report.get("config")
            if not isinstance(config, dict):
                raise ValueError("SNR point report has no configuration object")
            if (
                frozen_acceptance is not None
                and frozen_acceptance.get("schema") == (SNR_EXECUTION_ACCEPTANCE_SCHEMA)
                and _snr_execution_context_from_config(config)
                != frozen_acceptance.get("execution_context")
            ):
                raise ValueError(
                    "G5 SNR point receiver, decoder, or waveform configuration differs "
                    "from the signed acceptance contract"
                )
            if point_evidence_path is not None:
                manifest = config.get("waveform_evidence")
                if not isinstance(manifest, dict):
                    raise ValueError("SNR point did not retain requested waveform evidence")
                byte_count = manifest.get("byte_count")
                if (
                    isinstance(byte_count, bool)
                    or not isinstance(byte_count, int)
                    or byte_count < 1
                ):
                    raise ValueError("SNR point waveform evidence has an invalid byte count")
                staging_point_evidence_bytes += byte_count
                if staging_point_evidence_bytes > 2 * _MAX_WAVEFORM_EVIDENCE_BYTES:
                    raise ValueError(
                        "multi-SNR waveform evidence staging exceeds its disk size limit"
                    )
            points.append({"snr_db": snr, "experiment": report})

        if staging_evidence_path is not None:
            evidence_manifest = _combine_snr_curve_waveform_evidence(
                tuple(staging_point_evidence_paths),
                output_path=staging_evidence_path,
                snr_values=snr_values,
            )
            suffix = (
                "24"
                if device != "cpu" and channel_estimator == "finite_delay_dft"
                else "23"
                if device != "cpu"
                else "20"
                if channel_estimator == "finite_delay_dft"
                else "19"
            )
            for point in points:
                point_report = point["experiment"]
                point_config = point_report["config"]
                point_config["waveform_evidence"] = dict(evidence_manifest)
                point_report["schema"] = (
                    f"umcd.g5_ordered_impairments_independent_channels.v{suffix}"
                )

        curve_inference = _independent_snr_curve_cluster_inference(
            [point["experiment"] for point in points],
            snr_values,
            seed=INDEPENDENT_SNR_CURVE_BOOTSTRAP_SEED,
            draws=INDEPENDENT_SNR_CURVE_BOOTSTRAP_DRAWS,
            include_exact_seed_roster=curve_schema_version in {7, 8, 10, 13, 14, 15, 17, 25},
        )
        if staging_evidence_path is not None and evidence_path is not None:
            os.link(staging_evidence_path, evidence_path)
            staging_evidence_path.unlink()
    finally:
        if staging_directory is not None:
            try:
                if staging_evidence_path is not None:
                    staging_evidence_path.unlink(missing_ok=True)
                for point_path in staging_point_evidence_paths:
                    point_path.unlink(missing_ok=True)
                Path(staging_directory).rmdir()
            except OSError:
                # Do not mask the receiver or evidence-publication failure.
                pass
    schema_version = curve_schema_version
    acceptance = (
        _evaluate_independent_snr_curve_acceptance(
            frozen_acceptance,
            curve_inference=curve_inference,
            preregistration_signature=preregistration_signature,
        )
        if frozen_acceptance is not None
        else {
            "criteria_frozen_before_run": {
                "status": "descriptive_only",
                "reason": ("no acceptance contract was supplied before the SNR curve run"),
            },
            "pass": False,
        }
    )
    return {
        "schema": (f"umcd.g5_ordered_impairments_independent_channels_snr_curve.v{schema_version}"),
        "scope": (
            "paired diagnostic SNR curve over a seeded synthetic two-path PUSCH fixture; "
            "same channel, payload, oscillator, and standardized AWGN draws across points; "
            + (
                "uses a caller-supplied pre-run acceptance contract; "
                "not field or deployment qualification"
                if frozen_acceptance is not None
                else "no acceptance contract; not a field BLER estimate"
            )
        ),
        "inference_scope": (
            "simultaneous channel-cluster inference against a caller-supplied pre-run contract"
            if frozen_acceptance is not None
            else (
                "descriptive pointwise and simultaneous channel-cluster uncertainty; "
                "no acceptance claim"
            )
        ),
        "config": {
            "schema_sha256": _sha256(
                (
                    Path(__file__).resolve().parents[2]
                    / "datasets"
                    / "schema"
                    / f"ordered-impairments-snr-curve.v{curve_schema_version}.schema.json"
                ).read_bytes()
            ),
            "snr_db": list(snr_values),
            "channel_seeds": list(channel_seeds),
            "blocks_per_channel": blocks_per_channel,
            "channel_estimator": channel_estimator,
            "shared_random_streams_across_snr": [
                "channel",
                "payload",
                "TX phase noise",
                "RX phase noise",
                "standardized AWGN",
            ],
            "snr_definition": (
                "noise variance per complex sample = 10**(-SNR_dB/10), "
                "referenced to unit complex waveform power; measured TX waveform power is "
                "reported at each point"
            ),
            "transport_blocks_per_snr_point": len(channel_seeds) * blocks_per_channel,
            "transport_blocks_across_curve": requested_transport_blocks,
            "maximum_transport_blocks_across_curve": _MAX_INDEPENDENT_TRANSPORT_BLOCKS,
            "maximum_snr_points": _MAX_INDEPENDENT_SNR_POINTS,
            **({"execution_device": device} if device != "cpu" else {}),
            "waveform_evidence": evidence_manifest,
        },
        "acceptance": acceptance,
        "curve_inference": curve_inference,
        "points": points,
    }


def _independent_snr_curve_cluster_inference(
    reports: list[dict[str, Any]],
    snr_values: tuple[float, ...],
    *,
    seed: int,
    draws: int,
    include_exact_seed_roster: bool = False,
) -> dict[str, Any]:
    """Compute pointwise and simultaneous BLER bands by resampling channels."""
    conditions = ("clean", "uncorrected", "corrected")
    first_config = reports[0].get("config")
    channel_seeds = first_config.get("channel_seeds") if isinstance(first_config, dict) else None
    if (
        not isinstance(channel_seeds, list)
        or not channel_seeds
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in channel_seeds
        )
        or len(set(channel_seeds)) != len(channel_seeds)
    ):
        raise ValueError("SNR-curve inference requires a canonical channel-seed roster")
    for report in reports:
        config = report.get("config") if isinstance(report, dict) else None
        records = config.get("per_channel_draws") if isinstance(config, dict) else None
        if (
            not isinstance(config, dict)
            or config.get("channel_seeds") != channel_seeds
            or not isinstance(records, list)
            or [row.get("channel_seed") for row in records if isinstance(row, dict)]
            != channel_seeds
        ):
            raise ValueError("SNR-curve points do not share the exact channel-seed roster")
    channel_count = len(channel_seeds)
    blocks_per_channel = reports[0]["config"]["blocks_per_channel"]
    cells: dict[str, list[dict[str, Any]]] = {condition: [] for condition in conditions}
    if channel_count < 2:
        return {
            "status": "unavailable",
            "unit": "independent synthetic channel; transport blocks are nested",
            "cluster_count": channel_count,
            "bootstrap_seed": seed,
            "bootstrap_draws": draws,
            "family_size": len(snr_values) * (len(conditions) + 1),
            "method": (
                "channel-cluster percentile intervals with simultaneous max-deviation band "
                "over conditions and paired correction effects"
            ),
            "paired_contrast_definition": (
                "uncorrected macro-BLER minus corrected macro-BLER; positive values favor correction"
            ),
            "reason": "at least two independent channel clusters are required for resampling",
            "by_condition": cells,
            "by_paired_contrast": [],
            **({"channel_seeds": list(channel_seeds)} if include_exact_seed_roster else {}),
        }

    rates = np.empty((len(reports), len(conditions), channel_count), dtype=np.float64)
    for snr_index, report in enumerate(reports):
        records = report["config"]["per_channel_draws"]
        for condition_index, condition in enumerate(conditions):
            for channel_index, record in enumerate(records):
                outcomes = record["crc_pass_by_condition"][condition]
                rates[snr_index, condition_index, channel_index] = 1.0 - (
                    sum(outcomes) / blocks_per_channel
                )
    observed = rates.mean(axis=2)
    paired_improvement_rates = rates[:, 1, :] - rates[:, 2, :]
    observed_paired_improvement = paired_improvement_rates.mean(axis=1)
    rng = np.random.default_rng(seed)
    batches: list[np.ndarray] = []
    paired_improvement_batches: list[np.ndarray] = []
    batch_size = max(1, min(SNR_CURVE_BOOTSTRAP_BATCH_SIZE, draws))
    for start in range(0, draws, batch_size):
        size = min(batch_size, draws - start)
        sampled_channels = rng.integers(0, channel_count, size=(size, channel_count))
        sampled_means = rates[:, :, sampled_channels].mean(axis=-1)
        batches.append(np.moveaxis(sampled_means, -1, 0))
        sampled_improvements = paired_improvement_rates[:, sampled_channels].mean(axis=-1)
        paired_improvement_batches.append(np.moveaxis(sampled_improvements, -1, 0))
    bootstrap = np.concatenate(batches, axis=0)
    paired_improvement_bootstrap = np.concatenate(paired_improvement_batches, axis=0)
    lower_index = math.floor(0.025 * draws)
    upper_index = min(draws - 1, math.ceil(0.975 * draws) - 1)
    condition_deviations = np.max(np.abs(bootstrap - observed[None, :, :]), axis=(1, 2))
    paired_deviations = np.max(
        np.abs(paired_improvement_bootstrap - observed_paired_improvement[None, :]), axis=1
    )
    deviations = np.maximum(condition_deviations, paired_deviations)
    simultaneous_radius = float(np.quantile(deviations, 0.95, method="higher"))
    for condition_index, condition in enumerate(conditions):
        for snr_index, snr in enumerate(snr_values):
            ordered = np.sort(bootstrap[:, snr_index, condition_index])
            point_lower = float(ordered[lower_index])
            point_upper = float(ordered[upper_index])
            estimate = float(observed[snr_index, condition_index])
            cells[condition].append(
                {
                    "snr_db": snr,
                    "mean_channel_macro_bler": estimate,
                    "pointwise_95_percent_interval": [point_lower, point_upper],
                    "simultaneous_95_percent_interval": [
                        max(0.0, estimate - simultaneous_radius),
                        min(1.0, estimate + simultaneous_radius),
                    ],
                }
            )
    paired_contrast_rows = []
    for snr_index, snr in enumerate(snr_values):
        ordered = np.sort(paired_improvement_bootstrap[:, snr_index])
        estimate = float(observed_paired_improvement[snr_index])
        paired_contrast_rows.append(
            {
                "snr_db": snr,
                "mean_channel_macro_bler_improvement": estimate,
                "pointwise_95_percent_interval": [
                    float(ordered[lower_index]),
                    float(ordered[upper_index]),
                ],
                "simultaneous_95_percent_interval": [
                    max(-1.0, estimate - simultaneous_radius),
                    min(1.0, estimate + simultaneous_radius),
                ],
            }
        )
    return {
        "status": "available",
        "unit": "independent synthetic channel; transport blocks are nested",
        "cluster_count": channel_count,
        "bootstrap_seed": seed,
        "bootstrap_draws": draws,
        "family_size": len(snr_values) * (len(conditions) + 1),
        "method": (
            "channel-cluster percentile intervals with simultaneous max-deviation band "
            "over conditions and paired correction effects"
        ),
        "paired_contrast_definition": (
            "uncorrected macro-BLER minus corrected macro-BLER; positive values favor correction"
        ),
        "simultaneous_radius": simultaneous_radius,
        "by_condition": cells,
        "by_paired_contrast": paired_contrast_rows,
        **({"channel_seeds": list(channel_seeds)} if include_exact_seed_roster else {}),
    }
