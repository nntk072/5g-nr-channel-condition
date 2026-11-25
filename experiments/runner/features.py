"""Feature construction and channel-representation helpers for the experiment runner."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from datasets.channel import ChannelDataset
from evaluation.benchmark_partitions import (
    TEST_COHORTS,
)
from models.multipath import fit_sparse_multipath

_VALIDATION_THRESHOLD_POLICY = {
    "schema": "loslab.validation_threshold_selection.v1",
    "criterion": "Youden J = sensitivity - false-positive rate",
    "selection_partition": "validation",
    "positive_label": 1,
    "weighting": "unweighted validation rows",
    "tie_break": "first maximum in roc_curve threshold order",
    "test_labels_used_for_selection": False,
}


_MODEL_FEATURE_ROUTES = {
    "cnn": ("observed_cfr", "complex_cfr_grid_v1"),
    "cvnn": ("observed_cfr", "complex_cfr_grid_v1"),
    "raw_mlp": ("observed_cfr", "flattened_real_imag_cfr_v1"),
    "label_permuted": ("observed_cfr", "flattened_real_imag_cfr_v1"),
    "row_shuffled": ("observed_cfr", "flattened_real_imag_cfr_v1"),
    "source_only": ("source", "train_vocabulary_source_one_hot_v1"),
    "metadata_only": (
        "group/domain/source",
        "train_vocabulary_group_domain_source_one_hot_v1",
    ),
    "group_only": ("group", "train_vocabulary_group_one_hot_v1"),
    "domain_only": ("domain", "train_vocabulary_domain_one_hot_v1"),
    "power_only": ("observed_cfr", "log_power_per_link_subcarrier_v1"),
    "phase_randomized": ("observed_cfr", "iid_phase_randomized_flattened_real_imag_cfr_v1"),
    "observable_features": ("observed_cfr", "logmag_pdp_moments_v1"),
    "delay_k_proxy": ("observed_cfr", "normalized_delay_k_proxy_v1"),
    "sparse_multipath": ("observed_cfr", "omp_bic_sparse_cfr_features_v1"),
    "oracle_label_leak": ("data.y", "direct_binary_target_as_feature_leak_v1"),
}


_MODEL_METHODS = frozenset(
    {
        "rf",
        "histgb",
        "xgb",
        "mlp",
        "raw_mlp",
        "label_permuted",
        "row_shuffled",
        "source_only",
        "metadata_only",
        "group_only",
        "domain_only",
        "power_only",
        "phase_randomized",
        "oracle_label_leak",
        "observable_features",
        "delay_k_proxy",
        "sparse_multipath",
        "logreg",
        "cnn",
        "cvnn",
        "coral",
        "coral0",
    }
)


def _validate_evaluation_payload(payload: dict[str, Any], schema_filename: str) -> None:
    """Validate persisted calibration/prior metadata against its versioned contract."""
    import jsonschema

    schema_path = Path(__file__).resolve().parents[2] / "evaluation/schema" / schema_filename
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(payload))
    if errors:
        raise ValueError(
            f"evaluation payload violates {schema_filename}: "
            + "; ".join(error.message for error in errors[:10])
        )


def _validate_json_schema(payload: Any, schema: dict[str, Any], label: str) -> None:
    import jsonschema

    try:
        jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(
            payload
        )
    except jsonschema.ValidationError as error:
        raise ValueError(f"{label} violates its schema: {error.message}") from error


_CHANNEL_DATASET_BUILD_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["operation", "input_order"],
    "properties": {
        "operation": {"const": "ordered_channel_dataset_concatenation_v1"},
        "input_order": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["path", "sha256"],
                "properties": {
                    "path": {"type": "string", "minLength": 1},
                    "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}


def _identity_complex_cfr(observed_cfr: np.ndarray) -> np.ndarray:
    """Keep the audited complex CFR grid as the model's input matrix."""
    return observed_cfr


def _flatten_real_imag_cfr(observed_cfr: np.ndarray) -> np.ndarray:
    """Flatten each CFR sample in C order, real values before imaginary values."""
    return np.concatenate((observed_cfr.real, observed_cfr.imag), axis=1).reshape(
        len(observed_cfr), -1
    )


def _log_power_per_link_subcarrier(observed_cfr: np.ndarray) -> np.ndarray:
    """Expose only per-link/per-subcarrier channel power in a fixed dB scale."""
    observed = np.asarray(observed_cfr)
    if observed.ndim != 3 or not np.iscomplexobj(observed):
        raise ValueError("power-only features require complex [samples, links, subcarriers] CFR")
    if not np.isfinite(observed).all():
        raise ValueError("power-only CFR input must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        power = (
            np.asarray(observed, dtype=np.complex128).real ** 2
            + np.asarray(observed, dtype=np.complex128).imag ** 2
        )
    if not np.isfinite(power).all():
        raise ValueError("power-only CFR power is not representable in float64")
    log_power_db = 10.0 * np.log10(np.maximum(power, 1e-12))
    return log_power_db.reshape(len(observed), -1)


def _sparse_multipath_features(observed_cfr: np.ndarray) -> np.ndarray:
    """Use bounded receiver-input OMP/BIC fits as trainable sparse features.

    Frequency is expressed in normalized subcarrier-spacing units and the
    delay search is capped at half a normalized period. These are estimator
    coordinates, not physical seconds or oracle path delays.
    """
    observed = np.asarray(observed_cfr)
    if observed.ndim != 3 or not np.iscomplexobj(observed):
        raise ValueError("sparse multipath features require complex [samples, links, tones] CFR")
    if observed.shape[0] < 1 or observed.shape[1] < 1 or observed.shape[2] < 8:
        raise ValueError("sparse multipath features require nonempty CFR with at least eight tones")
    if not np.isfinite(observed).all():
        raise ValueError("sparse multipath CFR must be finite")
    tone_count = observed.shape[-1]
    frequencies = np.arange(tone_count, dtype=np.float64) - tone_count // 2
    pilot_mask = np.ones(tone_count, dtype=np.bool_)
    rows: list[list[float]] = []
    for sample in observed:
        sample_features: list[float] = []
        for link in sample:
            fit = fit_sparse_multipath(
                link,
                frequencies,
                pilot_mask,
                maximum_paths=3,
                maximum_delay_s=0.5,
                delay_grid_size=129,
            )
            signal_energy = float(np.vdot(link, link).real)
            residual_fraction = fit.residual_sum_squares / max(signal_energy, 1e-24)
            gain_power = np.abs(fit.gains) ** 2
            dominant_fraction = (
                float(np.max(gain_power) / np.sum(gain_power)) if gain_power.size else 0.0
            )
            sample_features.extend(
                [
                    fit.model_order / 3.0,
                    float(np.mean(fit.delays_s)) / 0.5 if fit.delays_s.size else 0.0,
                    float(np.std(fit.delays_s)) / 0.5 if fit.delays_s.size else 0.0,
                    residual_fraction,
                    dominant_fraction,
                ]
            )
        rows.append(sample_features)
    result = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("sparse multipath estimator produced nonfinite features")
    return result


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    header = json.dumps(
        {"dtype": value.dtype.str, "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _stable_single_link_sha256(path: str | Path) -> str:
    """Hash one stable regular-file snapshot without following file aliases."""
    file_path = Path(path)
    if not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("platform lacks safe single-link model artifact opening")
    path_before = os.lstat(file_path)
    if not stat.S_ISREG(path_before.st_mode) or path_before.st_nlink != 1:
        raise ValueError("model artifact must be a single-link regular file")
    descriptor = os.open(file_path, os.O_RDONLY | os.O_NOFOLLOW)
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("model artifact must be a single-link regular file")
        if (path_before.st_dev, path_before.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("model artifact path changed before opening")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        after = os.fstat(descriptor)
        path_after = os.lstat(file_path)
    finally:
        os.close(descriptor)

    def identity(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if identity(before) != identity(after) or identity(before) != identity(path_after):
        raise ValueError("model artifact changed during hashing")
    return digest.hexdigest()


def _label_semantics_by_source(data: ChannelDataset) -> str:
    semantics: dict[str, set[str]] = {}
    for source, metadata in zip(data.source, data.metadata, strict=True):
        source_name = str(source)
        label_semantic = metadata.get("geometry_label") or metadata.get("label_semantics")
        if label_semantic is None:
            label_semantic = (
                "SYNTHETIC_DEBUG_CONSTRUCTED"
                if source_name == "synthetic_debug"
                else f"UNSPECIFIED({source_name})"
            )
        semantics.setdefault(source_name, set()).add(str(label_semantic))
    if len(semantics) == 1:
        values = next(iter(semantics.values()))
        if len(values) == 1:
            return next(iter(values))
    return "; ".join(
        f"{source}={'|'.join(sorted(values))}" for source, values in sorted(semantics.items())
    )


def _normalized_label_semantics(value: str) -> str:
    normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "GEOMETRIC_DIRECT_PATH_ORACLE": "geometric_direct_path_oracle",
        "PROFILE_PROXY": "profile_proxy",
        "SYNTHETIC_DEBUG_CONSTRUCTED": "synthetic_debug_direct_component",
        "SYNTHETIC_DEBUG_DIRECT_COMPONENT": "synthetic_debug_direct_component",
        "UNSPECIFIED": "unspecified",
        "UNKNOWN": "unspecified",
        "UNKNOWN_LEGACY": "unspecified",
    }
    return aliases.get(normalized, value.strip().lower().replace(" ", "_"))


def _precomputed_observation_semantics(data: ChannelDataset) -> str:
    semantics = sorted(
        {
            str(metadata.get("receiver_observation"))
            for metadata in data.metadata
            if metadata.get("receiver_observation")
        }
    )
    return "; ".join(semantics) if semantics else "precomputed channel dataset observations"


def _sync_torch_device(device: str, enabled: bool) -> None:
    if enabled and device.startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)


def _learned_parameter_count(model: Any, training: dict[str, Any]) -> int | None:
    reported = training.get("trainable_parameters")
    if isinstance(reported, int) and not isinstance(reported, bool):
        return reported
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    coefficients = getattr(estimator, "coefs_", None)
    intercepts = getattr(estimator, "intercepts_", None)
    if coefficients is not None and intercepts is not None:
        return int(sum(array.size for array in (*coefficients, *intercepts)))
    return None


def _kish_effective_group_count(group_row_counts: Sequence[int]) -> float:
    """Return Kish effective group count from positive group row frequencies."""
    counts = list(group_row_counts)
    if any(not isinstance(count, int) or isinstance(count, bool) or count <= 0 for count in counts):
        raise ValueError("group row counts must be positive integers")
    total = sum(counts)
    squared = sum(count * count for count in counts)
    return total * total / squared if squared else 0.0


def _stable_input_bytes(path: str | Path, *, max_bytes: int = 8 * 1024**3) -> bytes:
    """Read one bounded, single-link input snapshot without following symlinks."""
    source = Path(path)
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise RuntimeError("benchmark replay requires no-follow file opening support")
    before_path = os.lstat(source)
    if not stat.S_ISREG(before_path.st_mode) or before_path.st_nlink != 1:
        raise ValueError(f"benchmark input must be a single-link regular file: {source}")
    if before_path.st_size > max_bytes:
        raise ValueError(f"benchmark input exceeds the {max_bytes}-byte limit: {source}")
    descriptor = os.open(
        source,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before_fd = os.fstat(descriptor)
        if (before_fd.st_dev, before_fd.st_ino) != (before_path.st_dev, before_path.st_ino):
            raise ValueError(f"benchmark input changed before it was opened: {source}")
        chunks: list[bytes] = []
        total = 0
        while total <= max_bytes:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        after_fd = os.fstat(descriptor)
        after_path = os.lstat(source)

        def signature(item: os.stat_result) -> tuple[int, ...]:
            return (
                item.st_dev,
                item.st_ino,
                item.st_mode,
                item.st_nlink,
                item.st_size,
                item.st_mtime_ns,
                item.st_ctime_ns,
            )

        if (
            total > max_bytes
            or not stat.S_ISREG(after_fd.st_mode)
            or after_fd.st_nlink != 1
            or not stat.S_ISREG(after_path.st_mode)
            or after_path.st_nlink != 1
            or signature(before_fd) != signature(after_fd)
            or signature(after_fd) != signature(after_path)
            or total != after_fd.st_size
        ):
            raise ValueError(f"benchmark input changed or exceeded its byte limit: {source}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _validate_npz_snapshot(payload: bytes, *, label: str, max_uncompressed_bytes: int) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if len(entries) > 32 or len(names) != len(set(names)):
                raise ValueError(f"{label} contains an invalid NPZ entry roster")
            if sum(entry.file_size for entry in entries) > max_uncompressed_bytes:
                raise ValueError(f"{label} exceeds the uncompressed dataset size limit")
    except zipfile.BadZipFile as error:
        raise ValueError(f"{label} is not a valid NPZ archive") from error


def _unsupported_benchmark_test_cohorts(cohort_summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    unsupported: list[dict[str, Any]] = []
    for cohort in TEST_COHORTS:
        summary = cohort_summary[cohort]
        reasons = []
        if summary["rows"] == 0:
            reasons.append("NO_ROWS")
        if summary["independent_components"] == 0:
            reasons.append("NO_INDEPENDENT_COMPONENTS")
        if reasons:
            dependencies = [
                "complete authoritative cohort and parent inventory",
                "independent physical-unit validation and trusted signer custody",
            ]
            if cohort == "real_external":
                dependencies.extend(
                    [
                        "authoritative measurement inventory and protocol validation",
                        "rights review and measurement validity evidence",
                    ]
                )
            unsupported.append(
                {
                    "cohort": cohort,
                    "status": "UNSUPPORTED",
                    "reason_codes": reasons,
                    "external_dependencies": dependencies,
                }
            )
    return unsupported


def _run_replay_rng_streams(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    streams: list[dict[str, Any]] = []
    for seed in config["seeds"]:
        value = int(seed)
        streams.append(
            {
                "name": f"run_seed_{value}",
                "library": "numpy/scikit-learn/torch",
                "algorithm": "common_runner_seed_v1",
                "seed_material": [value],
                "derived_seed": value,
                "purpose": "split/model fitting and seed-clustered evaluation",
            }
        )
        if config.get("prior_shift") is not None:
            streams.append(
                {
                    "name": f"calibration_partition_seed_{value}",
                    "library": "numpy",
                    "algorithm": "common_runner_seed_plus_907_v1",
                    "seed_material": [value, 907],
                    "derived_seed": value + 907,
                    "purpose": "train/calibration group split when no locked calibration partition exists",
                }
            )
    if config["observation"] == "comb_ls":
        value = int(config["observation_seed"])
        streams.append(
            {
                "name": "observation_noise",
                "library": "numpy",
                "algorithm": "comb_ls_estimate_rng_v1",
                "seed_material": [value],
                "derived_seed": value,
                "purpose": "comb-LS observation noise",
            }
        )
    phase_seed = int(config["phase_randomization_seed"])
    streams.append(
        {
            "name": "phase_randomization_control",
            "library": "numpy",
            "algorithm": "default_rng_uniform_phase_v1",
            "seed_material": [phase_seed],
            "derived_seed": phase_seed,
            "purpose": "phase-randomized falsification features",
        }
    )
    if "label_permuted" in config["methods"]:
        label_seed = int(config["label_permutation_seed"])
        streams.append(
            {
                "name": "label_permutation_base",
                "library": "numpy",
                "algorithm": "seedsequence_run_plus_base_permutation_v1",
                "seed_material": [label_seed],
                "derived_seed": label_seed,
                "purpose": "training-label permutation control",
            }
        )
    if "row_shuffled" in config["methods"]:
        row_seed = int(config["row_shuffle_seed"])
        streams.append(
            {
                "name": "row_shuffle_base",
                "library": "numpy",
                "algorithm": "seedsequence_run_plus_base_permutation_v1",
                "seed_material": [row_seed],
                "derived_seed": row_seed,
                "purpose": "training-feature row-shuffle control",
            }
        )
    return streams
