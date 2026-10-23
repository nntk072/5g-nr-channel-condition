"""Validated and immutable records for standardized stochastic TDL/CDL channels.

TDL taps and CDL clusters are profile-model entities, not geometric RT rays.
This family store therefore preserves their native coefficient axes and model
metadata without fabricating scene coordinates or geometric LOS labels.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, cast

import numpy as np

from .observations import FrequencyDomainModelInputs, FrequencyDomainObservation
from .targets import TargetDefinition, TargetValidity, TargetValue, definition_digest


class StandardizedFamily(StrEnum):
    """Pinned standardized channel families supported by these records."""

    TDL = "tdl"
    CDL = "cdl"


_SCALABLE_TDL_CDL_MODELS = frozenset({"A", "B", "C", "D", "E"})
_FIXED_TDL_MODELS = frozenset({"A10", "A30", "B100", "C60", "C300", "D10", "D30"})
_PROFILE_LOS_COMPONENT_MODELS = frozenset({"D", "E", "D10", "D30"})
_FIXED_TDL_FREQUENCY_RANGES = {
    "A10": {"FR2"},
    "A30": {"FR1", "FR2"},
    "B100": {"FR1"},
    "C60": {"FR2"},
    "C300": {"FR1"},
    "D10": {"FR2"},
    "D30": {"FR1", "FR2"},
}


def _numpy(value: Any, *, name: str, kind: str) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if kind == "complex" and not np.iscomplexobj(array):
        raise TypeError(f"{name} must be complex-valued")
    if kind == "real" and array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be real-valued")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.array(array, copy=True, order="C")


def _canonical_json(value: Any, *, name: str) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be finite JSON-compatible data") from exc


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    return value


def validate_standardized_channel_admission(
    family: StandardizedFamily,
    profile_model: str,
    h_path: np.ndarray,
    metadata: Mapping[str, Any],
) -> None:
    """Reject TDL/CDL records outside the pinned model's declared scope.

    Sionna's scalable TR 38.901 models do not enforce the 0.5–100 GHz
    carrier-frequency or 2 GHz maximum-bandwidth applicability limits. This
    check runs at record construction so generated diagnostics cannot silently
    become admitted dataset samples. Sampled bandwidth is defined here as the
    subcarrier count times subcarrier spacing, not a separately inferred
    occupied-RB bandwidth.
    """
    prefix = "standardized channel admission rejected: "
    if not isinstance(family, StandardizedFamily):
        raise TypeError("family must use StandardizedFamily")
    supported_models = (
        _SCALABLE_TDL_CDL_MODELS | _FIXED_TDL_MODELS
        if family is StandardizedFamily.TDL
        else _SCALABLE_TDL_CDL_MODELS
    )
    if profile_model not in supported_models:
        if family is StandardizedFamily.TDL:
            raise ValueError(prefix + "TDL profile must be scalable A–E or fixed TS 38.101-4")
        raise ValueError(prefix + "CDL profile must be one of A–E")
    expected_delay_semantics = (
        "fixed_ts_38_101_4_absolute_tap_delays"
        if profile_model in _FIXED_TDL_MODELS
        else "scalable_tr_38_901_profile"
    )
    if metadata.get("delay_profile_semantics") != expected_delay_semantics:
        raise ValueError(prefix + "delay profile semantics disagree with the admitted profile")
    if profile_model in _SCALABLE_TDL_CDL_MODELS and metadata.get("spec_version") not in {
        "16.1",
        "19.2",
    }:
        raise ValueError(prefix + "unsupported TR 38.901 parameter-table version")
    if (
        profile_model in _FIXED_TDL_MODELS
        and metadata.get("fixed_profile_spec_version") != "19.2.2"
    ):
        raise ValueError(prefix + "fixed TDL profile table must identify TS 38.101-4 V19.2.2")
    if h_path.ndim != 6 or h_path.shape[0] != 1 or h_path.shape[2] != 1:
        raise ValueError(prefix + "the supported TDL/CDL model requires one RX and one TX link")

    carrier_hz = metadata.get("carrier_frequency_hz")
    if (
        isinstance(carrier_hz, bool)
        or not isinstance(carrier_hz, (int, float))
        or not math.isfinite(carrier_hz)
        or not 0.5e9 <= carrier_hz <= 100e9
    ):
        raise ValueError(prefix + "carrier_frequency_hz must be within 0.5–100 GHz")
    if profile_model in _FIXED_TDL_MODELS:
        if 0.5e9 <= carrier_hz <= 7.125e9:
            frequency_range = "FR1"
        elif 24.25e9 <= carrier_hz <= 71e9:
            frequency_range = "FR2"
        else:
            raise ValueError(
                prefix + "fixed TS 38.101-4 TDL profiles require an FR1 or FR2 carrier frequency"
            )
        if frequency_range not in _FIXED_TDL_FREQUENCY_RANGES[profile_model]:
            raise ValueError(
                prefix + f"fixed TDL-{profile_model} is not applicable in {frequency_range}"
            )

    subcarrier_count = metadata.get("num_subcarriers")
    spacing_hz = metadata.get("subcarrier_spacing_hz")
    sampled_bandwidth_hz = metadata.get("sampled_bandwidth_hz")
    if (
        isinstance(subcarrier_count, bool)
        or not isinstance(subcarrier_count, int)
        or subcarrier_count < 8
        or isinstance(spacing_hz, bool)
        or not isinstance(spacing_hz, (int, float))
        or not math.isfinite(spacing_hz)
        or spacing_hz <= 0
        or isinstance(sampled_bandwidth_hz, bool)
        or not isinstance(sampled_bandwidth_hz, (int, float))
        or not math.isfinite(sampled_bandwidth_hz)
    ):
        raise ValueError(prefix + "subcarrier grid and sampled bandwidth metadata are required")
    expected_bandwidth_hz = subcarrier_count * spacing_hz
    if not math.isclose(sampled_bandwidth_hz, expected_bandwidth_hz, rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError(prefix + "sampled_bandwidth_hz disagrees with subcarrier grid")
    if sampled_bandwidth_hz > 2e9:
        raise ValueError(prefix + "standardized dataset sampled bandwidth must be at most 2 GHz")
    if profile_model in {"A10", "D10"}:
        channel_bandwidth_hz = metadata.get("channel_bandwidth_hz")
        if (
            isinstance(channel_bandwidth_hz, bool)
            or not isinstance(channel_bandwidth_hz, (int, float))
            or not math.isfinite(channel_bandwidth_hz)
            or channel_bandwidth_hz <= 200e6
        ):
            raise ValueError(
                prefix + f"fixed TDL-{profile_model} requires channel_bandwidth_hz > 200 MHz"
            )

    rx_antennas = int(h_path.shape[1])
    tx_antennas = int(h_path.shape[3])
    declared_rx = metadata.get("rx_antenna_count")
    declared_tx = metadata.get("tx_antenna_count")
    if declared_rx != rx_antennas or declared_tx != tx_antennas:
        raise ValueError(prefix + "declared antenna counts disagree with channel axes")

    if family is StandardizedFamily.TDL:
        if "rx_corr_mat_real" not in metadata or "rx_corr_mat_imag" not in metadata:
            raise ValueError(prefix + "TDL receive-correlation provenance is required")
        matrix_real = metadata.get("rx_corr_mat_real")
        matrix_imag = metadata.get("rx_corr_mat_imag")
        if (matrix_real is None) != (matrix_imag is None):
            raise ValueError(prefix + "receive correlation matrix real/imag metadata is incomplete")
        if matrix_real is not None:
            matrix = np.asarray(matrix_real, dtype=np.float64) + 1j * np.asarray(
                matrix_imag, dtype=np.float64
            )
            if matrix.shape != (rx_antennas, rx_antennas) or not np.isfinite(matrix).all():
                raise ValueError(prefix + "receive correlation matrix shape or values are invalid")
            if not np.allclose(matrix, matrix.conj().T, rtol=0.0, atol=1e-6):
                raise ValueError(prefix + "receive correlation matrix must be Hermitian")
            if not np.allclose(np.diag(matrix), 1.0, rtol=0.0, atol=1e-6):
                raise ValueError(prefix + "receive correlation matrix must have unit diagonal")
            if np.linalg.eigvalsh(matrix).min() <= 1e-7:
                raise ValueError(prefix + "receive correlation matrix must be positive definite")
        return

    direction = metadata.get("direction")
    if direction not in {"downlink", "uplink"}:
        raise ValueError(prefix + "CDL direction must be uplink or downlink")
    rx_array_key, tx_array_key = (
        ("ut_array", "bs_array") if direction == "downlink" else ("bs_array", "ut_array")
    )

    def declared_array_count(key: str) -> int:
        array = metadata.get(key)
        if not isinstance(array, Mapping):
            raise ValueError(prefix + f"CDL metadata must declare {key}")
        integer_fields = (
            "num_rows_per_panel",
            "num_cols_per_panel",
            "num_rows",
            "num_cols",
        )
        values = [array.get(field) for field in integer_fields]
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values
        ):
            raise ValueError(prefix + f"CDL {key} has invalid panel dimensions")
        polarization = array.get("polarization")
        if polarization not in {"single", "dual"}:
            raise ValueError(prefix + f"CDL {key} has invalid polarization")
        polarization_type = array.get("polarization_type")
        allowed_types = {"V", "H"} if polarization == "single" else {"VH", "cross"}
        if polarization_type not in allowed_types:
            raise ValueError(prefix + f"CDL {key} has invalid polarization type")
        if array.get("antenna_pattern") not in {"omni", "38.901", "38.901-handheld"}:
            raise ValueError(prefix + f"CDL {key} has invalid antenna pattern")
        for field in (
            "panel_vertical_spacing",
            "panel_horizontal_spacing",
            "element_vertical_spacing",
            "element_horizontal_spacing",
        ):
            spacing = array.get(field)
            if spacing is not None and (
                isinstance(spacing, bool)
                or not isinstance(spacing, (int, float))
                or not math.isfinite(spacing)
                or spacing <= 0
            ):
                raise ValueError(prefix + f"CDL {key} has invalid antenna spacing")
        polarization_count = 1 if polarization == "single" else 2
        rows_per_panel, cols_per_panel, rows, cols = (cast(int, value) for value in values)
        return rows_per_panel * cols_per_panel * rows * cols * polarization_count

    if declared_array_count(rx_array_key) != rx_antennas:
        raise ValueError(prefix + "CDL receive array configuration disagrees with channel axes")
    if declared_array_count(tx_array_key) != tx_antennas:
        raise ValueError(prefix + "CDL transmit array configuration disagrees with channel axes")


@dataclass(frozen=True, slots=True, eq=False)
class StandardizedChannelRecord:
    """One independent stochastic draw with oracle-only channel fields.

    ``h_path`` axes are [rx, rx_ant, tx, tx_ant, cluster, time], ``tau_s``
    axes are [rx, tx, cluster], and ``cfr`` axes are
    [rx, rx_ant, tx, tx_ant, time, subcarrier]. Delays are profile-relative
    excess delays, and ``profile_los_component`` is never geometric LOS truth.
    """

    sample_id: str
    run_id: str
    sample_index: int
    family: StandardizedFamily
    profile_model: str
    profile_los_component: bool
    h_path: np.ndarray
    tau_s: np.ndarray
    cfr: np.ndarray
    subcarrier_offsets_hz: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("sample_id", "run_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.sample_index, bool) or not isinstance(self.sample_index, int):
            raise TypeError("sample_index must be an integer")
        if self.sample_index < 0:
            raise ValueError("sample_index must be nonnegative")
        if not isinstance(self.family, StandardizedFamily):
            raise TypeError("family must use StandardizedFamily")
        allowed_models = (
            _SCALABLE_TDL_CDL_MODELS | _FIXED_TDL_MODELS
            if self.family is StandardizedFamily.TDL
            else _SCALABLE_TDL_CDL_MODELS
        )
        if self.profile_model not in allowed_models:
            if self.family is StandardizedFamily.TDL:
                raise ValueError("TDL profile must be scalable A–E or fixed TS 38.101-4")
            raise ValueError("CDL profile must be one of A–E")
        if not isinstance(self.profile_los_component, bool):
            raise TypeError("profile_los_component must be boolean")
        if self.profile_los_component != (self.profile_model in _PROFILE_LOS_COMPONENT_MODELS):
            raise ValueError("profile LOS category disagrees with the standardized profile family")

        h_path = _numpy(self.h_path, name="h_path", kind="complex")
        tau_s = _numpy(self.tau_s, name="tau_s", kind="real").astype(np.float64, copy=False)
        cfr = _numpy(self.cfr, name="cfr", kind="complex")
        offsets = _numpy(
            self.subcarrier_offsets_hz, name="subcarrier_offsets_hz", kind="real"
        ).astype(np.float64, copy=False)
        if h_path.ndim != 6 or min(h_path.shape) < 1:
            raise ValueError(
                "h_path must have non-empty [rx, rx_ant, tx, tx_ant, cluster, time] axes"
            )
        if h_path.shape[0] != 1 or h_path.shape[2] != 1:
            raise ValueError("pinned TDL/CDL adapters require one TX and one RX link")
        if tau_s.shape != (1, 1, h_path.shape[-2]) or np.any(tau_s < 0):
            raise ValueError("tau_s must be nonnegative [1, 1, cluster]")
        expected_cfr_shape = (*h_path.shape[:4], h_path.shape[-1], len(offsets))
        if cfr.shape != expected_cfr_shape:
            raise ValueError(f"cfr must have shape {expected_cfr_shape}")
        if len(offsets) < 8 or np.any(np.diff(offsets) <= 0) or not np.any(offsets == 0.0):
            raise ValueError(
                "subcarrier offsets must be increasing, include DC, and have length >= 8"
            )

        delays = tau_s[:, None, :, None, :, None, None]
        frequencies = offsets.reshape(1, 1, 1, 1, 1, 1, -1)
        expected_cfr = np.sum(
            h_path[..., None] * np.exp(-2j * np.pi * delays * frequencies), axis=4
        )
        error = np.max(np.abs(cfr - expected_cfr))
        scale = max(float(np.max(np.abs(expected_cfr))), 1e-8)
        if error > 1e-7 + 5e-5 * scale:
            raise ValueError("cfr does not match the stored path coefficients and delays")

        raw_metadata = dict(self.metadata)
        if raw_metadata.get("geometric_los_label") is not None:
            raise ValueError("standardized profile records may not claim geometric LOS labels")
        if raw_metadata.get("profile_los_component") is not self.profile_los_component:
            raise ValueError("metadata profile LOS category disagrees with the record")
        if (
            raw_metadata.get("profile_model", raw_metadata.get("profile_proxy"))
            != self.profile_model
        ):
            raise ValueError("metadata profile model disagrees with the record")
        if raw_metadata.get("delay_reference") != "profile_excess_delay":
            raise ValueError("standardized channels must declare profile-relative excess delays")
        spacing = raw_metadata.get("subcarrier_spacing_hz")
        if not isinstance(spacing, (int, float)) or spacing <= 0:
            raise ValueError("metadata must declare a positive subcarrier spacing")
        if not np.allclose(np.diff(offsets), spacing, rtol=1e-10, atol=1e-6):
            raise ValueError("subcarrier offsets disagree with the declared spacing")
        if raw_metadata.get("num_subcarriers") != len(offsets):
            raise ValueError("metadata subcarrier count disagrees with the CFR axis")
        if raw_metadata.get("num_time_steps") != h_path.shape[-1]:
            raise ValueError("metadata time-step count disagrees with the channel axes")
        validate_standardized_channel_admission(
            self.family, self.profile_model, h_path, raw_metadata
        )
        snapshot = json.loads(_canonical_json(raw_metadata, name="metadata"))
        h_path.setflags(write=False)
        tau_s.setflags(write=False)
        cfr.setflags(write=False)
        offsets.setflags(write=False)
        object.__setattr__(self, "h_path", h_path)
        object.__setattr__(self, "tau_s", tau_s)
        object.__setattr__(self, "cfr", cfr)
        object.__setattr__(self, "subcarrier_offsets_hz", offsets)
        object.__setattr__(self, "metadata", _freeze_json(snapshot))


@dataclass(frozen=True, slots=True, eq=False)
class StandardizedChannelRun:
    """One reproducible generator run, grouped for leakage-safe splitting."""

    run_id: str
    records: tuple[StandardizedChannelRecord, ...]

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        if not records or any(record.run_id != self.run_id for record in records):
            raise ValueError("a standardized run needs records that all share its run_id")
        sample_ids = [record.sample_id for record in records]
        indices = [record.sample_index for record in records]
        if len(set(sample_ids)) != len(sample_ids) or len(set(indices)) != len(indices):
            raise ValueError("standardized run sample IDs and indices must be unique")
        if set(indices) != set(range(len(records))):
            raise ValueError("sample_index values must cover the run's contiguous batch range")
        first = records[0]
        if any(
            record.family is not first.family
            or record.profile_model != first.profile_model
            or record.metadata != first.metadata
            for record in records[1:]
        ):
            raise ValueError("one standardized run cannot mix profile or generator metadata")
        object.__setattr__(
            self, "records", tuple(sorted(records, key=lambda item: item.sample_index))
        )


@dataclass(frozen=True, slots=True, eq=False)
class StandardizedLearningExample:
    """A noisy receiver view and separately held profile-proxy target."""

    observation: FrequencyDomainObservation
    targets: Mapping[str, TargetValue]
    source_run_id: str
    observation_seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.observation, FrequencyDomainObservation):
            raise TypeError("observation must be a FrequencyDomainObservation")
        targets = dict(self.targets)
        if not targets or any(not isinstance(target, TargetValue) for target in targets.values()):
            raise TypeError("learning examples require named TargetValue records")
        if any(target.entity_id != self.observation.sample_id for target in targets.values()):
            raise ValueError("target sample IDs must match their receiver observation")
        if not self.source_run_id.strip():
            raise ValueError("source_run_id must be non-empty")
        if (
            isinstance(self.observation_seed, bool)
            or not isinstance(self.observation_seed, int)
            or self.observation_seed < 0
        ):
            raise ValueError("observation_seed must be a nonnegative integer")
        object.__setattr__(self, "targets", MappingProxyType(targets))

    @property
    def sample_id(self) -> str:
        return self.observation.sample_id

    def model_inputs(self) -> FrequencyDomainModelInputs:
        """Return receiver-visible data without sample IDs, targets, or oracle truth."""
        return self.observation.model_inputs


@dataclass(frozen=True, slots=True, eq=False)
class StandardizedLearningRun:
    """Receiver/target child records joined by sample ID to a channel-oracle run."""

    run_id: str
    examples: tuple[StandardizedLearningExample, ...]

    def __post_init__(self) -> None:
        examples = tuple(self.examples)
        if not isinstance(self.run_id, str) or not self.run_id.strip() or not examples:
            raise ValueError("learning run needs a non-empty run_id and at least one example")
        sample_ids = tuple(example.sample_id for example in examples)
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("learning run sample IDs must be unique")
        if any(example.source_run_id != self.run_id for example in examples):
            raise ValueError("learning examples must reference their parent channel run")
        object.__setattr__(self, "examples", examples)


def standardized_learning_example_from_record(
    record: StandardizedChannelRecord,
    *,
    snr_db: float,
    seed: int,
    time_index: int = 0,
) -> StandardizedLearningExample:
    """Create a noisy CFR view and a typed profile-proxy target.

    This direct frequency-domain observation uses complex AWGN. It is not a
    PUSCH/DMRS waveform, channel estimator, or geometric LOS label.
    """
    if not isinstance(record, StandardizedChannelRecord):
        raise TypeError("record must be a StandardizedChannelRecord")
    if not math.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(time_index, bool) or not isinstance(time_index, int):
        raise TypeError("time_index must be an integer")
    if time_index < 0 or time_index >= record.cfr.shape[-2]:
        raise ValueError("time_index is outside the stored channel trajectory")
    noiseless = np.asarray(record.cfr[0, :, 0, :, time_index, :], dtype=np.complex128)
    signal_power = float(np.mean(np.abs(noiseless) ** 2))
    if signal_power <= 0 or not math.isfinite(signal_power):
        raise ValueError("cannot define relative-SNR noise for a zero-power channel")
    noise_variance = signal_power / (10.0 ** (snr_db / 10.0))
    rng = np.random.default_rng(seed)
    noise = np.sqrt(noise_variance / 2.0) * (
        rng.standard_normal(noiseless.shape) + 1j * rng.standard_normal(noiseless.shape)
    )
    inputs = FrequencyDomainModelInputs(
        observed_cfr=(noiseless + noise).astype(np.complex64),
        noise_variance=noise_variance,
        snr_db=snr_db,
        carrier_hz=float(record.metadata["carrier_frequency_hz"]),
        subcarrier_offsets_hz=record.subcarrier_offsets_hz,
    )
    observation = FrequencyDomainObservation(record.sample_id, inputs)
    definition = TargetDefinition(
        semantic="standardized_profile_los_category_proxy",
        units="binary_class",
        source="Sionna standardized TDL/CDL profile model",
        reference=(
            "profile_los_component false maps to 0 and true maps to 1; not geometric LOS truth"
        ),
        definition_sha256=definition_digest(
            "profile_los_proxy:v2; profile_los_component=false:0,true:1; "
            "source=standardized_profile_model; not_geometric_visibility"
        ),
    )
    target = TargetValue(
        entity_id=record.sample_id,
        definition=definition,
        values=np.asarray([int(record.profile_los_component)], dtype=np.uint8),
        valid_mask=np.asarray([True]),
        reason_codes=(TargetValidity.VALID,),
        confidence="exact standardized profile category; not physical visibility",
    )
    return StandardizedLearningExample(
        observation=observation,
        targets={"profile_los_proxy": target},
        source_run_id=record.run_id,
        observation_seed=seed,
    )


def standardized_learning_run_from_channel_run(
    run: StandardizedChannelRun, *, snr_db: float, seed: int, time_index: int = 0
) -> StandardizedLearningRun:
    """Create independently seeded receiver observations for each oracle draw."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    examples = []
    for record in run.records:
        child_seed = int(
            np.random.SeedSequence([seed, record.sample_index]).generate_state(1, dtype=np.uint32)[
                0
            ]
        )
        examples.append(
            standardized_learning_example_from_record(
                record, snr_db=snr_db, seed=child_seed, time_index=time_index
            )
        )
    return StandardizedLearningRun(run.run_id, tuple(examples))


def standardized_run_from_channel(
    channel: Any,
    *,
    family: StandardizedFamily,
    run_id: str | None = None,
) -> StandardizedChannelRun:
    """Split a Sionna TDL/CDL adapter batch into immutable per-draw records."""
    if not isinstance(family, StandardizedFamily):
        raise TypeError("family must use StandardizedFamily")
    metadata = dict(channel.metadata)
    profile_model = metadata.get("profile_model", metadata.get("profile_proxy"))
    if not isinstance(profile_model, str):
        raise ValueError("adapter metadata must identify its standardized profile model")
    if metadata.get("geometric_los_label") is not None:
        raise ValueError("adapter output unexpectedly contains a geometric LOS label")
    batch_size = int(metadata.get("batch_size", 0))
    h_all = _numpy(channel.h_path, name="h_path", kind="complex")
    tau_all = _numpy(channel.tau_s, name="tau_s", kind="real")
    cfr_all = _numpy(
        channel.channel_frequency_response, name="channel_frequency_response", kind="complex"
    )
    if batch_size < 1 or h_all.shape[0] != batch_size or tau_all.shape[0] != batch_size:
        raise ValueError("adapter batch metadata and channel arrays disagree")
    if cfr_all.shape[0] != batch_size:
        raise ValueError("adapter batch metadata and CFR array disagree")
    subcarrier_spacing = float(metadata["subcarrier_spacing_hz"])
    subcarrier_count = int(metadata["num_subcarriers"])
    offsets = (
        np.arange(subcarrier_count, dtype=np.float64) - subcarrier_count // 2
    ) * subcarrier_spacing
    canonical_config = _canonical_json(metadata, name="adapter metadata")
    resolved_run_id = run_id or (
        f"{family.value}-{profile_model}-{hashlib.sha256(canonical_config).hexdigest()[:20]}"
    )
    profile_los = bool(metadata["profile_los_component"])
    records = tuple(
        StandardizedChannelRecord(
            sample_id=f"{resolved_run_id}-sample-{index:08d}",
            run_id=resolved_run_id,
            sample_index=index,
            family=family,
            profile_model=profile_model,
            profile_los_component=profile_los,
            h_path=h_all[index],
            tau_s=tau_all[index],
            cfr=cfr_all[index],
            subcarrier_offsets_hz=offsets,
            metadata={
                **metadata,
                "delay_reference": "profile_excess_delay",
            },
        )
        for index in range(batch_size)
    )
    return StandardizedChannelRun(resolved_run_id, records)


@dataclass(frozen=True, slots=True)
class StoredStandardizedRun:
    """Immutable pointer to one manifest-last standardized channel run."""

    run_id: str
    content_sha256: str
    relative_path: str
    sample_ids: tuple[str, ...]
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class StoredStandardizedLearningRun:
    """Immutable pointer to separately stored observation and target children."""

    run_id: str
    content_sha256: str
    relative_path: str
    sample_ids: tuple[str, ...]
    schema_version: int = 1


def _publish_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            temp_path = stream.name
        try:
            os.link(temp_path, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)


def _target_bytes(targets: Mapping[str, TargetValue]) -> bytes:
    payload: dict[str, Any] = {"schema": "standardized_targets.v1", "targets": {}}
    for name, target in sorted(targets.items()):
        definition = target.definition
        payload["targets"][name] = {
            "entity_id": target.entity_id,
            "definition": {
                "semantic": definition.semantic,
                "units": definition.units,
                "source": definition.source,
                "reference": definition.reference,
                "definition_sha256": definition.definition_sha256,
            },
            "values": target.values.tolist(),
            "valid_mask": target.valid_mask.tolist(),
            "reason_codes": [reason.value for reason in target.reason_codes],
            "confidence": target.confidence,
        }
    return _canonical_json(payload, name="standardized targets") + b"\n"


def _observation_bytes(example: StandardizedLearningExample, run_id: str) -> bytes:
    inputs = example.model_inputs()
    metadata = {
        "schema": "standardized_observation.v1",
        "sample_id": example.sample_id,
        "source_run_id": run_id,
        "observation_seed": example.observation_seed,
        "observation_semantics": "synthetic_noisy_cfr_estimate",
        "receiver_model": "true_cfr_plus_complex_awgn; not_nr_waveform_or_channel_estimator",
        "noise_variance": inputs.noise_variance,
        "snr_db": inputs.snr_db,
        "carrier_hz": inputs.carrier_hz,
    }
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        observed_cfr=inputs.observed_cfr,
        subcarrier_offsets_hz=inputs.subcarrier_offsets_hz,
        metadata_json=np.frombuffer(
            _canonical_json(metadata, name="observation metadata"), dtype=np.uint8
        ),
    )
    return stream.getvalue()


def write_standardized_learning_run(
    root: str | Path, run: StandardizedLearningRun
) -> StoredStandardizedLearningRun:
    """Persist receiver observations and targets apart from physical truth."""
    base = Path(root)
    references: list[dict[str, Any]] = []
    for example in run.examples:
        observation_payload = _observation_bytes(example, run.run_id)
        observation_hash = hashlib.sha256(observation_payload).hexdigest()
        observation_path = f"observations/{observation_hash[:2]}/{observation_hash}.npz"
        _publish_immutable(base / observation_path, observation_payload)
        target_payload = _target_bytes(example.targets)
        target_hash = hashlib.sha256(target_payload).hexdigest()
        target_path = f"targets/{target_hash[:2]}/{target_hash}.json"
        _publish_immutable(base / target_path, target_payload)
        references.append(
            {
                "sample_id": example.sample_id,
                "observation": {
                    "content_sha256": observation_hash,
                    "relative_path": observation_path,
                },
                "targets": {"content_sha256": target_hash, "relative_path": target_path},
            }
        )
    core = {
        "schema": "standardized_learning_run.v1",
        "schema_version": 1,
        "run_id": run.run_id,
        "records": references,
    }
    manifest_hash = hashlib.sha256(_canonical_json(core, name="learning run manifest")).hexdigest()
    manifest = {**core, "content_sha256": manifest_hash}
    relative_path = (
        f"manifests/learning-runs/{hashlib.sha256(run.run_id.encode()).hexdigest()}.json"
    )
    _publish_immutable(
        base / relative_path,
        _canonical_json(manifest, name="learning run manifest") + b"\n",
    )
    return StoredStandardizedLearningRun(
        run_id=run.run_id,
        content_sha256=manifest_hash,
        relative_path=relative_path,
        sample_ids=tuple(example.sample_id for example in run.examples),
    )


def read_standardized_learning_run(
    root: str | Path, manifest: StoredStandardizedLearningRun | str | Path
) -> StandardizedLearningRun:
    """Read a learning view and validate independent observation/target hashes."""
    base = Path(root)
    if isinstance(manifest, StoredStandardizedLearningRun):
        stored = manifest
        relative = PurePosixPath(stored.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("standardized learning manifest path must remain inside the store")
        manifest_path = base.joinpath(*relative.parts)
    else:
        manifest_path = Path(manifest)
        if not manifest_path.is_absolute():
            manifest_path = base / manifest_path
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored = StoredStandardizedLearningRun(
            run_id=raw["run_id"],
            content_sha256=raw["content_sha256"],
            relative_path=manifest_path.relative_to(base).as_posix(),
            sample_ids=tuple(item["sample_id"] for item in raw["records"]),
            schema_version=raw["schema_version"],
        )
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if stored.schema_version != 1 or raw_manifest.get("schema") != "standardized_learning_run.v1":
        raise ValueError("unsupported standardized learning run schema")
    core = {key: raw_manifest[key] for key in ("schema", "schema_version", "run_id", "records")}
    digest = hashlib.sha256(_canonical_json(core, name="learning run manifest")).hexdigest()
    if digest != raw_manifest.get("content_sha256") or digest != stored.content_sha256:
        raise ValueError("standardized learning manifest hash is invalid")
    if raw_manifest["run_id"] != stored.run_id:
        raise ValueError("standardized learning run ID disagrees with its manifest pointer")

    examples: list[StandardizedLearningExample] = []
    for reference in raw_manifest["records"]:
        sample_id = reference["sample_id"]
        observation_ref = reference["observation"]
        observation_path = PurePosixPath(observation_ref["relative_path"])
        if observation_path.is_absolute() or ".." in observation_path.parts:
            raise ValueError("standardized observation path must remain inside the store")
        observation_bytes = base.joinpath(*observation_path.parts).read_bytes()
        observation_hash = hashlib.sha256(observation_bytes).hexdigest()
        if (
            observation_hash != observation_ref["content_sha256"]
            or observation_path.name != f"{observation_hash}.npz"
        ):
            raise ValueError("standardized observation content hash is invalid")
        with np.load(io.BytesIO(observation_bytes), allow_pickle=False) as archive:
            metadata = json.loads(archive["metadata_json"].tobytes().decode("utf-8"))
            if (
                metadata.get("schema") != "standardized_observation.v1"
                or metadata.get("sample_id") != sample_id
                or metadata.get("source_run_id") != stored.run_id
            ):
                raise ValueError("standardized observation identity is invalid")
            inputs = FrequencyDomainModelInputs(
                observed_cfr=archive["observed_cfr"],
                noise_variance=metadata["noise_variance"],
                snr_db=metadata["snr_db"],
                carrier_hz=metadata["carrier_hz"],
                subcarrier_offsets_hz=archive["subcarrier_offsets_hz"],
            )
        target_ref = reference["targets"]
        target_path = PurePosixPath(target_ref["relative_path"])
        if target_path.is_absolute() or ".." in target_path.parts:
            raise ValueError("standardized target path must remain inside the store")
        target_bytes = base.joinpath(*target_path.parts).read_bytes()
        target_hash = hashlib.sha256(target_bytes).hexdigest()
        if target_hash != target_ref["content_sha256"] or target_path.name != f"{target_hash}.json":
            raise ValueError("standardized target content hash is invalid")
        target_payload = json.loads(target_bytes.decode("utf-8"))
        if target_payload.get("schema") != "standardized_targets.v1":
            raise ValueError("unsupported standardized target schema")
        targets: dict[str, TargetValue] = {}
        for name, item in target_payload["targets"].items():
            definition = TargetDefinition(**item["definition"])
            targets[name] = TargetValue(
                entity_id=item["entity_id"],
                definition=definition,
                values=np.asarray(item["values"]),
                valid_mask=np.asarray(item["valid_mask"], dtype=bool),
                reason_codes=tuple(TargetValidity(value) for value in item["reason_codes"]),
                confidence=item["confidence"],
            )
        observation = FrequencyDomainObservation(sample_id, inputs)
        examples.append(
            StandardizedLearningExample(
                observation=observation,
                targets=targets,
                source_run_id=stored.run_id,
                observation_seed=int(metadata["observation_seed"]),
            )
        )
    sample_ids = tuple(example.sample_id for example in examples)
    if sample_ids != stored.sample_ids:
        raise ValueError("standardized learning sample order disagrees with run pointer")
    return StandardizedLearningRun(stored.run_id, tuple(examples))


def _record_bytes(record: StandardizedChannelRecord) -> bytes:
    metadata = {
        "schema": "standardized_channel_record.v1",
        "sample_id": record.sample_id,
        "run_id": record.run_id,
        "sample_index": record.sample_index,
        "family": record.family.value,
        "profile_model": record.profile_model,
        "profile_los_component": record.profile_los_component,
        "metadata": _plain_json(record.metadata),
    }
    arrays: dict[str, np.ndarray] = {
        "h_path": record.h_path,
        "tau_s": record.tau_s,
        "cfr": record.cfr,
        "subcarrier_offsets_hz": record.subcarrier_offsets_hz,
        "metadata_json": np.frombuffer(
            _canonical_json(metadata, name="record metadata"), dtype=np.uint8
        ),
    }
    stream = io.BytesIO()
    np.savez_compressed(stream, **cast(Any, arrays))
    return stream.getvalue()


def write_standardized_run(root: str | Path, run: StandardizedChannelRun) -> StoredStandardizedRun:
    """Store per-draw physical channels and atomically publish an immutable run index."""
    base = Path(root)
    record_refs: list[dict[str, str]] = []
    for record in run.records:
        payload = _record_bytes(record)
        digest = hashlib.sha256(payload).hexdigest()
        relative_path = f"records/{digest[:2]}/{digest}.npz"
        _publish_immutable(base / relative_path, payload)
        record_refs.append(
            {
                "sample_id": record.sample_id,
                "content_sha256": digest,
                "relative_path": relative_path,
            }
        )
    core = {
        "schema": "standardized_channel_run.v1",
        "schema_version": 1,
        "run_id": run.run_id,
        "records": record_refs,
    }
    manifest_payload = _canonical_json(core, name="run manifest")
    digest = hashlib.sha256(manifest_payload).hexdigest()
    manifest = {**core, "content_sha256": digest}
    relative_path = f"manifests/runs/{hashlib.sha256(run.run_id.encode()).hexdigest()}.json"
    _publish_immutable(base / relative_path, _canonical_json(manifest, name="run manifest") + b"\n")
    return StoredStandardizedRun(
        run_id=run.run_id,
        content_sha256=digest,
        relative_path=relative_path,
        sample_ids=tuple(record.sample_id for record in run.records),
    )


def read_standardized_run(
    root: str | Path, manifest: StoredStandardizedRun | str | Path
) -> StandardizedChannelRun:
    """Load a standardized run with manifest, record hash, and schema validation."""
    base = Path(root)
    if isinstance(manifest, StoredStandardizedRun):
        stored = manifest
        relative = PurePosixPath(stored.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("standardized manifest path must remain inside the store")
        manifest_path = base.joinpath(*relative.parts)
    else:
        manifest_path = Path(manifest)
        if not manifest_path.is_absolute():
            manifest_path = base / manifest_path
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored = StoredStandardizedRun(
            run_id=raw["run_id"],
            content_sha256=raw["content_sha256"],
            relative_path=manifest_path.relative_to(base).as_posix(),
            sample_ids=tuple(item["sample_id"] for item in raw["records"]),
            schema_version=raw["schema_version"],
        )
    if stored.schema_version != 1:
        raise ValueError(f"unsupported standardized run schema version: {stored.schema_version}")
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw_manifest.get("schema") != "standardized_channel_run.v1":
        raise ValueError("unsupported standardized run manifest")
    core = {key: raw_manifest[key] for key in ("schema", "schema_version", "run_id", "records")}
    digest = hashlib.sha256(_canonical_json(core, name="run manifest")).hexdigest()
    if digest != raw_manifest.get("content_sha256") or digest != stored.content_sha256:
        raise ValueError("standardized run manifest hash is invalid")
    if raw_manifest["run_id"] != stored.run_id:
        raise ValueError("standardized run ID does not match its manifest pointer")
    records: list[StandardizedChannelRecord] = []
    for reference in raw_manifest["records"]:
        relative_record = PurePosixPath(reference["relative_path"])
        if relative_record.is_absolute() or ".." in relative_record.parts:
            raise ValueError("standardized record path must remain inside the store")
        payload = base.joinpath(*relative_record.parts).read_bytes()
        record_hash = hashlib.sha256(payload).hexdigest()
        if (
            record_hash != reference["content_sha256"]
            or relative_record.name != f"{record_hash}.npz"
        ):
            raise ValueError("standardized record content hash is invalid")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            metadata = json.loads(archive["metadata_json"].tobytes().decode("utf-8"))
            if metadata.get("schema") != "standardized_channel_record.v1":
                raise ValueError("unsupported standardized channel record")
            if metadata.get("sample_id") != reference["sample_id"]:
                raise ValueError("standardized record identity disagrees with run manifest")
            record = StandardizedChannelRecord(
                sample_id=metadata["sample_id"],
                run_id=metadata["run_id"],
                sample_index=metadata["sample_index"],
                family=StandardizedFamily(metadata["family"]),
                profile_model=metadata["profile_model"],
                profile_los_component=metadata["profile_los_component"],
                h_path=archive["h_path"],
                tau_s=archive["tau_s"],
                cfr=archive["cfr"],
                subcarrier_offsets_hz=archive["subcarrier_offsets_hz"],
                metadata=metadata["metadata"],
            )
            if record.sample_id != reference["sample_id"] or record.run_id != stored.run_id:
                raise ValueError("standardized sample identity disagrees with run manifest")
            records.append(record)
    if tuple(record.sample_id for record in records) != stored.sample_ids:
        raise ValueError("standardized sample order disagrees with run pointer")
    return StandardizedChannelRun(stored.run_id, tuple(records))
