"""Fail-closed schema audit for values allowed across model-input boundaries."""

from __future__ import annotations

import sys
from dataclasses import fields, is_dataclass
from numbers import Real
from typing import Any

import numpy as np

_INPUT_FIELDS: dict[tuple[str, str], frozenset[str]] = {
    ("datasets.channel", "ChannelModelInputs"): frozenset({"observed_cfr"}),
    ("datasets.analytical", "AnalyticalModelInputs"): frozenset(
        {"frequencies_hz", "channel_frequency_response"}
    ),
    ("datasets.observations", "FrequencyDomainModelInputs"): frozenset(
        {"observed_cfr", "noise_variance", "snr_db", "carrier_hz", "subcarrier_offsets_hz"}
    ),
    ("datasets.nr", "NRModelInputs"): frozenset(
        {
            "received_grid",
            "h_hat",
            "error_variance",
            "pilot_mask",
            "fft_bin_indices",
            "receiver_config",
            "received_grid_mask",
        }
    ),
}

_NR_RECEIVER_CONFIG_FIELDS = frozenset(
    {
        "n_size_bwp",
        "subcarrier_spacing_khz",
        "num_antenna_ports",
        "num_layers",
        "n_cell_id",
        "dmrs_n_id",
        "dmrs_n_scid",
        "dmrs_config_type",
        "dmrs_length",
        "dmrs_additional_position",
        "num_cdm_groups_without_data",
        "mapping_type",
        "symbol_allocation",
        "precoding",
        "interpolation_type",
        "n_size_grid",
        "n_start_grid",
        "n_start_bwp",
    }
)


def _field_names(value: Any) -> frozenset[str]:
    if not is_dataclass(value) or isinstance(value, type):
        raise TypeError("model input payloads must use an audited dataclass contract")
    return frozenset(item.name for item in fields(value))


def _is_registered_class(value: Any, module_name: str, class_name: str) -> bool:
    module = sys.modules.get(module_name)
    return module is not None and getattr(module, class_name, None) is type(value)


def _validated_array(
    value: Any,
    *,
    name: str,
    value_kind: str,
    ndim: int | None = None,
    nonnegative: bool = False,
) -> tuple[int, ...]:
    """Admit only dense numeric arrays with the declared inference-time domain."""
    is_torch_tensor = False
    try:
        import torch

        is_torch_tensor = torch.is_tensor(value)
    except ImportError:
        torch = None

    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biufc":
            raise TypeError(f"{name} must be a numeric array")
        shape = value.shape
        is_complex = value.dtype.kind == "c"
        is_boolean = value.dtype.kind == "b"
        is_integer = value.dtype.kind in "iu"
        finite = bool(np.isfinite(value).all())
        has_negative = (
            bool(np.any(value < 0))
            if value_kind != "complex" and not is_complex and not is_boolean
            else False
        )
    elif is_torch_tensor:
        assert torch is not None
        if value.layout != torch.strided:
            raise TypeError(f"{name} must be a dense numeric array")
        shape = tuple(value.shape)
        is_complex = value.is_complex()
        is_boolean = value.dtype == torch.bool
        is_integer = not value.is_floating_point() and not is_complex and not is_boolean
        if not (value.is_floating_point() or is_complex or is_boolean or is_integer):
            raise TypeError(f"{name} must be a numeric array")
        finite = bool(torch.isfinite(value).all())
        has_negative = (
            bool((value < 0).any())
            if value_kind != "complex" and not is_complex and not is_boolean
            else False
        )
    else:
        raise TypeError(f"{name} must be a NumPy array or dense PyTorch tensor")

    if ndim is not None and len(shape) != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if any(size <= 0 for size in shape):
        raise ValueError(f"{name} must have nonempty dimensions")
    if value_kind == "complex" and not is_complex:
        raise TypeError(f"{name} must be complex-valued")
    if value_kind == "real" and (is_complex or is_boolean):
        raise TypeError(f"{name} must be real-valued")
    if value_kind == "boolean" and not is_boolean:
        raise TypeError(f"{name} must be Boolean-valued")
    if value_kind == "integer" and (not is_integer or is_boolean):
        raise TypeError(f"{name} must be integer-valued")
    if not finite:
        raise ValueError(f"{name} must contain only finite values")
    if nonnegative and has_negative:
        raise ValueError(f"{name} must be nonnegative")
    return tuple(shape)


def _validated_real_scalar(
    value: Any,
    *,
    name: str,
    nonnegative: bool = False,
    positive: bool = False,
) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real numeric scalar")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")


def audit_model_input_payload(value: Any) -> Any:
    """Check inference-only field rosters and values at model-input boundaries.

    Tuples are accepted for time-aligned observations. Arbitrary dataclasses,
    dictionaries, objects, and non-numeric sidecars are rejected so adding
    provenance or target data to a model path requires an explicit review.
    """
    if isinstance(value, tuple):
        if not value:
            raise ValueError("model input sequence must be nonempty")
        for item in value:
            audit_model_input_payload(item)
        return value

    identity = (type(value).__module__, type(value).__qualname__)
    expected = _INPUT_FIELDS.get(identity)
    if expected is None or not _is_registered_class(value, *identity):
        raise TypeError(f"unrecognized model-input contract: {identity[0]}.{identity[1]}")
    observed = _field_names(value)
    if observed != expected:
        extra = sorted(observed - expected)
        missing = sorted(expected - observed)
        raise ValueError(
            f"model-input field roster drift for {identity[1]}: "
            f"extra={extra}, missing={missing}"
        )

    if identity == ("datasets.nr", "NRModelInputs"):
        config = getattr(value, "receiver_config")
        config_identity = (type(config).__module__, type(config).__qualname__)
        if config_identity != ("receivers.nr.pusch", "NRPUSCHConfig") or not (
            _is_registered_class(config, *config_identity)
        ):
            raise TypeError("NR model inputs require the audited NRPUSCHConfig contract")
        config_fields = _field_names(config)
        if config_fields != _NR_RECEIVER_CONFIG_FIELDS:
            raise ValueError("NRPUSCHConfig field roster drifted across the model-input boundary")

    if identity == ("datasets.channel", "ChannelModelInputs"):
        _validated_array(
            value.observed_cfr,
            name="observed_cfr",
            value_kind="complex",
            ndim=3,
        )
    elif identity == ("datasets.analytical", "AnalyticalModelInputs"):
        frequencies_shape = _validated_array(
            value.frequencies_hz,
            name="frequencies_hz",
            value_kind="real",
            ndim=1,
        )
        response_shape = _validated_array(
            value.channel_frequency_response,
            name="channel_frequency_response",
            value_kind="complex",
            ndim=1,
        )
        if frequencies_shape != response_shape:
            raise ValueError("analytical frequency and CFR arrays must have the same shape")
    elif identity == ("datasets.observations", "FrequencyDomainModelInputs"):
        channel_shape = _validated_array(
            value.observed_cfr,
            name="observed_cfr",
            value_kind="complex",
            ndim=3,
        )
        offsets_shape = _validated_array(
            value.subcarrier_offsets_hz,
            name="subcarrier_offsets_hz",
            value_kind="real",
            ndim=1,
        )
        if offsets_shape[0] != channel_shape[-1]:
            raise ValueError("subcarrier_offsets_hz must match the observed CFR grid")
        offsets = np.asarray(value.subcarrier_offsets_hz)
        if np.any(np.diff(offsets) <= 0) or not np.any(offsets == 0.0):
            raise ValueError("subcarrier offsets must be increasing and include DC")
        _validated_real_scalar(value.noise_variance, name="noise_variance", nonnegative=True)
        _validated_real_scalar(value.snr_db, name="snr_db")
        _validated_real_scalar(value.carrier_hz, name="carrier_hz", positive=True)
    elif identity == ("datasets.nr", "NRModelInputs"):
        grid_shape = _validated_array(
            value.received_grid, name="received_grid", value_kind="complex", ndim=5
        )
        channel_shape = _validated_array(
            value.h_hat, name="h_hat", value_kind="complex", ndim=7
        )
        error_shape = _validated_array(
            value.error_variance,
            name="error_variance",
            value_kind="real",
            nonnegative=True,
        )
        pilot_shape = _validated_array(
            value.pilot_mask, name="pilot_mask", value_kind="boolean", ndim=4
        )
        if channel_shape[:3] != grid_shape[:3] or channel_shape[5] != grid_shape[3]:
            raise ValueError(
                "NR channel estimates must match received-grid batch/antenna/symbol axes"
            )
        if pilot_shape != channel_shape[3:]:
            raise ValueError("pilot_mask must match the channel-estimate resource axes")
        if len(error_shape) != 7 or any(
            variance_axis not in (1, estimate_axis)
            for variance_axis, estimate_axis in zip(error_shape, channel_shape, strict=True)
        ):
            raise ValueError("error_variance must be broadcastable to h_hat")
        if value.received_grid_mask is not None:
            mask_shape = _validated_array(
                value.received_grid_mask,
                name="received_grid_mask",
                value_kind="boolean",
                ndim=5,
            )
            if mask_shape != grid_shape:
                raise ValueError("received_grid_mask must match received_grid axes")
        indices = value.fft_bin_indices
        if not isinstance(indices, tuple) or any(
            isinstance(index, bool) or not isinstance(index, int) for index in indices
        ):
            raise TypeError("fft_bin_indices must be a tuple of integer FFT-bin indices")
        if (
            len(indices) != channel_shape[6]
            or len(set(indices)) != len(indices)
            or any(index < 0 or index >= grid_shape[4] for index in indices)
        ):
            raise ValueError(
                "fft_bin_indices must uniquely map each estimated subcarrier to the grid"
            )

    return value
