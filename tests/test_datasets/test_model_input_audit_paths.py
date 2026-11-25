"""Registered model-input contracts: roster, axis, and inference-boundary refusals."""

from __future__ import annotations

import builtins
import sys
from dataclasses import make_dataclass

import datasets.nr as nr_module
import numpy as np
import pytest
from datasets.model_input_audit import audit_model_input_payload
from datasets.nr import NRModelInputs
from datasets.observations import FrequencyDomainModelInputs
from evaluation.scoring.caez_forecast import CAEZForecastModelInputs
from receivers.nr import NRPUSCHConfig


def _nr_inputs(**changes):
    values = {
        "received_grid": np.ones((1, 1, 1, 1, 8), dtype=np.complex64),
        "h_hat": np.ones((1, 1, 1, 1, 1, 1, 8), dtype=np.complex64),
        "error_variance": np.ones((1, 1, 1, 1, 1, 1, 8), dtype=np.float32),
        "pilot_mask": np.ones((1, 1, 1, 8), dtype=bool),
        "fft_bin_indices": tuple(range(8)),
        "receiver_config": NRPUSCHConfig(),
        "received_grid_mask": None,
    }
    values.update(changes)
    return NRModelInputs(**values)


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"receiver_config": object()}, TypeError, "audited NRPUSCHConfig contract"),
        (
            {"h_hat": np.ones((1, 1, 2, 1, 1, 1, 8), dtype=np.complex64)},
            ValueError,
            "batch/antenna/symbol axes",
        ),
        ({"pilot_mask": np.ones((1, 1, 1, 7), dtype=bool)}, ValueError, "pilot_mask must match"),
        (
            {"error_variance": np.ones((1, 1, 1, 1, 1, 1, 7), dtype=np.float32)},
            ValueError,
            "broadcastable to h_hat",
        ),
        (
            {"received_grid_mask": np.ones((1, 1, 1, 1, 7), dtype=bool)},
            ValueError,
            "received_grid_mask must match",
        ),
        ({"fft_bin_indices": [0, 1, 2, 3, 4, 5, 6, 7]}, TypeError, "tuple of integer FFT-bin"),
        ({"fft_bin_indices": (0, 1, 2, 3, 4, 5, 6, True)}, TypeError, "tuple of integer FFT-bin"),
        ({"fft_bin_indices": (0, 1, 2, 3, 4, 5, 6, 8)}, ValueError, "uniquely map"),
        ({"fft_bin_indices": (0, 1, 2, 3, 4, 5, 6, 6)}, ValueError, "uniquely map"),
    ],
)
def test_nr_model_inputs_reject_invalid_receiver_and_axis_contracts(changes, error, message):
    with pytest.raises(error, match=message):
        audit_model_input_payload(_nr_inputs(**changes))


def test_nr_model_input_contract_roster_drift_is_rejected(monkeypatch):
    fake = make_dataclass(
        "NRModelInputs",
        [("unexpected", object)],
    )
    fake.__module__ = "datasets.nr"
    monkeypatch.setattr(nr_module, "NRModelInputs", fake)
    with pytest.raises(ValueError, match="field roster drift"):
        audit_model_input_payload(fake(unexpected=object()))


def test_caez_forecast_inputs_require_a_nonempty_finite_real_matrix():
    valid = CAEZForecastModelInputs(np.ones((2, 3), dtype=np.float64))
    assert audit_model_input_payload(valid) is valid
    with pytest.raises(ValueError, match="nonempty dimensions"):
        audit_model_input_payload(CAEZForecastModelInputs(np.empty((0, 3))))
    with pytest.raises(TypeError, match="real-valued"):
        audit_model_input_payload(CAEZForecastModelInputs(np.ones((2, 3), dtype=np.complex64)))


def test_nr_receiver_configuration_roster_drift_is_rejected(monkeypatch):
    module = sys.modules["receivers.nr.pusch"]
    fake_config = make_dataclass(
        "NRPUSCHConfig",
        [("unexpected", object)],
    )
    fake_config.__module__ = "receivers.nr.pusch"
    monkeypatch.setattr(module, "NRPUSCHConfig", fake_config)
    with pytest.raises(ValueError, match="NRPUSCHConfig field roster drifted"):
        audit_model_input_payload(_nr_inputs(receiver_config=fake_config(unexpected=object())))


def test_optional_torch_probe_fails_closed_when_torch_import_is_unavailable(monkeypatch):
    fake_tensor = type("Tensor", (), {"__module__": "torch.fake"})()
    original_import = builtins.__import__

    def import_without_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch is not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_torch)
    with pytest.raises(TypeError, match="NumPy array or dense PyTorch tensor"):
        from datasets.model_input_audit import _validated_array

        _validated_array(fake_tensor, name="input", value_kind="real")


@pytest.mark.parametrize(
    "offsets",
    [np.asarray([0.0, -1.0, 1.0]), np.asarray([-2.0, -1.0, 1.0])],
    ids=["non-increasing", "missing-dc"],
)
def test_frequency_domain_inputs_require_monotone_offsets_including_dc(offsets):
    values = FrequencyDomainModelInputs(
        observed_cfr=np.ones((1, 1, offsets.size), dtype=np.complex64),
        noise_variance=0.1,
        snr_db=10.0,
        carrier_hz=3.5e9,
        subcarrier_offsets_hz=np.arange(-1.0, 2.0),
    )
    object.__setattr__(values, "subcarrier_offsets_hz", offsets)
    with pytest.raises(ValueError, match="increasing and include DC"):
        audit_model_input_payload(values)


def test_nr_received_grid_mask_must_match_all_grid_axes():
    assert audit_model_input_payload(
        _nr_inputs(received_grid_mask=np.ones((1, 1, 1, 1, 8), dtype=bool))
    )
    with pytest.raises(ValueError, match="received_grid_mask must match"):
        audit_model_input_payload(
            _nr_inputs(received_grid_mask=np.ones((2, 1, 1, 1, 8), dtype=bool))
        )
