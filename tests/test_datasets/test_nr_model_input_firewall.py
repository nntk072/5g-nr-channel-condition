"""Adversarial tests for receiver-only NR model inputs."""

from dataclasses import fields

import numpy as np
from datasets.nr import NRChannelRecord, NRObservation, TargetDefinition
from receivers.nr import NRChannelEstimate, NRPUSCHConfig


def test_oracle_csi_targets_and_metadata_never_enter_nr_model_inputs():
    received_grid = np.ones((1, 1, 1, 14, 12), dtype=np.complex64)
    estimate_values = np.full((1, 1, 1, 1, 1, 14, 12), 0.5 + 0.25j, dtype=np.complex64)
    estimate = NRChannelEstimate(
        h_hat=estimate_values,
        error_variance=np.full(estimate_values.shape, 0.1, dtype=np.float32),
        pilot_mask=np.ones((1, 1, 14, 12), dtype=np.bool_),
        metadata={"receiver": "PUSCHLSChannelEstimator", "oracle_csi": "must stay private"},
    )
    observation = NRObservation(
        sample_id="private-sample-id",
        received_grid=received_grid,
        channel_estimate=estimate,
        fft_bin_indices=tuple(range(12)),
        receiver_config=NRPUSCHConfig(n_size_bwp=1, subcarrier_spacing_khz=30),
    )
    oracle_csi = np.full(estimate_values.shape, 1e6 + 3e6j, dtype=np.complex64)
    record = NRChannelRecord(
        observation=observation,
        targets={"future_oracle_csi": oracle_csi},
        target_masks={"future_oracle_csi": np.ones(oracle_csi.shape, dtype=np.bool_)},
        provenance={"future_oracle_csi": oracle_csi, "scene_id": "private-scene"},
        target_definitions={
            "future_oracle_csi": TargetDefinition(
                semantic="future perfect CSI", units="complex channel gain", source="simulator"
            )
        },
    )

    model_inputs = record.observation.model_inputs()
    exposed_fields = {field.name for field in fields(model_inputs)}

    assert exposed_fields == {
        "received_grid",
        "h_hat",
        "error_variance",
        "pilot_mask",
        "fft_bin_indices",
        "receiver_config",
        "received_grid_mask",
    }
    np.testing.assert_array_equal(model_inputs.h_hat, estimate_values)
    assert not np.shares_memory(model_inputs.h_hat, record.targets["future_oracle_csi"])
    assert not hasattr(model_inputs, "sample_id")
    assert not hasattr(model_inputs, "target_masks")
    assert not hasattr(model_inputs, "target_definitions")
    assert not hasattr(model_inputs, "provenance")
