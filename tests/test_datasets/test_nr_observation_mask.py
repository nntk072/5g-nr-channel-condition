"""Inference-visible NR resource-grid observation-mask contracts."""

from __future__ import annotations

import hashlib
import io

import numpy as np
import pytest
from datasets.nr import NRChannelRecord, NRObservation, TargetDefinition
from datasets.nr_store import NRStoredRecord, read_nr_record, write_nr_record
from receivers.nr import NRChannelEstimate, NRPUSCHConfig


def _observation(*, received_grid_mask: np.ndarray | None = None) -> NRObservation:
    received_grid = np.ones((1, 1, 1, 14, 12), dtype=np.complex64)
    estimate = NRChannelEstimate(
        h_hat=np.ones((1, 1, 1, 1, 1, 14, 12), dtype=np.complex64),
        error_variance=np.zeros((1, 1, 1, 1, 1, 14, 12), dtype=np.float32),
        pilot_mask=np.ones((1, 1, 14, 12), dtype=bool),
        metadata={"receiver": "test"},
    )
    return NRObservation(
        sample_id="nr-mask-test",
        received_grid=received_grid,
        channel_estimate=estimate,
        fft_bin_indices=tuple(range(12)),
        receiver_config=NRPUSCHConfig(n_size_bwp=1, subcarrier_spacing_khz=30),
        received_grid_mask=received_grid_mask,
    )


def test_observation_defaults_to_all_received_and_exposes_mask_to_model_inputs():
    observation = _observation()

    assert observation.received_grid_mask.shape == observation.received_grid.shape
    assert observation.received_grid_mask.all()
    np.testing.assert_array_equal(
        observation.model_inputs().received_grid_mask, observation.received_grid_mask
    )


def test_observation_preserves_explicit_resource_element_missingness(tmp_path):
    mask = np.ones((1, 1, 1, 14, 12), dtype=bool)
    mask[..., 3, 7] = False
    observation = _observation(received_grid_mask=mask)
    record = NRChannelRecord(
        observation=observation,
        targets={"target": np.asarray([1.0], dtype=np.float32)},
        target_masks={"target": np.asarray([True])},
        provenance={"source": "test"},
        target_definitions={
            "target": TargetDefinition.versioned(
                semantic="observation mask test target",
                units="fixture value",
                source="mask test fixture",
                reference="fixed test scalar",
                aggregation_rule="one scalar per record",
                definition_version="tests.nr-mask.target.v1",
            )
        },
    )

    assert not observation.model_inputs().received_grid_mask[..., 3, 7].item()
    restored = read_nr_record(tmp_path, write_nr_record(tmp_path, record))
    np.testing.assert_array_equal(restored.observation.received_grid_mask, mask)
    np.testing.assert_array_equal(restored.observation.model_inputs().received_grid_mask, mask)

    current_manifest = write_nr_record(tmp_path, record)
    with np.load(tmp_path / current_manifest.relative_path, allow_pickle=False) as archive:
        legacy_buffer = io.BytesIO()
        np.savez_compressed(
            legacy_buffer,
            **{name: archive[name] for name in archive.files if name != "received_grid_mask"},
        )
    legacy_payload = legacy_buffer.getvalue()
    legacy_path = tmp_path / "records" / "legacy-v1.npz"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_bytes(legacy_payload)
    legacy_manifest = NRStoredRecord(
        sample_id=observation.sample_id,
        content_sha256=hashlib.sha256(legacy_payload).hexdigest(),
        relative_path="records/legacy-v1.npz",
    )
    legacy = read_nr_record(tmp_path, legacy_manifest)
    assert legacy.observation.received_grid_mask.all()


@pytest.mark.parametrize(
    ("mask", "error", "message"),
    [
        (np.ones((1, 1, 14, 12), dtype=bool), ValueError, "match the received_grid axes"),
        (np.ones((1, 1, 1, 14, 12), dtype=np.float32), TypeError, "boolean dtype"),
    ],
)
def test_observation_rejects_invalid_received_grid_mask(mask, error, message):
    with pytest.raises(error, match=message):
        _observation(received_grid_mask=mask)
