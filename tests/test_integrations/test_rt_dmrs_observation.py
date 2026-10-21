"""Pure input and frequency-grid tests for the RT-to-DMRS observation adapter."""

from __future__ import annotations

import numpy as np
import pytest
from datasets.channel import ChannelDataset
from integrations.sionna.rt_dmrs import _label_semantics, resample_centered_cfr


def _labelled_dataset(*, labels, declared="unspecified_legacy"):
    count = len(labels)
    return ChannelDataset(
        h=np.ones((count, 1, 16), dtype=np.complex64),
        y=np.asarray([index % 2 for index in range(count)], dtype=np.uint8),
        group=np.asarray([f"g{index}" for index in range(count)]),
        domain=np.asarray(["scene"] * count),
        source=np.asarray(["sionna_rt"] * count),
        metadata=tuple({"geometry_label": label} for label in labels),
        label_semantics=declared,
    )


def test_rt_dmrs_label_semantics_resolves_legacy_metadata_and_declared_value():
    legacy = _labelled_dataset(
        labels=("GEOMETRIC_DIRECT_PATH_ORACLE", "GEOMETRIC_DIRECT_PATH_ORACLE")
    )
    declared = _labelled_dataset(
        labels=("GEOMETRIC_DIRECT_PATH_ORACLE", "GEOMETRIC_DIRECT_PATH_ORACLE"),
        declared="geometric_direct_path_oracle",
    )

    assert _label_semantics(legacy) == "geometric_direct_path_oracle"
    assert _label_semantics(declared) == "geometric_direct_path_oracle"


def test_rt_dmrs_label_semantics_rejects_mixed_or_unknown_targets():
    mixed = _labelled_dataset(labels=("GEOMETRIC_DIRECT_PATH_ORACLE", "PROFILE_PROXY"))
    unknown = _labelled_dataset(labels=("UNKNOWN", "UNKNOWN"))

    with pytest.raises(ValueError, match="consistent label semantic"):
        _label_semantics(mixed)
    with pytest.raises(ValueError, match="consistent label semantic"):
        _label_semantics(unknown)


def test_centered_cfr_resampling_preserves_overlapping_subcarriers():
    source_frequencies = (np.arange(128) - 64) * 30_000.0
    channels = np.stack(
        [
            (1.0 + source_frequencies / 1e6) + 1j * (0.5 - source_frequencies / 2e6),
            (-0.25 + source_frequencies / 3e6) + 1j * (1.0 + source_frequencies / 4e6),
        ]
    )[:, None, :].astype(np.complex64)

    actual = resample_centered_cfr(
        channels,
        source_spacing_hz=30_000.0,
        target_subcarriers=72,
        target_spacing_hz=30_000.0,
    )

    np.testing.assert_array_equal(actual, channels[:, 0, 28:100])


@pytest.mark.parametrize(
    ("channels", "overrides", "message"),
    [
        (np.ones((2, 2, 128), dtype=np.complex64), {}, "samples, 1"),
        (np.ones((2, 1, 128), dtype=np.float64), {}, "complex"),
        (np.ones((2, 1, 128), dtype=np.complex64), {"source_spacing_hz": 0.0}, "source_spacing_hz"),
        (
            np.ones((2, 1, 128), dtype=np.complex64),
            {"target_subcarriers": 128, "target_spacing_hz": 60_000.0},
            "inside the source CFR bandwidth",
        ),
    ],
)
def test_centered_cfr_resampling_rejects_invalid_grids(channels, overrides, message):
    arguments = {
        "source_spacing_hz": 30_000.0,
        "target_subcarriers": 72,
        "target_spacing_hz": 30_000.0,
    }
    arguments.update(overrides)
    with pytest.raises(ValueError, match=message):
        resample_centered_cfr(channels, **arguments)
