import numpy as np
import pytest
from datasets.channel import ChannelDataset
from datasets.synthetic import debug_multipath
from propagation.physics import (
    add_estimation_noise,
    frequency_response,
    observable_features,
    subcarrier_frequencies,
)


def test_debug_has_both_classes():
    d = debug_multipath(n_scenes=3, links_per_scene=12)
    assert set(d.y) == {0, 1}
    assert len(d.y) == 36


def test_debug_los_contains_reflections():
    d = debug_multipath(n_scenes=2, links_per_scene=6)
    assert all(m["reflected_paths"] >= 5 for m in d.metadata)
    assert all(m["direct_path"] == bool(y) for m, y in zip(d.metadata, d.y))


def test_debug_deterministic():
    a = debug_multipath(seed=31)
    b = debug_multipath(seed=31)
    assert np.array_equal(a.h, b.h)
    assert np.array_equal(a.y, b.y)


def test_debug_seeds_change_channels():
    a = debug_multipath(seed=1)
    b = debug_multipath(seed=2)
    assert not np.array_equal(a.h, b.h)


def test_debug_invalid_counts():
    with pytest.raises(ValueError):
        debug_multipath(n_scenes=1)
    with pytest.raises(ValueError):
        debug_multipath(links_per_scene=3)


def test_storage_roundtrip(tmp_path):
    d = debug_multipath(n_scenes=3, links_per_scene=10)
    path = tmp_path / "set.npz"
    d.save(path)
    loaded = ChannelDataset.load(path)
    assert np.allclose(loaded.h, d.h)
    assert np.array_equal(loaded.y, d.y)
    assert loaded.metadata == d.metadata


def test_dataset_rejects_nans():
    d = debug_multipath(n_scenes=3, links_per_scene=10)
    h = d.h.copy()
    h[0, 0, 0] = np.nan + 1j * np.nan
    with pytest.raises(ValueError, match="finite"):
        ChannelDataset(h, d.y, d.group, d.domain, d.source, d.metadata)


def test_concat_fails_on_mismatch():
    a = debug_multipath(n_subcarriers=64)
    b = debug_multipath(n_subcarriers=32)
    with pytest.raises(ValueError):
        ChannelDataset.concatenate([a, b])


def test_select_preserves_metadata():
    d = debug_multipath()
    x = d.select(np.array([7, 2, 1]))
    assert x.metadata[0] == d.metadata[7]
    assert len(x.y) == 3


def test_no_model_metadata_in_features():
    d = debug_multipath()
    f1 = observable_features(d.h)
    # Deliberately change every label, group and metadata field; features unchanged.
    changed = ChannelDataset(
        d.h,
        1 - d.y,
        np.array(["x" + str(i) for i in range(len(d.y))]),
        np.repeat("arbitrary", len(d.y)),
        d.source,
        tuple({"oracle": int(i)} for i in range(len(d.y))),
    )
    assert np.array_equal(f1, observable_features(changed.h))


def test_feature_gain_invariance():
    d = debug_multipath()
    assert np.allclose(observable_features(d.h), observable_features(d.h * 7j), atol=1e-6)


def test_feature_phase_opt_in():
    d = debug_multipath()
    assert observable_features(d.h, include_phase=True).shape[1] > observable_features(d.h).shape[1]


def test_channel_sum_known():
    f = subcarrier_frequencies(16, 30e3)
    a = np.asarray([1 + 0j, 0.3 + 0.5j])
    tau = np.asarray([0.0, 1e-6])
    result = frequency_response(a, tau, f)
    assert np.allclose(result[8], sum(a))


def test_awgn_finite():
    h = np.ones((8, 1, 32), dtype=np.complex64)
    out = add_estimation_noise(h, 10, np.random.default_rng(1))
    assert out.shape == h.shape and np.all(np.isfinite(out))
    assert not np.array_equal(out, h)
