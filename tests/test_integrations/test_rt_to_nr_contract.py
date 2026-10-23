from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from integrations.sionna.nr_timing import nr_normal_cp_symbol_start_times  # noqa: E402
from integrations.sionna.rt_to_nr import (  # noqa: E402
    paths_to_nr_ofdm_channel,
    paths_to_nr_ofdm_channel_at_times,
)


class FakePaths:
    def __init__(self):
        self.kwargs = None

    def cfr(self, **kwargs):
        self.kwargs = kwargs
        times = kwargs["num_time_steps"]
        frequencies = len(kwargs["frequencies"])
        return torch.ones((1, 2, 1, 1, times, frequencies), dtype=torch.complex64)


class FakeTimedPaths:
    def __init__(self):
        self.doppler = torch.tensor([[[100.0, float("nan")]]])
        self.valid = torch.tensor([[[True, False]]])

    def cir(self, **kwargs):
        assert kwargs["normalize_delays"] is False
        coefficients = np.asarray([1.0 + 0.0j, 2.0 + 1.0j], dtype=np.complex64).reshape(
            1, 1, 1, 1, 2, 1
        )
        delays = np.asarray([0.0, 1e-6], dtype=np.float64).reshape(1, 1, 1, 1, 2)
        return coefficients, delays


def test_rt_paths_map_to_nr_axes_and_repeat_static_snapshot():
    paths = FakePaths()
    grid = SimpleNamespace(fft_size=8, num_ofdm_symbols=14, subcarrier_spacing=30e3)

    result = paths_to_nr_ofdm_channel(
        paths,
        grid,
        batch_size=3,
        sampling_frequency_hz=15e3,
    )

    assert result.channel.shape == (3, 1, 2, 1, 1, 14, 8)
    assert result.channel.is_complex()
    assert torch.all(result.channel == 1)
    assert torch.equal(result.frequencies_hz, torch.arange(-4, 4) * 30e3)
    assert paths.kwargs["normalize_delays"] is False
    assert paths.kwargs["normalize"] is False
    assert result.metadata["static_snapshot_repeated"] is True


def test_rt_paths_require_static_or_symbol_aligned_time_samples():
    with pytest.raises(ValueError, match="match the OFDM symbol count"):
        paths_to_nr_ofdm_channel(
            FakePaths(),
            SimpleNamespace(fft_size=8, num_ofdm_symbols=14, subcarrier_spacing=30e3),
            num_time_steps=2,
        )


def test_rt_paths_reject_invalid_frequency_batch_and_grid_values():
    grid = SimpleNamespace(fft_size=8, num_ofdm_symbols=14, subcarrier_spacing=30e3)
    with pytest.raises(ValueError, match="sampling_frequency_hz"):
        paths_to_nr_ofdm_channel(FakePaths(), grid, sampling_frequency_hz=0)
    with pytest.raises(TypeError, match="batch_size"):
        paths_to_nr_ofdm_channel(FakePaths(), grid, batch_size=True)
    with pytest.raises(ValueError, match="resource_grid"):
        paths_to_nr_ofdm_channel(
            FakePaths(),
            SimpleNamespace(fft_size=0, num_ofdm_symbols=14, subcarrier_spacing=30e3),
        )


def test_rt_paths_keep_dynamic_ofdm_symbol_axis():
    result = paths_to_nr_ofdm_channel(
        FakePaths(),
        SimpleNamespace(fft_size=8, num_ofdm_symbols=4, subcarrier_spacing=15e3),
        num_time_steps=4,
        sampling_frequency_hz=14e3,
    )
    assert result.channel.shape == (1, 1, 2, 1, 1, 4, 8)
    assert result.metadata["static_snapshot_repeated"] is False


def test_rt_paths_evaluate_at_explicit_times_with_invalid_paths_masked():
    result = paths_to_nr_ofdm_channel_at_times(
        FakeTimedPaths(),
        SimpleNamespace(fft_size=4, num_ofdm_symbols=2, subcarrier_spacing=15e3),
        (0.0, 1e-3),
    )

    expected = np.exp(2j * np.pi * 100.0 * np.asarray([0.0, 1e-3]))
    assert result.channel.shape == (1, 1, 1, 1, 1, 2, 4)
    np.testing.assert_allclose(
        result.channel.numpy()[0, 0, 0, 0, 0, :, :],
        np.broadcast_to(expected[:, None], (2, 4)),
    )
    assert result.metadata["sample_times_s"] == [0.0, 1e-3]
    assert result.metadata["normalize_delays"] is False


def test_rt_doppler_uses_cp_aware_nr_symbol_start_times_in_seconds():
    slot_index = 0
    times_s = nr_normal_cp_symbol_start_times(
        30_000.0,
        14,
        slot_index_in_subframe=slot_index,
    )
    doppler_hz = 2_400.0
    paths = FakeTimedPaths()
    paths.doppler = torch.tensor([[[doppler_hz, float("nan")]]])
    grid = SimpleNamespace(fft_size=8, num_ofdm_symbols=14, subcarrier_spacing=30e3)

    result = paths_to_nr_ofdm_channel_at_times(paths, grid, times_s)

    center_bin = grid.fft_size // 2
    observed = result.channel.numpy()[0, 0, 0, 0, 0, :, center_bin]
    expected = np.exp(2j * np.pi * doppler_hz * np.asarray(times_s))
    assert not np.allclose(np.diff(times_s), 1.0 / 30_000.0, rtol=0.0, atol=1e-15)
    assert result.metadata["sample_times_s"] == list(times_s)
    np.testing.assert_allclose(observed, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(
    ("times", "message"),
    [
        ((), "non-empty finite vector"),
        ((0.0, float("nan")), "non-empty finite vector"),
        ((0.0, 0.0), "strictly increasing"),
        ((0.0,), "length must match"),
    ],
)
def test_rt_paths_at_times_reject_invalid_sampling_times(times, message):
    with pytest.raises(ValueError, match=message):
        paths_to_nr_ofdm_channel_at_times(
            FakeTimedPaths(),
            SimpleNamespace(fft_size=4, num_ofdm_symbols=2, subcarrier_spacing=15e3),
            times,
        )
