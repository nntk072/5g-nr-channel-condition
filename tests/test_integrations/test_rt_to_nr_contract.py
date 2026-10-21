from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from integrations.sionna.rt_to_nr import paths_to_nr_ofdm_channel  # noqa: E402


class FakePaths:
    def __init__(self):
        self.kwargs = None

    def cfr(self, **kwargs):
        self.kwargs = kwargs
        times = kwargs["num_time_steps"]
        frequencies = len(kwargs["frequencies"])
        return torch.ones((1, 2, 1, 1, times, frequencies), dtype=torch.complex64)


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
