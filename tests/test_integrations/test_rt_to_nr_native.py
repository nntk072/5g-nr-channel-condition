"""Native CPU/LLVM physical check for the Sionna RT-to-NR frequency bridge."""

import numpy as np
import pytest
from scipy.constants import speed_of_light

pytest.importorskip("mitsuba")


@pytest.mark.native_sionna
def test_rt_absolute_delay_cfr_maps_to_nr_baseband_bins():
    import mitsuba as mi

    mi.set_variant("llvm_ad_mono_polarized")
    rt = pytest.importorskip("sionna.rt")
    from integrations.sionna.rt_to_nr import paths_to_nr_ofdm_channel
    from sionna.phy.ofdm import ResourceGrid

    distance_m = 10.0
    scene = rt.load_scene()
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.add(rt.Transmitter("tx", [0.0, 0.0, 0.0]))
    scene.add(rt.Receiver("rx", [distance_m, 0.0, 0.0]))
    scene.get("tx").look_at(scene.get("rx"))
    scene.get("rx").look_at(scene.get("tx"))
    paths = rt.PathSolver(deterministic=True)(
        scene,
        max_depth=0,
        los=True,
        specular_reflection=False,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
    )
    grid = ResourceGrid(
        num_ofdm_symbols=14,
        fft_size=24,
        subcarrier_spacing=30e3,
        cyclic_prefix_length=0,
        device="cpu",
    )

    result = paths_to_nr_ofdm_channel(paths, grid)
    cir, delays_s = paths.cir(normalize_delays=False, out_type="numpy")
    frequencies_hz = result.frequencies_hz.numpy()
    expected = np.sum(
        cir[0, 0, 0, 0, :, 0, None]
        * np.exp(-2j * np.pi * delays_s[0, 0, :, None] * frequencies_hz[None, :]),
        axis=0,
    )

    assert result.channel.shape == (1, 1, 1, 1, 1, 14, 24)
    np.testing.assert_allclose(
        result.channel[0, 0, 0, 0, 0, 0].numpy(), expected, rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(delays_s[0, 0, 0], distance_m / speed_of_light, atol=1e-12)
    assert result.metadata["normalize_delays"] is False
    assert result.metadata["frequency_reference"] == "NR baseband subcarrier offsets"
    assert np.allclose(
        result.channel[0, 0, 0, 0, 0, 0].numpy(), result.channel[0, 0, 0, 0, 0, 13].numpy()
    )
