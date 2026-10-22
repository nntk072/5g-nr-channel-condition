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
    from integrations.sionna.adapter import GridConfig, rt_cfr
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
    odd_grid = ResourceGrid(
        num_ofdm_symbols=1,
        fft_size=25,
        subcarrier_spacing=30e3,
        cyclic_prefix_length=0,
        device="cpu",
    )
    odd_result = paths_to_nr_ofdm_channel(paths, odd_grid)
    odd_frequencies = (np.arange(25) - 12) * 30e3
    odd_reference = np.sum(
        cir[0, 0, 0, 0, :, 0, None]
        * np.exp(-2j * np.pi * delays_s[0, 0, :, None] * odd_frequencies[None, :]),
        axis=0,
    )
    np.testing.assert_allclose(odd_result.frequencies_hz.numpy(), odd_frequencies, rtol=0, atol=0)
    np.testing.assert_allclose(
        odd_result.channel[0, 0, 0, 0, 0, 0].numpy(),
        odd_reference,
        rtol=1e-5,
        atol=1e-8,
    )
    np.testing.assert_allclose(delays_s[0, 0, 0], distance_m / speed_of_light, atol=1e-12)
    assert result.metadata["normalize_delays"] is False
    assert result.metadata["frequency_reference"] == "NR baseband subcarrier offsets"
    assert np.allclose(
        result.channel[0, 0, 0, 0, 0, 0].numpy(), result.channel[0, 0, 0, 0, 0, 13].numpy()
    )

    # The legacy dataset adapter must use the same absolute-delay convention
    # as the canonical NR bridge, rather than silently removing the ToA slope.
    adapter_channel = rt_cfr(paths, GridConfig(n_subcarriers=24, spacing_hz=30e3))
    adapter_reference = paths.cfr(
        frequencies=(np.arange(24) - 12) * 30e3,
        num_time_steps=1,
        normalize_delays=False,
        normalize=False,
        out_type="numpy",
    ).reshape(1, 1, 24)
    np.testing.assert_allclose(adapter_channel, adapter_reference, rtol=1e-5, atol=1e-8)
    path_gain = paths.a
    path_gain = np.asarray(path_gain[0].numpy()) + 1j * np.asarray(path_gain[1].numpy())
    path_delays = np.asarray(paths.tau.numpy(), dtype=float)
    if path_delays.ndim == 3:
        path_delays = path_delays[:, None, :, None, :]
    frequency_offsets = (np.arange(24) - 12) * 30e3
    carrier_frequency_hz = float(np.asarray(scene.frequency.numpy()).item())
    baseband_gains = path_gain * np.exp(-2j * np.pi * carrier_frequency_hz * path_delays)
    manual_cfr = np.sum(
        baseband_gains[..., :, None]
        * np.exp(-2j * np.pi * path_delays[..., :, None] * frequency_offsets),
        axis=-2,
    ).reshape(1, 1, 24)
    np.testing.assert_allclose(adapter_channel, manual_cfr, rtol=1e-5, atol=1e-8)
    normalized_reference = paths.cfr(
        frequencies=(np.arange(24) - 12) * 30e3,
        num_time_steps=1,
        normalize_delays=True,
        normalize=False,
        out_type="numpy",
    ).reshape(1, 1, 24)
    assert np.max(np.abs(adapter_channel - normalized_reference)) > 1e-3


@pytest.mark.native_sionna
def test_rt_direct_and_reflected_paths_sum_coherently_on_nr_grid():
    import mitsuba as mi

    mi.set_variant("llvm_ad_mono_polarized")
    rt = pytest.importorskip("sionna.rt")
    from integrations.sionna.adapter import (
        GridConfig,
        direct_indirect_diagnostics,
        direct_path_mask,
        rt_cfr,
    )
    from integrations.sionna.rt_to_nr import paths_to_nr_ofdm_channel
    from sionna.phy.ofdm import ResourceGrid

    scene = rt.load_scene(rt.scene.simple_reflector, merge_shapes=False)
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.add(rt.Transmitter("tx", [-10.0, 0.0, 10.0]))
    scene.add(rt.Receiver("rx", [10.0, 0.0, 10.0]))
    scene.get("tx").look_at([0.0, 0.0, 0.0])
    scene.get("rx").look_at([0.0, 0.0, 0.0])
    paths = rt.PathSolver(deterministic=True)(
        scene,
        max_depth=1,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
        seed=1,
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

    assert np.count_nonzero(paths.valid.numpy()) == 2
    np.testing.assert_allclose(
        result.channel[0, 0, 0, 0, 0, 0].numpy(), expected, rtol=1e-5, atol=1e-8
    )
    assert np.ptp(np.abs(result.channel[0, 0, 0, 0, 0, 0].numpy())) > 0

    # Partition coefficients from this exact Paths object; coherent
    # reconstruction must conserve the frequency response at one delay origin.
    rt_grid = GridConfig(n_subcarriers=24, spacing_hz=30e3)
    valid = np.asarray(paths.valid.numpy(), dtype=bool)
    direct_mask = direct_path_mask(paths)
    indirect_mask = valid & ~direct_mask
    assert direct_mask.sum() == 1 and indirect_mask.sum() == 1
    full_cfr = rt_cfr(paths, rt_grid)
    direct_cfr = rt_cfr(paths, rt_grid, path_mask=direct_mask)
    indirect_cfr = rt_cfr(paths, rt_grid, path_mask=indirect_mask)
    np.testing.assert_allclose(full_cfr, direct_cfr + indirect_cfr, rtol=1e-5, atol=1e-8)
    path_gain = paths.a
    path_gain = np.asarray(path_gain[0].numpy()) + 1j * np.asarray(path_gain[1].numpy())
    path_delays = np.asarray(paths.tau.numpy(), dtype=float)
    if path_delays.ndim == 3:
        path_delays = path_delays[:, None, :, None, :]
    carrier_frequency_hz = float(np.asarray(scene.frequency.numpy()).item())
    frequency_offsets = (np.arange(24) - 12) * 30e3
    baseband_gains = path_gain * np.exp(-2j * np.pi * carrier_frequency_hz * path_delays)
    manual_cfr = np.sum(
        baseband_gains[..., :, None]
        * np.exp(-2j * np.pi * path_delays[..., :, None] * frequency_offsets),
        axis=-2,
    ).reshape(1, 1, 24)
    np.testing.assert_allclose(full_cfr, manual_cfr, rtol=1e-4, atol=3e-8)
    diagnostic = direct_indirect_diagnostics(full_cfr, direct_cfr)
    np.testing.assert_allclose(
        diagnostic["residual_indirect_power"], np.mean(np.abs(indirect_cfr) ** 2), rtol=1e-5
    )


@pytest.mark.native_sionna
def test_same_paths_direct_indirect_partition_preserves_explicit_mimo_axes():
    import mitsuba as mi

    mi.set_variant("llvm_ad_mono_polarized")
    rt = pytest.importorskip("sionna.rt")
    from integrations.sionna.adapter import GridConfig, direct_path_mask, rt_cfr

    scene = rt.load_scene(rt.scene.simple_reflector, merge_shapes=False)
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=2, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=3, pattern="iso", polarization="V")
    scene.add(rt.Transmitter("tx", [-10.0, 0.0, 10.0]))
    scene.add(rt.Receiver("rx", [10.0, 0.0, 10.0]))
    scene.get("tx").look_at([0.0, 0.0, 0.0])
    scene.get("rx").look_at([0.0, 0.0, 0.0])
    paths = rt.PathSolver(deterministic=True)(
        scene,
        max_depth=1,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
        synthetic_array=False,
        seed=1,
    )

    valid = np.asarray(paths.valid.numpy(), dtype=bool)
    direct_mask = direct_path_mask(paths)
    indirect_mask = valid & ~direct_mask
    assert paths.a[0].shape[:4] == (1, 3, 1, 2)
    assert direct_mask.shape == valid.shape
    assert direct_mask.any() and indirect_mask.any()

    grid = GridConfig(n_subcarriers=24, spacing_hz=30e3)
    full = rt_cfr(paths, grid)
    direct = rt_cfr(paths, grid, path_mask=direct_mask)
    indirect = rt_cfr(paths, grid, path_mask=indirect_mask)
    assert full.shape == (1, 6, 24)
    np.testing.assert_allclose(full, direct + indirect, rtol=1e-5, atol=1e-8)


@pytest.mark.native_sionna
def test_rt_reflected_path_depends_on_surface_radio_material():
    """Changing a reflector's material changes paths and the observed NR CFR."""
    import mitsuba as mi

    mi.set_variant("llvm_ad_mono_polarized")
    rt = pytest.importorskip("sionna.rt")
    from integrations.sionna.rt_to_nr import paths_to_nr_ofdm_channel
    from sionna.phy.ofdm import ResourceGrid

    grid = ResourceGrid(
        num_ofdm_symbols=14,
        fft_size=24,
        subcarrier_spacing=30e3,
        cyclic_prefix_length=0,
        device="cpu",
    )
    channels = {}
    path_counts = {}
    for relative_permittivity in (1.0, 4.0):
        scene = rt.load_scene(rt.scene.simple_reflector, merge_shapes=False)
        scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
        scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
        scene.add(rt.Transmitter("tx", [-10.0, 0.0, 10.0]))
        scene.add(rt.Receiver("rx", [10.0, 0.0, 10.0]))
        scene.get("tx").look_at([0.0, 0.0, 0.0])
        scene.get("rx").look_at([0.0, 0.0, 0.0])
        scene.get("reflector").radio_material = rt.RadioMaterial(
            name=f"material-eps-{relative_permittivity:g}",
            relative_permittivity=relative_permittivity,
            conductivity=0.0,
        )
        paths = rt.PathSolver(deterministic=True)(
            scene,
            max_depth=1,
            los=True,
            specular_reflection=True,
            diffuse_reflection=False,
            refraction=False,
            diffraction=False,
            seed=1,
        )
        path_counts[relative_permittivity] = int(np.count_nonzero(paths.valid.numpy()))
        channels[relative_permittivity] = paths_to_nr_ofdm_channel(paths, grid).channel.numpy()

    # With epsilon_r=1 and zero conductivity the boundary is electromagnetically
    # matched; epsilon_r=4 produces a reflected path in this fixed geometry.
    assert path_counts == {1.0: 1, 4.0: 2}
    assert np.mean(np.abs(channels[4.0] - channels[1.0])) > 1e-6


@pytest.mark.native_sionna
def test_rt_cpu_box_scene_generates_balanced_geometry_labeled_records():
    import mitsuba as mi

    mi.set_variant("llvm_ad_mono_polarized")
    pytest.importorskip("sionna.rt")
    from integrations.sionna.adapter import GridConfig, RayScene, generate_raytraced

    data = generate_raytraced(
        RayScene(
            name="box_one_screen",
            tx_position=(-2.0, 0.0, 2.5),
            rx_center=(2.0, 0.0, 2.5),
            radius_m=4.0,
            block_size_m=10.0,
            max_depth=1,
            samples_per_src=20,
            per_class=2,
            max_attempts=30,
            require_los_multipath=False,
        ),
        grid=GridConfig(n_subcarriers=16),
        seed=42,
    )

    assert data.h.shape == (4, 1, 16)
    np.testing.assert_array_equal(np.bincount(data.y), [2, 2])
    assert set(data.source) == {"sionna_rt"}
    assert set(data.domain) == {"rt_box_one_screen"}
    assert all(row["geometry_label"] == "GEOMETRIC_DIRECT_PATH_ORACLE" for row in data.metadata)
    assert all(row["valid_paths_full"] > 0 for row in data.metadata)
    assert all(row["mitsuba_variant"] == "llvm_ad_mono_polarized" for row in data.metadata)
    assert all(row["scene_asset_manifest_complete"] for row in data.metadata)
    assert all(len(row["scene_asset_set_sha256"]) == 64 for row in data.metadata)
    assert all(row["scene_config"]["samples_per_src"] == 20 for row in data.metadata)
    assert all(row["grid_config"]["n_subcarriers"] == 16 for row in data.metadata)
    assert all(row["solver_config"]["deterministic"] for row in data.metadata)
    assert all(
        row["valid_paths_direct_oracle"] == 0
        for row, label in zip(data.metadata, data.y, strict=True)
        if label == 0
    )
    report = data.generation_report
    assert report["scene_asset_manifest"]["complete"] is True
    assert report["scene_asset_manifest"]["asset_count"] >= 1
    assert report["accepted_counts"]["total"] == 4
    assert report["attempt_counts"]["depth_zero_oracle_calls"] == (
        report["attempt_counts"]["candidate_geometric_los"]
        + report["attempt_counts"]["candidate_geometric_nlos"]
    )
    assert report["attempt_counts"]["full_solver_empty_paths"] >= 0
