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


@pytest.mark.native_sionna
def test_rt_direct_and_reflected_paths_sum_coherently_on_nr_grid():
    import mitsuba as mi

    mi.set_variant("llvm_ad_mono_polarized")
    rt = pytest.importorskip("sionna.rt")
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
