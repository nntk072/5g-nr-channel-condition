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
    path_gain = paths.a
    path_gain = np.asarray(path_gain[0].numpy()) + 1j * np.asarray(path_gain[1].numpy())
    valid_paths = np.asarray(paths.valid.numpy(), dtype=bool)
    wavelength_m = float(np.asarray(scene.wavelength.numpy()).reshape(-1)[0])
    valid_gains = path_gain[valid_paths]
    friis_amplitude = wavelength_m / (4 * np.pi * distance_m)
    assert valid_gains.size == 1
    np.testing.assert_allclose(np.abs(valid_gains[0]), friis_amplitude, rtol=1e-5, atol=0)
    assert abs(np.angle(valid_gains[0])) < 1e-5
    assert result.metadata["normalize_delays"] is False
    assert result.metadata["frequency_reference"] == "NR baseband subcarrier offsets"
    assert np.allclose(
        result.channel[0, 0, 0, 0, 0, 0].numpy(), result.channel[0, 0, 0, 0, 0, 13].numpy()
    )

    # A common physical timing-origin shift must remain visible in the
    # absolute-delay response, including both carrier phase and the baseband
    # subcarrier phase slope. Delay normalization deliberately removes it.
    delay_shift_s = 0.37e-9
    original_tau = paths.tau
    paths._tau = original_tau + delay_shift_s
    shifted_result = paths_to_nr_ofdm_channel(paths, grid)
    carrier_frequency_hz = float(np.asarray(scene.frequency.numpy()).item())
    expected_shift = np.exp(-2j * np.pi * (carrier_frequency_hz + frequencies_hz) * delay_shift_s)
    shifted_ratio = (
        shifted_result.channel[0, 0, 0, 0, 0, 0].numpy() / result.channel[0, 0, 0, 0, 0, 0].numpy()
    )
    np.testing.assert_allclose(shifted_ratio, expected_shift, rtol=2e-4, atol=2e-5)
    _, shifted_delays_s = paths.cir(normalize_delays=False, out_type="numpy")
    np.testing.assert_allclose(
        shifted_delays_s[0, 0, 0], delays_s[0, 0, 0] + delay_shift_s, atol=2e-13
    )
    normalized_before = paths.cfr(
        frequencies=frequencies_hz,
        num_time_steps=1,
        normalize_delays=True,
        normalize=False,
        out_type="numpy",
    )
    paths._tau = original_tau
    normalized_after = paths.cfr(
        frequencies=frequencies_hz,
        num_time_steps=1,
        normalize_delays=True,
        normalize=False,
        out_type="numpy",
    )
    np.testing.assert_allclose(normalized_before, normalized_after, rtol=1e-5, atol=1e-8)

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
@pytest.mark.parametrize("tx_azimuth_rad", [0.0, np.pi / 2.0])
def test_explicit_array_delays_and_cfr_match_oriented_element_geometry(tx_azimuth_rad):
    """Verify element ordering, orientation, spherical delay, and Friis phase."""
    import mitsuba as mi

    mi.set_variant("llvm_ad_mono_polarized")
    rt = pytest.importorskip("sionna.rt")

    carrier_hz = 3.5e9
    wavelength_m = speed_of_light / carrier_hz
    tx_center = np.asarray([0.0, 0.0, 0.0])
    rx_position = np.asarray([10.0, 1.0, 0.0])
    scene = rt.load_scene()
    scene.frequency = carrier_hz
    scene.tx_array = rt.PlanarArray(
        num_rows=1,
        num_cols=2,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.add(rt.Transmitter("tx", tx_center.tolist(), orientation=[tx_azimuth_rad, 0.0, 0.0]))
    scene.add(rt.Receiver("rx", rx_position.tolist()))
    paths = rt.PathSolver(deterministic=True)(
        scene,
        max_depth=0,
        los=True,
        specular_reflection=False,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
        synthetic_array=False,
    )

    valid = np.asarray(paths.valid.numpy(), dtype=bool)
    assert valid.shape == (1, 1, 1, 2, 1)
    assert valid.all()
    local_element_positions = (
        np.asarray([[0.0, -0.25, 0.0], [0.0, 0.25, 0.0]], dtype=np.float64) * wavelength_m
    )
    cosine = np.cos(tx_azimuth_rad)
    sine = np.sin(tx_azimuth_rad)
    rotation = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    element_positions = tx_center + local_element_positions @ rotation.T
    distances_m = np.linalg.norm(rx_position - element_positions, axis=1)
    expected_delays_s = distances_m / speed_of_light
    expected_gains = wavelength_m / (4.0 * np.pi * distances_m)
    expected_cfr = expected_gains * np.exp(-2j * np.pi * carrier_hz * expected_delays_s)

    delays_s = np.asarray(paths.tau.numpy(), dtype=np.float64)[0, 0, 0, :, 0]
    gains = (
        np.asarray(paths.a[0].numpy())[0, 0, 0, :, 0]
        + 1j * np.asarray(paths.a[1].numpy())[0, 0, 0, :, 0]
    )
    np.testing.assert_allclose(delays_s, expected_delays_s, rtol=0, atol=2e-12)
    np.testing.assert_allclose(np.abs(gains), expected_gains, rtol=2e-5, atol=0)

    cfr = paths.cfr(
        frequencies=np.asarray([0.0]),
        normalize_delays=False,
        normalize=False,
        out_type="numpy",
    )
    measured_cfr = cfr[0, 0, 0, :, 0, 0]
    np.testing.assert_allclose(measured_cfr, expected_cfr, rtol=1e-4, atol=1e-9)


@pytest.mark.native_sionna
@pytest.mark.parametrize(("polarization", "fresnel_mode"), [("V", "TM"), ("H", "TE")])
def test_rt_reflected_path_gain_matches_analytic_fresnel_slab(polarization, fresnel_mode):
    """V and H links match the analytic TM and TE slab coefficients, respectively."""
    import mitsuba as mi

    mi.set_variant("llvm_ad_mono_polarized")
    rt = pytest.importorskip("sionna.rt")
    from integrations.sionna.adapter import direct_path_mask
    from scipy.constants import c, epsilon_0

    frequency_hz = 3.5e9
    relative_permittivity = 4.0
    conductivity_s_per_m = 0.01
    slab_thickness_m = 0.1
    direct_distance_m = 20.0
    reflected_distance_m = 2.0 * np.sqrt(10.0**2 + 10.0**2)
    scene = rt.load_scene(rt.scene.simple_reflector, merge_shapes=False)
    scene.frequency = frequency_hz
    scene.tx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, pattern="iso", polarization=polarization
    )
    scene.rx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, pattern="iso", polarization=polarization
    )
    scene.add(rt.Transmitter("tx", [-10.0, 0.0, 10.0]))
    scene.add(rt.Receiver("rx", [10.0, 0.0, 10.0]))
    scene.get("tx").look_at([0.0, 0.0, 0.0])
    scene.get("rx").look_at([0.0, 0.0, 0.0])
    scene.get("reflector").radio_material = rt.RadioMaterial(
        name=f"analytic-{fresnel_mode.lower()}-fresnel-slab",
        relative_permittivity=relative_permittivity,
        conductivity=conductivity_s_per_m,
        thickness=slab_thickness_m,
        scattering_coefficient=0.0,
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

    valid = np.asarray(paths.valid.numpy(), dtype=bool).reshape(-1)
    direct = direct_path_mask(paths).reshape(-1)
    reflected = valid & ~direct
    assert np.count_nonzero(valid) == 2
    assert np.count_nonzero(direct) == np.count_nonzero(reflected) == 1
    gains = np.asarray(paths.a[0].numpy()).reshape(-1) + 1j * np.asarray(
        paths.a[1].numpy()
    ).reshape(-1)
    delays_s = np.asarray(paths.tau.numpy(), dtype=float).reshape(-1)
    np.testing.assert_allclose(delays_s[direct], direct_distance_m / c, rtol=0, atol=1e-12)
    np.testing.assert_allclose(delays_s[reflected], reflected_distance_m / c, rtol=0, atol=1e-12)
    wavelength_m = c / frequency_hz
    np.testing.assert_allclose(
        np.abs(gains[direct][0]), wavelength_m / (4.0 * np.pi * direct_distance_m), rtol=1e-5
    )

    # Independent textbook TE/TM Fresnel coefficient for a finite dielectric slab.
    cos_theta = 10.0 / np.sqrt(10.0**2 + 10.0**2)
    angular_frequency = 2.0 * np.pi * frequency_hz
    complex_permittivity = relative_permittivity - 1j * conductivity_s_per_m / (
        angular_frequency * epsilon_0
    )
    transmitted_cosine_term = np.sqrt(complex_permittivity - (1.0 - cos_theta**2))
    if fresnel_mode == "TE":
        interface_coefficient = (cos_theta - transmitted_cosine_term) / (
            cos_theta + transmitted_cosine_term
        )
    else:
        interface_coefficient = (complex_permittivity * cos_theta - transmitted_cosine_term) / (
            complex_permittivity * cos_theta + transmitted_cosine_term
        )
    slab_phase = 2.0 * np.pi * slab_thickness_m / wavelength_m * transmitted_cosine_term
    round_trip = np.exp(-2j * slab_phase)
    expected_coefficient = (
        interface_coefficient * (1.0 - round_trip) / (1.0 - interface_coefficient**2 * round_trip)
    )

    measured_coefficient = (
        gains[reflected][0] / gains[direct][0] * reflected_distance_m / direct_distance_m
    )
    np.testing.assert_allclose(measured_coefficient, expected_coefficient, rtol=2e-4, atol=2e-5)


@pytest.mark.native_sionna
@pytest.mark.parametrize("synthetic_array", [True, False])
@pytest.mark.parametrize(
    ("tx_polarization", "rx_polarization"),
    [("V", "V"), ("V", "VH"), ("VH", "V"), ("VH", "VH")],
)
def test_same_paths_direct_indirect_partition_preserves_mimo_axes(
    synthetic_array, tx_polarization, rx_polarization
):
    import mitsuba as mi

    mi.set_variant("llvm_ad_mono_polarized")
    rt = pytest.importorskip("sionna.rt")
    from integrations.sionna.adapter import GridConfig, direct_path_mask, rt_cfr
    from integrations.sionna.path_oracle import sionna_paths_to_oracle_table
    from sionna.rt.constants import InteractionType

    scene = rt.load_scene(rt.scene.simple_reflector, merge_shapes=False)
    scene.tx_array = rt.PlanarArray(
        num_rows=1, num_cols=2, pattern="iso", polarization=tx_polarization
    )
    scene.rx_array = rt.PlanarArray(
        num_rows=1, num_cols=3, pattern="iso", polarization=rx_polarization
    )
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
        synthetic_array=synthetic_array,
        seed=1,
    )

    valid = np.asarray(paths.valid.numpy(), dtype=bool)
    direct_mask = direct_path_mask(paths)
    indirect_mask = valid & ~direct_mask
    num_tx_antennas = 2 * (2 if tx_polarization == "VH" else 1)
    num_rx_antennas = 3 * (2 if rx_polarization == "VH" else 1)
    assert paths.a[0].shape[:4] == (1, num_rx_antennas, 1, num_tx_antennas)
    assert direct_mask.shape == valid.shape
    assert direct_mask.any() and indirect_mask.any()

    oracle = sionna_paths_to_oracle_table(paths, link_id="tx-rx", frame_index=3)
    assert oracle.doppler_hz is not None
    assert oracle.aoa_azimuth_elevation_rad is not None
    assert oracle.aod_azimuth_elevation_rad is not None
    assert oracle.path_kind is not None
    assert oracle.interaction_offsets is not None
    assert oracle.interaction_types is not None
    assert oracle.object_ids is not None
    assert oracle.primitive_ids is not None
    assert oracle.interaction_vertices_m is not None
    rx_count = int(paths.a[0].shape[1])
    tx_count = int(paths.a[0].shape[3])
    assert oracle.group_count == rx_count * tx_count
    tau = np.asarray(paths.tau.numpy())
    doppler = np.asarray(paths.doppler.numpy())
    theta_t = np.asarray(paths.theta_t.numpy())
    phi_t = np.asarray(paths.phi_t.numpy())
    theta_r = np.asarray(paths.theta_r.numpy())
    phi_r = np.asarray(paths.phi_r.numpy())
    path_valid = valid
    if valid.ndim == 3:
        path_valid = valid[:, None, :, None, :]
        tau = tau[:, None, :, None, :]
        doppler = doppler[:, None, :, None, :]
        theta_t = theta_t[:, None, :, None, :]
        phi_t = phi_t[:, None, :, None, :]
        theta_r = theta_r[:, None, :, None, :]
        phi_r = phi_r[:, None, :, None, :]
    path_valid = np.broadcast_to(path_valid, paths.a[0].shape)
    assert oracle.path_count == int(path_valid.sum())
    tau = np.broadcast_to(tau, paths.a[0].shape)
    doppler = np.broadcast_to(doppler, paths.a[0].shape)
    theta_t = np.broadcast_to(theta_t, paths.a[0].shape)
    phi_t = np.broadcast_to(phi_t, paths.a[0].shape)
    theta_r = np.broadcast_to(theta_r, paths.a[0].shape)
    phi_r = np.broadcast_to(phi_r, paths.a[0].shape)
    gains = paths.a[0].numpy() + 1j * paths.a[1].numpy()
    interactions = np.asarray(paths.interactions.numpy())
    objects = np.asarray(paths.objects.numpy())
    primitives = np.asarray(paths.primitives.numpy())
    vertices = np.asarray(paths.vertices.numpy())
    if interactions.ndim == 4:
        interactions = interactions[:, :, None, :, None, :]
        objects = objects[:, :, None, :, None, :]
        primitives = primitives[:, :, None, :, None, :]
        vertices = vertices[:, :, None, :, None, :, :]
    interactions = np.broadcast_to(interactions, (interactions.shape[0],) + paths.a[0].shape)
    objects = np.broadcast_to(objects, (objects.shape[0],) + paths.a[0].shape)
    primitives = np.broadcast_to(primitives, (primitives.shape[0],) + paths.a[0].shape)
    vertices = np.broadcast_to(vertices, (vertices.shape[0],) + paths.a[0].shape + (3,))
    for rx_port in range(rx_count):
        for tx_port in range(tx_count):
            group = rx_port * tx_count + tx_port
            mask = path_valid[0, rx_port, 0, tx_port]
            group_slice = oracle.group_slice(group)
            assert group_slice.stop - group_slice.start == int(mask.sum())
            np.testing.assert_allclose(
                oracle.delay_s[group_slice], tau[0, rx_port, 0, tx_port, mask]
            )
            np.testing.assert_allclose(
                oracle.complex_gain[group_slice], gains[0, rx_port, 0, tx_port, mask]
            )
            np.testing.assert_allclose(
                oracle.doppler_hz[group_slice], doppler[0, rx_port, 0, tx_port, mask]
            )
            expected_aoa = np.stack(
                (
                    (phi_r[0, rx_port, 0, tx_port, mask].astype(np.float64) + np.pi) % (2 * np.pi)
                    - np.pi,
                    np.pi / 2 - theta_r[0, rx_port, 0, tx_port, mask],
                ),
                axis=-1,
            )
            expected_aod = np.stack(
                (
                    (phi_t[0, rx_port, 0, tx_port, mask].astype(np.float64) + np.pi) % (2 * np.pi)
                    - np.pi,
                    np.pi / 2 - theta_t[0, rx_port, 0, tx_port, mask],
                ),
                axis=-1,
            )
            np.testing.assert_allclose(oracle.aoa_azimuth_elevation_rad[group_slice], expected_aoa)
            np.testing.assert_allclose(oracle.aod_azimuth_elevation_rad[group_slice], expected_aod)
            np.testing.assert_array_equal(
                oracle.interaction_order[group_slice],
                np.count_nonzero(
                    interactions[:, 0, rx_port, 0, tx_port, mask] != int(InteractionType.NONE),
                    axis=0,
                ),
            )
            expected_type_matrix = interactions[:, 0, rx_port, 0, tx_port][:, mask]
            expected_object_matrix = objects[:, 0, rx_port, 0, tx_port][:, mask]
            expected_primitive_matrix = primitives[:, 0, rx_port, 0, tx_port][:, mask]
            expected_vertex_tensor = vertices[:, 0, rx_port, 0, tx_port][:, mask]
            for local_path in range(int(mask.sum())):
                path_row = group_slice.start + local_path
                interaction_slice = slice(
                    int(oracle.interaction_offsets[path_row]),
                    int(oracle.interaction_offsets[path_row + 1]),
                )
                interaction_mask = expected_type_matrix[:, local_path] != int(InteractionType.NONE)
                np.testing.assert_array_equal(
                    oracle.interaction_types[interaction_slice],
                    expected_type_matrix[interaction_mask, local_path],
                )
                np.testing.assert_array_equal(
                    oracle.object_ids[interaction_slice],
                    expected_object_matrix[interaction_mask, local_path],
                )
                np.testing.assert_array_equal(
                    oracle.primitive_ids[interaction_slice],
                    expected_primitive_matrix[interaction_mask, local_path],
                )
                np.testing.assert_allclose(
                    oracle.interaction_vertices_m[interaction_slice],
                    expected_vertex_tensor[interaction_mask, local_path],
                )
            assert all(
                (kind == "direct") == (order == 0)
                for kind, order in zip(
                    oracle.path_kind[group_slice], oracle.interaction_order[group_slice]
                )
            )

    grid = GridConfig(n_subcarriers=24, spacing_hz=30e3)
    full = rt_cfr(paths, grid)
    direct = rt_cfr(paths, grid, path_mask=direct_mask)
    indirect = rt_cfr(paths, grid, path_mask=indirect_mask)
    assert full.shape == (1, num_tx_antennas * num_rx_antennas, 24)
    np.testing.assert_allclose(full, direct + indirect, rtol=1e-5, atol=1e-8)


@pytest.mark.native_sionna
@pytest.mark.parametrize("synthetic_array", [True, False])
def test_empty_paths_produce_finite_zero_cfr_for_both_array_modes(synthetic_array):
    import mitsuba as mi

    mi.set_variant("llvm_ad_mono_polarized")
    rt = pytest.importorskip("sionna.rt")
    from integrations.sionna.adapter import GridConfig, direct_path_mask, rt_cfr
    from integrations.sionna.path_oracle import sionna_paths_to_oracle_table

    scene = rt.load_scene()
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=2, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=3, pattern="iso", polarization="V")
    scene.add(rt.Transmitter("tx", [0.0, 0.0, 1.0]))
    scene.add(rt.Receiver("rx", [10.0, 0.0, 1.0]))
    paths = rt.PathSolver(deterministic=True)(
        scene,
        los=False,
        specular_reflection=False,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
        synthetic_array=synthetic_array,
    )

    valid = np.asarray(paths.valid.numpy(), dtype=bool)
    direct_mask = direct_path_mask(paths)
    assert valid.size == 0
    assert direct_mask.shape == valid.shape
    assert not direct_mask.any()
    oracle = sionna_paths_to_oracle_table(paths, link_id="empty-link")
    assert oracle.group_count == 6
    assert oracle.path_count == 0
    assert set(oracle.solve_status) == {"no_path"}

    grid = GridConfig(n_subcarriers=24, spacing_hz=30e3)
    full = rt_cfr(paths, grid)
    direct = rt_cfr(paths, grid, path_mask=direct_mask)
    indirect = rt_cfr(paths, grid, path_mask=valid & ~direct_mask)
    assert full.shape == direct.shape == indirect.shape == (1, 6, 24)
    assert np.all(np.isfinite(full))
    assert np.count_nonzero(full) == 0
    np.testing.assert_array_equal(full, direct + indirect)


@pytest.mark.native_sionna
def test_path_oracle_selects_explicit_multi_device_link():
    import mitsuba as mi

    mi.set_variant("llvm_ad_mono_polarized")
    rt = pytest.importorskip("sionna.rt")
    from integrations.sionna.path_oracle import sionna_paths_to_oracle_table

    scene = rt.load_scene()
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.add(rt.Transmitter("tx0", [0.0, 0.0, 1.0]))
    scene.add(rt.Transmitter("tx1", [2.0, 0.0, 1.0]))
    scene.add(rt.Receiver("rx0", [10.0, 0.0, 1.0]))
    scene.add(rt.Receiver("rx1", [12.0, 0.0, 1.0]))
    paths = rt.PathSolver(deterministic=True)(
        scene,
        max_depth=0,
        los=True,
        specular_reflection=False,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
        synthetic_array=True,
    )

    oracle = sionna_paths_to_oracle_table(
        paths,
        link_id="tx0-to-rx1",
        rx_device_index=1,
        tx_device_index=0,
    )
    assert oracle.group_count == 1
    assert oracle.path_count == int(np.count_nonzero(paths.valid.numpy()[1, 0]))
    expected_mask = np.asarray(paths.valid.numpy()[1, 0], dtype=bool)
    np.testing.assert_allclose(oracle.delay_s, paths.tau.numpy()[1, 0, expected_mask])
    expected_gain = paths.a[0].numpy()[1, 0, 0, 0] + 1j * paths.a[1].numpy()[1, 0, 0, 0]
    np.testing.assert_allclose(oracle.complex_gain, expected_gain[expected_mask])


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
