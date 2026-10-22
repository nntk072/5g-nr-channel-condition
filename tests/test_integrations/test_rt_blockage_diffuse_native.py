"""Native CPU/LLVM checks for blocked-direct and diffuse RT paths."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mitsuba")


def _trace_box_one_screen(
    *,
    tx_position: list[float],
    rx_position: list[float],
    scattering_coefficient: float,
):
    import mitsuba as mi

    mi.set_variant("llvm_ad_mono_polarized")
    rt = pytest.importorskip("sionna.rt")

    scene = rt.load_scene(rt.scene.box_one_screen, merge_shapes=False)
    scene.frequency = 3.5e9
    scene.get("screen").radio_material.scattering_coefficient = scattering_coefficient
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.add(rt.Transmitter("tx", tx_position))
    scene.add(rt.Receiver("rx", rx_position))
    scene.get("tx").look_at(scene.get("rx"))
    scene.get("rx").look_at(scene.get("tx"))

    direct = rt.PathSolver(deterministic=True)(
        scene,
        max_depth=0,
        los=True,
        specular_reflection=False,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
        seed=13,
    )
    full = rt.PathSolver(deterministic=True)(
        scene,
        max_depth=3,
        los=True,
        specular_reflection=True,
        diffuse_reflection=True,
        refraction=True,
        diffraction=False,
        samples_per_src=2000,
        seed=13,
    )
    valid = np.asarray(full.valid.numpy(), dtype=bool).reshape(-1)
    interactions = np.asarray(full.interactions.numpy()).reshape(full.interactions.shape[0], -1)
    diffuse_paths = np.any((interactions & rt.InteractionType.DIFFUSE) != 0, axis=0)
    return int(np.count_nonzero(direct.valid.numpy())), valid, diffuse_paths


@pytest.mark.native_sionna
def test_box_screen_separates_direct_visibility_from_indirect_paths_and_diffuse_scattering():
    same_side_direct, same_side_valid, same_side_diffuse = _trace_box_one_screen(
        tx_position=[-2.0, -1.0, 2.5],
        rx_position=[-2.0, 1.0, 2.5],
        scattering_coefficient=0.0,
    )
    blocked_direct, blocked_valid, blocked_diffuse = _trace_box_one_screen(
        tx_position=[-2.0, 0.0, 2.5],
        rx_position=[2.0, 0.0, 2.5],
        scattering_coefficient=0.0,
    )
    diffuse_direct, diffuse_valid, diffuse_paths = _trace_box_one_screen(
        tx_position=[-2.0, 0.0, 2.5],
        rx_position=[2.0, 0.0, 2.5],
        scattering_coefficient=0.5,
    )

    assert same_side_direct == 1
    assert np.count_nonzero(same_side_valid) > 1
    assert not np.any(same_side_diffuse & same_side_valid)
    assert blocked_direct == 0
    assert np.count_nonzero(blocked_valid) > 0
    assert not np.any(blocked_diffuse & blocked_valid)
    assert diffuse_direct == 0
    assert np.count_nonzero(diffuse_valid) > 0
    assert np.count_nonzero(diffuse_paths & diffuse_valid) > 0


@pytest.mark.native_sionna
def test_receiver_outside_closed_metal_box_has_no_solved_paths():
    direct, valid, diffuse = _trace_box_one_screen(
        tx_position=[0.0, -1.0, 2.5],
        rx_position=[6.0, 0.0, 2.5],
        scattering_coefficient=0.0,
    )

    assert direct == 0
    assert np.count_nonzero(valid) == 0
    assert np.count_nonzero(diffuse) == 0


@pytest.mark.native_sionna
def test_blocked_link_separates_single_and_multi_interaction_cfr_contributions():
    """A depth-three NLOS solve conserves its direct/order-resolved CFR sum."""
    import mitsuba as mi

    mi.set_variant("llvm_ad_mono_polarized")
    rt = pytest.importorskip("sionna.rt")
    from integrations.sionna.adapter import GridConfig, direct_path_mask, rt_cfr

    scene = rt.load_scene(rt.scene.box_one_screen, merge_shapes=False)
    scene.frequency = 3.5e9
    scene.get("screen").radio_material.scattering_coefficient = 0.0
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.add(rt.Transmitter("tx", [-2.0, 0.0, 2.5]))
    scene.add(rt.Receiver("rx", [2.0, 0.0, 2.5]))
    scene.get("tx").look_at(scene.get("rx"))
    scene.get("rx").look_at(scene.get("tx"))
    paths = rt.PathSolver(deterministic=True)(
        scene,
        max_depth=3,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=True,
        diffraction=False,
        samples_per_src=2000,
        seed=13,
    )

    valid = np.asarray(paths.valid.numpy(), dtype=bool)
    interactions = np.asarray(paths.interactions.numpy())
    interaction_count = np.count_nonzero(interactions.reshape(interactions.shape[0], -1), axis=0)
    assert interaction_count.shape == valid.reshape(-1).shape
    assert np.count_nonzero(valid) > 0
    assert not direct_path_mask(paths).any(), "the screen blocks the zero-interaction path"
    one_interaction = valid.reshape(-1) & (interaction_count == 1)
    multiple_interactions = valid.reshape(-1) & (interaction_count >= 2)
    assert one_interaction.any()
    assert multiple_interactions.any()

    grid = GridConfig(n_subcarriers=24, spacing_hz=30e3)
    full = rt_cfr(paths, grid)
    one_order = rt_cfr(paths, grid, path_mask=one_interaction.reshape(valid.shape))
    multiple_orders = rt_cfr(paths, grid, path_mask=multiple_interactions.reshape(valid.shape))
    np.testing.assert_allclose(full, one_order + multiple_orders, rtol=1e-5, atol=1e-8)
