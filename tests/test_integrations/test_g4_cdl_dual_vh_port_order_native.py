"""Native Sionna PanelArray dual-VH port-order contract for frozen G4 families."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna.phy")

from sionna.phy.channel.tr38901 import PanelArray  # noqa: E402


@pytest.mark.native_sionna
def test_one_panel_two_element_dual_vh_native_port_order() -> None:
    array = PanelArray(
        num_rows_per_panel=1,
        num_cols_per_panel=2,
        polarization="dual",
        polarization_type="VH",
        antenna_pattern="omni",
        carrier_frequency=3.5e9,
        element_horizontal_spacing=0.5,
        device="cpu",
    )
    assert array.num_ant == 4
    assert array.ant_ind_pol1.tolist() == [0, 1]
    assert array.ant_ind_pol2.tolist() == [2, 3]
    assert torch.equal(array.ant_pos[0], array.ant_pos[2])
    assert torch.equal(array.ant_pos[1], array.ant_pos[3])
    assert array.ant_pos[0, 1] < array.ant_pos[1, 1]
